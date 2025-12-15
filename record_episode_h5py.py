import os
import json
import time
import cv2
import numpy as np
import argparse
import glob
import h5py
import DobotDllType as dType

# --- CONFIGURATION & SAFETY BOUNDS ---
# Prevents robot from hitting table or base. Adjust these to your physical setup!
SAFETY_LIMITS = {
    "x_min": -150.0, "x_max": 250.0,  # Don't hit robot base / don't over-extend
    "y_min": -200.0, "y_max": 200.0, # Side-to-side reach
    "z_min": -100.0,  "z_max": 150.0, # Table height (0 is roughly base height, -60 is usually table)
    "r_min": -150.0, "r_max": 100.0
}

# 7-DOF PADDING (Standardizing 4DOF Dobot to 7DOF VLA format)
# Indices: 0:X, 1:Y, 2:Z, 3:Roll(Dobot R), 4:Pitch(Dummy), 5:Yaw(Dummy), 6:Gripper
dummy_pitch = 0.0
dummy_yaw = 0.0

# --- CLI ARGUMENT PARSER ---
parser = argparse.ArgumentParser(description="Dobot HDF5 Data Collector")
parser.add_argument("--job", type=str, required=True, help="Name of the JSON file inside /job_recipes (without .json)")
args = parser.parse_args()

# --- GLOBALS ---
RECIPE_DIR = "job_recipes"
DATASET_DIR = "dataset_hdf5"
api = dType.load()
cam1 = None  # Top Down
cam2 = None  # Side View (Fake Wrist)

def get_next_episode_path(job_id):
    """Finds the next sequential filename: episode_001.h5, episode_002.h5..."""
    base_dir = os.path.join(DATASET_DIR, job_id)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return os.path.join(base_dir, "episode_001.h5")
    
    existing = glob.glob(os.path.join(base_dir, "episode_*.h5"))
    ids = []
    for path in existing:
        try:
            # Extract number from "episode_005.h5"
            filename = os.path.basename(path)
            num = int(filename.replace("episode_", "").replace(".h5", ""))
            ids.append(num)
        except ValueError:
            pass
            
    next_id = max(ids) + 1 if ids else 1
    return os.path.join(base_dir, f"episode_{next_id:03d}.h5")

def check_safety(x, y, z, r):
    """Clamps target to safe workspace bounds."""
    orig = (x, y, z, r)
    x = np.clip(x, SAFETY_LIMITS["x_min"], SAFETY_LIMITS["x_max"])
    y = np.clip(y, SAFETY_LIMITS["y_min"], SAFETY_LIMITS["y_max"])
    z = np.clip(z, SAFETY_LIMITS["z_min"], SAFETY_LIMITS["z_max"])
    r = np.clip(r, SAFETY_LIMITS["r_min"], SAFETY_LIMITS["r_max"])
    
    if (x,y,z,r) != orig:
        print(f"!!! SAFETY CLAMP TRIGGERED !!! Requested {orig} -> Clamped to {(x,y,z,r)}")
    return x, y, z, r

def initialize_hardware():
    global cam1, cam2
    print("\n--- Connecting to Hardware ---")
    
    # 1. Connect Robot
    state = dType.ConnectDobot(api, "", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("CRITICAL ERROR: Could not connect to Dobot.")
        exit()
    
    dType.ClearAllAlarmsState(api)
    dType.SetQueuedCmdClear(api)
    dType.SetPTPCommonParams(api, 100, 100, isQueued=0) # Velocity/Accel %
    
    # # 2. Homing
    # print("Homing Robot (Please wait ~20s)...")
    # dType.SetHOMECmd(api, temp=0, isQueued=1)
    # dType.SetQueuedCmdStartExec(api)
    # time.sleep(20) # Simple wait for homing to finish
    # 2. Homing
    print("Moving to Safe Start Position...")
    move_robot_block(200, 0, 50, 0)

    # 3. Open Cameras (Top + Side)
    # NOTE: Adjust indices (0, 1, 2) based on your USB ports
    print("Opening Cameras...")
    cam1 = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Top View
    cam2 = cv2.VideoCapture(2, cv2.CAP_DSHOW) # Side View (Treat as Wrist)
    
    # Set Resolution (Optional, keeps file size manageable)
    cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cam1.isOpened() or not cam2.isOpened():
        print("CRITICAL ERROR: Cameras failed to open.")
        dType.DisconnectDobot(api)
        exit()
        
    # Warmup buffer
    for _ in range(5): 
        cam1.read(); cam2.read(); time.sleep(0.1)
    print("Hardware Ready.")

def move_robot_block(x, y, z, r):
    """Moves robot and waits until arrival (Blocking)."""
    x, y, z, r = check_safety(x, y, z, r)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, r, isQueued=0)
    
    # Wait loop
    target = np.array([x, y, z])
    start_t = time.time()
    while True:
        pose = dType.GetPose(api)
        curr = np.array(pose[0:3])
        if np.linalg.norm(curr - target) < 2.0: break # Arrived within 2mm
        if time.time() - start_t > 5.0: break # Timeout safety
        time.sleep(0.005)

def save_episode_hdf5(filepath, buffer_data, instruction):
    """
    Writes the RAM buffer to HDF5.
    Structure follows a standard VLA-compatible format.
    """
    print(f"Saving episode to {filepath}...")
    
    # Unpack buffer
    # buffer_data is list of dicts: {'top': img, 'side': img, 'state': [...], 'action': [...]}
    n_steps = len(buffer_data)
    
    # Arrays to store
    img_top_arr = []
    img_side_arr = []
    state_arr = []  # Proprioception
    action_arr = [] # Target Action
    
    for step in buffer_data:
        img_top_arr.append(step['top'])
        img_side_arr.append(step['side'])
        state_arr.append(step['state'])
        action_arr.append(step['action'])
        
    with h5py.File(filepath, 'w') as f:
        # Metadata
        f.attrs['instruction'] = instruction
        f.attrs['sim'] = False
        f.attrs['robot'] = "dobot_magician"
        
        # Observations Group
        obs = f.create_group('observations')
        
        # Images (Compressed)
        # Note: 'top' -> mapped to 'full_image', 'side' -> mapped to 'wrist_image'
        obs.create_dataset('images/top', data=np.array(img_top_arr), compression="gzip")
        obs.create_dataset('images/side', data=np.array(img_side_arr), compression="gzip")
        
        # Robot State (Proprioception)
        obs.create_dataset('state', data=np.array(state_arr))
        
        # Actions (The "Label")
        f.create_dataset('action', data=np.array(action_arr))
        
    print("Save Complete.")

def interpolate_path(start, end, step_mm):
    start = np.array(start)
    end = np.array(end)
    dist = np.linalg.norm(end - start)
    if dist < 1.0: return [end.tolist()]
    steps = int(dist / step_mm)
    if steps < 1: return [end.tolist()]
    return np.linspace(start, end, steps).tolist()

def run_episode(recipe_path):
    # 1. Load User Recipe
    with open(recipe_path, 'r') as f:
        job = json.load(f)
    
    instruction = job['instruction']
    settings = job['settings']
    episode_path = get_next_episode_path(job['job_id'])
    
    print(f"\n>>> STARTING: {job['job_id']}")
    print(f">>> File: {episode_path}")
    print(f">>> Task: '{instruction}'")
    
    # RAM Buffer (Prevent corruption by writing only at end)
    data_buffer = []
    current_gripper = 0.0 # 0.0 = Open, 1.0 = Closed
    
    # 2. Execute Segments
    for segment in job['segments']:
        seg_desc = segment.get('desc', '')
        print(f"Executing: {seg_desc}")
        
        if segment['type'] == 'action':
            # Handle Gripper
            current_gripper = float(segment['suction'])
            dType.SetEndEffectorSuctionCup(api, 1, int(current_gripper), isQueued=0)
            time.sleep(0.5) # Allow suction to engage
            
            # Record a few frames of "staying still" to capture the state change
            # This helps the model learn that "action command" = "gripper closes"
            curr_pose = dType.GetPose(api) # [x,y,z,r,joint1,joint2,joint3,joint4]
            curr_xyzr = list(curr_pose[0:4])
            
            # 7-DOF Padded State [x, y, z, r, pitch, yaw, gripper]
            state_vec = curr_xyzr + [dummy_pitch, dummy_yaw, current_gripper]
            
            for _ in range(5):
                ret1, frame1 = cam1.read()
                ret2, frame2 = cam2.read()
                if not ret1 or not ret2: continue
                
                data_buffer.append({
                    'top': frame1,
                    'side': frame2,
                    'state': state_vec,
                    'action': state_vec # Action is "stay here and hold gripper"
                })
                time.sleep(0.05)
            
        elif segment['type'] == 'move':
            # Interpolated Move
            start_pos = dType.GetPose(api)[0:4]
            end_pos = segment['target']
            
            # Generate waypoints
            path = interpolate_path(start_pos, end_pos, settings['step_size_mm'])
            
            for i in range(len(path)):
                target = path[i] # This is where we want to go (The Action)
                
                # Move Physical Robot
                move_robot_block(*target)
                
                # Capture Data AFTER move
                ret1, frame1 = cam1.read()
                ret2, frame2 = cam2.read()
                
                # Get ACTUAL pose (Proprioception)
                curr_pose = dType.GetPose(api)
                curr_xyzr = list(curr_pose[0:4])
                
                # Construct 7-DOF Vectors
                # State: Where I am NOW
                state_vec = curr_xyzr + [dummy_pitch, dummy_yaw, current_gripper]
                
                # Action: Where I am going NEXT (or where I just tried to go)
                # In VLA, action usually means "target for next timestep". 
                # For simplicity in this script, Action = The Target Coordinate we just moved to.
                # (You can shift this by -1 index in the dataloader if needed)
                target_xyzr = list(target)
                action_vec = target_xyzr + [dummy_pitch, dummy_yaw, current_gripper]
                
                if ret1 and ret2:
                    data_buffer.append({
                        'top': frame1,
                        'side': frame2,
                        'state': state_vec,
                        'action': action_vec
                    })
                
                # Show Feed
                cv2.imshow("Recorder", cv2.resize(frame1, (0,0), fx=0.5, fy=0.5))
                cv2.waitKey(1)
                
                # Rate limit
                time.sleep(settings['wait_time_sec'])

    # 3. Save Data
    save_episode_hdf5(episode_path, data_buffer, instruction)
    print("Episode Done.")

if __name__ == "__main__":
    recipe_file = os.path.join(RECIPE_DIR, f"{args.job}.json")
    if not os.path.exists(recipe_file):
        print(f"Error: Recipe file not found: {recipe_file}")
        exit()

    try:
        initialize_hardware()
        run_episode(recipe_file)
    except KeyboardInterrupt:
        print("\nAborted by user.")
    finally:
        # Cleanup
        dType.SetEndEffectorSuctionCup(api, 0, 0, isQueued=0)
        dType.DisconnectDobot(api)
        if cam1: cam1.release()
        if cam2: cam2.release()
        cv2.destroyAllWindows()