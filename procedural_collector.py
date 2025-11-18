import os
import json
import time
import cv2
import numpy as np
import argparse
import glob

# Dobot Setup
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import DobotDllType as dType

# --- CLI ARGUMENT PARSER ---
parser = argparse.ArgumentParser(description="Dobot Data Collector")
parser.add_argument("--job", type=str, required=True, help="Name of the JSON file inside /job_recipes (without .json)")
args = parser.parse_args()

# --- CONSTANTS & GLOBALS ---
RECIPE_DIR = "job_recipes"
DATASET_DIR = "dataset"
api = dType.load()
cam1 = None
cam2 = None

def get_episode_path(job_id):
    """
    Checks the dataset folder, finds the highest episode number,
    and returns the path for the NEXT episode.
    """
    base_dir = os.path.join(DATASET_DIR, job_id)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return os.path.join(base_dir, "episode_001")
    
    # Find all existing episode folders
    existing = glob.glob(os.path.join(base_dir, "episode_*"))
    if not existing:
        return os.path.join(base_dir, "episode_001")
    
    # Extract numbers: "episode_005" -> 5
    ids = []
    for path in existing:
        try:
            ids.append(int(path.split("_")[-1]))
        except ValueError:
            pass
            
    next_id = max(ids) + 1 if ids else 1
    return os.path.join(base_dir, f"episode_{next_id:03d}")

def initialize_hardware():
    global cam1, cam2
    print("\n--- Connecting to Hardware ---")
    
    # Robot
    state = dType.ConnectDobot(api, "", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("CRITICAL ERROR: Could not connect to Dobot.")
        exit()
    
    dType.ClearAllAlarmsState(api)
    dType.SetQueuedCmdClear(api)
    dType.SetPTPCommonParams(api, 100, 100, isQueued=0)
    
    # Homing
    print("Homing Robot (Wait ~20s)...")
    dType.SetHOMECmd(api, temp=0, isQueued=1)
    dType.SetQueuedCmdStartExec(api)
    # Wait for homing (simple sleep is robust enough here)
    time.sleep(20)

    # Cameras
    print("Opening Cameras...")
    cam1 = cv2.VideoCapture(1, cv2.CAP_DSHOW) 
    cam2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    
    if not cam1.isOpened() or not cam2.isOpened():
        print("CRITICAL ERROR: Cameras failed to open.")
        exit()
        
    # Warmup
    for _ in range(10): 
        cam1.read(); cam2.read(); time.sleep(0.05)
    print("Hardware Ready.")

def move_to_absolute(x, y, z, r):
    """Move and block until arrival."""
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, r, isQueued=0)
    target = np.array([x, y, z])
    start = time.time()
    while True:
        pose = dType.GetPose(api)
        curr = np.array(pose[0:3])
        if np.linalg.norm(curr - target) < 2.0: break
        if time.time() - start > 4.0: break
        time.sleep(0.005)

def capture_and_save(save_dir, step_index, next_target, segment_desc, instruction, gripper_state):
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()
    
    if not ret1 or not ret2: return

    # Get current state
    pose = dType.GetPose(api)
    curr_xyzr = list(pose[0:4])
    
    # CALCULATE ACTION (The "Label")
    # Vector from Current -> Next Target
    action_vector = np.subtract(next_target, curr_xyzr).tolist()

    # Filenames
    base_name = f"step_{step_index:04d}"
    img1_path = f"{base_name}_cam1.jpg"
    img2_path = f"{base_name}_cam2.jpg"
    
    cv2.imwrite(os.path.join(save_dir, img1_path), frame1)
    cv2.imwrite(os.path.join(save_dir, img2_path), frame2)
    
    # JSON Payload (OpenVLA Ready)
    data = {
        "language_instruction": instruction,  # <-- CRITICAL FOR VLA
        "current_pose": curr_xyzr,
        "action": {
            "delta_vector": action_vector,
            "gripper_closed": gripper_state,
            "terminate_episode": 0
        },
        "next_pose_absolute": next_target,
        "segment_desc": segment_desc,
        "images": {
            "cam1": img1_path,
            "cam2": img2_path
        }
    }
    
    with open(os.path.join(save_dir, f"{base_name}_data.json"), 'w') as f:
        json.dump(data, f, indent=2)
    
    # Visualization
    viz_frame = np.hstack((cv2.resize(frame1, (0,0), fx=0.5, fy=0.5), 
                           cv2.resize(frame2, (0,0), fx=0.5, fy=0.5)))
    cv2.putText(viz_frame, f"Step: {step_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(viz_frame, segment_desc, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
    cv2.imshow("Data Collector", viz_frame)
    cv2.waitKey(1)
    
    print(f"Step {step_index:03d} | Action: {np.round(action_vector, 1)} | {segment_desc}")

def interpolate_path(start, end, step_mm):
    start = np.array(start)
    end = np.array(end)
    dist = np.linalg.norm(end - start)
    if dist < 1.0: return [end.tolist()]
    steps = int(dist / step_mm)
    if steps < 1: return [end.tolist()]
    return np.linspace(start, end, steps).tolist()

def run_episode(recipe_path):
    # 1. Load Recipe
    with open(recipe_path, 'r') as f:
        job = json.load(f)
    
    instruction = job['instruction']
    settings = job['settings']
    
    # 2. Setup Episode Folder
    episode_dir = get_episode_path(job['job_id'])
    os.makedirs(episode_dir)
    print(f"\n>>> STARTING RECORDING: {job['job_id']}")
    print(f">>> Output Folder: {episode_dir}")
    print(f">>> Instruction: '{instruction}'")
    
    global_step = 0
    current_gripper = 0 # 0=Open, 1=Closed
    
    # 3. Execute Segments
    for segment in job['segments']:
        desc = segment.get('desc', 'Moving')
        
        if segment['type'] == 'action':
            # Update Gripper State
            current_gripper = segment['suction']
            dType.SetEndEffectorSuctionCup(api, 1, current_gripper, isQueued=0)
            print(f" -- Action: Gripper -> {current_gripper}")
            time.sleep(0.5)
            
        elif segment['type'] == 'move':
            # Plan Path
            start_pos = dType.GetPose(api)[0:4]
            end_pos = segment['target']
            path = interpolate_path(start_pos, end_pos, settings['step_size_mm'])
            
            # Execute Path
            for i in range(len(path) - 1):
                next_target = path[i+1]
                
                # Move
                move_to_absolute(*path[i])
                time.sleep(settings['wait_time_sec'])
                
                # Record
                capture_and_save(episode_dir, global_step, next_target, desc, instruction, current_gripper)
                global_step += 1
            
            # Finalize Segment
            move_to_absolute(*end_pos)

    print(f"\nEpisode Complete. Saved {global_step} frames.")

if __name__ == "__main__":
    # Locate Recipe
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
        dType.SetEndEffectorSuctionCup(api, 0, 0, isQueued=0)
        dType.DisconnectDobot(api)
        if cam1: cam1.release()
        if cam2: cam2.release()
        cv2.destroyAllWindows()