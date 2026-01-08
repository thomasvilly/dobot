import os
import time
import cv2
import numpy as np
import h5py
import glob
import random
import DobotDllType as dType

# --- CONFIGURATION ---
DATASET_DIR = "dataset_hdf5/interactive_session_v2"
CAM_INDEX = 1
EXPOSURE_VAL = -6
BLOCK_COLOR = "blue"

# Workspace Constraints (Dobot Magician)
# Z_SAFE: Height to hover/transport
# Z_PICK: Height to grab the block (tuned to your -75 setup)
Z_SAFE = -50.0   
Z_PICK = -75.0  
Z_HOVER = 0.0

# Areas
PICK_ZONE  = {"x": (180, 240), "y": (-120, 0)}
PLACE_ZONE = {"x": (180, 240), "y": (80, 200)}
HOME_ZONE = {"x": (180, 240), "y": (0, 80)}

# Recording Settings
RECORDING_HZ = 5      # Capture 15 frames per second
MOVE_DURATION = 3.0    # Seconds to move between points (Transport)
ACTION_DURATION = 1.0  # Seconds to move up/down (Pick/Place)

# OpenVLA Standards
SUCTION_ON = 1.0
SUCTION_OFF = -1.0

# Safety Limits
SAFETY = {
    "x_min": 140, "x_max": 280,
    "y_min": -250, "y_max": 250,
    "z_min": -100, "z_max": 150,
    "r_min": -150, "r_max": 150
}

# --- GLOBALS ---
api = dType.load()
cam = None

def initialize():
    global cam, api
    print("--- Initializing Hardware ---")

    # 1. Camera
    print(f"Opening Camera (Index {CAM_INDEX})...")
    cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if EXPOSURE_VAL != 0:
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
        cam.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VAL)
    
    if not cam.isOpened():
        print("[CRITICAL] Camera Failed! Exiting.")
        exit()
    
    for _ in range(10): cam.read()
    print(f"--> Camera Ready.")

    # 2. Robot
    print("Connecting to Dobot...")
    state = dType.ConnectDobot(api, "", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("[ERROR] Robot Connect Failed!")
        cam.release()
        exit()
    
    # STOP and CLEAR queue on startup (Best Practice from your "Correct Code")
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    
    # Set velocity/accel for queued moves
    dType.SetPTPCommonParams(api, 30, 30, isQueued=1)

    print("--> Hardware Ready.")

def set_suction_hardware(val, queued=1):
    """
    queued=1: Adds command to buffer (happens in sync with motion)
    queued=0: Happens immediately (for Ghost Pointer/Setup)
    """
    enable = 1
    suck = 1 if val > 0 else 0
    
    # We return the index so we can track it if needed
    return dType.SetEndEffectorSuctionCup(api, enable, suck, isQueued=queued)[0]

def get_real_state():
    pose = dType.GetPose(api) # [x, y, z, r, j1, j2, j3, j4]
    
    # Get suction state from hardware check (or memory)
    # Note: Dobot DLL doesn't always return suction state perfectly in GetPose
    # So we often trust the command, but let's try reading:
    grip_resp = dType.GetEndEffectorSuctionCup(api)
    suck_state = SUCTION_ON if grip_resp[0] == 1 else SUCTION_OFF
    
    return list(pose[0:4]) + [0.0, 0.0, suck_state]

# --- S-CURVE TRAJECTORY GENERATION ---
def generate_s_curve(start, end, steps):
    """
    Generates a path from start to end using Cosine interpolation.
    This creates a smooth velocity profile (Starts slow, fast middle, ends slow).
    Input: start [x,y,z,r], end [x,y,z,r], steps (int)
    Output: List of [x,y,z,r] arrays
    """
    start = np.array(start)
    end = np.array(end)
    
    # Cosine S-Curve: 0.0 to 1.0
    t = np.linspace(0, 1, steps)
    s_curve = (1 - np.cos(t * np.pi)) / 2
    
    path = []
    for s in s_curve:
        # Interpolate
        point = start + (end - start) * s
        path.append(point)
        
    return path

# --- PLANNING ---
def plan_dynamic_episode(home_loc, pick_loc, place_loc):
    """
    Pre-calculates the ENTIRE episode trajectory (Pick + Place).
    Returns a list of 'waypoints', where each waypoint is:
    {'pos': [x,y,z,r], 'suction': float}
    """
    full_plan = []
    
    # Helper to add segments
    def add_segment(start, end, duration_s, suction_val):
        steps = int(duration_s * RECORDING_HZ)
        points = generate_s_curve(start, end, steps)
        for p in points:
            full_plan.append({'pos': p, 'suction': suction_val})

    # Define Key Positions (x, y, z, r)
    # We keep R=0 for simplicity, but you can change it
    p_home = [home_loc[0], home_loc[1], Z_HOVER, 0.0]
    p_pick_hover  = [pick_loc[0],  pick_loc[1],  Z_SAFE, 0.0]
    p_pick_down   = [pick_loc[0],  pick_loc[1],  Z_PICK, 0.0]
    p_place_hover = [place_loc[0], place_loc[1], Z_HOVER, 0.0]
    p_place_down  = [place_loc[0], place_loc[1], Z_SAFE, 0.0]

    add_segment(p_home, p_pick_hover, ACTION_DURATION, SUCTION_OFF)

    # --- SEGMENT 1: APPROACH (Hover -> Pick Down) ---
    add_segment(p_pick_hover, p_pick_down, ACTION_DURATION, SUCTION_OFF)
    
    # --- GRASP WAIT (Stay at bottom, turn suction ON) ---
    # Add a few frames of "staying still" to ensure grasp
    for _ in range(int(0.5 * RECORDING_HZ)):
        full_plan.append({'pos': p_pick_down, 'suction': SUCTION_ON})

    # --- SEGMENT 2: LIFT (Pick Down -> Pick Hover) ---
    add_segment(p_pick_down, p_pick_hover, ACTION_DURATION, SUCTION_ON)

    # --- SEGMENT 3: TRANSPORT (Pick Hover -> Place Hover) ---
    add_segment(p_pick_hover, p_place_hover, MOVE_DURATION, SUCTION_ON)

    # --- SEGMENT 4: LOWER (Place Hover -> Place Down) ---
    add_segment(p_place_hover, p_place_down, ACTION_DURATION, SUCTION_ON)

    # --- RELEASE WAIT (Stay at bottom, turn suction OFF) ---
    for _ in range(int(0.5 * RECORDING_HZ)):
        full_plan.append({'pos': p_place_down, 'suction': SUCTION_OFF})

    # --- SEGMENT 5: RETREAT (Place Down -> Place Hover) ---
    add_segment(p_place_down, p_place_hover, ACTION_DURATION, SUCTION_OFF)

    return full_plan

def run_ghost_pointer_workflow():
    """
    Safe 'Ring Buffer' Streaming. 
    Prevents 'Bricking' by ensuring we never send more commands than the robot can hold.
    """
    print("\n=== NEW EPISODE SETUP ===")
    
    # [Setup Code Identical to before...] 
    # 1. Generate Random Positions
    pick_x = random.uniform(*PICK_ZONE["x"])
    pick_y = random.uniform(*PICK_ZONE["y"])
    place_x = random.uniform(*PLACE_ZONE["x"])
    place_y = random.uniform(*PLACE_ZONE["y"])
    home_x = random.uniform(*HOME_ZONE["x"])
    home_y = random.uniform(*HOME_ZONE["y"])
    
    pick_target  = [pick_x, pick_y]
    place_target = [place_x, place_y]
    home_target  = [home_x, home_y]

    # [Ghost Pointer Display Code - KEEP THIS THE SAME AS BEFORE]
    print("--> Showing PICK location...")
    set_suction_hardware(SUCTION_OFF, queued=0)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, pick_x, pick_y, Z_SAFE, 0, isQueued=0)
    time.sleep(2.0)
    input(f"--> Place {BLOCK_COLOR} block under suction cup. Press ENTER.")
    
    print("--> Showing PLACE location...")
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, place_x, place_y, Z_SAFE, 0, isQueued=0)
    time.sleep(2.0)
    input(f"--> Place plate here. Press ENTER.")

    # Move to Home (Immediate)
    print("--> Moving to Start Position...")
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, home_x, home_y, Z_HOVER, 0, isQueued=0)
    time.sleep(1.0) 

    # --- THE CRITICAL FIX START ---
    print("--> Generating Plan...")
    plan = plan_dynamic_episode(home_target, pick_target, place_target)
    total_steps = len(plan)
    
    # Clear Queue before starting
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    
    buffer = []
    
    # RING BUFFER STATE
    sent_cursor = 0      # How many commands we have sent to the robot
    BUFFER_LIMIT = 1    # Keep max 20 commands in robot memory (Safe Zone)
    
    print("--> Starting Active Streaming...")
    dType.SetQueuedCmdStartExec(api) # Start the queue processor immediately
    
    # STREAMING LOOP
    while True:
        loop_start = time.time()
        
        # 1. Get Robot Progress
        # current_cmd_index is the command the robot is *currently* executing
        current_cmd_index = dType.GetQueuedCmdCurrentIndex(api)[0]
        
        # 2. Fill the Buffer (Spoon feed)
        # While we have commands left in the plan AND there is space in the robot's brain
        while (sent_cursor < total_steps) and (sent_cursor - current_cmd_index < BUFFER_LIMIT):
            
            # Send the next waypoint
            waypoint = plan[sent_cursor]
            target_pos = waypoint['pos']
            target_suction = waypoint['suction']
            
            x = np.clip(target_pos[0], SAFETY["x_min"], SAFETY["x_max"])
            y = np.clip(target_pos[1], SAFETY["y_min"], SAFETY["y_max"])
            z = np.clip(target_pos[2], SAFETY["z_min"], SAFETY["z_max"])
            
            # Send Motion (Queued)
            # We ignore the return index for now, we rely on the count
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, 0, isQueued=1)
            
            # Send Suction (Queued)
            set_suction_hardware(target_suction, queued=1)
            
            sent_cursor += 1 # Mark as sent
            
        # 3. Capture Data (Continuous)
        # We only record if the robot is actually working (index > 0)
        # or if we just started.
        ret, frame = cam.read()
        if ret:
            real_state = get_real_state()
            action_vec = real_state 
            
            buffer.append({
                'top': frame,
                'state': real_state,
                'action': action_vec
            })
            
        # 4. Exit Condition
        # If we have sent everything AND the robot has finished everything
        if (sent_cursor >= total_steps) and (current_cmd_index >= sent_cursor):
            print("--> All commands executed.")
            break
            
        # 5. Loop Timing (5Hz)
        elapsed = time.time() - loop_start
        dt = 1.0 / RECORDING_HZ
        if dt > elapsed:
            time.sleep(dt - elapsed)

    # Cleanup
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    set_suction_hardware(SUCTION_OFF, queued=0)
    
    print(f"--> Episode Complete. Captured {len(buffer)} frames.")
    return buffer

def save_hdf5(buffer):
    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)
    
    existing = glob.glob(os.path.join(DATASET_DIR, "episode_*.h5"))
    next_id = len(existing) + 1
    filename = os.path.join(DATASET_DIR, f"episode_{next_id:03d}.h5")
    
    print(f"--> Saving to {filename}...")
    
    imgs = [b['top'] for b in buffer]
    states = [b['state'] for b in buffer]
    actions = [b['action'] for b in buffer]
    
    with h5py.File(filename, 'w') as f:
        f.attrs['instruction'] = f"Pick up the {BLOCK_COLOR} block and place it on the plate"
        obs = f.create_group('observations')
        obs.create_dataset('images/top', data=np.array(imgs), compression="gzip")
        obs.create_dataset('state', data=np.array(states))
        f.create_dataset('action', data=np.array(actions))
        
    print("--> Saved.")

# --- MAIN ---
def main():
    initialize()
    
    while True:
        buffer = run_ghost_pointer_workflow()
        
        # The "Augmented Resetting" Feedback Loop
        ans = input("\nWas this episode SUCCESSFUL? [Y]es / [N]o / [Q]uit: ").upper()
        
        if ans == 'Y':
            save_hdf5(buffer)
        elif ans == 'Q':
            dType.DisconnectDobot(api)
            cam.release()
            break
        else:
            print("--> Discarding data (Failure).")

if __name__ == "__main__":
    main()