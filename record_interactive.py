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
Z_SAFE = 50.0   
Z_PICK = -75.0  

# Areas
PICK_ZONE  = {"x": (180, 240), "y": (-50, 50)}
PLACE_ZONE = {"x": (180, 240), "y": (100, 180)}

# Recording Settings
RECORDING_HZ = 15      # Capture 15 frames per second
MOVE_DURATION = 3.0    # Seconds to move between points (Transport)
ACTION_DURATION = 1.0  # Seconds to move up/down (Pick/Place)

# OpenVLA Standards
SUCTION_ON = 1.0
SUCTION_OFF = -1.0

# Safety Limits
SAFETY = {
    "x_min": 140, "x_max": 280,
    "y_min": -200, "y_max": 200,
    "z_min": -100, "z_max": 150,
    "r_min": -150, "r_max": 150
}

# --- GLOBALS ---
api = dType.load()
cam = None

# --- HARDWARE INTERFACE ---
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
    
    # Warmup
    for _ in range(10): cam.read()
    print(f"--> Camera Ready.")

    # 2. Robot
    print("Connecting to Dobot...")
    state = dType.ConnectDobot(api, "", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("[ERROR] Robot Connect Failed!")
        cam.release()
        exit()
    
    dType.ClearAllAlarmsState(api)
    dType.SetPTPCommonParams(api, 100, 100, isQueued=0)

    print("--> Hardware Ready. (Assuming Robot is already Homed)")

def set_suction_hardware(val):
    """
    Maps -1.0/1.0 to actual Dobot suction commands.
    val > 0: SUCK (Pick)
    val <= 0: BLOW/OFF (Release)
    """
    if val > 0:
        # Enable=1, Suck=1
        dType.SetEndEffectorSuctionCup(api, 1, 1, isQueued=0)
    else:
        # Enable=1, Suck=0 (Blow to release)
        dType.SetEndEffectorSuctionCup(api, 1, 0, isQueued=0)

def get_real_state():
    pose = dType.GetPose(api) # [x, y, z, r, j1, j2, j3, j4]
    
    # Get suction state from hardware check (or memory)
    # Note: Dobot DLL doesn't always return suction state perfectly in GetPose
    # So we often trust the command, but let's try reading:
    grip_resp = dType.GetEndEffectorSuctionCup(api)
    # Logic: if enabled(1) and sucked(1) -> 1.0, else -1.0
    suck_state = SUCTION_ON if (grip_resp[0] == 1 and grip_resp[1] == 1) else SUCTION_OFF
    
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
def plan_dynamic_episode(pick_loc, place_loc):
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
    p_pick_hover  = [pick_loc[0],  pick_loc[1],  Z_SAFE, 0.0]
    p_pick_down   = [pick_loc[0],  pick_loc[1],  Z_PICK, 0.0]
    p_place_hover = [place_loc[0], place_loc[1], Z_SAFE, 0.0]
    p_place_down  = [place_loc[0], place_loc[1], Z_PICK, 0.0]

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

# --- EXECUTION ---
def run_ghost_pointer_workflow():
    """
    The Human-Friendly "Show me" loop.
    """
    print("\n=== NEW EPISODE SETUP ===")
    
    # 1. Generate Random Positions
    pick_x = random.uniform(*PICK_ZONE["x"])
    pick_y = random.uniform(*PICK_ZONE["y"])
    
    place_x = random.uniform(*PLACE_ZONE["x"])
    place_y = random.uniform(*PLACE_ZONE["y"])
    
    pick_target  = [pick_x, pick_y, Z_PICK]
    place_target = [place_x, place_y, Z_PICK]

    # 2. GHOST POINTER: Show Pick Location
    print("--> Showing PICK location...")
    set_suction_hardware(SUCTION_OFF)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, pick_x, pick_y, Z_SAFE, 0, isQueued=0)
    time.sleep(1.5) # Wait for move
    
    # Lower slightly to indicate "Here"
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, pick_x, pick_y, Z_SAFE - 20, 0, isQueued=0)
    input(f"--> Place {BLOCK_COLOR} block under the suction cup. Press ENTER when ready.")
    
    # Move back up to safe
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, pick_x, pick_y, Z_SAFE, 0, isQueued=0)

    # 3. GHOST POINTER: Show Place Location
    print("--> Showing PLACE location...")
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, place_x, place_y, Z_SAFE, 0, isQueued=0)
    time.sleep(1.5)
    print("--> (I will place it here)")
    time.sleep(0.5)

    # 4. Return to Start Position (Pick Hover) to begin
    print("--> Moving to Start Position...")
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, pick_x, pick_y, Z_SAFE, 0, isQueued=0)
    time.sleep(2.0)
    
    # 5. Flush Camera Buffer
    for _ in range(5): cam.read()

    # 6. GENERATE PLAN & EXECUTE
    print("--> Planning Smooth Trajectory...")
    plan = plan_dynamic_episode(pick_target, place_target)
    
    print(f"--> Starting Recording ({len(plan)} steps at {RECORDING_HZ}Hz)... GO!")
    
    buffer = []
    
    # THE STREAMING LOOP (Non-Blocking)
    dt = 1.0 / RECORDING_HZ
    
    for step_idx, waypoint in enumerate(plan):
        loop_start = time.time()
        
        # A. Send Command (Async)
        target_pos = waypoint['pos']
        target_suction = waypoint['suction']
        
        # Safety Clip
        x = np.clip(target_pos[0], SAFETY["x_min"], SAFETY["x_max"])
        y = np.clip(target_pos[1], SAFETY["y_min"], SAFETY["y_max"])
        z = np.clip(target_pos[2], SAFETY["z_min"], SAFETY["z_max"])
        
        # Send Motion
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, 0, isQueued=0)
        
        # Send Suction (Only if it changes, or just spam it safely)
        set_suction_hardware(target_suction)

        # B. Capture Data
        ret, frame = cam.read()
        if not ret:
            print("[WARN] dropped frame")
            continue
            
        real_state = get_real_state()
        
        # Action is the TARGET we just sent + Suction Command
        # Format: [x, y, z, r, 0, 0, suction] -> 7 DOF
        action_vec = [x, y, z, 0.0, 0.0, 0.0, target_suction]
        
        buffer.append({
            'top': frame,
            'state': real_state,
            'action': action_vec
        })
        
        # C. Frequency Control
        elapsed = time.time() - loop_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    # Stop Suction at end just in case
    set_suction_hardware(SUCTION_OFF)
    print("--> Episode Complete.")
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
        f.attrs['instruction'] = f"Pick up the {BLOCK_COLOR} block and place it on the target"
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