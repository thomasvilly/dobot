import os
import time
import cv2
import numpy as np
import h5py
import glob
import random
import DobotDllType as dType

# --- CONFIGURATION ---
DATASET_DIR = "dataset_hdf5/interactive_session_fixed"
CAM_INDEX = 1
EXPOSURE_VAL = -6
BLOCK_COLOR = "blue"

# Workspace
Z_SAFE = -40.0   
Z_PICK = -75.0  
Z_HOVER = 50.0 

PICK_ZONE  = {"x": (80, 160), "y": (-120, 0)}
PLACE_ZONE = {"x": (140, 220), "y": (80, 200)}
HOME_ZONE  = {"x": (120, 180), "y": (0, 80)}

RECORDING_HZ = 10
MOVE_DURATION = 3.0
ACTION_DURATION = 1.0

SAFETY = {
    "x_min": 140, "x_max": 280,
    "y_min": -250, "y_max": 250,
    "z_min": -100, "z_max": 150,
    "r_min": -150, "r_max": 150
}

api = dType.load()
cam = None

# --- HELPERS ---
def move_wait(x, y, z):
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    time.sleep(0.5) # Wait for Clear
    
    last_idx = dType.SetCPCmd(api, 1, x, y, z, 100.0, isQueued=1)[0]
    dType.SetQueuedCmdStartExec(api)
    
    while dType.GetQueuedCmdCurrentIndex(api)[0] < last_idx:
        time.sleep(0.1)
        
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)

def initialize():
    global cam, api
    print("--- Initializing ---")
    
    cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if EXPOSURE_VAL != 0:
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
        cam.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VAL)
    if not cam.isOpened(): exit()
    for _ in range(10): cam.read()

    state = dType.ConnectDobot(api, "", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("[ERROR] Connect Failed")
        exit()

    dType.ClearAllAlarmsState(api)
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    time.sleep(1.0)
    
    # LOWERED SPEED to 20 to fix Jitter/Fast Move
    dType.SetCPParams(api, 50, 100, 50, 1, isQueued=1)
    
    print("--> Ready.")

def set_suction_hardware(val, queued=1):
    enable = 1
    suck = 1 if val > 0 else 0
    return dType.SetEndEffectorSuctionCup(api, enable, suck, isQueued=queued)[0]

def get_real_state():
    pose = dType.GetPose(api)
    grip_resp = dType.GetEndEffectorSuctionCup(api)
    if isinstance(grip_resp, list) and len(grip_resp) > 0:
        suck_state = 1.0 if grip_resp[0] == 1 else -1.0
    else:
        suck_state = -1.0
    return list(pose[0:4]) + [0.0, 0.0, suck_state]

def generate_s_curve(start, end, steps, accel_fraction=0.15):
    """
    Trapezoidal Velocity Profile (The "Table Top").
    
    Args:
        start, end: Coordinates
        steps: Total number of points
        accel_fraction: 0.15 means 15% accel, 70% flat speed, 15% decel.
                        Smaller number = Steeper acceleration (flatter top).
    """
    start = np.array(start)
    end = np.array(end)
    
    # 1. Define the number of steps for each phase
    # Ensure at least 1 step for accel/decel to avoid errors
    n_accel = max(1, int(steps * accel_fraction))
    n_decel = max(1, int(steps * accel_fraction))
    n_flat  = steps - n_accel - n_decel
    
    # 2. Build the Velocity Profile (The "Table")
    # / (Ramp Up)
    v_ramp_up = np.linspace(0, 1, n_accel)
    # - (Flat Top)
    v_flat    = np.ones(n_flat)
    # \ (Ramp Down)
    v_ramp_down = np.linspace(1, 0, n_decel)
    
    # Combine them
    velocity_profile = np.concatenate([v_ramp_up, v_flat, v_ramp_down])
    
    # 3. Integrate Velocity to get Position (The Path)
    # Cumulative sum creates the S-shape from the trapezoid
    position_profile = np.cumsum(velocity_profile)
    
    # 4. Normalize exactly to 0.0 -> 1.0 range
    s_curve = (position_profile - position_profile.min()) / (position_profile.max() - position_profile.min())
    
    # Safety: Force exact start/end
    s_curve[0] = 0.0
    s_curve[-1] = 1.0

    # 5. Interpolate (Just in case rounding errors changed the length)
    if len(s_curve) != steps:
        old_x = np.linspace(0, 1, len(s_curve))
        new_x = np.linspace(0, 1, steps)
        s_curve = np.interp(new_x, old_x, s_curve)

    # 6. Apply to coordinates
    path = []
    for s in s_curve:
        path.append(start + (end - start) * s)
        
    return path

def plan_dynamic_episode(home_loc, pick_loc, place_loc):
    full_plan = []
    
    def add_segment(start, end, duration_s, suction_val):
        steps = int(duration_s * RECORDING_HZ)
        steps = max(2, steps) 
        points = generate_s_curve(start, end, steps)
        for p in points:
            full_plan.append({'pos': p, 'suction': suction_val})

    p_home        = [home_loc[0], home_loc[1], Z_HOVER, 0.0]
    p_pick_hover  = [pick_loc[0], pick_loc[1], Z_SAFE, 0.0]
    p_pick_down   = [pick_loc[0], pick_loc[1], Z_PICK, 0.0]
    p_place_hover = [place_loc[0], place_loc[1], Z_SAFE, 0.0]
    p_place_down  = [place_loc[0], place_loc[1], Z_SAFE, 0.0]
    p_place_done = [place_loc[0], place_loc[1], Z_HOVER-20, 0.0]

    add_segment(p_home, p_pick_hover, ACTION_DURATION, -1.0)
    add_segment(p_pick_hover, p_pick_down, ACTION_DURATION, -1.0)
    for _ in range(int(1.0 * RECORDING_HZ)):
        full_plan.append({'pos': p_pick_down, 'suction': 1.0})
    add_segment(p_pick_down, p_pick_hover, ACTION_DURATION, 1.0)
    add_segment(p_pick_hover, p_place_hover, MOVE_DURATION, 1.0)
    add_segment(p_place_hover, p_place_down, ACTION_DURATION, 1.0)
    for _ in range(int(0.5 * RECORDING_HZ)):
        full_plan.append({'pos': p_place_down, 'suction': -1.0})
    add_segment(p_place_down, p_place_done, ACTION_DURATION, -1.0)

    return full_plan

def run_safe_workflow():
    print("\n=== NEW EPISODE ===")
    
    pick_x = random.uniform(*PICK_ZONE["x"])
    pick_y = random.uniform(*PICK_ZONE["y"])
    place_x = random.uniform(*PLACE_ZONE["x"])
    place_y = random.uniform(*PLACE_ZONE["y"])
    home_x = random.uniform(*HOME_ZONE["x"])
    home_y = random.uniform(*HOME_ZONE["y"])
    
    print("--> Showing Pick...")
    set_suction_hardware(-1.0, queued=1) 
    move_wait(pick_x, pick_y, Z_SAFE)    
    input("Place Block. Press ENTER.")
    
    print("--> Showing Place...")
    move_wait(place_x, place_y, Z_SAFE)  
    input("Check Zone. Press ENTER.")
    
    print("--> Going Home...")
    move_wait(home_x, home_y, Z_HOVER)   
    
    plan = plan_dynamic_episode([home_x, home_y], [pick_x, pick_y], [place_x, place_y])
    
    # 4. STREAM (Ring Buffer)
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    # CRITICAL: Wait for reset
    time.sleep(1.0)
    
    buffer = []
    total_steps = len(plan)
    sent_cursor = 0
    BUFFER_LIMIT = 20
    
    # We track the ACTUAL index the API gives us for the last command
    last_queued_api_index = 0

    print("--> Clearing Camera Buffer... (Move hand away!)")
    time.sleep(1.0) # Give you time to retreat
    for _ in range(10): # Flush old frames (aggressive flush)
        cam.read()
    
    print("--> Executing...")
    dType.SetQueuedCmdStartExec(api)
    
    while True:
        loop_start = time.time()
        
        current_idx = dType.GetQueuedCmdCurrentIndex(api)[0]
        
        # Feed Buffer
        while sent_cursor < total_steps and (last_queued_api_index - current_idx < BUFFER_LIMIT):
            wp = plan[sent_cursor]
            x = np.clip(wp['pos'][0], SAFETY["x_min"], SAFETY["x_max"])
            y = np.clip(wp['pos'][1], SAFETY["y_min"], SAFETY["y_max"])
            z = np.clip(wp['pos'][2], SAFETY["z_min"], SAFETY["z_max"])
            
            # Store the Return Value (The Ticket Number)
            last_queued_api_index = dType.SetCPCmd(api, 1, x, y, z, 100.0, isQueued=1)[0]
            set_suction_hardware(wp['suction'], queued=1)
            
            sent_cursor += 1
            
        ret, frame = cam.read()
        if ret:
            state = get_real_state()
            buffer.append({'top': frame, 'state': state, 'action': state})
            
        # DEBUG PRINT (Helps you see if it's counting)
        if sent_cursor % 10 == 0:
            print(f"   [Step {sent_cursor}/{total_steps}] Robot Index: {current_idx} / {last_queued_api_index}", end='\r')

        # Exit Condition: 
        # We verify 'current_idx' against the REAL 'last_queued_api_index'
        if sent_cursor >= total_steps and current_idx >= last_queued_api_index:
            print(f"\n--> Success. Final Index: {current_idx}")
            break
            
        elapsed = time.time() - loop_start
        dt = 1.0 / RECORDING_HZ
        if dt > elapsed: time.sleep(dt - elapsed)
            
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    print(f"--> Done. Captured {len(buffer)} frames.")
    return buffer

def save_hdf5(buffer):
    if not buffer: return
    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)
    
    next_id = len(glob.glob(os.path.join(DATASET_DIR, "episode_*.h5"))) + 1
    fname = os.path.join(DATASET_DIR, f"episode_{next_id:03d}.h5")
    
    print(f"--> Saving {fname}...")
    with h5py.File(fname, 'w') as f:
        f.create_dataset('observations/images/top', data=np.array([b['top'] for b in buffer]), compression="gzip")
        f.create_dataset('observations/state', data=np.array([b['state'] for b in buffer]))
        f.create_dataset('action', data=np.array([b['action'] for b in buffer]))
    print("Saved.")

def main():
    initialize()
    while True:
        buf = run_safe_workflow()
        ans = input("Save? [Y/N/Q]: ").upper()
        if ans == 'Y': save_hdf5(buf)
        elif ans == 'Q': break
    
    dType.DisconnectDobot(api)

if __name__ == "__main__":
    main()