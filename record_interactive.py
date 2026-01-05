import os
import time
import cv2
import numpy as np
import h5py
import glob
import random
import DobotDllType as dType

# --- CONFIGURATION ---
DATASET_DIR = "dataset_hdf5/interactive_session"
CAM_INDEX = 1  # Using Index 1 (matches your working test)
EXPOSURE_VAL = -6
BLOCK_COLOR = "blue" 

# Workspace constraints
# Z set to -75 as requested
TARGET_BOX = {"x": (100, 200), "y": (30, 130), "z": -75} 
START_BOX  = {"x": (40, 100), "y": (-30, 30), "z": 50}  

# Task Settings
STEP_SIZE_MM = 10.0
NOISE_MM = 2.0
WAIT_TIME = 0.1

# Safety Limits 
SAFETY = {
    "x_min": 140, "x_max": 280,
    "y_min": -200, "y_max": 200,
    "z_min": -100, "z_max": 150, # Adjusted z_min to -100
    "r_min": -150, "r_max": 150
}

# --- GLOBALS ---
api = dType.load()
cam = None

# --- HARDWARE INTERFACE ---
def initialize():
    global cam, api
    print("--- Initializing Hardware ---")

    # 1. OPEN CAMERA FIRST
    print(f"Opening Top Camera (Index {CAM_INDEX})...")
    cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    
    # Apply settings
    if EXPOSURE_VAL != 0:
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
        cam.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VAL)
    
    if not cam.isOpened():
        print("[CRITICAL] Camera Failed to Open! Exiting.")
        exit()
    
    ret, _ = cam.read()
    if not ret:
        print("[CRITICAL] Camera opened but returned no frame.")
        exit()
    print(f"--> Camera Ready (Exposure: {EXPOSURE_VAL})")

    # 2. CONNECT ROBOT SECOND
    print("Connecting to Robot...")
    state = dType.ConnectDobot(api, "", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("[ERROR] Robot Connect Failed!")
        cam.release()
        exit()
    
    dType.ClearAllAlarmsState(api)
    dType.SetPTPCommonParams(api, 100, 100, isQueued=0)

    # 3. HOMING SEQUENCE
    print("--> Homing Robot (This takes 20 seconds)...")
    dType.SetHOMECmd(api, temp=0, isQueued=1)
    dType.SetQueuedCmdStartExec(api)
    time.sleep(20) 
    dType.SetQueuedCmdClear(api)
    print("--> Homing Complete.")
    print("Hardware Ready.\n")

def reconnect_camera():
    global cam
    if cam is not None:
        cam.release()
    
    print(f"Connecting to Camera Index {CAM_INDEX}...")
    cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    cam.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VAL)
    
    if not cam.isOpened():
        print("[WARN] Camera open failed. Will retry in loop.")
    else:
        for _ in range(5): cam.read()
        print("Camera Reconnected.")

def move_robot(x, y, z, r):
    x = np.clip(x, SAFETY["x_min"], SAFETY["x_max"])
    y = np.clip(y, SAFETY["y_min"], SAFETY["y_max"])
    z = np.clip(z, SAFETY["z_min"], SAFETY["z_max"])
    r = np.clip(r, SAFETY["r_min"], SAFETY["r_max"])
    
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, r, isQueued=0)
    
    target = np.array([x,y,z])
    start = time.time()
    while time.time() - start < 5.0:
        curr = np.array(dType.GetPose(api)[0:3])
        if np.linalg.norm(curr - target) < 2.0: break
        time.sleep(0.01)

def set_gripper(enable: bool, suck: bool = False):
    is_sucked = 1 if suck else 0
    is_enabled = 1 if enable else 0
    dType.SetEndEffectorSuctionCup(api, is_enabled, is_sucked, isQueued=0)
    time.sleep(0.2)

def get_state():
    pose = dType.GetPose(api) 
    grip_resp = dType.GetEndEffectorSuctionCup(api)
    grip = float(grip_resp[0]) if isinstance(grip_resp, list) else float(grip_resp)
    return list(pose[0:4]) + [0.0, 0.0, grip]

# --- PATH GENERATION ---
def generate_task():
    tx = random.uniform(*TARGET_BOX["x"])
    ty = random.uniform(*TARGET_BOX["y"])
    tz = TARGET_BOX["z"]
    
    sx = random.uniform(*START_BOX["x"])
    sy = random.uniform(*START_BOX["y"])
    sz = START_BOX["z"]
    return [sx, sy, sz, 0.0], [tx, ty, tz, 0.0]

def generate_noisy_path(start, end):
    start = np.array(start)
    end = np.array(end)
    dist = np.linalg.norm(end - start)
    steps = int(max(dist / STEP_SIZE_MM, 2))
    
    raw_path = np.linspace(start, end, steps)
    
    noisy_path = []
    for i, pt in enumerate(raw_path):
        if i == 0 or i == (len(raw_path) - 1):
            noisy_path.append(pt) 
        else:
            jitter = np.random.normal(0, NOISE_MM, 4)
            jitter[3] = 0 
            noisy_path.append(pt + jitter)
            
    return noisy_path

# --- CORE RECORDING ---
def execute_episode(start_pos, target_pos):
    buffer = []
    path = generate_noisy_path(start_pos, target_pos)
    
    # --- STEP 1: SHOW TARGET & WAIT ---
    print("\n--> Moving to TARGET...")
    
    # NEW: Add +10mm to Z so you can fit the block under
    show_pos = list(target_pos)
    show_pos[2] += 10.0 
    move_robot(*show_pos)
    
    # Force Release (Blow)
    set_gripper(enable=True, suck=False)
    time.sleep(0.5)
    set_gripper(enable=False)
    
    input(f"--> Target shown (Z={show_pos[2]:.1f}). Place {BLOCK_COLOR} block. Press ENTER.")

    # --- STEP 2: MOVE TO START & AUTO-START ---
    print("--> Moving to START...")
    move_robot(*start_pos)
    
    print("--> Get Ready... Recording in 2 seconds...")
    time.sleep(2.0) 

    # Discards frames to clear the stale buffer
    for _ in range(10):
        cam.read()
    
    print("--> Recording Path...")
    
    # --- STEP 3: EXECUTE APPROACH ---
    for i in range(len(path)):
        target_waypoint = path[i]
        
        ret, frame = cam.read()
        if not ret: 
            print("[WARN] Frame drop. Reconnecting...")
            reconnect_camera()
            ret, frame = cam.read()
            if not ret: continue

        current_state_vec = get_state()
        action_vec = list(target_waypoint) + [0.0, 0.0, 0.0] 
        
        # Only saving TOP image now
        buffer.append({
            'top': frame, 
            'state': current_state_vec, 
            'action': action_vec
        })
        
        move_robot(*target_waypoint)
        time.sleep(WAIT_TIME)
        
    # --- STEP 4: GRASP ---
    ret, frame = cam.read()
    current_state_vec = get_state()
    grip_action_vec = list(target_pos) + [0.0, 0.0, 1.0] 
    
    buffer.append({
        'top': frame,
        'state': current_state_vec, 
        'action': grip_action_vec
    })
    
    print("--> Grasping...")
    set_gripper(enable=True, suck=True)
    time.sleep(0.5)
    
    # --- STEP 5: SMOOTH LIFT ---
    print("--> Lifting Smoothly...")
    LIFT_HEIGHT = 50.0
    lift_steps = int(LIFT_HEIGHT / STEP_SIZE_MM)
    z_path = np.linspace(target_pos[2], target_pos[2] + LIFT_HEIGHT, lift_steps)
    
    for z_next in z_path:
        lift_waypoint = list(target_pos)
        lift_waypoint[2] = z_next
        
        ret, frame = cam.read()
        if not ret: continue
        
        current_state_vec = get_state()
        action_vec = list(lift_waypoint) + [0.0, 0.0, 1.0] 
        
        buffer.append({
            'top': frame,
            'state': current_state_vec, 
            'action': action_vec
        })
        
        move_robot(*lift_waypoint)
        time.sleep(0.05)
    
    return buffer

def save_hdf5(buffer):
    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)
    
    existing = glob.glob(os.path.join(DATASET_DIR, "episode_*.h5"))
    next_id = len(existing) + 1
    filename = os.path.join(DATASET_DIR, f"episode_{next_id:03d}.h5")
    
    print(f"--> Saving to {filename}...")
    
    imgs = [b['top'] for b in buffer]
    # Removed side array
    states = [b['state'] for b in buffer]
    actions = [b['action'] for b in buffer]
    
    with h5py.File(filename, 'w') as f:
        f.attrs['instruction'] = f"Pick up the {BLOCK_COLOR} block" 
        obs = f.create_group('observations')
        obs.create_dataset('images/top', data=np.array(imgs), compression="gzip")
        # Removed images/side dataset
        obs.create_dataset('state', data=np.array(states))
        f.create_dataset('action', data=np.array(actions))
        
    print("--> Saved.")

# --- MAIN LOOP ---
def main():
    initialize()
    
    while True:
        print("\n=== GENERATING NEW TASK ===")
        start_pos, target_pos = generate_task()
        
        while True:
            buffer = execute_episode(start_pos, target_pos)
            
            cmd = input("\n[S]ave | [D]elete & Retry | [N]ew Path | [Q]uit: ").upper()
            
            if cmd == 'S':
                save_hdf5(buffer)
                break 
            elif cmd == 'D':
                print("--> Retrying SAME path...")
                continue 
            elif cmd == 'N':
                print("--> Discarding path.")
                break 
            elif cmd == 'Q':
                dType.DisconnectDobot(api)
                if cam: cam.release()
                return

if __name__ == "__main__":
    main()