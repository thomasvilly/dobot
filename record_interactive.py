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
CAM_INDEX = 1  # Top-Down Camera Index
EXPOSURE_VAL = -5 # <--- NEW: Hardcoded Exposure

# Workspace constraints (Safe box for random generation)
# Adjust these to match your table!
TARGET_BOX = {"x": (200, 240), "y": (-50, 50), "z": -60}  # Where block appears
START_BOX  = {"x": (160, 190), "y": (-80, 80), "z": 50}   # Where robot starts

# Task Settings
STEP_SIZE_MM = 10.0   # Distance between waypoints
NOISE_MM = 5.0        # How much to "jitter" the path (The Overfit Fix)
WAIT_TIME = 0.2       # Pause between steps (for clean images)

# Safety Limits (Hard Clamps)
SAFETY = {
    "x_min": 140, "x_max": 280,
    "y_min": -200, "y_max": 200,
    "z_min": -65,  "z_max": 150,
    "r_min": -150, "r_max": 150
}

# --- GLOBALS ---
api = dType.load()
cam = None

# --- HARDWARE INTERFACE ---
def initialize():
    global cam, api
    print("--- Initializing Hardware ---")
    state = dType.ConnectDobot(api, "", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("[ERROR] Robot Connect Failed!")
        exit()
    
    dType.ClearAllAlarmsState(api)
    dType.SetPTPCommonParams(api, 100, 100, isQueued=0)
    
    print(f"Opening Top Camera (Index {CAM_INDEX})...")
    cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # --- NEW: FORCE EXPOSURE SETTINGS ---
    # 1. Disable Auto Exposure (1 = Manual)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    # 2. Set Exposure Value
    cam.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VAL)
    print(f"--> Exposure Locked to {EXPOSURE_VAL}")
    # ------------------------------------
    
    if not cam.isOpened():
        print("[ERROR] Camera Failed! Check USB.")
        dType.DisconnectDobot(api)
        exit()
        
    # Warmup
    for _ in range(5): cam.read()
    print("Hardware Ready.\n")

def move_robot(x, y, z, r):
    # Clamp Safety
    x = np.clip(x, SAFETY["x_min"], SAFETY["x_max"])
    y = np.clip(y, SAFETY["y_min"], SAFETY["y_max"])
    z = np.clip(z, SAFETY["z_min"], SAFETY["z_max"])
    r = np.clip(r, SAFETY["r_min"], SAFETY["r_max"])
    
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, r, isQueued=0)
    
    # Block until arrival
    target = np.array([x,y,z])
    start = time.time()
    while time.time() - start < 5.0:
        curr = np.array(dType.GetPose(api)[0:3])
        if np.linalg.norm(curr - target) < 2.0: break
        time.sleep(0.01)

def set_gripper(enable: bool):
    state = 1 if enable else 0
    dType.SetEndEffectorSuctionCup(api, 1, state, isQueued=0)
    time.sleep(0.2)

def get_state():
    # Returns 7-DOF [x,y,z,r, 0, 0, grip]
    pose = dType.GetPose(api) # [x,y,z,r, j1, j2, j3, j4]
    grip = float(dType.GetEndEffectorSuctionCup(api))
    return list(pose[0:4]) + [0.0, 0.0, grip]

# --- PATH GENERATION ---
def generate_task():
    """Generates random Start and Target positions."""
    tx = random.uniform(*TARGET_BOX["x"])
    ty = random.uniform(*TARGET_BOX["y"])
    tz = TARGET_BOX["z"]
    
    sx = random.uniform(*START_BOX["x"])
    sy = random.uniform(*START_BOX["y"])
    sz = START_BOX["z"]
    
    return [sx, sy, sz, 0.0], [tx, ty, tz, 0.0]

def generate_noisy_path(start, end):
    """Generates waypoints with Gaussian noise."""
    start = np.array(start)
    end = np.array(end)
    dist = np.linalg.norm(end - start)
    steps = int(max(dist / STEP_SIZE_MM, 2))
    
    # Linear Interpolation
    raw_path = np.linspace(start, end, steps)
    
    # Inject Noise (Except at Start and End)
    noisy_path = []
    for i, pt in enumerate(raw_path):
        if i == 0 or i == (len(raw_path) - 1):
            noisy_path.append(pt) # Keep anchors precise
        else:
            # Add random jitter (mm) to X, Y, Z
            jitter = np.random.normal(0, NOISE_MM, 4)
            jitter[3] = 0 # Don't jitter rotation
            noisy_path.append(pt + jitter)
            
    return noisy_path

# --- CORE RECORDING ---
def execute_episode(start_pos, target_pos):
    buffer = []
    path = generate_noisy_path(start_pos, target_pos)
    
    print("--> Resetting to Start...")
    set_gripper(False)
    move_robot(*start_pos)
    input("--> Place GREEN Block at Target. Remove Hands. Press ENTER.")
    
    print("--> Recording...")
    
    for i in range(len(path)):
        target_waypoint = path[i]
        
        ret, frame = cam.read()
        if not ret: print("[WARN] Frame drop"); continue
        
        current_state_vec = get_state()
        action_vec = list(target_waypoint) + [0.0, 0.0, 0.0] 
        
        buffer.append({
            'top': frame,
            'state': current_state_vec,
            'action': action_vec
        })
        
        move_robot(*target_waypoint)
        time.sleep(WAIT_TIME)
        
    # Grasp Action
    ret, frame = cam.read()
    current_state_vec = get_state()
    grip_action_vec = list(target_pos) + [0.0, 0.0, 1.0] 
    
    buffer.append({
        'top': frame,
        'state': current_state_vec, 'action': grip_action_vec
    })
    
    print("--> Grasping...")
    set_gripper(True)
    time.sleep(0.5)
    
    # Lift
    lift_pos = list(target_pos)
    lift_pos[2] += 50 
    move_robot(*lift_pos)
    
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
        # --- NEW: UPDATED INSTRUCTION ---
        f.attrs['instruction'] = "Pick up the green block" 
        # --------------------------------
        obs = f.create_group('observations')
        obs.create_dataset('images/top', data=np.array(imgs), compression="gzip")
        obs.create_dataset('state', data=np.array(states))
        f.create_dataset('action', data=np.array(actions))
        
    print("--> Saved.")

# --- MAIN LOOP ---
def main():
    initialize()
    
    while True:
        print("\n=== GENERATING NEW TASK ===")
        start_pos, target_pos = generate_task()
        
        print(f"Target: {target_pos}")
        move_robot(*target_pos)
        
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
                return

if __name__ == "__main__":
    main()