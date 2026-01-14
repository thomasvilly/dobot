import os
import time
import cv2
import numpy as np
import h5py
import glob
import random
import DobotDllType as dType

# --- CONFIGURATION ---
DATASET_DIR = "dataset_hdf5/simple_session"
CAM_INDEX = 1
EXPOSURE_VAL = -6

# Workspace
Z_SAFE = -40.0   
Z_PICK = -75.0  
Z_HOVER = 50.0 

PICK_ZONE  = {"x": (80, 160), "y": (-120, 0)}
PLACE_ZONE = {"x": (140, 220), "y": (80, 200)}
HOME_ZONE  = {"x": (120, 180), "y": (0, 80)}

api = dType.load()
cam = None

def initialize():
    global cam, api
    print("--- Initializing ---")
    
    cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if EXPOSURE_VAL != 0:
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
        cam.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VAL)
    
    state = dType.ConnectDobot(api, "", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("[ERROR] Connect Failed")
        exit()

    dType.ClearAllAlarmsState(api)
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    
    # Standard PTP Speed
    dType.SetPTPCommonParams(api, 30, 30, isQueued=1) 
    dType.SetPTPJumpParams(api, 20, 100, isQueued=1)
    
    print("--> Ready.")

def set_suction(val, queued=1):
    return dType.SetEndEffectorSuctionCup(api, 1, 1 if val > 0 else 0, isQueued=queued)[0]

def move_ptp(x, y, z, mode=dType.PTPMode.PTPMOVJXYZMode):
    # Just queues the command, returns index
    return dType.SetPTPCmd(api, mode, x, y, z, 0, isQueued=1)[0]

def move_and_wait(x, y, z):
    """
    Setup moves: Queue -> Execute -> Wait for Finish.
    """
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    
    last_idx = move_ptp(x, y, z)
    dType.SetQueuedCmdStartExec(api)
    
    # Wait for the specific index to finish
    while dType.GetQueuedCmdCurrentIndex(api)[0] < last_idx:
        time.sleep(0.1)
        
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)

def get_real_state():
    pose = dType.GetPose(api)
    grip = dType.GetEndEffectorSuctionCup(api)[0]
    return list(pose[0:4]) + [0.0, 0.0, 1.0 if grip == 1 else -1.0]

def run_simple_episode():
    print("\n=== NEW EPISODE (Simple) ===")
    
    # 1. Random Targets
    pick_x = random.uniform(*PICK_ZONE["x"])
    pick_y = random.uniform(*PICK_ZONE["y"])
    place_x = random.uniform(*PLACE_ZONE["x"])
    place_y = random.uniform(*PLACE_ZONE["y"])
    home_x = random.uniform(*HOME_ZONE["x"])
    home_y = random.uniform(*HOME_ZONE["y"])
    
    # 2. Setup
    print("--> Moving to Pick...")
    move_and_wait(pick_x, pick_y, Z_SAFE)
    input("Place Block. Press ENTER.")
    
    print("--> Moving to Place...")
    move_and_wait(place_x, place_y, Z_SAFE)
    input("Clear Zone. Press ENTER.")
    
    print("--> Moving Home...")
    move_and_wait(home_x, home_y, Z_HOVER)
    
    # 3. QUEUE THE MISSION
    print("--> Uploading Command List...")
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    time.sleep(0.5)
    
    # We capture 'last_index' to know exactly when to stop recording
    move_ptp(pick_x, pick_y, Z_SAFE)      
    move_ptp(pick_x, pick_y, Z_PICK)      
    set_suction(1.0)                      
    dType.SetWAITCmd(api, 500, isQueued=1)
    move_ptp(pick_x, pick_y, Z_SAFE)      
    move_ptp(place_x, place_y, Z_SAFE)    
    move_ptp(place_x, place_y, Z_SAFE)    
    set_suction(-1.0)                     
    
    # CRITICAL: Save the Ticket Number of the LAST command
    last_index = move_ptp(place_x, place_y, Z_HOVER-20)

    # 4. EXECUTE & RECORD
    print("--> Clearing Camera Buffer...")
    for _ in range(5): cam.read()
    
    buffer = []
    print(f"--> Action! (Waiting for Index {last_index})")
    dType.SetQueuedCmdStartExec(api)
    
    while True:
        # Capture
        ret, frame = cam.read()
        if ret:
            state = get_real_state()
            buffer.append({'top': frame, 'state': state, 'action': state})
        
        # Check Status
        current_index = dType.GetQueuedCmdCurrentIndex(api)[0]
        
        # STOP Condition: Robot has reached the ticket number of the final command
        if current_index >= last_index:
            print(f"--> Robot Finished (Index {current_index} >= {last_index})")
            break
            
        time.sleep(0.05) 

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
        buf = run_simple_episode()
        ans = input("Save? [Y/N/Q]: ").upper()
        if ans == 'Y': save_hdf5(buf)
        elif ans == 'Q': break
    
    dType.DisconnectDobot(api)

if __name__ == "__main__":
    main()