import os
import time
import threading
import cv2
import numpy as np
# The library to listen to keyboard in the background
from pynput import keyboard 

# Dobot imports
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import DobotDllType as dType

# --- CONFIGURATION ---
Target_HZ = 10.0
Move_Step_XYZ = 10.0  # Move 10mm per key press
Move_Step_R = 5.0     # Rotate 5 degrees per key press
Save_Data = False     # Set to True when you are actually recording data

# --- GLOBAL SHARED VARIABLES ---
# We need a lock to stop the two threads from fighting over the USB cable
dobot_lock = threading.Lock()

# Shared state for the robot
current_pose = [0, 0, 0, 0] # x, y, z, r
keep_running = True

# Camera Setup (Adjust indices if needed. 0 is usually built-in webcam)
# We assume 1 and 2 are your external cameras
cam1_index = 1
cam2_index = 2

# Initialize Dobot
api = dType.load()

def initialize_robot():
    print("Connecting to Dobot...")
    state = dType.ConnectDobot(api, "", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("Failed to connect!")
        exit()
    
    print("Connected. Clearing Alarms and Homing...")
    # 1. Clear any red lights
    dType.ClearAllAlarmsState(api)
    
    # 2. Clear queues
    dType.SetQueuedCmdClear(api)
    dType.SetQueuedCmdStopExec(api)
    
    # 3. Set Parameters
    dType.SetPTPCommonParams(api, 100, 100, isQueued=0)
    
    # 4. PERFORM HOMING (Crucial!)
    # This will make the robot rotate and touch the base limits to find zero.
    # WARNING: Ensure the robot area is clear!
    print("Homing robot (this takes about 20 seconds)...")
    dType.SetHOMECmd(api, temp=0, isQueued=1)
    dType.SetQueuedCmdStartExec(api)
    
    # Wait for homing to finish
    # We just sleep for safety, but you can poll execution status if you prefer
    time.sleep(20) 
    
    print("Homing Complete. Robot Initialized.")

# --- THREAD 1: THE CONTROLLER (TELEOPERATION) ---
def controller_thread():
    global keep_running

    # --- TUNING PARAMETERS ---
    Control_Hz = 20.0         # Send commands 20 times per second (Smooth)
    Control_Period = 1.0 / Control_Hz
    
    # How many mm to move per "tick" (Start small!)
    # 2.0mm * 20Hz = 40mm/sec movement speed
    XYZ_Speed = 2.0           
    R_Speed = 1.0             

    # State of keys (True = Pressed, False = Released)
    pressed_keys = {
        'w': False, 's': False, 
        'a': False, 'd': False, 
        'q': False, 'e': False, 
        'r': False, 'f': False
    }

    # --- 1. The Input Listener (Non-blocking) ---
    def on_press(key):
        try:
            if hasattr(key, 'char') and key.char in pressed_keys:
                pressed_keys[key.char] = True
        except AttributeError: pass

    def on_release(key):
        try:
            if hasattr(key, 'char') and key.char in pressed_keys:
                pressed_keys[key.char] = False
        except AttributeError: pass

    # Start the listener in a non-blocking way
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print(f"Joystick Control Active at {Control_Hz}Hz. Hold keys to move.")

    # --- 2. The Control Loop (The Brain) ---
    while keep_running:
        loop_start = time.time()
        
        # Calculate the Composite Vector (Mixes multiple keys)
        d_x, d_y, d_z, d_r = 0, 0, 0, 0
        
        if pressed_keys['w']: d_x += XYZ_Speed
        if pressed_keys['s']: d_x -= XYZ_Speed
        if pressed_keys['a']: d_y -= XYZ_Speed
        if pressed_keys['d']: d_y += XYZ_Speed
        if pressed_keys['q']: d_z += XYZ_Speed
        if pressed_keys['e']: d_z -= XYZ_Speed
        if pressed_keys['r']: d_r += R_Speed
        if pressed_keys['f']: d_r -= R_Speed

        # Only talk to robot if we actually want to move
        if d_x != 0 or d_y != 0 or d_z != 0 or d_r != 0:
            with dobot_lock:
                # 1. Clear Alarms (Safety)
                dType.ClearAllAlarmsState(api)
                
                # 2. Get Current Pose
                pose = dType.GetPose(api)
                curr_x, curr_y, curr_z, curr_r = pose[0], pose[1], pose[2], pose[3]

                # 3. Calculate Target
                target_x = curr_x + d_x
                target_y = curr_y + d_y
                target_z = curr_z + d_z
                target_r = curr_r + d_r

                # 4. Send Absolute Command
                # Using isQueued=0 means "Abandon previous command, do this NOW"
                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, 
                                target_x, target_y, target_z, target_r, isQueued=0)
        
        # Sleep to maintain steady rate
        elapsed = time.time() - loop_start
        if elapsed < Control_Period:
            time.sleep(Control_Period - elapsed)

    listener.stop()

# --- THREAD 2: THE RECORDER (10HZ HEARTBEAT) ---
def recorder_loop():
    global keep_running, current_pose
    
    print("DEBUG: Attempting to open Camera 1...")
    # CHANGE 1: Add cv2.CAP_DSHOW
    cap1 = cv2.VideoCapture(cam1_index, cv2.CAP_DSHOW)
    
    print("DEBUG: Attempting to open Camera 2...")
    # CHANGE 2: Add cv2.CAP_DSHOW
    cap2 = cv2.VideoCapture(cam2_index, cv2.CAP_DSHOW)
    
    # CHANGE 3: Check if they actually opened
    if not cap1.isOpened():
        print(f"CRITICAL ERROR: Camera {cam1_index} failed to open.")
        keep_running = False
        return
    if not cap2.isOpened():
        print(f"CRITICAL ERROR: Camera {cam2_index} failed to open.")
        keep_running = False
        return

    print("DEBUG: Cameras opened. Starting loop...")
    print("Starting Recording Loop (10Hz)... Press 'ESC' in video window to stop.")
    
    prev_pose = None
    prev_img1 = None
    prev_img2 = None

    try:
        while keep_running:
            loop_start = time.time()
            
            # 1. READ SENSORS (Cameras)
            # We do this outside the lock because it takes time and doesn't use the robot
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                print("Frame drop!")
                continue

            # 2. READ ROBOT STATE (With Lock)
            with dobot_lock:
                # GetPose returns [x, y, z, r, j1, j2, j3, j4]
                raw_pose = dType.GetPose(api)
                curr_xyzr = np.array(raw_pose[0:4]) # Keep only Cartesian X,Y,Z,R
            
            # 3. CALCULATE ACTION (Delta)
            # The "Action" that resulted in the current state is (Current - Previous)
            if prev_pose is not None:
                actual_action = curr_xyzr - prev_pose
                
                # --- DATA SAVING BLOCK ---
                if Save_Data:
                    # Here you save:
                    # Input: prev_img1, prev_img2, prev_pose
                    # Label: actual_action
                    # timestamp = time.time()
                    pass 
                # -------------------------

                print(f"Pose: {curr_xyzr} | Action taken: {actual_action}")

            # 4. UPDATE PREVIOUS STATE
            prev_pose = curr_xyzr
            prev_img1 = frame1
            prev_img2 = frame2
            
            # 5. VISUALIZATION
            # Combine images horizontally for display
            if frame1.shape == frame2.shape:
                combined = np.hstack((frame1, frame2))
                cv2.imshow("Dual Camera Feed", combined)
            else:
                cv2.imshow("Cam 1", frame1)
                cv2.imshow("Cam 2", frame2)

            # Exit logic
            if cv2.waitKey(1) & 0xFF == 27: # ESC key
                keep_running = False
                break
            
            # 6. SLEEP TO MAINTAIN 10HZ
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / Target_HZ) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    initialize_robot()
    
    # Start the Controller Thread (Daemon means it dies when main program dies)
    t_control = threading.Thread(target=controller_thread, daemon=True)
    t_control.start()
    
    # Run the Recorder Loop in the Main Thread
    recorder_loop()
    
    # Cleanup
    print("Disconnecting...")
    dType.DisconnectDobot(api)