# initialize the dobot
import os 
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import DobotDllType as dType
import time
import threading
import cv2
import numpy as np
import random

#Useful global variables
# --- These are status strings that you might see, so we're defining them here ---
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}

#always begin with this line, or you can't connect to the robot at all. Just don't
#remove this line and keep it at the top of your code
api = dType.load()

"""
These coordinates are to the left of the robot's x axis and slight above the xy plane, viewed from
the top. This is a useful home position when dealing with the vision labs, since it moves
the robot out of the way. You can change the coordinates here if you really want.
"""
home_pos = [200,100,50] # to validate?

def initialize_robot(api):
    #detect the robot's com port
    com_port = dType.SearchDobot(api)
    print(dType.SearchDobot(api))
    #if we can't find it, then we can't continue, so exit
    if "COM" not in com_port[0]:
        print("Error: The robot either isn't on or isn't responding. Exiting now")
        exit()
    
    
    #we've found it, so let's try to connect
    state = dType.DobotConnect.DobotConnect_NoError
    for i in range(0,len(com_port)):
        state_full = dType.ConnectDobot(api, com_port[i], 115200)
        state = state_full[0]
        print("STATE FULL:")
        print(state_full)
        #If the connection failed at this point, we also can't proceed, so we need to exit
        if state == dType.DobotConnect.DobotConnect_NoError:
            print("Connected!")
            name = "elrond" #dType.GetDeviceName(api)
            if name[0] == "Not a dobot":
                dType.DisconnectDobot(api)
                continue
            else:
                break
            
    if state != dType.DobotConnect.DobotConnect_NoError:
            print("Can not connect! Exiting")
            exit()    
    """
        stop any queued commands and clear the queue. You HAVE TO do this every time you initialize the robot
        If there are queued commands in the queue, then they will execute first. This can
        cause the robot to go well outside of its allowable range. The simplest way to do this
        is to stop anything that might be running or might try to run, then clear the queue.
        
        Other than at startup, during normal operation you shouldn't have to do this.
    """
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    
    #Set the robot's max speed and acceleration. We're keeping these to 50% of max for safety
    dType.SetPTPCommonParams(api, 50, 50, isQueued=1) # to validate -- i think this wont matter
    
    """
        Home the robot. 
    """
    #Set the home position
    dType.SetHOMEParams(api, home_pos[0], home_pos[1], home_pos[2], 0, isQueued=1)
    
    cmdIndx = -1
    """
        Enqueue the home command. This command always begins by moving the robot back to an initialization
        position so that the encoders are reset, then it will move the robot to its home position,
        and finally it will undergo a quick procedure to validate that its encoders are properly set. You definitely
        want to run this every time you initialize the robot
    """
    execCmd = dType.SetHOMECmd(api, temp=0, isQueued=1)[0]
    
    #Execute the three enqueued commands: set the speed/acceleration, set the home position, and move to home
    dType.SetQueuedCmdStartExec(api)
    
    #Allow the homing command to complete. The robot will beep and the LED will turn green
    #when it's ready to go
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(25)
        
    #OK, the robot is ready to move!

# initialize the camera

# Open webcam
cap = cv2.VideoCapture(0)

#Before running and commands, always run this
initialize_robot(api)

"""
    TESTING____________________________________________________________________________________________TESTING
"""

# --- Constants ---
TARGET_HZ = 10.0
TARGET_PERIOD = 1.0 / TARGET_HZ

# --- WARNING: EXAMPLE JOINT LIMITS ---
# --- REPLACE THESE WITH YOUR ROBOT'S ACTUAL SAFE LIMITS ---
J1_LIMITS = (-120.0, 120.0) # i think up to 130 is ok
J2_LIMITS = (0.0, 85.0) # some wiggle
J3_LIMITS = (-20.0, 60.0) # some wiggle
J4_LIMITS = (-180.0, 180.0) # unknown

print("Starting 10Hz control loop. Press 'q' in the video window to stop.")

# Helper function to keep targets within safe limits
def clamp_angles(j1, j2, j3, j4):
    j1 = max(J1_LIMITS[0], min(J1_LIMITS[1], j1))
    j2 = max(J2_LIMITS[0], min(J2_LIMITS[1], j2))
    j3 = max(J3_LIMITS[0], min(J3_LIMITS[1], j3))
    j4 = max(J4_LIMITS[0], min(J4_LIMITS[1], j4))
    return (j1, j2, j3, j4)

try:
    while True:
        loop_start_time = time.time()

        # 1. Get current state (image and joint angles)
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('VLA Project Feed', frame)
        
        # Get the CURRENT position from the robot
        pose = dType.GetPose(api)
        current_joints = pose[4:8]
        # current_joints.angle will be a list/tuple: [j1, j2, j3, j4]

        # 2. Get the VLA's predicted action (the DELTAs)
        # instruction = "pick up the red block" # (This comes from your high-level plan)
        # delta_action = vla_model.predict(frame, instruction) 
        
        # --- Placeholder for VLA model ---
        # Let's simulate a small random "nudge"
        delta_action = [
            random.uniform(-0.5, 0.5),  # Small delta for j1
            random.uniform(-0.5, 0.5),  # Small delta for j2
            random.uniform(-0.5, 0.5),  # Small delta for j3
            random.uniform(-0.5, 0.5),  # Small delta for j4
            0                           # No gripper action
        ]
        # --- End Placeholder ---

        # 3. Calculate the NEW target position
        target_j1 = current_joints[0] + delta_action[0]
        target_j2 = current_joints[1] + delta_action[1]
        target_j3 = current_joints[2] + delta_action[2]
        target_j4 = current_joints[3] + delta_action[3]
        
        # 4. CRITICAL: Clamp to safety limits
        (target_j1, target_j2, target_j3, target_j4) = clamp_angles(
            target_j1, target_j2, target_j3, target_j4
        )

        # 5. Send the new ABSOLUTE target to the robot
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJANGLEMode, 
                        target_j1, target_j2, target_j3, target_j4, isQueued=1)

        print(f"Current J1: {current_joints[0]:.2f} | Delta J1: {delta_action[0]:.2f} | Target J1: {target_j1:.2f}")

        # --- 10Hz Rate Control ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        elapsed_time = time.time() - loop_start_time
        sleep_time = TARGET_PERIOD - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

finally:    
    print("Cleaning up... Clearing command queue!")
    # TELL THE ROBOT TO CLEAR ITS BUFFER
    dType.SetQueuedCmdClear(api)
    dType.SetQueuedCmdStopExec(api) # Optional, but good practice
    cap.release()
    cv2.destroyAllWindows()
    dType.DisconnectDobot(api)

# questions:
# camera positioning, home position
# "smoothness" of operation while running at 10Hz
# XYZ positions vs Joint movement: we will not be using these "move" functions
# 