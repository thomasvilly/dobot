import os 
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import DobotDllType as dType
import json_numpy
import numpy as np
import requests
import time

json_numpy.patch()

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

t_ = time.time()

# initialize the camera

# Open webcam
cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2) # Ensure this is your second camera index

print(f"Took {time.time()-t_}s to startup camera")
t_ = time.time()

#Before running and commands, always run this
initialize_robot(api)

print(f"Took {time.time()-t_}s to startup DOBOT")

"""
    TESTING____________________________________________________________________________________________TESTING
"""

# --- Constants ---
TARGET_HZ = 0.8
TARGET_PERIOD = 1.0 / TARGET_HZ
SCALE_FACTOR = 500.0 # to scale outputs from the VLA

# --- Your high-level instruction ---
instruction = "pick up the watch"

# --- NETWORKING ---
# The IP found from 'ip addr show'
VLA_SERVER_IP = "129.97.8.241" 
VLA_SERVER_URL = f"http://{VLA_SERVER_IP}:8000/act"

# --- CARTESIAN LIMITS (in mm) ---
X_LIMITS = (-150.0, 250.0)  # Min/Max X (forward/back)
Y_LIMITS = (-200.0, 200.0) # Min/Max Y (left/right)
Z_LIMITS = (-100.0, 150.0)  # Min/Max Z (up/down)
R_LIMITS = (-150.0, 100.0) # Min/Max R (wrist rotation)

def clamp_pose(x, y, z, r):
    x = max(X_LIMITS[0], min(X_LIMITS[1], x))
    y = max(Y_LIMITS[0], min(Y_LIMITS[1], y))
    z = max(Z_LIMITS[0], min(Z_LIMITS[1], z))
    r = max(R_LIMITS[0], min(R_LIMITS[1], r))
    return (x, y, z, r)

print(f"Starting {TARGET_HZ}Hz control loop. Press 'q' in the video window to stop.")

try:
    while True:
        loop_start_time = time.time()
        
        # 1. Get image from local cameras
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        
        if not ret or not ret2:
            print("Frame drop")
            break
            
        # --- STITCHING LOGIC START ---
        # Resize to 224x224 so the stitched result is 448x224 (OpenVLA standard)
        im1 = cv2.resize(frame, (224, 224))
        im2 = cv2.resize(frame2, (224, 224))
        stitched = np.hstack((im1, im2))
        
        cv2.imshow('VLA Project Feed', stitched)
        # --- STITCHING LOGIC END ---
        
        # (We don't need joint angles for the VLA, just for the Dobot)
        pose = dType.GetPose(api)
        current_x = pose[0]
        current_y = pose[1]
        current_z = pose[2]
        current_r = pose[3]

        # 2. Package the payload for the server
        frame_rgb = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
        payload = {
            "image": frame_rgb,
            "instruction": instruction,
            "unnorm_key": "bridge_orig" # Optional: as seen in the example
        }

        # 3. "CALL" THE VLA over the network
        # This sends the image/instruction to the Linux box
        # and WAITS for the action to be sent back.
        try:
            response = requests.post(VLA_SERVER_URL, json=payload, timeout=0.5)
            response.raise_for_status() # Raise an error for bad responses
            
            # The 'action' (deltas) comes back from the server
            action_deltas = response.json() 
            print(response.json())

        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            print(f"Error calling VLA server: {e}")
            continue # Skip this loop cycle
        
        # scale the action deltas
        action_deltas = [a*SCALE_FACTOR for a in action_deltas]
        action_deltas[-1] = action_deltas[-1] / SCALE_FACTOR # not for grippper
        print(f"Open VLA proposed action = dx: {action_deltas[0]}, dy: {action_deltas[1]}, dz: {action_deltas[2]}, d_yaw: {action_deltas[5]}, gripper: {action_deltas[6]}")
        
        dx        = action_deltas[0]
        dy        = action_deltas[1]
        dz        = action_deltas[2]
        # d_roll  = action_deltas[3]  <- IGNORE
        # d_pitch = action_deltas[4]  <- IGNORE
        d_yaw     = action_deltas[5]
        gripper = action_deltas[6]  # <- Use this for your gripper logic

        # 4. Use the action deltas
        target_x = current_x + dx
        target_y = current_y + dy
        target_z = current_z + dz
        target_r = current_r + d_yaw
        target_x, target_y, target_z, target_r = clamp_pose(target_x, target_y, target_z, target_r)
        print(f"Action taken = dx: {target_x}, dy: {target_y}, dz: {target_z}, d_yaw: {target_r}, gripper: {gripper}")
        
        # 5. Command the Dobot
        # CHANGED: isQueued=0 (Must be 0 for immediate control)
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 
                target_x, target_y, target_z, target_r, isQueued=0)        

        # 6. End effector: switched to Suction Cup
        # Set a threshold. If the VLA is "more than 50% closed",
        # Command to close gripper (or turn on suction)
        gripper_value = 1 if gripper > 0.5 else 0
        if gripper_value:
            print("TRYING TO GRIP")
        
        # CHANGED: Swapped to SuctionCup logic
        dType.SetEndEffectorSuctionCup(api, 1, gripper_value, isQueued=0)
        
        # suction_value = 1 if gripper > 0.5 else 0
        # dType.SetEndEffectorSuctionCup(api, 1, suction_value, isQueued=1)
            
        # --- 10Hz Rate Control ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        elapsed_time = time.time() - loop_start_time
        sleep_time = TARGET_PERIOD - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print("Warning: Loop (including network) took too long!")

finally:    
    print("Cleaning up... Clearing command queue!")
    # TELL THE ROBOT TO CLEAR ITS BUFFER
    dType.SetQueuedCmdClear(api)
    dType.SetQueuedCmdStopExec(api) # Optional, but good practice
    cap.release()
    cap2.release()
    cv2.destroyAllWindows()
    dType.DisconnectDobot(api)