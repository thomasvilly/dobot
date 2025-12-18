import socket
import struct
import json
import cv2
import numpy as np
import DobotDllType as dType
import time

# --- CONFIG ---
HOST = '192.168.208.1'  # Your Windows IP
PORT = 65432

def update_monitor(frame1, frame2):
    """Helper to update the OpenCV window with index labels."""
    if frame1 is None or frame2 is None: return

    # Resize for display
    view1 = cv2.resize(frame1, (320, 240))
    view2 = cv2.resize(frame2, (320, 240))
    
    # Add Labels
    cv2.putText(view1, "Idx 0: Overhead", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(view2, "Idx 1: Wrist", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Stack and Show
    combined_view = np.hstack((view1, view2))
    cv2.imshow("Robot Server Monitor", combined_view)
    cv2.waitKey(1) # Keeps window responsive

def init_robot():
    print("--- Initializing Robot (Windows Side) ---")
    api = dType.load()
    state = dType.ConnectDobot(api, "COM3", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        raise Exception("Failed to connect to Dobot")
    
    dType.ClearAllAlarmsState(api)
    dType.SetQueuedCmdClear(api)
    
    # --- CRITICAL FIX: REDUCE SPEED & ACCELERATION ---
    # Default is 100, 100. We drop to 50% to prevent "Jerk" alarms.
    dType.SetPTPCommonParams(api, 50, 50, isQueued=0) 
    dType.SetPTPJointParams(api, 50, 50, 50, 50, 50, 50, 50, 50, isQueued=0)
    dType.SetPTPCoordinateParams(api, 50, 50, 50, 50, isQueued=0)
    # -------------------------------------------------
    
    print("Opening Cameras...")
    cam_overhead = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    cam_wrist = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cam_overhead.isOpened() or not cam_wrist.isOpened():
         print("WARNING: One or both cameras failed to open.")

    cam_overhead.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam_overhead.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam_wrist.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam_wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return api, cam_overhead, cam_wrist

def move_robot(api, cam_overhead, cam_wrist, x, y, z, r):
    # 1. Safety Clear (Fixes 'Red Light' stops)
    dType.ClearAllAlarmsState(api)
    
    # 2. Clamp 'r' (Rotation)
    # The servo breaks if you go beyond -150/150.
    r = max(-150, min(150, r))
    
    # 3. Send Command (Immediate Mode)
    last_index = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, r, isQueued=0)[0]
    
    target = np.array([x, y, z])
    start = time.time()
    
    # 4. "Locking" Loop
    while True:
        # Update Cameras
        ret1, frame1 = cam_overhead.read()
        ret2, frame2 = cam_wrist.read()
        if ret1 and ret2: update_monitor(frame1, frame2)

        # Get Current Position
        pose = dType.GetPose(api)
        curr = np.array(pose[0:3])
        
        # STOP CONDITION 1: We are there (within 2mm)
        dist = np.linalg.norm(curr - target)
        if dist < 2.0: 
            break
            
        # STOP CONDITION 2: Timeout (Robot stuck/alarmed)
        # Increased to 5.0s because we slowed down the robot
        if time.time() - start > 5.0: 
            print("WARN: Move timeout (Robot stuck?)")
            break
        
        time.sleep(0.005)

def set_gripper(api, state):
    enable = 1 if state > 0.5 else 0
    dType.SetEndEffectorSuctionCup(api, 1, enable, isQueued=0)

def main():
    api, cam_overhead, cam_wrist = init_robot()
    print(f"✅ Server Listening on {HOST}:{PORT}...")
    
    # Create the socket once
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        
        # --- PERMANENT SERVER LOOP ---
        while True:
            print(f"Waiting for new connection...")
            try:
                conn, addr = s.accept() # Blocks here until Linux connects
            except Exception as e:
                print(f"Accept error: {e}")
                time.sleep(1)
                continue
            
            with conn:
                print(f"Connected by {addr}")
                
                # SAFETY: Clear alarms/queues on new connection
                # This fixes the "stuck" state if the previous run crashed
                dType.ClearAllAlarmsState(api)
                dType.SetQueuedCmdClear(api)

                # --- CLIENT SESSION LOOP ---
                while True:
                    try:
                        # 1. Receive Command Header
                        raw_len = conn.recv(4)
                        if not raw_len: 
                            print("Client closed connection cleanly.")
                            break
                        msg_len = struct.unpack('>I', raw_len)[0]
                        
                        # 2. Receive Body
                        data = b''
                        while len(data) < msg_len:
                            packet = conn.recv(msg_len - len(data))
                            if not packet: raise ConnectionResetError("Partial packet")
                            data += packet
                        
                        command = json.loads(data.decode('utf-8'))
                        response = {"status": "ok"}
                        cmd_type = command.get("cmd")
                        
                        # 3. Process Command
                        if cmd_type == "GET_IMAGE":
                            ret1, frame1 = cam_overhead.read()
                            ret2, frame2 = cam_wrist.read()
                            
                            if ret1 and ret2:
                                update_monitor(frame1, frame2)
                                _, buf1 = cv2.imencode('.jpg', frame1)
                                _, buf2 = cv2.imencode('.jpg', frame2)
                                img1_bytes = buf1.tobytes()
                                img2_bytes = buf2.tobytes()
                                
                                payload = (
                                    struct.pack('>I', len(img1_bytes)) + img1_bytes +
                                    struct.pack('>I', len(img2_bytes)) + img2_bytes
                                )
                                conn.sendall(payload)
                                continue 
                            else:
                                response = {"status": "error", "msg": "Camera read failed"}

                        elif cmd_type == "MOVE":
                            move_robot(api, cam_overhead, cam_wrist, 
                                       command['x'], command['y'], command['z'], command['r'])                
                        
                        elif cmd_type == "GRIP":
                            set_gripper(api, command['state'])
                        
                        elif cmd_type == "GET_POSE":
                             pose = dType.GetPose(api)
                             response["pose"] = list(pose)

                        # 4. Send JSON Response (For non-image commands)
                        resp_bytes = json.dumps(response).encode('utf-8')
                        conn.sendall(struct.pack('>I', len(resp_bytes)) + resp_bytes)
                    
                    except (ConnectionResetError, ConnectionAbortedError):
                        print("Connection lost (Client crashed or disconnected).")
                        break
                    except Exception as e:
                        print(f"Error in session: {e}")
                        break
            
            print("Session ended. Cleaning up and re-listening...")
            # Loop goes back to top -> s.accept()

if __name__ == "__main__":
    main()