import socket
import struct
import json
import cv2
import numpy as np
import DobotDllType as dType
import time

# --- CONFIG ---
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 65432      # Port to listen on

# --- HARDWARE SETUP ---
def init_robot():
    print("--- Initializing Robot (Windows Side) ---")
    api = dType.load()
    state = dType.ConnectDobot(api, "COM3", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        raise Exception("Failed to connect to Dobot")
    
    dType.ClearAllAlarmsState(api)
    dType.SetQueuedCmdClear(api)
    dType.SetPTPCommonParams(api, 100, 100, isQueued=0)
    
    # Cameras (Top + Side)
    # Note: We open both but only send Top for now unless requested
    cam1 = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Top
    if not cam1.isOpened(): raise Exception("Failed to open Camera 1")
    
    return api, cam1

def move_robot(api, cam, x, y, z, r): # <-- ADD 'cam' HERE
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, r, isQueued=0)
    
    # Blocking Wait for Robot to Settle
    target = np.array([x, y, z])
    start = time.time()
    while True:
        pose = dType.GetPose(api)
        curr = np.array(pose[0:3])
        if np.linalg.norm(curr - target) < 2.0: break
        if time.time() - start > 5.0: break
        time.sleep(0.01)
        
    # --- CRITICAL FIX: FLUSH CAMERA BUFFER ---
    # Read and discard 3-5 frames to clear out old, blurry motion frames
    # This ensures the next GET_IMAGE command gets a fresh, stable frame.
    for _ in range(3):
        cam.read()

def set_gripper(api, state):
    enable = 1 if state > 0.5 else 0
    dType.SetEndEffectorSuctionCup(api, 1, enable, isQueued=0)

def main():
    api, cam = init_robot()
    print(f"✅ Server Listening on {PORT}...")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                # 1. Receive Command Length (4 bytes)
                raw_len = conn.recv(4)
                if not raw_len: break
                msg_len = struct.unpack('>I', raw_len)[0]
                
                # 2. Receive Command Data
                data = b''
                while len(data) < msg_len:
                    packet = conn.recv(msg_len - len(data))
                    if not packet: break
                    data += packet
                
                command = json.loads(data.decode('utf-8'))
                response = {"status": "ok"}
                
                # 3. Process Command
                cmd_type = command.get("cmd")
                
                if cmd_type == "GET_IMAGE":
                    ret, frame = cam.read()
                    if ret:
                        # Encode to JPEG to send over wire
                        _, buffer = cv2.imencode('.jpg', frame)
                        img_bytes = buffer.tobytes()
                        # Send: [Size of Image (4 bytes)] + [Image Bytes]
                        conn.sendall(struct.pack('>I', len(img_bytes)) + img_bytes)
                        continue # Image handled separately, skip JSON response
                    else:
                        response = {"status": "error", "msg": "Camera read failed"}

                elif cmd_type == "MOVE":
                    move_robot(api, cam, command['x'], command['y'], command['z'], command['r'])                
                elif cmd_type == "GRIP":
                    set_gripper(api, command['state'])
                
                # 4. Send JSON Response
                resp_bytes = json.dumps(response).encode('utf-8')
                conn.sendall(struct.pack('>I', len(resp_bytes)) + resp_bytes)

if __name__ == "__main__":
    main()