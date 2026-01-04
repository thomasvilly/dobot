import os
import sys
import time
import json
import socket
import struct
import argparse
import threading
import numpy as np
import cv2
import requests
import json_numpy
json_numpy.patch()

# Try to import the Dobot driver wrapper
try:
    import DobotDllType as dType
except ImportError:
    print("[!] Error: 'DobotDllType.py' not found. Make sure it is in the same directory.")
    sys.exit(1)

# --- GLOBAL CONFIG ---
# Safety & Movement Limits
MAX_STEP_MM = 20.0
MAX_RADIUS = 310.0
MIN_RADIUS = 140.0
Z_MIN = -90.0
Z_MAX = 150.0

# Client Logic Params
ACTION_SCALE = 1.0
GRIPPER_THRESHOLD = 0.3
INSTRUCTION = "pick up the blue block"

# ==============================================================================
# CLASS: DOBOT DRIVER (Hardware Layer)
# ==============================================================================
class DobotDriver:
    def __init__(self, port="COM3"):
        self.api = dType.load()
        self.port = port
        self.connected = False
        self.cam_overhead = None
        self.cam_wrist = None
        self.gripper_state = 0
        
        # Connect
        print(f"[Driver] Connecting to Dobot on {port}...")
        state = dType.ConnectDobot(self.api, port, 115200)[0]
        if state == dType.DobotConnect.DobotConnect_NoError:
            print("[Driver] Dobot Connected.")
            self.connected = True
            self._setup_robot()
        else:
            print("[Driver] Failed to connect to Dobot. Check USB/Power.")
            # We don't exit here to allow debugging cameras without robot if needed
            
        # Open Cameras
        self._setup_cameras()

    def _setup_robot(self):
        dType.ClearAllAlarmsState(self.api)
        dType.SetQueuedCmdClear(self.api)
        
        # Slow down for safety
        # dType.SetPTPCommonParams(self.api, 50, 50, isQueued=0)
        # dType.SetPTPJointParams(self.api, 50, 50, 50, 50, 50, 50, 50, 50, isQueued=0)
        # dType.SetPTPCoordinateParams(self.api, 50, 50, 50, 50, isQueued=0)

    def _setup_cameras(self):
        print("[Driver] Opening cameras...")
        # Index 0/1 might swap depending on USB ports
        self.cam_overhead = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        # Set Resolution (Standard 640x480)
        for cam in [self.cam_overhead]:
            if cam and cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                print("[Driver] Warning: A camera failed to open.")

    def get_pose(self):
        """Returns [x, y, z]"""
        if not self.connected: return [200, 0, 0]
        pose = dType.GetPose(self.api)
        return np.array(pose[:3])

    def get_images(self):
        """Returns (overhead_bgr, wrist_bgr)"""
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if self.cam_overhead and self.cam_overhead.isOpened():
            ret, frame = self.cam_overhead.read()
            if ret: img1 = frame
            
        return img1

    def move(self, x, y, z):
        if not self.connected: return
        
        # Clear Alarms before move (prevents 'Red Light' lockups)
        dType.ClearAllAlarmsState(self.api)
        
        # Execute PTP Move
        dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, 0, isQueued=0)
        
        # Wait for arrival (Simple blocking)
        target = np.array([x, y, z])
        start_t = time.time()
        while time.time() - start_t < 5.0:
            curr = self.get_pose()[:3]
            if np.linalg.norm(curr - target) < 2.0:
                break
            time.sleep(0.01)

    def grip(self, state):
        """state: 0 (Open) or 1 (Closed)"""
        if not self.connected: return
        self.gripper_state = state
        enable = 1 if state > 0.5 else 0
        dType.SetEndEffectorSuctionCup(self.api, 1, enable, isQueued=0)

    def close(self):
        dType.DisconnectDobot(self.api)
        if self.cam_overhead: self.cam_overhead.release()
        if self.cam_wrist: self.cam_wrist.release()


# ==============================================================================
# MODE A: HTTP CLIENT (Talks to Linux 'deploy.py')
# ==============================================================================
def check_safety_clamp(target_xyz, current_xyz):
    """Prevents the robot from hitting itself or moving too fast."""
    x, y, z = target_xyz
    
    # 1. Z-Limit
    z = np.clip(z, Z_MIN, Z_MAX)

    # 2. Radius Limit (Don't hit base, don't overstretch)
    radius = np.sqrt(x**2 + y**2)
    if radius > MAX_RADIUS:
        scale = MAX_RADIUS / radius
        x *= scale; y *= scale
    elif radius < MIN_RADIUS:
        scale = MIN_RADIUS / radius
        x *= scale; y *= scale

    # 3. Velocity Clamp (Prevent teleporting)
    cur_x, cur_y, cur_z = current_xyz
    dist_sq = (x - cur_x)**2 + (y - cur_y)**2 + (z - cur_z)**2
    if dist_sq > MAX_STEP_MM**2:
        scale = np.sqrt(MAX_STEP_MM**2 / dist_sq)
        x = cur_x + (x - cur_x) * scale
        y = cur_y + (y - cur_y) * scale
        z = cur_z + (z - cur_z) * scale

    return x, y, z

def run_http_client(driver, brain_ip, brain_port):
    url = f"http://{brain_ip}:{brain_port}/act"
    print(f"\n[Mode: Client] Connecting to Brain at {url}...")
    
    step = 0
    current_gripper = 0
    
    print(">>> Press Ctrl+C to STOP.")
    try:
        while True:
            t0 = time.time()
            
            # 1. Get Hardware State
            img_overhead = driver.get_images() # Wrist img unused for now
            pose_4d = driver.get_pose()
            
            # Show Feed
            cv2.imshow("Robot Eye", img_overhead)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # Convert to RGB for Model
            img_rgb = cv2.cvtColor(img_overhead, cv2.COLOR_BGR2RGB)
            
            # 2. Construct Payload
            payload = {
                    "full_image": img_rgb.tolist(),
                    "state": [pose_4d[0], pose_4d[1], pose_4d[2], float(current_gripper)],
                    "instruction": INSTRUCTION
            }
            
            # 3. Query Brain
            try:
                resp = requests.post(url, json=payload, timeout=5.0)
                if resp.status_code != 200:
                    print(f"[!] Server Error {resp.status_code}: {resp.text}")
                    continue
                action = np.array(resp.json())
            except Exception as e:
                print(f"[!] Network Error: {e}")
                time.sleep(0.5)
                continue
            
            # 4. Parse Action
            # Handle chunking (model might return [10, 4] list)
            print(action)
            action = action[:5]
            for i, act in enumerate(action):
                delta_xyz = act[:3] * ACTION_SCALE
                raw_gripper = act[-1]
                
                # Gripper Logic
                if raw_gripper > GRIPPER_THRESHOLD: current_gripper = 1
                elif raw_gripper < -GRIPPER_THRESHOLD: current_gripper = 0
                
                pose_4d = driver.get_pose() 
                current_xyz = pose_4d[:3]
                
                target_xyz = current_xyz + delta_xyz
                safe_x, safe_y, safe_z = check_safety_clamp(target_xyz, current_xyz)
                
                dt = time.time() - t0
                print(f"Step {step} ({dt:.3f}s) | Delta: {np.round(delta_xyz, 1)} | Grip: {raw_gripper:.2f}")
                driver.move(safe_x, safe_y, safe_z)
                driver.grip(current_gripper)
                step += 1
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cv2.destroyAllWindows()


# ==============================================================================
# MODE B: SOCKET SERVER (Legacy / Debug Mode)
# ==============================================================================
def run_socket_server(driver, port):
    print(f"\n[Mode: Server] Listening on 0.0.0.0:{port}...")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', port))
        s.listen()
        
        while True:
            print("Waiting for connection...")
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    try:
                        # 1. Header
                        raw_len = conn.recv(4)
                        if not raw_len: break
                        msg_len = struct.unpack('>I', raw_len)[0]
                        
                        # 2. Body
                        data = b''
                        while len(data) < msg_len:
                            packet = conn.recv(msg_len - len(data))
                            if not packet: break
                            data += packet
                        
                        cmd = json.loads(data.decode('utf-8'))
                        resp = {"status": "ok"}
                        ctype = cmd.get("cmd")
                        
                        # 3. Dispatch
                        if ctype == "GET_IMAGE":
                            img1, img2 = driver.get_images()
                            # Encode
                            _, b1 = cv2.imencode('.jpg', img1)
                            _, b2 = cv2.imencode('.jpg', img2)
                            b1_bytes = b1.tobytes()
                            b2_bytes = b2.tobytes()
                            
                            # Send Raw Bytes
                            payload = struct.pack('>I', len(b1_bytes)) + b1_bytes + \
                                      struct.pack('>I', len(b2_bytes)) + b2_bytes
                            conn.sendall(payload)
                            continue # Skip JSON response for images
                            
                        elif ctype == "GET_POSE":
                            resp["pose"] = driver.get_pose().tolist()
                            
                        elif ctype == "MOVE":
                            driver.move(cmd['x'], cmd['y'], cmd['z'], cmd.get('r', 0))
                            
                        elif ctype == "GRIP":
                            driver.grip(cmd['state'])
                            
                        # 4. Send JSON Response
                        resp_bytes = json.dumps(resp).encode('utf-8')
                        conn.sendall(struct.pack('>I', len(resp_bytes)) + resp_bytes)
                        
                    except Exception as e:
                        print(f"Session Error: {e}")
                        break
            print("Client disconnected.")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dobot Hybrid Controller")
    parser.add_argument("--mode", choices=["client", "server"], required=True, help="Run as 'client' (Talks to Linux) or 'server' (Listens for Linux)")
    parser.add_argument("--ip", type=str, default="192.168.219.198", help="Linux Brain IP (Client Mode)")
    parser.add_argument("--port", type=int, default=8777, help="Port (Client Mode=Brain Port, Server Mode=Local Port)")
    parser.add_argument("--dobot_port", type=str, default="COM3", help="Windows COM Port for Robot")
    
    args = parser.parse_args()
    
    # Initialize Driver
    driver = DobotDriver(port=args.dobot_port)
    
    try:
        if args.mode == "client":
            # Windows (Logic) -> HTTP -> Linux (Brain)
            run_http_client(driver, args.ip, args.port)
        else:
            # Windows (Hardware) <- Socket <- Linux (Logic)
            run_socket_server(driver, 65432) # Default socket port from your old script
    except KeyboardInterrupt:
        pass
    finally:
        driver.close()
        print("[System] Driver Closed.")