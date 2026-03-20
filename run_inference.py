import sys
import time
import cv2
import socket
import requests
import numpy as np
import json_numpy
from dataclasses import dataclass

# Patch json_numpy to allow seamless numpy array transmission over HTTP
json_numpy.patch()

try:
    import DobotDllType as dType
except ImportError:
    print("[!] Error: 'DobotDllType.py' not found in this directory.")
    sys.exit(1)

# --- HARDWARE CONFIG ---
PORT = "COM3" 
BRAIN_URL = "http://192.168.219.198:8777/act" # Replace with your actual Linux IP
INSTRUCTION = "pick up the blue block and put it on the plate"

# --- SAFETY LIMITS ---
MAX_STEP_MM = 20.0
Z_MIN, Z_MAX = -50.0, 150.0  # Synced with Linux Server
MAX_RADIUS, MIN_RADIUS = 310.0, 140.0

class DobotController:
    def __init__(self, port):
        self.api = dType.load()
        self.connected = False
        
        print(f"[Init] Connecting to Dobot on {port}...")
        res = dType.ConnectDobot(self.api, port, 115200)[0]
        if res == dType.DobotConnect.DobotConnect_NoError:
            self.connected = True
            dType.ClearAllAlarmsState(self.api)
            dType.SetQueuedCmdClear(self.api)
            print("[Init] Dobot Online.")
        
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("[Init] Warning: Camera failed to open.")

    def get_state(self):
        """Returns (RGB_Image, 4D_Pose)"""
        ret, frame = self.cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # OpenVLA was trained on RGB; OpenCV is BGR
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose = dType.GetPose(self.api) if self.connected else [200, 0, 0]
        # We use X, Y, Z and a placeholder for the current gripper state
        return img_rgb, np.array([pose[0], pose[1], pose[2]])

    def move_and_grip(self, x, y, z, gripper_val):
        """Executes hardware command with safety clamping."""
        if not self.connected: return

        # 1. Safety Clamp
        z = np.clip(z, Z_MIN, Z_MAX)
        r = np.sqrt(x**2 + y**2)
        if r > MAX_RADIUS or r < MIN_RADIUS:
            scale = np.clip(r, MIN_RADIUS, MAX_RADIUS) / r
            x *= scale; y *= scale

        # 2. Execute
        dType.ClearAllAlarmsState(self.api)
        dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, 0, isQueued=0)
        
        # 3. Gripper (Suction Cup)
        enable = 1 if gripper_val > 0.5 else 0
        dType.SetEndEffectorSuctionCup(self.api, 1, enable, isQueued=0)

    def close(self):
        if self.connected: dType.DisconnectDobot(self.api)
        self.cap.release()

def run_inference_loop():
    robot = DobotController(PORT)
    current_gripper = 0.0
    
    print(f"\n>>> Starting Live Inference")
    print(f">>> Task: {INSTRUCTION}\n")
    
    try:
        while True:
            # 1. Capture Observation
            img_rgb, current_xyz = robot.get_state()
            
            # 2. Build Flat Payload for deploy.py
            payload = {
                "full_image": img_rgb,
                "state": np.append(current_xyz, [current_gripper]),
                "instruction": INSTRUCTION
            }

            # 3. Query Linux VLA Server
            try:
                response = requests.post(BRAIN_URL, json=payload, timeout=10.0)
                result = response.json()
                if result == "error":
                    print("[!] Linux Server Error. Check terminal.")
                    continue
                
                actions = np.array(result) # Expected shape: [10, 4]
            except Exception as e:
                print(f"[Error] Connection failed: {e}")
                time.sleep(1)
                continue

            # 4. MINI-LOOP: Execute 3 steps from the predicted chunk
            # This makes the motion 3x more fluid and less "stuttery"
            print(f">>> Executing steps...")
            
            physical_gripper_on = False

            for i in range(1):
                # Extract delta and gripper from the i-th step of the chunk
                target_delta = actions[i][:3]
                raw_gripper = actions[i][-1]
                print(f"actions: {actions[i]}")
                thresh = 0.5

                if raw_gripper > thresh:
                    physical_gripper_on = True
                    print(">>> GRIPPER TRIGGERED: ON")
                elif raw_gripper < -thresh:
                    physical_gripper_on = False
                    print(">>> GRIPPER TRIGGERED: OFF")
                
                # AMPLIFIER: Fixes the "small increments" issue. 
                target_delta *= 1.0 

                # Calculate new target based on the LAST known position
                target_xyz = current_xyz + target_delta
                
                # Physical Move
                # Note: robot.move_and_grip contains the safety clamp logic
                robot.move_and_grip(target_xyz[0], target_xyz[1], target_xyz[2], 1.0 if physical_gripper_on else 0.0)
                
                # CRITICAL: Update current_xyz so the NEXT step in the loop 
                # starts from where the robot just moved to.
                current_xyz = target_xyz 

            time.sleep(0.5)

            # Visual debug (Update every 3 steps)
            cv2.imshow("Dobot View", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n[Terminating] Stopping robot...")
    finally:
        robot.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference_loop()