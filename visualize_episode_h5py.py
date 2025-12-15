import cv2
import h5py
import os
import numpy as np

# --- CONFIGURATION ---
DATASET_DIR = "dataset_hdf5"
JOB_NAME = "pick_up_red_block"
EPISODE_NAME = "episode_010.h5" # Make sure this file actually exists!

EPISODE_FILE = os.path.join(DATASET_DIR, JOB_NAME, EPISODE_NAME)

def visualize_hdf5():
    if not os.path.exists(EPISODE_FILE):
        print(f"Error: File not found: {EPISODE_FILE}")
        return

    print(f"Opening {EPISODE_FILE}...")
    
    with h5py.File(EPISODE_FILE, 'r') as f:
        # 1. Read the datasets
        # These look like arrays: [num_steps, height, width, channels]
        images_top = f['observations/images/top'][:]
        images_side = f['observations/images/side'][:]
        actions = f['action'][:] # [num_steps, 7]
        state = f['observations/state'][:]
        
        # Metadata
        instruction = f.attrs.get('instruction', "No Instruction")
        print(f"Instruction: {instruction}")
        print(f"Total Steps: {len(actions)}")
        print("\nPress SPACE to step forward. 'q' to quit.")

        # 2. Loop through steps
        for i in range(len(actions)):
            # Grab images for this step
            frame1 = images_top[i]
            frame2 = images_side[i]
            
            # Grab Action (The 7-DOF Vector we saved)
            # [x, y, z, r, pitch, yaw, gripper]
            vec = actions[i] 
            gripper_state = vec[6]
            
            # --- DRAWING THE ARROW (Top Down View) ---
            # Center of image
            cx, cy = frame1.shape[1] // 2, frame1.shape[0] // 2
            
            # SCALE for visualization (pixels per mm)
            scale = 2.0 
            
            # User's Coordinate Mapping (Verify this!)
            # Robot X (Forward) -> Image Y (Up/Down) ??
            # Robot Y (Left/Right) -> Image X (Left/Right) ??
            # Note: In the recording script, we used absolute coordinates for 'action'.
            # To draw a "direction arrow", we really want (Target - Current).
            # But here we visualize the Target Position relative to center (Just for sanity)
            
            # If you want to visualize the DELTA (movement), we need:
            # delta = action - state
            curr_pose = state[i]
            target_pose = actions[i]
            
            dx = target_pose[0] - curr_pose[0]
            dy = target_pose[1] - curr_pose[1]
            
            # Draw movement vector
            # Assuming: Camera Top is aligned such that Robot +X is Image -Y (Up)
            end_x = int(cx + (dy * scale)) 
            end_y = int(cy - (dx * scale))
            
            cv2.arrowedLine(frame1, (cx, cy), (end_x, end_y), (0, 0, 255), 3)
            
            # --- DISPLAY ---
            # Resize for screen
            disp1 = cv2.resize(frame1, (400, 400))
            disp2 = cv2.resize(frame2, (400, 400))
            
            # Overlay Text
            status_text = f"Step {i} | Grip: {int(gripper_state)}"
            cv2.putText(disp1, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Stack side-by-side
            combined = np.hstack((disp1, disp2))
            
            cv2.imshow("HDF5 Viewer", combined)
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_hdf5()