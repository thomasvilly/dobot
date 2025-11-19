import cv2
import json
import glob
import os
import numpy as np

DATASET_DIR = "dataset"
# Change this to the specific job/episode you want to check
EPISODE_PATH = os.path.join(DATASET_DIR, "watcard_to_measuring_tape", "episode_002")

def visualize_episode():
    # Find all JSON files and sort them
    files = sorted(glob.glob(os.path.join(EPISODE_PATH, "*.json")))
    
    if not files:
        print(f"No files found in {EPISODE_PATH}")
        return

    print("Press SPACE to step forward. 'q' to quit.")

    for i, json_path in enumerate(files):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Load Images
        img1_path = os.path.join(EPISODE_PATH, data['images']['cam1'])
        img2_path = os.path.join(EPISODE_PATH, data['images']['cam2'])
        
        if not os.path.exists(img1_path): continue

        frame1 = cv2.imread(img1_path)
        frame2 = cv2.imread(img2_path)
        
        # Get Action Vector (The Label)
        # vec = [dx, dy, dz, dr]
        vec = data['action']['delta_vector']
        
        # --- DRAWING THE ARROW ---
        # We project the 3D action vector onto the 2D image
        # This is an approximation, but usually:
        # Cam 1 (Top Down): X axis = Up/Down image, Y axis = Left/Right image
        # You might need to flip these multipliers based on your specific camera mounting!
        
        # Center of image
        cx, cy = frame1.shape[1] // 2, frame1.shape[0] // 2
        
        # SCALE for visualization (pixels per mm)
        scale = 5.0 
        
        # Calculate arrow endpoint
        # NOTE: You must verify if +X in robot frame corresponds to +X in pixel frame!
        end_x = int(cx + (vec[1] * scale)) # Using Y for horizontal
        end_y = int(cy - (vec[0] * scale)) # Using X for vertical (Minus because pixel Y is down)
        
        cv2.arrowedLine(frame1, (cx, cy), (end_x, end_y), (0, 0, 255), 2)
        
        # Resize for display
        disp1 = cv2.resize(frame1, (400, 400))
        disp2 = cv2.resize(frame2, (400, 400))
        combined = np.hstack((disp1, disp2))
        
        # Add Text
        text = f"Step {i} | Action: {np.round(vec, 1)}"
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Sanity Check", combined)
        key = cv2.waitKey(0) # Wait for key press
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_episode()