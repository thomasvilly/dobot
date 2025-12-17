import cv2
import h5py
import os
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
DATASET_DIR = "dataset_hdf5"
JOB_NAME = "pick_up_blue_block"
EPISODE_NAME = "episode_003.h5" 
CSV_OUTPUT = "episode_analysis.csv"

EPISODE_FILE = os.path.join(DATASET_DIR, JOB_NAME, EPISODE_NAME)

def export_to_csv(states, deltas, instruction):
    """Exports the episode data to a single spreadsheet for analysis."""
    data = []
    labels = ['X', 'Y', 'Z', 'Rot', 'Pit', 'Yaw', 'Grip']
    
    for i in range(len(states)):
        row = {'step': i, 'instruction': instruction}
        # Add Current State values
        for idx, label in enumerate(labels):
            row[f'state_{label}'] = states[i][idx]
        # Add Temporal Delta values
        for idx, label in enumerate(labels):
            row[f'delta_{label}'] = deltas[i][idx]
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"\n📊 Spreadsheet updated: {CSV_OUTPUT}")

def draw_text_sidebar(img, state, next_delta, step):
    h, w, _ = img.shape
    sidebar = np.zeros((h, 300, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    fs = 0.5
    
    cv2.putText(sidebar, f"STEP: {step:03d}", (15, 30), font, 0.6, (255, 255, 255), 1)
    labels = ['X', 'Y', 'Z', 'Rot', 'Pit', 'Yaw', 'Grip']
    
    # State Display
    y = 70
    cv2.putText(sidebar, "STATE [i]", (15, y), font, 0.5, (0, 255, 255), 1)
    for idx, label in enumerate(labels):
        y += 25
        cv2.putText(sidebar, f"{label}: {state[idx]:.2f}", (20, y), font, fs, (200, 200, 200), 1)

    # Delta Display
    y += 40
    cv2.putText(sidebar, "NEXT DELTA [i+1-i]", (15, y), font, 0.5, (0, 255, 0), 1)
    for idx, label in enumerate(labels):
        y += 25
        val = next_delta[idx]
        color = (0, 255, 0) if abs(val) > 0.01 else (100, 100, 100)
        cv2.putText(sidebar, f"{label}: {val:+.4f}", (20, y), font, fs, color, 1)

    return sidebar

def visualize_and_analyze():
    if not os.path.exists(EPISODE_FILE):
        print(f"Error: File not found: {EPISODE_FILE}")
        return

    with h5py.File(EPISODE_FILE, 'r') as f:
        images_top = f['observations/images/top'][:]
        state = f['observations/state'][:]
        instruction = f.attrs.get('instruction', "No Instruction")

        num_steps = len(state)
        all_deltas = []

        # 1. Pre-calculate all deltas for CSV export
        for i in range(num_steps):
            if i < num_steps - 1:
                all_deltas.append(state[i+1] - state[i])
            else:
                all_deltas.append(np.zeros_like(state[i]))
        
        # 2. Save to CSV immediately
        export_to_csv(state, all_deltas, instruction)

        # 3. Visual Loop
        print("Visualizer Active. Press 'q' to quit, any other key for next step.")
        for i in range(num_steps):
            draw_top = images_top[i].copy()
            next_delta = all_deltas[i]
            
            # Draw movement vector on image
            cx, cy = draw_top.shape[1] // 2, draw_top.shape[0] // 2
            ex, ey = int(cx + (next_delta[1] * 10)), int(cy + (next_delta[0] * 10))
            if np.linalg.norm(next_delta[:2]) > 0.1:
                cv2.arrowedLine(draw_top, (cx, cy), (ex, ey), (0, 0, 255), 2)
            
            d_top = cv2.resize(draw_top, (512, 512))
            sidebar = draw_text_sidebar(d_top, state[i], next_delta, i)
            layout = np.hstack((sidebar, d_top))
            
            cv2.imshow("Temporal Delta Analysis", layout)
            if cv2.waitKey(0) == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_and_analyze()