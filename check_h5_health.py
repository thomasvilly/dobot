import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# --- CONFIGURATION ---
# UPDATED PATH: Recursive search in D:\DOBOT\dataset_hdf5
DATA_DIR = r"C:\Code\dobot\dataset_hdf5\interactive_session" #r"D:\DOBOT\dataset_hdf5"
# ---------------------

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Path not found: {DATA_DIR}")
        return

    print(f"Scanning for HDF5 files recursively in: {DATA_DIR}")
    
    # Search for both .h5 and .hdf5 extensions
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.h5"), recursive=True)
    files += glob.glob(os.path.join(DATA_DIR, "**", "*.hdf5"), recursive=True)
    
    if not files:
        print("No .h5 or .hdf5 files found! Please check the folder structure.")
        return

    print(f"Found {len(files)} episodes.")
    print(f"Sample file: {files[0]}") # Print one to confirm it looks right
    print("Analyzing data health...")

    all_actions = []
    episode_lengths = []
    
    for fpath in files:
        try:
            with h5py.File(fpath, 'r') as f:
                # Check for various common action keys
                if 'action' in f:
                    act = f['action'][:]
                elif 'actions' in f:
                    act = f['actions'][:]
                elif 'rel_actions' in f:
                    act = f['rel_actions'][:]
                else:
                    print(f"Skipping {os.path.basename(fpath)}: No 'action' key found.")
                    continue
                
                all_actions.append(act)
                episode_lengths.append(len(act))
        except Exception as e:
            print(f"Error reading {os.path.basename(fpath)}: {e}")

    if not all_actions:
        print("No valid action data extracted.")
        return

    # Concatenate all steps
    flat_actions = np.concatenate(all_actions, axis=0)
    print(f"\nStats collected from {len(files)} episodes ({len(flat_actions)} total steps).")

    # --- ANALYSIS 1: VELOCITY (XYZ) ---
    # Assuming actions are [dx, dy, dz, dr, grip]
    # We take the first 3 for movement magnitude
    xyz_deltas = flat_actions[:, :3]
    magnitudes = np.linalg.norm(xyz_deltas, axis=1)
    
    # Check for "Dead" frames (movement < 0.1 units)
    # Note: If your data is in meters, 0.1 is huge. If mm, 0.1 is tiny.
    # Adjust threshold if needed. Assuming MM here based on Dobot.
    dead_frames = np.sum(magnitudes < 0.1) 
    dead_percent = (dead_frames / len(magnitudes)) * 100

    print(f"\n[Velocity Health]")
    print(f"  Mean Action Magnitude: {np.mean(magnitudes):.4f}")
    print(f"  Max Action Magnitude:  {np.max(magnitudes):.4f}")
    print(f"  'Dead' Frames (<0.1):  {dead_percent:.1f}%")
    
    if dead_percent > 30:
        print("  WARNING: High percentage of dead frames! Model will learn to sit still.")

    # --- ANALYSIS 2: GRIPPER ---
    # Assuming gripper is the last dimension
    gripper = flat_actions[:, -1]
    unique_grips, counts = np.unique(np.round(gripper, 2), return_counts=True)
    
    print(f"\n[Gripper Health]")
    for val, count in zip(unique_grips, counts):
        print(f"    Value {val}: {count} frames ({count/len(gripper)*100:.1f}%)")

    # --- PLOTTING ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Velocity Hist
    axs[0, 0].hist(magnitudes, bins=50, color='blue', alpha=0.7)
    axs[0, 0].set_title(f"Velocity Magnitude (Dead: {dead_percent:.1f}%)")
    axs[0, 0].set_xlabel("Movement Delta")
    if dead_percent > 30:
        axs[0, 0].set_facecolor('#ffe6e6') # Red warning

    # 2. XYZ Distribution
    axs[0, 1].hist(flat_actions[:, 0], bins=50, alpha=0.5, label='X', color='r')
    axs[0, 1].hist(flat_actions[:, 1], bins=50, alpha=0.5, label='Y', color='g')
    axs[0, 1].hist(flat_actions[:, 2], bins=50, alpha=0.5, label='Z', color='b')
    axs[0, 1].set_title("XYZ Components")
    axs[0, 1].legend()

    # 3. Episode Lengths
    axs[1, 0].hist(episode_lengths, bins=20, color='purple', alpha=0.7)
    axs[1, 0].set_title("Episode Lengths")

    # 4. Gripper
    if flat_actions.shape[1] >= 5:
        axs[1, 1].bar(unique_grips, counts, width=0.1, color='orange')
        axs[1, 1].set_title("Gripper States")

    plt.tight_layout()
    plt.show()

    velocities = np.diff(flat_actions[:, :3], axis=0) # Shape: (N-1, 3)
    
    # Now calculate magnitude of the MOVEMENT
    vel_magnitudes = np.linalg.norm(velocities, axis=1)
    
    # Check for TRUE Dead Frames (Movement < 0.5mm)
    true_dead_frames = np.sum(vel_magnitudes < 0.5)
    true_dead_percent = (true_dead_frames / len(vel_magnitudes)) * 100

    print(f"\n[TRUE Velocity Health]")
    print(f"  Mean Movement: {np.mean(vel_magnitudes):.4f} mm/step")
    print(f"  Max Movement:  {np.max(vel_magnitudes):.4f} mm/step")
    print(f"  Stationary Frames (<0.5mm): {true_dead_percent:.1f}%")

if __name__ == "__main__":
    main()