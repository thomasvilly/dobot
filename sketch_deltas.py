import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
FILE_PATH = "dataset_hdf5/simple_session/episode_004.h5" 
OUTPUT_PLOT = "debug_rolling_stride.png"

RECORDING_HZ = 10 
STRIDE = 3
GAUSSIAN_SIGMA = 1.0 

# --- REPLACEMENT FOR SCIPY ---
def gaussian_smooth_pure_numpy(data, sigma):
    """
    Recreates scipy.ndimage.gaussian_filter1d using only NumPy.
    Bypasses DLL blocking issues.
    """
    # 1. Generate the Gaussian Kernel
    # Standard radius is 4 * sigma
    radius = int(4 * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum() # Normalize so sum is 1.0 (maintains amplitude)
    
    smoothed = np.zeros_like(data)
    
    # 2. Convolve each column (X, Y, Z)
    for col in range(data.shape[1]):
        # Pad the edges so the line doesn't drop to zero at start/end
        # 'edge' padding repeats the first/last value
        padded = np.pad(data[:, col], radius, mode='edge')
        
        # 'valid' mode returns only the parts where kernel fully overlaps
        convolved = np.convolve(padded, kernel, mode='valid')
        smoothed[:, col] = convolved
        
    return smoothed

def calculate_deltas(states):
    n_steps = len(states)
    actions = []
    
    # --- ROLLING WINDOW (Stride 3, Iterate 1) ---
    for i in range(0, n_steps - STRIDE, 1): 
        curr_state = states[i]
        future_state = states[i + STRIDE]
        
        curr_xyz = curr_state[0:3]
        curr_grip = curr_state[6]
        
        future_xyz = future_state[0:3]
        future_grip = future_state[6]
        
        delta_xyz = future_xyz - curr_xyz
        grip_cmd = future_grip
        
        actions.append(np.concatenate([delta_xyz, [grip_cmd]]))
        
    return np.array(actions)

def create_visuals(file_path):
    if not os.path.exists(file_path): 
        print(f"File not found: {file_path}")
        return

    with h5py.File(file_path, 'r') as f:
        states = f['observations/state'][:]
        
        # 1. Calculate Rolling Deltas
        actions = calculate_deltas(states)
        
        if len(actions) == 0:
            print("Error: No actions generated. Check STRIDE vs. Episode Length.")
            return

        # 2. Apply Pure NumPy Smoothing
        print(f"Smoothing with Sigma {GAUSSIAN_SIGMA} (Pure NumPy)...")
        xyz_smooth = gaussian_smooth_pure_numpy(actions[:, 0:3], sigma=GAUSSIAN_SIGMA)
        
        dx, dy, dz = xyz_smooth[:, 0], xyz_smooth[:, 1], xyz_smooth[:, 2]
        grip = actions[:, 3]

        # Plot
        fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        fig.suptitle(f'Smoothed Actions (10Hz, Stride {STRIDE}, Sigma {GAUSSIAN_SIGMA})', fontsize=16)

        ax1.plot(dx, label='Delta X', color='#FF5733')
        ax1.plot(dy, label='Delta Y', color='#33FF57')
        ax1.plot(dz, label='Delta Z', color='#3357FF')
        ax1.set_ylabel('Delta (mm)', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Translation (Rolling Window + Gaussian)', fontsize=10)

        ax3.plot(grip, label='Gripper', color='black', linewidth=2)
        ax3.set_title('Gripper State', fontsize=10)

        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=300)
        print(f"-> Saved {OUTPUT_PLOT} (Total Frames: {len(actions)})")
        plt.show()

if __name__ == "__main__":
    create_visuals(FILE_PATH)