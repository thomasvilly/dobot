# Dobot Magician OpenVLA Fine-Tuning

This repository contains a robust data collection pipeline for fine-tuning Vision-Language-Action (VLA) models (specifically [OpenVLA](https://github.com/openvla/openvla)) using a Dobot Magician robot arm.

The system implements a "Stop-and-Go" recording strategy to ensure high-quality, blur-free image capture with precise ground-truth action labels.

## 🤖 Hardware Setup
* **Robot:** Dobot Magician (Suction Cup End Effector)
* **Vision:** 2x USB Webcams (Orthogonal setup: Top-down + Side-profile)
* **Platform:** Windows (Required for proprietary Dobot Drivers)

## 📦 Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Dobot SDK:**
    * Ensure `DobotDllType.py` and `DobotDll.dll` are in the root directory. These are proprietary files from the DobotStudio SDK.

## 🚀 Usage

### 1. Define a Job Recipe
Create a JSON file in the `job_recipes/` folder. This defines the "Script" the robot will follow.
* **`instruction`**: The language prompt the VLA will learn.
* **`segments`**: The physical waypoints (Move or Action).

*Example (`job_recipes/pick_red_block.json`):*
```json
{
  "job_id": "pick_red_block",
  "instruction": "Pick up the red block and place it on the tape measure",
  "settings": { "step_size_mm": 12.0, "wait_time_sec": 0.2 },
  "segments": [
    { "type": "move", "target": [220, 0, 0, 0], "desc": "Approach" },
    { "type": "action", "suction": 1, "desc": "Grab" },
    { "type": "move", "target": [200, 0, 50, 0], "desc": "Lift" }
  ]
}
````

### 2\. Run Data Collection

Run the collector script specifying the job name. The script handles homing, connection, and recording.

```bash
python collect_job.py --job pick_red_block
```

  * **Episode Management:** The script automatically detects existing data and increments the episode count (e.g., `episode_001`, `episode_002`).
  * **Reset:** After the job finishes, manually reset the scene (move the object back) and run the command again.

## 📂 Data Structure

The pipeline produces an **OpenVLA-ready** dataset structure.

```text
dataset/
└── pick_red_block/
    └── episode_001/
        ├── step_0000_cam1.jpg      # Top View
        ├── step_0000_cam2.jpg      # Side View
        └── step_0000_data.json     # Label & Metadata
```

### JSON Label Format

Each frame is paired with a JSON containing the **Language Instruction** and the **Action Label** (the vector required to reach the next step).

```json
{
  "language_instruction": "Pick up the red block...",
  "current_pose": [200.0, 100.0, 50.0, 0.0],
  "action": {
    "delta_vector": [-1.0, -10.5, -6.9, -1.6],
    "gripper_closed": 0
  }
}
```

## 🛠️ Methodology

We utilize a **Procedural Interpolation** approach:

1.  The script calculates a linear path between recipe waypoints.
2.  The robot moves in small increments (e.g., 12mm).
3.  The robot **stops completely**.
4.  Cameras capture the state; the vector to the *next* waypoint is calculated as the label.
5.  Repeat.

This eliminates motion blur and lag, creating a dataset ideal for Behavior Cloning.