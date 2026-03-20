"""
test_jog_pose.py — Quick diagnostic to verify JOG + GetPose work together
==========================================================================
Run this FIRST before using teleop_record.py.

This script:
  1. Connects to the Dobot
  2. Sends a JOG X+ command for 1 second
  3. Polls GetPose at ~20 Hz during that movement
  4. Sends JOG Idle to stop
  5. Prints all the pose readings so you can verify they were updating

If the poses are all identical → there's a problem (stale reads).
If the X values increase smoothly → JOG + GetPose works, you're good to go.
"""

import time
import sys
import DobotDllType as dType

def main():
    print("=" * 50)
    print("  JOG + GetPose Concurrent Read Test")
    print("=" * 50)

    # Connect
    api = dType.load()
    state = dType.ConnectDobot(api, "", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("[FATAL] Could not connect to Dobot.")
        sys.exit(1)
    print("[OK] Connected.\n")

    dType.ClearAllAlarmsState(api)
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)

    # Configure JOG — slow speed for safety
    dType.SetJOGCoordinateParams(api,
        20, 20,   # X vel, acc
        20, 20,   # Y vel, acc
        20, 20,   # Z vel, acc
        20, 20,   # R vel, acc
        isQueued=0)
    dType.SetJOGCommonParams(api, 100, 100, isQueued=0)

    # Read initial pose
    pose0 = dType.GetPose(api)
    print(f"[START] X={pose0[0]:.2f} Y={pose0[1]:.2f} Z={pose0[2]:.2f} R={pose0[3]:.2f}")
    print()

    # --- TEST 1: JOG X+ for 1.5 seconds, poll GetPose ---
    print("[TEST] Sending JOG X+ for 1.5 seconds...")
    print("       Polling GetPose at ~20 Hz during movement...\n")

    readings = []
    dType.SetJOGCmd(api, 0, 1, isQueued=0)  # isJoint=0, cmd=1 (X+)

    t0 = time.time()
    while time.time() - t0 < 1.5:
        pose = dType.GetPose(api)
        t = time.time() - t0
        readings.append((t, pose[0], pose[1], pose[2]))
        time.sleep(0.05)  # ~20 Hz

    # Stop
    dType.SetJOGCmd(api, 0, 0, isQueued=0)  # Idle
    time.sleep(0.3)

    pose_final = dType.GetPose(api)
    print(f"{'Time':>6s}  {'X':>8s}  {'Y':>8s}  {'Z':>8s}")
    print("-" * 36)
    for t, x, y, z in readings:
        print(f"{t:6.3f}  {x:8.2f}  {y:8.2f}  {z:8.2f}")
    print("-" * 36)
    print(f"[FINAL] X={pose_final[0]:.2f} Y={pose_final[1]:.2f} Z={pose_final[2]:.2f}")

    # --- ANALYSIS ---
    x_values = [r[1] for r in readings]
    x_delta = x_values[-1] - x_values[0]
    unique_x = len(set(f"{x:.2f}" for x in x_values))

    print(f"\n[RESULT] X moved {x_delta:.2f} mm over {len(readings)} readings.")
    print(f"         {unique_x} unique X values observed.")

    if unique_x <= 2:
        print("\n[FAIL] GetPose appears to return STALE values during JOG.")
        print("       The readings are nearly identical — pose is not updating.")
        print("       We may need an alternative approach (CP mode or polling thread).")
    elif x_delta < 1.0:
        print("\n[WARN] Very small movement detected. The robot may not have moved.")
        print("       Check that the workspace is clear and the arm can move in X+.")
    else:
        print("\n[PASS] GetPose successfully reads updated positions during JOG!")
        print("       You are good to use teleop_record.py.")

    # --- TEST 2: Return to start ---
    print(f"\n[TEST] Sending JOG X- to return (1.5 sec)...")
    dType.SetJOGCmd(api, 0, 2, isQueued=0)  # X-
    time.sleep(1.5)
    dType.SetJOGCmd(api, 0, 0, isQueued=0)  # Idle
    time.sleep(0.3)

    pose_return = dType.GetPose(api)
    print(f"[RETURN] X={pose_return[0]:.2f} Y={pose_return[1]:.2f} Z={pose_return[2]:.2f}")
    print(f"         (started at X={pose0[0]:.2f})")

    # Cleanup
    dType.DisconnectDobot(api)
    print("\n[OK] Test complete. Disconnected.")


if __name__ == "__main__":
    main()
