"""End-to-end demo of the simulated arm — no hardware needed.

    python examples/demo.py

Shows: loading config, moving to a Cartesian pose via IK, reading back the
(simulated) joint state, and printing the derived steps-per-degree that the
real motors will use.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import ArmController, load_config  # noqa: E402


def main():
    cfg = load_config()
    arm = ArmController.simulated(cfg)

    print("=== ARobot control stack — simulated ===\n")
    print("Link lengths (mm):", cfg.link_lengths)
    print("\nPer-joint motion constants:")
    for j in cfg.joints:
        print(f"  {j.name:9s} ratio={j.gear_ratio:5.1f}  "
              f"{cfg.steps_per_deg(j):8.1f} microsteps/deg  "
              f"{cfg.encoder_counts_per_deg(j):8.1f} enc-counts/deg")

    print("\nHome angles (deg):", np.round(arm.current_angles_deg(), 1))

    target = dict(x=250.0, y=80.0, z=300.0, pitch_deg=0.0)
    print(f"\nPlanning trapezoidal move to {target} ...")
    traj = arm.plan_to_pose(**target)
    print(f"  duration: {traj.duration:.2f} s over {len(traj)} samples")
    print(f"  peak joint speeds (deg/s): {np.round(traj.peak_vel, 1)}")
    arm.execute(traj)

    final = arm.current_angles_deg()
    print("Final joint angles (deg):", np.round(final, 1))

    # Confirm where the tool actually ended up.
    x, y, z, pitch = arm.kin.tip_pose_deg(final)
    print(f"Tool tip now at x={x:.1f} y={y:.1f} z={z:.1f} pitch={pitch:.1f}deg")
    print("\nMove complete (simulated). To visualize:  python -m sim.visualize")


if __name__ == "__main__":
    main()
