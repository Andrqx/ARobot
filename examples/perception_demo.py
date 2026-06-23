"""Perception demo — a simulated camera spots a cube and the arm goes to it.

    python examples/perception_demo.py

No hardware and no real camera: ``SimDetector`` stands in for the Pi camera.
Swap in ``ArucoCameraDetector`` later and nothing else changes.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import (  # noqa: E402
    ArmController, CameraExtrinsics, Detection, PerceptionPipeline,
    SimDetector, load_config,
)


def main():
    arm = ArmController.simulated(load_config())
    arm.home()

    # Pretend a camera is mounted 300mm in front of the base, looking back.
    # (Identity extrinsics here keeps the demo's numbers easy to read — the
    #  cube's camera coords already equal base coords.)
    extrinsics = CameraExtrinsics.identity()

    # The "camera" sees one cube sitting in the workspace.
    camera = SimDetector([Detection("cube", position_cam_mm=[300, 60, 250])])
    pipe = PerceptionPipeline(arm, camera, extrinsics)

    print("=== ARobot perception demo (simulated camera) ===\n")
    for t in pipe.targets_in_base():
        print(f"  saw '{t.target_id}' at base {np.round(t.position_base_mm, 1)} "
              f"(conf {t.confidence:.2f}, {t.distance_mm:.0f}mm away)")

    print("\nMoving to nearest target, hovering 40mm above it ...")
    target, traj = pipe.pick_nearest_and_move(pitch_deg=0, hover_mm=40)

    x, y, z, pitch = arm.kin.tip_pose_deg(arm.current_angles_deg())
    print(f"  picked '{target.target_id}'")
    print(f"  tool tip now at x={x:.1f} y={y:.1f} z={z:.1f} (target z+hover=290)")
    print(f"  move took {traj.duration:.2f}s over {len(traj.pos)} steps")
    print("\nReal camera later: swap SimDetector -> ArucoCameraDetector, done.")


if __name__ == "__main__":
    main()
