"""Export the arm to a URDF file for physics-sim training.

    python examples/export_urdf.py

Writes ``arobot.urdf`` you can load in PyBullet / MuJoCo / Gazebo. Uses
placeholder cylinder geometry until CAD meshes are listed in
config/geometry.yaml — then re-run and the URDF references your STLs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import RobotModel, load_config  # noqa: E402


def main():
    cfg = load_config()
    model = RobotModel.from_config(cfg)
    out = Path(__file__).resolve().parent.parent / "arobot.urdf"
    model.save_urdf(out)
    print(f"wrote {out}")
    print(f"links: {[l.name for l in model.links]}")
    meshed = [l.name for l in model.links if l.mesh]
    print(f"using CAD meshes for: {meshed or '(none yet — placeholder cylinders)'}")
    print("\nLoad it in PyBullet to start training:")
    print("  import pybullet as p, pybullet_data")
    print("  p.connect(p.GUI); p.loadURDF('arobot.urdf', useFixedBase=True)")


if __name__ == "__main__":
    main()
