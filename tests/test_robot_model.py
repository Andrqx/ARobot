"""Robot-model / URDF export tests.

We don't need a physics engine here — just prove the exported URDF is
well-formed XML with the expected chain, joints, and limits, and that CAD
meshes flow through when provided.
"""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import RobotModel, load_config  # noqa: E402


def test_model_chain_from_config():
    model = RobotModel.from_config(load_config())
    assert [l.name for l in model.links] == ["column", "upper", "forearm", "tool"]
    # base joint rotates about Z (yaw), the rest about Y (pitch)
    assert model.links[0].axis == "z"
    assert all(l.axis == "y" for l in model.links[1:])


def test_urdf_is_wellformed_xml_with_four_joints():
    model = RobotModel.from_config(load_config())
    urdf = model.to_urdf()
    root = ET.fromstring(urdf)  # raises if malformed
    assert root.tag == "robot"
    joints = root.findall("joint")
    assert len(joints) == 4
    assert all(j.get("type") == "revolute" for j in joints)
    # 5 links: base_link + 4 segments
    assert len(root.findall("link")) == 5


def test_urdf_joint_limits_match_config():
    cfg = load_config()
    model = RobotModel.from_config(cfg)
    root = ET.fromstring(model.to_urdf())

    by_name = {j.get("name"): j for j in root.findall("joint")}
    elbow_cfg = next(j for j in cfg.joints if j.name == "elbow")
    limit = by_name["elbow"].find("limit")
    assert math.isclose(float(limit.get("lower")), math.radians(elbow_cfg.min_deg), abs_tol=1e-3)
    assert math.isclose(float(limit.get("upper")), math.radians(elbow_cfg.max_deg), abs_tol=1e-3)


def test_cad_mesh_flows_into_urdf():
    cfg = load_config()
    geom = {"links": {"upper": {"mesh": "meshes/upper.stl", "mass_kg": 0.7}}}
    model = RobotModel.from_config(cfg, geometry=geom)
    urdf = model.to_urdf()
    assert 'filename="meshes/upper.stl"' in urdf
    upper = next(l for l in model.links if l.name == "upper")
    assert upper.mesh == "meshes/upper.stl"
    assert upper.mass_kg == 0.7


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll robot-model tests passed.")
