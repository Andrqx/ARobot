"""Kinematics tests — prove the math without any hardware.

The core guarantee: for any reachable pose, IK then FK must return the
original pose (a "round trip"). If that holds, the arm will go where you
tell it to in simulation, and the same math runs on the Pi.

Run:  pytest -q       (from the control/ folder)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import ArmKinematics, Unreachable, load_config  # noqa: E402

KIN = ArmKinematics(L1=150.0, L2=200.0, L3=180.0, L4=120.0)


def test_config_loads_and_derives_constants():
    cfg = load_config()
    assert len(cfg.joints) == 4
    elbow = next(j for j in cfg.joints if j.name == "elbow")
    # 200 full steps * 16 microsteps * 20.25 ratio / 360 deg (double belt 4.5^2)
    assert np.isclose(cfg.steps_per_deg(elbow), 200 * 16 * 20.25 / 360.0)
    # Output torque after reduction: NEMA 17 (0.3 N·m) * 20.25 * 0.85 efficiency
    assert np.isclose(elbow.output_torque_nm(), 0.3 * 20.25 * 0.85)
    # Wrist is a bus servo: torque already at the output (no extra reduction).
    wrist = next(j for j in cfg.joints if j.name == "wrist")
    assert wrist.driver_type == "bus_servo"
    assert np.isclose(wrist.output_torque_nm(), 2.9)


def test_forward_home_geometry():
    # Shoulder straight up, others zero: tip should be directly above base.
    points, pitch = KIN.forward_deg([0.0, 90.0, 0.0, 0.0])
    tip = points["tip"]
    assert np.isclose(tip[0], 0.0, atol=1e-6)
    assert np.isclose(tip[1], 0.0, atol=1e-6)
    # height = L1 + L2 + L3 + L4 all stacked vertically
    assert np.isclose(tip[2], 150 + 200 + 180 + 120, atol=1e-6)
    assert np.isclose(pitch, 90.0)


def test_ik_fk_round_trip():
    rng = np.random.default_rng(42)
    checked = 0
    for _ in range(500):
        x = rng.uniform(-350, 350)
        y = rng.uniform(-350, 350)
        z = rng.uniform(50, 500)
        pitch_deg = rng.uniform(-60, 60)
        try:
            q = KIN.inverse_deg(x, y, z, pitch_deg)
        except Unreachable:
            continue
        fx, fy, fz, fpitch = KIN.tip_pose_deg(q)
        assert np.allclose([fx, fy, fz], [x, y, z], atol=1e-4)
        assert np.isclose(fpitch, pitch_deg, atol=1e-4)
        checked += 1
    assert checked > 50, "too few reachable samples — check the geometry"


def test_elbow_up_and_down_both_valid():
    target = dict(x=250.0, y=0.0, z=300.0, pitch_deg=0.0)
    up = KIN.inverse_deg(**target, elbow_up=True)
    down = KIN.inverse_deg(**target, elbow_up=False)
    # Different joint solutions, same tool pose.
    assert not np.allclose(up, down)
    for q in (up, down):
        fx, fy, fz, _ = KIN.tip_pose_deg(q)
        assert np.allclose([fx, fy, fz], [250.0, 0.0, 300.0], atol=1e-4)


def test_unreachable_raises():
    far = (KIN.L1 + KIN.L2 + KIN.L3 + KIN.L4) * 2
    try:
        KIN.inverse_deg(far, 0.0, 0.0, 0.0)
    except Unreachable:
        return
    raise AssertionError("expected Unreachable for a target beyond max reach")


if __name__ == "__main__":
    # Allow running without pytest installed.
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll kinematics tests passed.")
