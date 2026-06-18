"""Teach-and-repeat tests — record, save/load, and replay programs."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import ArmController, Program, Waypoint, load_config  # noqa: E402


def test_program_save_load_roundtrip(tmp_path):
    prog = Program(name="t", description="demo")
    prog.add(Waypoint("a", [0, 90, 0, 0]))
    prog.add(Waypoint("b", [10, 80, -20, 5], pause_s=0.3, gripper="close"))

    path = prog.save(tmp_path / "t.json")
    loaded = Program.load(path)

    assert loaded.name == "t"
    assert loaded.description == "demo"
    assert len(loaded) == 2
    assert loaded.waypoints[1].joints_deg == [10, 80, -20, 5]
    assert loaded.waypoints[1].pause_s == 0.3
    assert loaded.waypoints[1].gripper == "close"


def test_record_waypoint_captures_current_pose():
    cfg = load_config()
    arm = ArmController.simulated(cfg)
    arm.move_to_pose(x=250, y=80, z=300, pitch_deg=0)
    wp = arm.record_waypoint("here")
    assert wp.name == "here"
    assert np.allclose(wp.joints_deg, arm.current_angles_deg(), atol=1e-2)


def test_run_program_reaches_every_waypoint():
    cfg = load_config()
    arm = ArmController.simulated(cfg)

    prog = Program(name="square")
    targets = [
        [0, 90, 0, 0],
        [30, 70, -40, 10],
        [-30, 60, -60, 20],
        [0, 90, 0, 0],
    ]
    for i, q in enumerate(targets):
        prog.add(Waypoint(f"p{i}", q))

    visited = []
    arm.run_program(prog, on_waypoint=lambda wp, traj: visited.append(wp.name))

    assert visited == ["p0", "p1", "p2", "p3"]
    assert np.allclose(arm.current_angles_deg(), targets[-1], atol=1e-3)


def test_bundled_demo_program_loads_and_runs():
    cfg = load_config()
    arm = ArmController.simulated(cfg)
    prog = Program.load("pick_place_demo")  # by bare name from programs/
    assert len(prog) == 7
    trajs = arm.run_program(prog)
    assert len(trajs) == 7
    # ends back at home
    assert np.allclose(arm.current_angles_deg(), [0, 90, 0, 0], atol=1e-3)


def test_homing_sets_flag():
    cfg = load_config()
    arm = ArmController.simulated(cfg)
    assert arm.homed is False
    arm.move_to_pose(x=250, y=80, z=300, pitch_deg=0)
    arm.home()
    assert arm.homed is True
    assert np.allclose(arm.current_angles_deg(), cfg.home_angles_deg(), atol=1e-3)


if __name__ == "__main__":
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"PASS {name}")
    print("\nAll teach tests passed.")
