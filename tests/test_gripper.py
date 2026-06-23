"""Gripper tests — the driver itself, controller wiring, teach capture, and
the server endpoint.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import (  # noqa: E402
    ArmController, ArmService, Program, ServoGripper, SimGripper, Waypoint,
    load_config,
)


# --- SimGripper -----------------------------------------------------------

def test_sim_gripper_open_close_state():
    g = SimGripper()
    assert g.state() == "open"
    g.close()
    assert g.state() == "close"
    g.open()
    assert g.state() == "open"


def test_sim_gripper_set_by_name():
    g = SimGripper()
    g.set("close")
    assert g.state() == "close"
    with pytest.raises(ValueError):
        g.set("halfway")


def test_sim_gripper_bad_start():
    with pytest.raises(ValueError):
        SimGripper(start="ajar")


# --- controller wiring ----------------------------------------------------

def test_simulated_arm_has_gripper():
    arm = ArmController.simulated(load_config())
    assert arm.gripper_state() == "open"
    arm.close_gripper()
    assert arm.gripper_state() == "close"


def test_set_gripper_none_is_noop():
    arm = ArmController.simulated(load_config())
    arm.set_gripper(None)
    assert arm.gripper_state() == "open"


def test_set_gripper_without_gripper_raises():
    cfg = load_config()
    from arm_control.kinematics import ArmKinematics
    from arm_control.drivers import SimDriver
    arm = ArmController(cfg, ArmKinematics.from_config(cfg),
                        [SimDriver(home_deg=j.home_deg) for j in cfg.joints],
                        gripper=None)
    assert arm.gripper_state() is None
    with pytest.raises(RuntimeError):
        arm.set_gripper("close")


def test_record_waypoint_captures_gripper_state():
    arm = ArmController.simulated(load_config())
    arm.close_gripper()
    wp = arm.record_waypoint("grabbed")
    assert wp.gripper == "close"


def test_run_program_actuates_gripper():
    arm = ArmController.simulated(load_config())
    prog = Program(name="grab")
    prog.add(Waypoint("reach", [0, 90, 0, 0], gripper="open"))
    prog.add(Waypoint("grab", [10, 80, -20, 5], gripper="close"))
    arm.run_program(prog)
    assert arm.gripper_state() == "close"


def test_pick_place_demo_ends_open():
    arm = ArmController.simulated(load_config())
    arm.run_program(Program.load("pick_place_demo"))
    # last waypoint releases the part
    assert arm.gripper_state() == "open"


# --- ServoGripper stub ----------------------------------------------------

def test_servo_gripper_is_stub():
    with pytest.raises(NotImplementedError):
        ServoGripper(servo_id=1)


# --- server ---------------------------------------------------------------

def test_state_includes_gripper():
    svc = ArmService.simulated(load_config())
    status, body = svc.dispatch("GET", "/state", None)
    assert status == 200
    assert body["gripper"] == "open"


def test_gripper_endpoint_closes():
    svc = ArmService.simulated(load_config())
    status, body = svc.dispatch("POST", "/gripper", {"state": "close"})
    assert status == 200
    assert body["state"]["gripper"] == "close"


def test_gripper_endpoint_missing_field_is_400():
    svc = ArmService.simulated(load_config())
    status, body = svc.dispatch("POST", "/gripper", {})
    assert status == 400


def test_gripper_endpoint_bad_value_is_400():
    svc = ArmService.simulated(load_config())
    status, body = svc.dispatch("POST", "/gripper", {"state": "halfway"})
    assert status == 400


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {name}: {e}")
    print("\ngripper tests done.")
