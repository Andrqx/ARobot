"""Perception tests — coordinate transforms, the detector seam, and the
detection -> base -> motion pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import (  # noqa: E402
    ArmController, CameraExtrinsics, Detection, PerceptionPipeline,
    SimDetector, ArucoCameraDetector, load_config,
)


def make_arm():
    return ArmController.simulated(load_config())


# --- Detection -------------------------------------------------------------

def test_detection_rejects_bad_shape():
    with pytest.raises(ValueError):
        Detection("x", [1, 2])


# --- CameraExtrinsics ------------------------------------------------------

def test_identity_extrinsics_passthrough():
    ext = CameraExtrinsics.identity()
    assert np.allclose(ext.to_base([10, 20, 30]), [10, 20, 30])


def test_extrinsics_translation_only():
    ext = CameraExtrinsics.from_pose([100, 0, 500])
    assert np.allclose(ext.to_base([0, 0, 0]), [100, 0, 500])
    assert np.allclose(ext.to_base([10, 0, 0]), [110, 0, 500])


def test_extrinsics_yaw_90_rotates_x_into_y():
    # Yaw +90deg about Z: camera +x should land on base +y.
    ext = CameraExtrinsics.from_pose([0, 0, 0], rpy_deg=[0, 0, 90])
    assert np.allclose(ext.to_base([10, 0, 0]), [0, 10, 0], atol=1e-9)


def test_extrinsics_rotation_then_translation():
    ext = CameraExtrinsics.from_pose([5, 5, 5], rpy_deg=[0, 0, 90])
    assert np.allclose(ext.to_base([10, 0, 0]), [5, 15, 5], atol=1e-9)


# --- SimDetector -----------------------------------------------------------

def test_simdetector_reads_and_updates():
    det = SimDetector([Detection("a", [0, 0, 100])])
    assert [d.target_id for d in det.read()] == ["a"]
    det.set([Detection("b", [1, 1, 1]), Detection("c", [2, 2, 2])])
    assert [d.target_id for d in det.read()] == ["b", "c"]


# --- Pipeline transforms + selection --------------------------------------

def test_targets_in_base_applies_extrinsics():
    src = SimDetector([Detection("cube", [10, 0, 0])])
    ext = CameraExtrinsics.from_pose([300, 0, 200], rpy_deg=[0, 0, 90])
    pipe = PerceptionPipeline(make_arm(), src, ext)
    targets = pipe.targets_in_base()
    assert len(targets) == 1
    assert np.allclose(targets[0].position_base_mm, [300, 10, 200], atol=1e-9)


def test_min_confidence_filters_detections():
    src = SimDetector([
        Detection("sure", [0, 0, 100], confidence=0.9),
        Detection("maybe", [0, 0, 100], confidence=0.2),
    ])
    pipe = PerceptionPipeline(make_arm(), src, min_confidence=0.5)
    ids = [t.target_id for t in pipe.targets_in_base()]
    assert ids == ["sure"]


def test_nearest_picks_closest_to_base():
    src = SimDetector([
        Detection("far", [600, 0, 0]),
        Detection("near", [200, 0, 0]),
    ])
    pipe = PerceptionPipeline(make_arm(), src, CameraExtrinsics.identity())
    assert pipe.nearest().target_id == "near"


def test_nearest_none_when_empty():
    pipe = PerceptionPipeline(make_arm(), SimDetector([]))
    assert pipe.nearest() is None


# --- Pipeline drives the arm ----------------------------------------------

def test_move_to_detection_lands_tool_on_target():
    arm = make_arm()
    # Identity extrinsics: camera coords == base coords. Reachable point.
    src = SimDetector([Detection("cube", [300, 0, 250])])
    pipe = PerceptionPipeline(arm, src, CameraExtrinsics.identity())
    pipe.pick_nearest_and_move(pitch_deg=0)
    x, y, z, _ = arm.kin.tip_pose_deg(arm.current_angles_deg())
    assert np.allclose([x, y, z], [300, 0, 250], atol=1e-1)


def test_hover_offset_raises_goal_in_z():
    arm = make_arm()
    src = SimDetector([Detection("cube", [300, 0, 250])])
    pipe = PerceptionPipeline(arm, src, CameraExtrinsics.identity())
    pipe.pick_nearest_and_move(pitch_deg=0, hover_mm=40)
    _, _, z, _ = arm.kin.tip_pose_deg(arm.current_angles_deg())
    assert abs(z - 290) < 1e-1   # 250 + 40 hover


def test_pick_nearest_and_move_returns_target_and_traj():
    arm = make_arm()
    src = SimDetector([Detection("cube", [300, 0, 250])])
    pipe = PerceptionPipeline(arm, src, CameraExtrinsics.identity())
    result = pipe.pick_nearest_and_move(pitch_deg=0)
    assert result is not None
    target, traj = result
    assert target.target_id == "cube"
    assert len(traj.pos) > 0


def test_pick_nearest_and_move_none_when_nothing_seen():
    pipe = PerceptionPipeline(make_arm(), SimDetector([]))
    assert pipe.pick_nearest_and_move() is None


def test_unreachable_target_propagates():
    from arm_control.kinematics import Unreachable
    arm = make_arm()
    src = SimDetector([Detection("moon", [5000, 0, 300])])
    pipe = PerceptionPipeline(arm, src, CameraExtrinsics.identity())
    with pytest.raises(Unreachable):
        pipe.pick_nearest_and_move(pitch_deg=0)


# --- hardware stub ---------------------------------------------------------

def test_aruco_detector_is_stub():
    with pytest.raises(NotImplementedError):
        ArucoCameraDetector(marker_length_mm=30.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {name}: {e}")
    print("\nperception tests done.")
