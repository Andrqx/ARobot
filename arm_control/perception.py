"""Perception — turn "the camera sees a thing" into "move the arm there".

This is roadmap item #8, built the same way as the rest of the stack:
**simulation-first, with one clean seam to hardware.** The arm doesn't care
*how* a target was found — it only needs a 3D point. So perception is split
into three small pieces:

1. :class:`Detection` — a target the camera found, as ``(x, y, z)`` in the
   **camera's** own coordinate frame (millimetres).
2. :class:`CameraExtrinsics` — the fixed rigid transform that says where the
   camera sits relative to the arm's base. It converts a point from camera
   coordinates into **base/world** coordinates (the frame ``move_to_pose``
   uses). You measure this once with a hand-eye calibration.
3. :class:`DetectionSource` — the seam. ``SimDetector`` makes up detections so
   the whole pipeline runs on your laptop today; ``ArucoCameraDetector`` is the
   Pi-camera stub you fill in later. **Nothing downstream changes.**

:class:`PerceptionPipeline` ties them to an ``ArmController``::

    pipe = PerceptionPipeline(arm, SimDetector([Detection("cube", [0, 0, 400])]),
                              CameraExtrinsics.from_pose([300, 0, 600], rpy_deg=[0, 90, 0]))
    pipe.pick_nearest_and_move(pitch_deg=-90, hover_mm=40)   # hover 40mm above it

Start with markers (ArUco), not neural nets — the geometry below is identical
whatever the detector is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .kinematics import Unreachable


# ----------------------------------------------------------------------
# 1. What the camera reports
# ----------------------------------------------------------------------

@dataclass
class Detection:
    """One thing the camera found, in the CAMERA's coordinate frame (mm).

    ``confidence`` is a 0..1 score the detector assigns; the pipeline can
    ignore low-confidence hits.
    """
    target_id: str | int
    position_cam_mm: np.ndarray
    confidence: float = 1.0

    def __post_init__(self):
        self.position_cam_mm = np.asarray(self.position_cam_mm, dtype=float)
        if self.position_cam_mm.shape != (3,):
            raise ValueError("position_cam_mm must be an (x, y, z) triple")


# ----------------------------------------------------------------------
# 2. Where the camera is, relative to the arm base
# ----------------------------------------------------------------------

def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from roll/pitch/yaw (radians), applied Z*Y*X."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass
class CameraExtrinsics:
    """Rigid transform mapping camera-frame points into base-frame points.

    ``p_base = R @ p_cam + t``, where ``t`` is the camera origin expressed in
    base coordinates (mm) and ``R`` is its orientation. Get these from a
    hand-eye calibration once the camera is mounted; until then,
    :meth:`from_pose` lets you describe the mount by hand.
    """
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=float).reshape(3)

    @classmethod
    def identity(cls) -> "CameraExtrinsics":
        """Camera frame == base frame (handy for tests / a top-down rig)."""
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_pose(cls, position_mm, rpy_deg=(0.0, 0.0, 0.0)) -> "CameraExtrinsics":
        """Build from the camera's position and roll/pitch/yaw (degrees)."""
        R = _rotation_from_rpy(*np.radians(rpy_deg))
        return cls(R, np.asarray(position_mm, dtype=float))

    def to_base(self, p_cam) -> np.ndarray:
        """Convert a camera-frame point (mm) into base/world coordinates."""
        return self.R @ np.asarray(p_cam, dtype=float) + self.t


# ----------------------------------------------------------------------
# 3. The seam: where detections come from
# ----------------------------------------------------------------------

class DetectionSource(ABC):
    """Anything that can report the targets currently visible."""

    @abstractmethod
    def read(self) -> list[Detection]:
        """Return the detections seen right now (camera frame)."""


class SimDetector(DetectionSource):
    """Scripted detections for the simulator — works today, no camera.

    Hand it a list of :class:`Detection` and it returns them; update the list
    at any time with :meth:`set` to simulate the scene changing.
    """

    def __init__(self, detections: list[Detection] | None = None):
        self._detections = list(detections or [])

    def set(self, detections: list[Detection]) -> None:
        self._detections = list(detections)

    def read(self) -> list[Detection]:
        return list(self._detections)


class ArucoCameraDetector(DetectionSource):
    """Pi-camera ArUco-marker detector — stub until the camera is wired.

    When you build it (on the Pi):
      * grab a frame (picamera2 / OpenCV ``VideoCapture``),
      * ``cv2.aruco.detectMarkers`` + ``estimatePoseSingleMarkers`` with the
        marker size and the camera's calibrated intrinsics,
      * return one :class:`Detection` per marker, position in the camera frame.
    The rest of the pipeline is unchanged.
    """

    def __init__(self, *, marker_length_mm: float, camera_matrix=None,
                 dist_coeffs=None, dictionary: str = "DICT_4X4_50"):
        raise NotImplementedError(
            "ArucoCameraDetector: install opencv-python (+ picamera2) on the Pi "
            "and implement frame-grab + marker pose. Until then use SimDetector."
        )

    def read(self) -> list[Detection]:  # pragma: no cover
        raise NotImplementedError


# ----------------------------------------------------------------------
# Pipeline: detections -> base coordinates -> motion
# ----------------------------------------------------------------------

@dataclass
class BaseTarget:
    """A detection resolved into base/world coordinates (mm)."""
    target_id: str | int
    position_base_mm: np.ndarray
    confidence: float = 1.0
    distance_mm: float = field(default=0.0)


class PerceptionPipeline:
    """Reads detections, converts them to base coordinates, drives the arm."""

    def __init__(self, controller, source: DetectionSource,
                 extrinsics: CameraExtrinsics | None = None,
                 min_confidence: float = 0.0):
        self.arm = controller
        self.source = source
        self.extrinsics = extrinsics or CameraExtrinsics.identity()
        self.min_confidence = min_confidence

    def targets_in_base(self) -> list[BaseTarget]:
        """All current detections, transformed into base coordinates.

        Filters out anything below ``min_confidence`` and tags each with its
        straight-line distance from the base origin (used to pick the nearest).
        """
        out = []
        for d in self.source.read():
            if d.confidence < self.min_confidence:
                continue
            p = self.extrinsics.to_base(d.position_cam_mm)
            out.append(BaseTarget(d.target_id, p, d.confidence,
                                  float(np.linalg.norm(p))))
        return out

    def nearest(self, targets: list[BaseTarget] | None = None) -> BaseTarget | None:
        """The closest visible target to the base, or ``None`` if none seen."""
        targets = self.targets_in_base() if targets is None else targets
        return min(targets, key=lambda t: t.distance_mm) if targets else None

    def move_to(self, target, *, pitch_deg: float = 0.0, hover_mm: float = 0.0,
                elbow_up: bool = True):
        """Move the tool to a target (a BaseTarget, Detection, or xyz point).

        ``hover_mm`` lifts the goal straight up in Z — use it to approach from
        above instead of crashing into the object. Raises
        :class:`~arm_control.kinematics.Unreachable` if it's out of reach.
        """
        x, y, z = self._resolve(target)
        return self.arm.move_to_pose(float(x), float(y), float(z + hover_mm),
                                     pitch_deg=pitch_deg, elbow_up=elbow_up)

    def pick_nearest_and_move(self, *, pitch_deg: float = 0.0, hover_mm: float = 0.0,
                              elbow_up: bool = True):
        """Find the nearest target and move to it.

        Returns ``(BaseTarget, Trajectory)`` on success, or ``None`` if nothing
        is visible. An unreachable target still raises ``Unreachable`` — caller
        decides how to react.
        """
        target = self.nearest()
        if target is None:
            return None
        traj = self.move_to(target, pitch_deg=pitch_deg, hover_mm=hover_mm,
                            elbow_up=elbow_up)
        return target, traj

    # --- helpers -------------------------------------------------------

    def _resolve(self, target) -> np.ndarray:
        """Coerce a BaseTarget / Detection / xyz into a base-frame point."""
        if isinstance(target, BaseTarget):
            return target.position_base_mm
        if isinstance(target, Detection):
            return self.extrinsics.to_base(target.position_cam_mm)
        return np.asarray(target, dtype=float).reshape(3)
