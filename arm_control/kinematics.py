"""Forward and inverse kinematics for the 4-DOF arm.

Joint model (base -> tip):
    q[0] = base   : yaw, rotates the whole arm about vertical Z
    q[1] = shoulder: pitch
    q[2] = elbow   : pitch
    q[3] = wrist   : pitch

Because joints 2-4 all pitch in the same vertical plane, the arm reduces to:
    * a base yaw that picks the plane, plus
    * a planar 3-link chain inside that plane.

That structure gives a clean *closed-form* inverse kinematics solution
(no iterative solver), so it runs cheaply on the Pi.

All angles here are in RADIANS. Use the ``*_deg`` wrappers at the boundary
if you prefer degrees (the rest of the stack works in degrees).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Unreachable(Exception):
    """Raised when a requested pose lies outside the arm's reach."""


@dataclass
class ArmKinematics:
    L1: float  # base height: base plane -> shoulder pivot
    L2: float  # upper arm:   shoulder   -> elbow
    L3: float  # forearm:     elbow      -> wrist
    L4: float  # tool:        wrist      -> tip

    @classmethod
    def from_config(cls, cfg) -> "ArmKinematics":
        return cls(*cfg.link_lengths)

    # --- Forward kinematics -------------------------------------------

    def forward(self, q):
        """Joint angles (rad) -> world XYZ of every joint + end pitch.

        Returns (points, pitch) where ``points`` is a dict of np.array([x,y,z])
        for base/shoulder/elbow/wrist/tip, and ``pitch`` is the tool angle
        (rad) measured from horizontal in the arm plane.
        """
        t1, t2, t3, t4 = q

        # Absolute link angles within the vertical arm plane.
        a2 = t2
        a3 = t2 + t3
        a4 = t2 + t3 + t4  # == end-effector pitch

        # Build the chain in plane coordinates (r = radial out, z = up).
        er = self.L2 * np.cos(a2)
        ez = self.L1 + self.L2 * np.sin(a2)
        wr = er + self.L3 * np.cos(a3)
        wz = ez + self.L3 * np.sin(a3)
        tr = wr + self.L4 * np.cos(a4)
        tz = wz + self.L4 * np.sin(a4)

        c1, s1 = np.cos(t1), np.sin(t1)

        def world(r, z):
            return np.array([r * c1, r * s1, z])

        points = {
            "base": np.array([0.0, 0.0, 0.0]),
            "shoulder": np.array([0.0, 0.0, self.L1]),
            "elbow": world(er, ez),
            "wrist": world(wr, wz),
            "tip": world(tr, tz),
        }
        return points, a4

    def tip_pose(self, q):
        """Joint angles (rad) -> (x, y, z, pitch) of the tool tip."""
        points, pitch = self.forward(q)
        x, y, z = points["tip"]
        return float(x), float(y), float(z), float(pitch)

    # --- Inverse kinematics -------------------------------------------

    def inverse(self, x, y, z, pitch, elbow_up=True):
        """Target (x, y, z, pitch in rad) -> joint angles (rad).

        ``pitch`` is the desired tool approach angle in the vertical plane
        (0 = horizontal, +pi/2 = pointing straight up).

        Raises ``Unreachable`` if the target is out of range.
        """
        # 1) Base yaw selects the working plane.
        t1 = np.arctan2(y, x)
        r = np.hypot(x, y)

        # 2) Step back along the tool to find the wrist pivot in-plane.
        wr = r - self.L4 * np.cos(pitch)
        wz = z - self.L4 * np.sin(pitch)

        # 3) Solve the 2-link (upper arm + forearm) sub-problem from the
        #    shoulder pivot at (0, L1) to the wrist (wr, wz).
        pr = wr
        pz = wz - self.L1
        dist2 = pr * pr + pz * pz

        cos_elbow = (dist2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        if not -1.0 <= cos_elbow <= 1.0:
            raise Unreachable(
                f"target ({x:.1f},{y:.1f},{z:.1f}) pitch={np.degrees(pitch):.0f}deg "
                f"is outside reach [{abs(self.L2 - self.L3):.0f}..{self.L2 + self.L3:.0f} mm]"
            )

        t3 = np.arccos(cos_elbow)
        if elbow_up:
            t3 = -t3

        t2 = np.arctan2(pz, pr) - np.arctan2(
            self.L3 * np.sin(t3), self.L2 + self.L3 * np.cos(t3)
        )

        # 4) Wrist makes up whatever pitch is left over.
        t4 = pitch - (t2 + t3)

        return np.array([t1, t2, t3, t4])

    # --- Degree-friendly convenience wrappers -------------------------

    def forward_deg(self, q_deg):
        points, pitch = self.forward(np.radians(q_deg))
        return points, float(np.degrees(pitch))

    def tip_pose_deg(self, q_deg):
        x, y, z, pitch = self.tip_pose(np.radians(q_deg))
        return x, y, z, np.degrees(pitch)

    def inverse_deg(self, x, y, z, pitch_deg, elbow_up=True):
        q = self.inverse(x, y, z, np.radians(pitch_deg), elbow_up)
        return np.degrees(q)

    def reach(self) -> tuple[float, float]:
        """(min, max) planar distance the wrist can reach from the shoulder."""
        return abs(self.L2 - self.L3), self.L2 + self.L3
