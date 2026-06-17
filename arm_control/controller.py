"""High-level arm controller: ties kinematics + drivers + trajectory together.

This is the object your scripts (and later, the vision system) talk to:

    arm = ArmController.simulated(load_config())
    arm.move_to_pose(x=250, y=0, z=300, pitch_deg=0)

Today it runs entirely in simulation. To go live, build it with real
``StepperEncoderDriver`` instances instead of ``SimDriver`` — nothing else
in your code changes.
"""

from __future__ import annotations

import numpy as np

from .config import ArmConfig
from .drivers import JointDriver, SimDriver
from .kinematics import ArmKinematics, Unreachable
from .trajectory import Trajectory, interpolate, plan_trapezoidal


class LimitViolation(Exception):
    """A planned joint angle exceeds that joint's mechanical limits."""


class ArmController:
    def __init__(self, cfg: ArmConfig, kin: ArmKinematics, drivers: list[JointDriver]):
        if len(drivers) != len(cfg.joints):
            raise ValueError("need exactly one driver per joint")
        self.cfg = cfg
        self.kin = kin
        self.drivers = drivers

    @classmethod
    def simulated(cls, cfg: ArmConfig, backlash_deg: float = 0.0) -> "ArmController":
        """Build a fully simulated arm — no hardware required."""
        kin = ArmKinematics.from_config(cfg)
        drivers = [SimDriver(home_deg=j.home_deg, backlash_deg=backlash_deg)
                   for j in cfg.joints]
        return cls(cfg, kin, drivers)

    # --- State --------------------------------------------------------

    def current_angles_deg(self) -> np.ndarray:
        return np.array([d.read_deg() for d in self.drivers])

    def _check_limits(self, q_deg) -> None:
        bad = []
        for joint, angle in zip(self.cfg.joints, q_deg):
            if not joint.in_range(angle):
                bad.append(f"{joint.name}={angle:.1f}deg "
                           f"(limit {joint.min_deg:.0f}..{joint.max_deg:.0f})")
        if bad:
            raise LimitViolation("; ".join(bad))

    # --- Motion -------------------------------------------------------

    def command_angles_deg(self, q_deg) -> None:
        for driver, angle in zip(self.drivers, q_deg):
            driver.move_to_deg(float(angle))

    def plan_to_pose(self, x, y, z, pitch_deg=0.0, *, elbow_up=True,
                     dt=0.01) -> Trajectory:
        """Plan a synchronized, velocity/accel-limited move to a tool pose.

        Returns a Trajectory (time-stamped positions + velocities) without
        executing it — useful for previewing/plotting before committing.
        """
        try:
            q_goal = self.kin.inverse_deg(x, y, z, pitch_deg, elbow_up=elbow_up)
        except Unreachable as e:
            raise Unreachable(str(e)) from None

        self._check_limits(q_goal)
        q_start = self.current_angles_deg()
        return plan_trapezoidal(
            q_start, q_goal,
            v_max=self.cfg.max_vel_deg_s(),
            a_max=self.cfg.max_accel_deg_s2(),
            dt=dt,
        )

    def execute(self, traj: Trajectory) -> Trajectory:
        """Stream a planned trajectory to the joint drivers."""
        for q in traj.pos:
            self.command_angles_deg(q)
        return traj

    def move_to_pose(self, x, y, z, pitch_deg=0.0, *, elbow_up=True, dt=0.01):
        """Plan (trapezoidal) and execute a move to a Cartesian tool pose.

        Returns the executed Trajectory.
        """
        traj = self.plan_to_pose(x, y, z, pitch_deg, elbow_up=elbow_up, dt=dt)
        return self.execute(traj)

    def go_home(self, dt=0.01):
        """Return every joint to its configured home angle (profiled)."""
        q_start = self.current_angles_deg()
        traj = plan_trapezoidal(
            q_start, self.cfg.home_angles_deg(),
            v_max=self.cfg.max_vel_deg_s(),
            a_max=self.cfg.max_accel_deg_s2(),
            dt=dt,
        )
        return self.execute(traj)
