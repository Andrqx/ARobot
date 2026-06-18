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
from .teach import Program, Waypoint
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
        self.homed = False

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

    def home(self, dt=0.01):
        """Drive every joint to its configured home angle and mark as homed.

        In sim this just moves to the home pose. On hardware this is where a
        real homing routine goes (seek limit switches / encoder index), after
        which joint zero is trusted and taught programs are repeatable.
        """
        traj = plan_trapezoidal(
            self.current_angles_deg(), self.cfg.home_angles_deg(),
            v_max=self.cfg.max_vel_deg_s(),
            a_max=self.cfg.max_accel_deg_s2(),
            dt=dt,
        )
        self.execute(traj)
        self.homed = True
        return traj

    go_home = home  # backward-compatible alias

    # --- Teach and repeat ---------------------------------------------

    def record_waypoint(self, name: str, pause_s: float = 0.0) -> Waypoint:
        """Snapshot the current joint angles as a named waypoint."""
        return Waypoint(
            name=name,
            joints_deg=[round(float(a), 3) for a in self.current_angles_deg()],
            pause_s=pause_s,
        )

    def run_program(self, program: Program, dt=0.01, on_waypoint=None):
        """Replay a taught program, moving through each waypoint in order.

        Returns the list of executed Trajectories. ``on_waypoint(wp, traj)`` is
        called after each waypoint is reached (e.g. to log or actuate a gripper).
        """
        trajs = []
        for wp in program.waypoints:
            q_goal = np.asarray(wp.joints_deg, dtype=float)
            self._check_limits(q_goal)
            traj = plan_trapezoidal(
                self.current_angles_deg(), q_goal,
                v_max=self.cfg.max_vel_deg_s(),
                a_max=self.cfg.max_accel_deg_s2(),
                dt=dt,
            )
            self.execute(traj)
            trajs.append(traj)
            if on_waypoint is not None:
                on_waypoint(wp, traj)
        return trajs
