"""Joint-space trajectory generation.

Two tools:

1. ``interpolate`` — quick eased point-to-point sampling (geometry only, no
   timing). Handy for static visualization.

2. ``plan_trapezoidal`` — a proper **time-parameterized, velocity/acceleration
   limited** trajectory with a trapezoidal speed profile, *synchronized* across
   all joints so they start and finish together. This is what the arm should
   actually execute: it never commands more speed or acceleration than the
   motors can deliver, which keeps the closed-loop steppers from losing steps
   and keeps inertial torque spikes out of the PETG joints and belts.

Trapezoidal profile (per joint):

    velocity
      ^      _________________        <- cruise at v
      |     /                 \\
      |    /                   \\      area under curve = distance
      |___/                     \\___ -> time
          accel    cruise    decel

For a short move the cruise phase vanishes and the profile becomes a triangle.

Synchronization: each joint has its own (v_max, a_max). We find every joint's
minimum move time, take the slowest as the move duration T, then *slow the
faster joints down* (keeping their acceleration limit, lowering their cruise
velocity) so all of them take exactly T. Result: one clean, coordinated motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- Simple eased interpolation (no timing) ---------------------------

def interpolate(q_start, q_goal, steps: int = 50, ease: bool = True):
    """Return a list of intermediate joint vectors from start to goal.

    q_start / q_goal: array-like joint angles (any unit).
    steps: number of samples including both endpoints.
    ease:  cosine smoothing (zero velocity at the ends) if True.
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    ts = np.linspace(0.0, 1.0, max(2, steps))
    if ease:
        ts = 0.5 * (1.0 - np.cos(np.pi * ts))
    return [q_start + (q_goal - q_start) * t for t in ts]


# --- Time-parameterized trapezoidal trajectory ------------------------

@dataclass
class Trajectory:
    """A sampled, time-stamped motion.

    t:   (N,)   time stamps in seconds
    pos: (N, J) joint positions (deg)
    vel: (N, J) joint velocities (deg/s)
    duration: total move time (s)
    """
    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    duration: float

    def __len__(self) -> int:
        return len(self.t)

    @property
    def waypoints(self):
        """Position rows as a list of joint vectors (for command streaming)."""
        return [self.pos[i] for i in range(len(self.t))]

    @property
    def peak_vel(self):
        return np.max(np.abs(self.vel), axis=0)


def _min_time(dist: float, v_max: float, a_max: float) -> float:
    """Shortest time to move ``dist`` honoring v_max and a_max."""
    d = abs(dist)
    if d == 0.0:
        return 0.0
    d_accel = v_max * v_max / (2.0 * a_max)  # distance to reach v_max
    if 2.0 * d_accel <= d:                    # trapezoid: reaches cruise
        t_accel = v_max / a_max
        t_cruise = (d - 2.0 * d_accel) / v_max
        return 2.0 * t_accel + t_cruise
    return 2.0 * np.sqrt(d / a_max)           # triangle: never reaches v_max


def _cruise_vel_for_time(dist: float, a_max: float, T: float) -> float:
    """Cruise velocity that makes a fixed-accel trapezoid last exactly T.

    Solves  d = v*T - v^2/a  ->  v^2 - a*T*v + a*d = 0, smaller root.
    """
    d = abs(dist)
    if d == 0.0 or T == 0.0:
        return 0.0
    disc = max(a_max * a_max * T * T - 4.0 * a_max * d, 0.0)
    return (a_max * T - np.sqrt(disc)) / 2.0


def _sample_axis(dist: float, a_max: float, v: float, T: float, t: float):
    """Position & velocity along one axis at time t (monotonic 0 -> |dist|)."""
    d = abs(dist)
    if d == 0.0 or T == 0.0:
        return 0.0, 0.0
    t_accel = v / a_max if a_max > 0 else 0.0
    d_accel = 0.5 * a_max * t_accel * t_accel
    t_cruise = max(T - 2.0 * t_accel, 0.0)

    if t < t_accel:
        return 0.5 * a_max * t * t, a_max * t
    if t < t_accel + t_cruise:
        return d_accel + v * (t - t_accel), v
    if t <= T:
        td = T - t
        return d - 0.5 * a_max * td * td, a_max * td
    return d, 0.0


def plan_trapezoidal(q_start, q_goal, v_max, a_max, dt: float = 0.01) -> Trajectory:
    """Synchronized trapezoidal trajectory between two joint configurations.

    q_start, q_goal: joint angle vectors (deg)
    v_max, a_max:    per-joint limits (deg/s, deg/s^2)
    dt:              sample period (s)
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    v_max = np.asarray(v_max, dtype=float)
    a_max = np.asarray(a_max, dtype=float)

    delta = q_goal - q_start
    dist = np.abs(delta)
    sign = np.sign(delta)
    J = len(delta)

    # Slowest joint sets the move duration; everyone synchronizes to it.
    t_min = np.array([_min_time(dist[j], v_max[j], a_max[j]) for j in range(J)])
    T = float(t_min.max()) if J else 0.0

    if T == 0.0:  # nothing moves
        return Trajectory(
            t=np.array([0.0]),
            pos=q_start[None, :].copy(),
            vel=np.zeros((1, J)),
            duration=0.0,
        )

    # Re-time each joint to last exactly T (keep accel, lower cruise velocity).
    v_sync = np.array([_cruise_vel_for_time(dist[j], a_max[j], T) for j in range(J)])

    n = int(np.ceil(T / dt)) + 1
    t = np.linspace(0.0, T, n)
    pos = np.zeros((n, J))
    vel = np.zeros((n, J))
    for j in range(J):
        for i, ti in enumerate(t):
            x, xd = _sample_axis(dist[j], a_max[j], v_sync[j], T, ti)
            pos[i, j] = q_start[j] + sign[j] * x
            vel[i, j] = sign[j] * xd

    # Guarantee the final sample lands exactly on the goal (no fp drift).
    pos[-1] = q_goal
    vel[-1] = 0.0
    return Trajectory(t=t, pos=pos, vel=vel, duration=T)
