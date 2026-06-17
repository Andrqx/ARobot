"""Trajectory tests — prove the trapezoidal planner respects the motor limits.

The guarantees we check:
  * endpoints are exact (start -> goal),
  * no joint ever exceeds its velocity or acceleration limit,
  * all joints start and finish together (synchronized),
  * a zero-distance move is handled cleanly.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import load_config, plan_trapezoidal  # noqa: E402

V_MAX = np.array([60.0, 45.0, 90.0, 120.0])
A_MAX = np.array([120.0, 90.0, 180.0, 360.0])


def test_endpoints_exact():
    start = np.array([0.0, 90.0, 0.0, 0.0])
    goal = np.array([45.0, 60.0, -90.0, 30.0])
    traj = plan_trapezoidal(start, goal, V_MAX, A_MAX, dt=0.005)
    assert np.allclose(traj.pos[0], start)
    assert np.allclose(traj.pos[-1], goal)
    assert np.allclose(traj.vel[0], 0.0)
    assert np.allclose(traj.vel[-1], 0.0)


def test_respects_velocity_and_accel_limits():
    start = np.zeros(4)
    goal = np.array([160.0, 140.0, -200.0, 100.0])  # large move -> hits cruise
    traj = plan_trapezoidal(start, goal, V_MAX, A_MAX, dt=0.002)

    peak_v = np.max(np.abs(traj.vel), axis=0)
    assert np.all(peak_v <= V_MAX + 1e-6), f"velocity exceeded: {peak_v} vs {V_MAX}"

    # Acceleration via numerical derivative of velocity.
    dt = traj.t[1] - traj.t[0]
    accel = np.diff(traj.vel, axis=0) / dt
    peak_a = np.max(np.abs(accel), axis=0)
    assert np.all(peak_a <= A_MAX + 1.0), f"accel exceeded: {peak_a} vs {A_MAX}"


def test_joints_synchronized():
    # Every joint must reach its goal exactly at the final timestamp, i.e.
    # they all finish together rather than one finishing early.
    start = np.zeros(4)
    goal = np.array([90.0, 10.0, -120.0, 5.0])  # very different distances
    traj = plan_trapezoidal(start, goal, V_MAX, A_MAX, dt=0.005)

    # Halfway in time, no joint should already be sitting at its goal
    # (which would mean it finished early — not synchronized).
    mid = len(traj) // 2
    moving = np.abs(goal - start) > 1e-6
    still_moving = np.abs(traj.pos[mid] - goal) > 1e-3
    assert np.all(still_moving[moving]), "a joint finished early (not synchronized)"


def test_zero_move():
    q = np.array([10.0, 20.0, 30.0, 40.0])
    traj = plan_trapezoidal(q, q, V_MAX, A_MAX)
    assert traj.duration == 0.0
    assert np.allclose(traj.pos[0], q)


def test_short_move_is_triangular():
    # A tiny move should never reach cruise velocity (triangle profile):
    # peak velocity stays well under the limit.
    start = np.zeros(4)
    goal = np.array([2.0, 0.0, 0.0, 0.0])
    traj = plan_trapezoidal(start, goal, V_MAX, A_MAX, dt=0.001)
    assert np.max(np.abs(traj.vel[:, 0])) < V_MAX[0]


def test_planner_uses_config_limits():
    cfg = load_config()
    assert cfg.max_vel_deg_s() == [60.0, 45.0, 90.0, 120.0]
    assert cfg.max_accel_deg_s2() == [120.0, 90.0, 180.0, 360.0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll trajectory tests passed.")
