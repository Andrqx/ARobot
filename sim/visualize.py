"""3D visualization of the arm using matplotlib.

Run a quick demo move:

    python -m sim.visualize

This draws the arm and animates a smooth move to a target pose, so you can
*see* the kinematics working long before any parts are printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running as a script (python sim/visualize.py) as well as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import ArmKinematics, load_config  # noqa: E402
from arm_control.trajectory import interpolate  # noqa: E402
from sim.stl_loader import load_stl  # noqa: E402

JOINT_ORDER = ["base", "shoulder", "elbow", "wrist", "tip"]
LINK_ORDER = ["column", "upper", "forearm", "tool"]
_ROOT = Path(__file__).resolve().parent.parent


def load_link_meshes(geometry_path=None) -> dict:
    """Load the STL meshes named in ``config/geometry.yaml``, by link name.

    Returns ``{link_name: (n, 3, 3) array}``. Links whose ``mesh:`` is null (or
    whose file is missing) are skipped — so an empty dict simply means "no CAD
    exported yet" and the caller falls back to the stick-figure skeleton.
    """
    import yaml

    gp = Path(geometry_path) if geometry_path else _ROOT / "config" / "geometry.yaml"
    if not gp.exists():
        return {}
    data = yaml.safe_load(gp.read_text()) or {}
    meshes_dir = data.get("meshes_dir", "meshes")
    out = {}
    for name, g in (data.get("links") or {}).items():
        mesh = (g or {}).get("mesh")
        if not mesh:
            continue
        p = Path(mesh)
        if not p.is_absolute():
            p = _ROOT / p
            if not p.exists():                      # allow a bare filename too
                p = _ROOT / meshes_dir / Path(mesh).name
        if p.exists():
            out[name] = load_stl(p)
    return out


def _world_triangles(verts, R, p):
    """Transform local mesh triangles (n,3,3) into world coordinates."""
    world = verts.reshape(-1, 3) @ R.T + p
    return world.reshape(-1, 3, 3)


def add_meshes(ax, kin: ArmKinematics, q_deg, link_meshes,
               color="#9bbcdf", alpha=0.92):
    """Draw each link's CAD mesh at the pose ``q_deg``; return the collections."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    frames = kin.link_frames_deg(q_deg)
    collections = {}
    for name, verts in link_meshes.items():
        R, pos = frames[name]
        tris = _world_triangles(verts, R, pos)
        coll = Poly3DCollection(list(tris), facecolor=color,
                                edgecolor="#5a7ca8", linewidths=0.2, alpha=alpha)
        ax.add_collection3d(coll)
        collections[name] = coll
    return collections


def _segments(kin: ArmKinematics, q_deg):
    points, _ = kin.forward_deg(q_deg)
    xs = [points[name][0] for name in JOINT_ORDER]
    ys = [points[name][1] for name in JOINT_ORDER]
    zs = [points[name][2] for name in JOINT_ORDER]
    return xs, ys, zs


def _setup_axes(kin: ArmKinematics):
    import matplotlib.pyplot as plt  # imported lazily so tests don't need a display

    reach = kin.L1 + kin.L2 + kin.L3 + kin.L4
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_zlim(0, reach)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("ARobot — 4-DOF arm")
    return fig, ax


def plot_pose(kin: ArmKinematics, q_deg, ax=None, link_meshes=None):
    """Draw a single static pose.

    If ``link_meshes`` is given (or found in geometry.yaml), the CAD shells are
    rendered with a faint skeleton overlay; otherwise just the skeleton.
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = _setup_axes(kin)
    if link_meshes is None:
        link_meshes = load_link_meshes()

    skeleton_alpha = 0.35 if link_meshes else 1.0
    xs, ys, zs = _segments(kin, q_deg)
    ax.plot(xs, ys, zs, "-o", lw=3, ms=6, color="#2a7de1", alpha=skeleton_alpha)
    ax.plot([xs[-1]], [ys[-1]], [zs[-1]], "o", ms=9, color="#e23a3a")  # tool tip
    if link_meshes:
        add_meshes(ax, kin, q_deg, link_meshes)
    if created:
        plt.show()
    return ax


def animate_move(kin: ArmKinematics, q_start_deg, q_goal_deg, steps=60, save=None,
                 link_meshes=None):
    """Animate a smooth move between two joint configurations.

    Renders CAD meshes if available (from ``link_meshes`` or geometry.yaml),
    otherwise the stick-figure skeleton.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if link_meshes is None:
        link_meshes = load_link_meshes()

    path = interpolate(q_start_deg, q_goal_deg, steps=steps, ease=True)
    fig, ax = _setup_axes(kin)
    skeleton_alpha = 0.35 if link_meshes else 1.0
    (line,) = ax.plot([], [], [], "-o", lw=3, ms=6, color="#2a7de1",
                      alpha=skeleton_alpha)
    (tip,) = ax.plot([], [], [], "o", ms=9, color="#e23a3a")
    mesh_colls = add_meshes(ax, kin, path[0], link_meshes) if link_meshes else {}

    def update(frame):
        qd = path[frame]
        xs, ys, zs = _segments(kin, qd)
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        tip.set_data([xs[-1]], [ys[-1]])
        tip.set_3d_properties([zs[-1]])
        artists = [line, tip]
        if mesh_colls:
            frames = kin.link_frames_deg(qd)
            for name, coll in mesh_colls.items():
                R, pos = frames[name]
                coll.set_verts(list(_world_triangles(link_meshes[name], R, pos)))
                artists.append(coll)
        return artists

    anim = FuncAnimation(fig, update, frames=len(path), interval=30, blit=False)
    if save:
        anim.save(save, writer="pillow", fps=30)
        print(f"saved animation -> {save}")
    else:
        plt.show()
    return anim


def plot_profile(traj, joint_names=None, save=None):
    """Plot position & velocity vs time for a planned Trajectory.

    Lets you *see* the trapezoidal velocity profile and confirm every joint
    starts and finishes together.
    """
    import matplotlib.pyplot as plt

    J = traj.pos.shape[1]
    names = joint_names or [f"joint {i}" for i in range(J)]

    fig, (ax_p, ax_v) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for j in range(J):
        ax_p.plot(traj.t, traj.pos[:, j], label=names[j])
        ax_v.plot(traj.t, traj.vel[:, j], label=names[j])

    ax_p.set_ylabel("position (deg)")
    ax_p.set_title(f"Trapezoidal move — duration {traj.duration:.2f} s")
    ax_p.legend(loc="best", fontsize=8)
    ax_p.grid(alpha=0.3)

    ax_v.set_ylabel("velocity (deg/s)")
    ax_v.set_xlabel("time (s)")
    ax_v.legend(loc="best", fontsize=8)
    ax_v.grid(alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110)
        print(f"saved profile plot -> {save}")
    else:
        plt.show()
    return fig


def _demo():
    cfg = load_config()
    kin = ArmKinematics.from_config(cfg)

    q_home = np.array(cfg.home_angles_deg())

    # Pick a reachable target out in front of the arm.
    target = dict(x=250.0, y=80.0, z=300.0, pitch_deg=0.0)
    q_goal = kin.inverse_deg(**target)
    print("home angles (deg):", np.round(q_home, 1))
    print("goal angles (deg):", np.round(q_goal, 1))
    x, y, z, pitch = kin.tip_pose_deg(q_goal)
    print(f"FK check of goal -> x={x:.1f} y={y:.1f} z={z:.1f} pitch={pitch:.1f}")

    meshes = load_link_meshes()
    if meshes:
        print(f"CAD meshes loaded for: {', '.join(meshes)}")
    else:
        print("No CAD meshes yet (geometry.yaml mesh fields are null) — "
              "drawing the skeleton. Export STLs to meshes/ to see real parts.")

    animate_move(kin, q_home, q_goal, link_meshes=meshes)


if __name__ == "__main__":
    _demo()
