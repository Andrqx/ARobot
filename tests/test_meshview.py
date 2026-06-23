"""Mesh-visualizer tests — link frames, the STL loader, and a headless render.

The render test forces matplotlib's non-interactive 'Agg' backend so it runs
without a display (and in CI).
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import ArmKinematics, load_config  # noqa: E402
from sim.stl_loader import load_stl  # noqa: E402


def make_kin():
    return ArmKinematics.from_config(load_config())


# --- link_frames consistency with forward kinematics ----------------------

def test_link_frame_origins_match_joint_points():
    kin = make_kin()
    q = [20.0, 70.0, -40.0, 15.0]
    points, _ = kin.forward_deg(q)
    frames = kin.link_frames_deg(q)
    assert np.allclose(frames["column"][1], points["base"])
    assert np.allclose(frames["upper"][1], points["shoulder"])
    assert np.allclose(frames["forearm"][1], points["elbow"])
    assert np.allclose(frames["tool"][1], points["wrist"])


def test_tool_frame_reconstructs_tip():
    # Walking L4 along the tool frame's local +X must land on the FK tip.
    kin = make_kin()
    q = [-30.0, 80.0, -50.0, 10.0]
    points, _ = kin.forward_deg(q)
    R, p = kin.link_frames_deg(q)["tool"]
    tip = p + R @ np.array([kin.L4, 0.0, 0.0])
    assert np.allclose(tip, points["tip"], atol=1e-6)


def test_upper_frame_reconstructs_elbow():
    kin = make_kin()
    q = [10.0, 100.0, -70.0, 0.0]
    points, _ = kin.forward_deg(q)
    R, p = kin.link_frames_deg(q)["upper"]
    elbow = p + R @ np.array([kin.L2, 0.0, 0.0])
    assert np.allclose(elbow, points["elbow"], atol=1e-6)


def test_frames_are_orthonormal_right_handed():
    kin = make_kin()
    for R, _ in kin.link_frames_deg([15, 60, -30, 20]).values():
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


# --- STL loader: ASCII + binary round-trips --------------------------------

# One unit triangle.
_TRI = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])


def _write_ascii_stl(path, tris):
    lines = ["solid test"]
    for t in tris:
        lines.append("facet normal 0 0 0")
        lines.append("  outer loop")
        for v in t:
            lines.append(f"    vertex {v[0]} {v[1]} {v[2]}")
        lines.append("  endloop")
        lines.append("endfacet")
    lines.append("endsolid test")
    Path(path).write_text("\n".join(lines))


def _write_binary_stl(path, tris):
    buf = bytearray(b"\0" * 80)              # header
    buf += struct.pack("<I", len(tris))
    for t in tris:
        buf += struct.pack("<3f", 0, 0, 0)   # normal
        for v in t:
            buf += struct.pack("<3f", *v)
        buf += struct.pack("<H", 0)          # attribute
    Path(path).write_bytes(bytes(buf))


def test_load_ascii_stl(tmp_path):
    p = tmp_path / "t.stl"
    _write_ascii_stl(p, _TRI)
    tris = load_stl(p)
    assert tris.shape == (1, 3, 3)
    assert np.allclose(tris, _TRI)


def test_load_binary_stl(tmp_path):
    p = tmp_path / "t.stl"
    _write_binary_stl(p, _TRI)
    tris = load_stl(p)
    assert tris.shape == (1, 3, 3)
    assert np.allclose(tris, _TRI)


def test_binary_detection_not_fooled_by_solid_header(tmp_path):
    # Binary header literally starting with 'solid' must still parse as binary.
    p = tmp_path / "t.stl"
    buf = bytearray(b"solid" + b"\0" * 75)
    buf += struct.pack("<I", 1)
    buf += struct.pack("<3f", 0, 0, 0)
    for v in _TRI[0]:
        buf += struct.pack("<3f", *v)
    buf += struct.pack("<H", 0)
    p.write_bytes(bytes(buf))
    assert np.allclose(load_stl(p), _TRI)


def test_malformed_ascii_raises(tmp_path):
    p = tmp_path / "bad.stl"
    p.write_text("solid x\nvertex 0 0 0\nvertex 1 0 0\nendsolid x")  # 2 verts
    with pytest.raises(ValueError):
        load_stl(p)


# --- headless render smoke tests ------------------------------------------

def test_plot_pose_with_meshes_no_display(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sim.visualize import plot_pose

    kin = make_kin()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Hand it a synthetic mesh per link so the mesh path is exercised.
    link_meshes = {name: _TRI * 50 for name in ["column", "upper", "forearm", "tool"]}
    plot_pose(kin, [0, 90, 0, 0], ax=ax, link_meshes=link_meshes)
    assert len(ax.collections) == 4   # one Poly3DCollection per link
    plt.close(fig)


def test_load_link_meshes_empty_when_all_null():
    # The shipped geometry.yaml has every mesh: null -> nothing to load.
    from sim.visualize import load_link_meshes
    assert load_link_meshes() == {}


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
    print("\nmeshview tests done.")
