"""Robot description + URDF export — the CAD-integration seam.

Formalizes the arm as a kinematic chain of links, each with a hook for its
CAD mesh. Two payoffs:

  * **CAD drop-in:** when you export STLs from SolidWorks, list them in
    ``config/geometry.yaml`` and they replace the placeholder shapes — no
    code changes.
  * **Training:** ``to_urdf()`` emits a standard URDF you can load straight
    into a physics simulator (PyBullet / MuJoCo / Gazebo) to train the arm.
    Build it now with primitive geometry; swap in CAD meshes later.

The kinematic chain mirrors ``kinematics.py``:
    base_link --(base, yaw/Z)--> column(L1)
              --(shoulder, pitch/Y)--> upper(L2)
              --(elbow, pitch/Y)--> forearm(L3)
              --(wrist, pitch/Y)--> tool(L4)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import ArmConfig, load_config

DEFAULT_GEOMETRY = Path(__file__).resolve().parent.parent / "config" / "geometry.yaml"

# Maps each chain segment to the joint that drives it and its length source.
_SEGMENTS = [
    ("column", "base", "base_height_mm", "z"),    # rotates about vertical
    ("upper", "shoulder", "upper_arm_mm", "y"),   # pitch
    ("forearm", "elbow", "forearm_mm", "y"),      # pitch
    ("tool", "wrist", "tool_mm", "y"),            # pitch
]


@dataclass
class Link:
    name: str
    length_m: float
    mass_kg: float
    axis: str             # "z" or "y" — the driving joint's rotation axis
    joint_name: str
    mesh: str | None = None


@dataclass
class RobotModel:
    links: list[Link]
    cfg: ArmConfig

    @classmethod
    def from_config(cls, cfg: ArmConfig | None = None, geometry=None) -> "RobotModel":
        cfg = cfg or load_config()
        geom = _load_geometry(geometry)
        link_lengths = dict(zip(
            ("base_height_mm", "upper_arm_mm", "forearm_mm", "tool_mm"),
            cfg.link_lengths,
        ))
        links = []
        for seg_name, joint_name, length_key, axis in _SEGMENTS:
            g = geom.get(seg_name, {})
            links.append(Link(
                name=seg_name,
                length_m=link_lengths[length_key] / 1000.0,
                mass_kg=float(g.get("mass_kg", 0.3)),
                axis=axis,
                joint_name=joint_name,
                mesh=g.get("mesh"),
            ))
        return cls(links=links, cfg=cfg)

    def joint(self, name):
        return next(j for j in self.cfg.joints if j.name == name)

    # --- URDF export ---------------------------------------------------

    def to_urdf(self, robot_name: str = "arobot") -> str:
        out = [f'<?xml version="1.0"?>', f'<robot name="{robot_name}">']
        out.append(_base_link())

        parent = "base_link"
        # joint origin offset within the parent link (start of next segment)
        origin = "0 0 0"
        for link in self.links:
            out.append(_link_xml(link))
            out.append(_joint_xml(link, parent, origin, self.joint(link.joint_name)))
            parent = f"{link.name}_link"
            # next joint sits at the far end of this link, along its axis
            if link.axis == "z":
                origin = f"0 0 {link.length_m:.4f}"
            else:  # link extends along +x after a pitch joint
                origin = f"{link.length_m:.4f} 0 0"
        out.append("</robot>")
        return "\n".join(out)

    def save_urdf(self, path: str | Path, robot_name: str = "arobot") -> Path:
        path = Path(path)
        path.write_text(self.to_urdf(robot_name))
        return path


# --- helpers ----------------------------------------------------------

def _load_geometry(geometry) -> dict:
    if isinstance(geometry, dict):
        return geometry.get("links", geometry)
    path = Path(geometry) if geometry else DEFAULT_GEOMETRY
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("links", {})


def _rod_inertia(mass: float, length: float) -> tuple[float, float, float]:
    """Diagonal inertia of a uniform rod (rough; refine from CAD)."""
    i_trans = mass * length * length / 12.0 if length > 0 else 1e-4
    i_axial = max(mass * 1e-4, 1e-5)
    return i_axial, i_trans, i_trans


def _base_link() -> str:
    return (
        '  <link name="base_link">\n'
        '    <inertial><mass value="2.0"/>\n'
        '      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>\n'
        '    </inertial>\n'
        '    <visual><geometry><cylinder radius="0.06" length="0.04"/></geometry></visual>\n'
        '  </link>'
    )


def _geometry_tag(link: Link) -> str:
    if link.mesh:
        return f'<geometry><mesh filename="{link.mesh}"/></geometry>'
    # placeholder cylinder spanning the link length
    return f'<geometry><cylinder radius="0.025" length="{link.length_m:.4f}"/></geometry>'


def _link_xml(link: Link) -> str:
    ixx, iyy, izz = _rod_inertia(link.mass_kg, link.length_m)
    half = link.length_m / 2.0
    # place the visual so the cylinder spans from the joint outward
    if link.axis == "z":
        origin = f'0 0 {half:.4f}'
        rpy = "0 0 0"
    else:
        origin = f'{half:.4f} 0 0'
        rpy = "0 1.5708 0"  # rotate cylinder (default +z) to lie along +x
    return (
        f'  <link name="{link.name}_link">\n'
        f'    <inertial><origin xyz="{origin}" rpy="{rpy}"/>\n'
        f'      <mass value="{link.mass_kg:.3f}"/>\n'
        f'      <inertia ixx="{ixx:.5f}" ixy="0" ixz="0" iyy="{iyy:.5f}" iyz="0" izz="{izz:.5f}"/>\n'
        f'    </inertial>\n'
        f'    <visual><origin xyz="{origin}" rpy="{rpy}"/>{_geometry_tag(link)}</visual>\n'
        f'    <collision><origin xyz="{origin}" rpy="{rpy}"/>{_geometry_tag(link)}</collision>\n'
        f'  </link>'
    )


def _joint_xml(link: Link, parent: str, origin: str, joint_cfg) -> str:
    axis_vec = "0 0 1" if link.axis == "z" else "0 1 0"
    lower = math.radians(joint_cfg.min_deg)
    upper = math.radians(joint_cfg.max_deg)
    effort = joint_cfg.output_torque_nm()
    velocity = math.radians(joint_cfg.max_vel_deg_s)
    return (
        f'  <joint name="{link.joint_name}" type="revolute">\n'
        f'    <parent link="{parent}"/>\n'
        f'    <child link="{link.name}_link"/>\n'
        f'    <origin xyz="{origin}" rpy="0 0 0"/>\n'
        f'    <axis xyz="{axis_vec}"/>\n'
        f'    <limit lower="{lower:.4f}" upper="{upper:.4f}" '
        f'effort="{effort:.2f}" velocity="{velocity:.3f}"/>\n'
        f'  </joint>'
    )
