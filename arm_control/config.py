"""Load the arm's hardware configuration and derive control constants.

The YAML file is the single source of truth for geometry and gearing.
Everything the motors need (steps-per-degree, encoder-counts-per-degree)
is *computed* from it so there is only ever one number to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Default config location: ../config/arm_config.yaml relative to this file.
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "arm_config.yaml"


@dataclass
class Joint:
    name: str
    axis: str          # "yaw" or "pitch"
    gear_ratio: float  # output revs per motor rev (reduction)
    min_deg: float
    max_deg: float
    home_deg: float
    driver_type: str = "stepper"     # "stepper" or "bus_servo"
    motor_torque_nm: float = 0.0     # motor holding/stall torque before reduction
    max_vel_deg_s: float = 60.0      # output speed limit (deg/s) for motion planning
    max_accel_deg_s2: float = 120.0  # output accel limit (deg/s^2) for motion planning

    def clamp(self, angle_deg: float) -> float:
        """Clamp an angle to this joint's mechanical limits."""
        return max(self.min_deg, min(self.max_deg, angle_deg))

    def in_range(self, angle_deg: float) -> bool:
        return self.min_deg <= angle_deg <= self.max_deg

    def output_torque_nm(self, efficiency: float = 0.85) -> float:
        """Gross torque at the joint output after reduction and efficiency.

        For a bus servo the motor torque already includes its internal
        reduction, so gear_ratio is 1.0 and efficiency is treated as ~1.
        """
        eff = 1.0 if self.driver_type == "bus_servo" else efficiency
        return self.motor_torque_nm * self.gear_ratio * eff


@dataclass
class Motor:
    full_steps_per_rev: int
    microsteps: int
    encoder_counts_per_rev: int

    @property
    def microsteps_per_rev(self) -> float:
        return self.full_steps_per_rev * self.microsteps


@dataclass
class ArmConfig:
    base_height_mm: float
    upper_arm_mm: float
    forearm_mm: float
    tool_mm: float
    joints: list[Joint]
    motor: Motor

    @property
    def link_lengths(self) -> tuple[float, float, float, float]:
        """(L1, L2, L3, L4) = base height, upper arm, forearm, tool."""
        return (self.base_height_mm, self.upper_arm_mm,
                self.forearm_mm, self.tool_mm)

    # --- Derived motion constants -------------------------------------

    def steps_per_deg(self, joint: Joint) -> float:
        """Microsteps the motor must take to move this joint one output degree."""
        return self.motor.microsteps_per_rev * joint.gear_ratio / 360.0

    def encoder_counts_per_deg(self, joint: Joint) -> float:
        """Encoder counts seen per output degree (for closed-loop feedback)."""
        return self.motor.encoder_counts_per_rev * joint.gear_ratio / 360.0

    def home_angles_deg(self) -> list[float]:
        return [j.home_deg for j in self.joints]

    def max_vel_deg_s(self) -> list[float]:
        return [j.max_vel_deg_s for j in self.joints]

    def max_accel_deg_s2(self) -> list[float]:
        return [j.max_accel_deg_s2 for j in self.joints]


def load_config(path: str | Path = DEFAULT_CONFIG) -> ArmConfig:
    data = yaml.safe_load(Path(path).read_text())
    links = data["links"]
    joints = [Joint(**j) for j in data["joints"]]
    motor = Motor(**data["motor"])
    return ArmConfig(
        base_height_mm=links["base_height_mm"],
        upper_arm_mm=links["upper_arm_mm"],
        forearm_mm=links["forearm_mm"],
        tool_mm=links["tool_mm"],
        joints=joints,
        motor=motor,
    )
