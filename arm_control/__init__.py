"""ARobot — 4-DOF robotic arm control stack.

Pure-Python, runs in simulation today and on the Raspberry Pi later.
"""

from .config import ArmConfig, Joint, Motor, load_config
from .kinematics import ArmKinematics, Unreachable
from .trajectory import interpolate, plan_trapezoidal, Trajectory
from .drivers import JointDriver, SimDriver, StepperEncoderDriver, BusServoDriver
from .teach import Program, Waypoint
from .robot_model import RobotModel, Link
from .controller import ArmController

__all__ = [
    "ArmConfig",
    "Joint",
    "Motor",
    "load_config",
    "ArmKinematics",
    "Unreachable",
    "interpolate",
    "plan_trapezoidal",
    "Trajectory",
    "JointDriver",
    "SimDriver",
    "StepperEncoderDriver",
    "BusServoDriver",
    "Program",
    "Waypoint",
    "RobotModel",
    "Link",
    "ArmController",
]
