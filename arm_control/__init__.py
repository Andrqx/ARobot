"""ARobot — 4-DOF robotic arm control stack.

Pure-Python, runs in simulation today and on the Raspberry Pi later.
"""

from .config import ArmConfig, Joint, Motor, load_config
from .kinematics import ArmKinematics, Unreachable
from .trajectory import interpolate, plan_trapezoidal, Trajectory
from .drivers import JointDriver, SimDriver, StepperEncoderDriver, BusServoDriver
from .gripper import GripperDriver, SimGripper, ServoGripper
from .teach import Program, Waypoint
from .robot_model import RobotModel, Link
from .controller import ArmController
from .server import ArmService, serve
from .perception import (
    Detection, CameraExtrinsics, DetectionSource, SimDetector,
    ArucoCameraDetector, BaseTarget, PerceptionPipeline,
)

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
    "GripperDriver",
    "SimGripper",
    "ServoGripper",
    "Program",
    "Waypoint",
    "RobotModel",
    "Link",
    "ArmController",
    "ArmService",
    "serve",
    "Detection",
    "CameraExtrinsics",
    "DetectionSource",
    "SimDetector",
    "ArucoCameraDetector",
    "BaseTarget",
    "PerceptionPipeline",
]
