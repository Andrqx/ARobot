"""Joint drivers — the hardware seam.

The control code only ever talks to the ``JointDriver`` interface, so the
exact same kinematics/controller logic runs:

    * today, against ``SimDriver`` (no hardware, perfect tracking), and
    * later, against ``StepperEncoderDriver`` on the Pi (closed-loop
      stepper + encoder, with backlash correction).

When you finish building, you implement ``StepperEncoderDriver`` and change
ONE line in the controller — nothing above it changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class JointDriver(ABC):
    """Abstract single-joint actuator measured in output degrees."""

    @abstractmethod
    def move_to_deg(self, angle_deg: float) -> None:
        """Command the joint toward an absolute output angle (degrees)."""

    @abstractmethod
    def read_deg(self) -> float:
        """Return the joint's current measured output angle (degrees)."""


class SimDriver(JointDriver):
    """Pure-software joint for the simulator.

    Optionally models backlash (lost motion on direction reversal) so you can
    preview how much your encoders will have to correct for once built.
    """

    def __init__(self, home_deg: float = 0.0, backlash_deg: float = 0.0):
        self._commanded = home_deg
        self._measured = home_deg
        self._backlash = backlash_deg
        self._last_dir = 0

    def move_to_deg(self, angle_deg: float) -> None:
        direction = 0
        if angle_deg > self._commanded:
            direction = 1
        elif angle_deg < self._commanded:
            direction = -1

        self._commanded = angle_deg

        # Model backlash: on a direction reversal, the first bit of motion is
        # "eaten" by the slack before the output actually moves.
        measured = angle_deg
        if self._backlash and direction != 0 and direction != self._last_dir:
            measured -= direction * (self._backlash / 2.0)
        self._measured = measured

        if direction != 0:
            self._last_dir = direction

    def read_deg(self) -> float:
        return self._measured


class StepperEncoderDriver(JointDriver):
    """Closed-loop stepper + encoder driver for the Raspberry Pi.

    Intentionally a stub until the hardware exists. When you build:
      * drive STEP/DIR via pigpio / RPi.GPIO (or a HAT),
      * read the encoder each loop,
      * run a small correction loop so commanded == measured despite backlash.
    """

    def __init__(self, *, step_pin: int, dir_pin: int,
                 steps_per_deg: float, encoder, deadband_deg: float = 0.1):
        raise NotImplementedError(
            "StepperEncoderDriver: wire up GPIO + encoder when the arm is built. "
            "Until then use SimDriver."
        )

    def move_to_deg(self, angle_deg: float) -> None:  # pragma: no cover
        raise NotImplementedError

    def read_deg(self) -> float:  # pragma: no cover
        raise NotImplementedError


class BusServoDriver(JointDriver):
    """Waveshare serial bus servo (e.g. ST3215) for the wrist.

    These are smart servos: absolute position feedback built in (no separate
    encoder), commanded over a half-duplex serial bus by ID. Stub until the
    wrist servo is chosen and wired.

    When you build:
      * pick the model (sets resolution + torque),
      * talk to it over the Waveshare bus (serial), addressing it by ``servo_id``,
      * map output degrees <-> the servo's raw position units.
    """

    def __init__(self, *, servo_id: int, port: str = "/dev/ttyAMA0",
                 counts_per_rev: int = 4096):
        raise NotImplementedError(
            "BusServoDriver: choose the Waveshare servo and wire the serial bus "
            "when the wrist is built. Until then use SimDriver."
        )

    def move_to_deg(self, angle_deg: float) -> None:  # pragma: no cover
        raise NotImplementedError

    def read_deg(self) -> float:  # pragma: no cover
        raise NotImplementedError
