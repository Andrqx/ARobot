"""Gripper — the end-effector that actually grabs things.

Same seam pattern as the joint drivers: the control code only talks to the
:class:`GripperDriver` interface, so taught pick-and-place programs run
identically in sim and on hardware.

  * **``SimGripper``** — pure software, tracks open/closed; works today.
  * **``ServoGripper``** — stub for a real servo-driven gripper on the Pi.

The two command strings are ``"open"`` and ``"close"`` — the same values the
:class:`~arm_control.teach.Waypoint` ``gripper`` field already uses, so a taught
program drives the gripper with no translation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

OPEN = "open"
CLOSE = "close"
_VALID = {OPEN, CLOSE}


class GripperDriver(ABC):
    """Abstract two-state (or proportional) end-effector."""

    @abstractmethod
    def open(self) -> None:
        """Open the gripper (release)."""

    @abstractmethod
    def close(self) -> None:
        """Close the gripper (grasp)."""

    @abstractmethod
    def state(self) -> str:
        """Current state: ``"open"`` or ``"close"``."""

    def set(self, command: str) -> None:
        """Drive to ``"open"`` or ``"close"`` by name."""
        if command == OPEN:
            self.open()
        elif command == CLOSE:
            self.close()
        else:
            raise ValueError(
                f"unknown gripper command {command!r} (expected 'open' or 'close')"
            )


class SimGripper(GripperDriver):
    """Software gripper for the simulator — just remembers its state."""

    def __init__(self, start: str = OPEN):
        if start not in _VALID:
            raise ValueError(f"start must be 'open' or 'close', got {start!r}")
        self._state = start

    def open(self) -> None:
        self._state = OPEN

    def close(self) -> None:
        self._state = CLOSE

    def state(self) -> str:
        return self._state


class ServoGripper(GripperDriver):
    """Servo-driven gripper for the Pi — stub until the hardware exists.

    When you build it:
      * pick the gripper + servo (sets the open/close travel),
      * command the servo to ``open_deg`` / ``close_deg`` over its bus,
      * optionally read current/load to detect a successful grasp.
    Until then use :class:`SimGripper`.
    """

    def __init__(self, *, servo_id: int, port: str = "/dev/ttyAMA0",
                 open_deg: float = 0.0, close_deg: float = 60.0):
        raise NotImplementedError(
            "ServoGripper: choose the gripper servo and wire it when the "
            "end-effector is built. Until then use SimGripper."
        )

    def open(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def state(self) -> str:  # pragma: no cover
        raise NotImplementedError
