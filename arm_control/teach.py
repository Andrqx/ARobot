"""Teach-and-repeat: record poses, save them as sequences, play them back.

This is the "training" layer in the teach-by-demonstration sense:
  * jog/move the arm to a pose, ``record`` it as a named waypoint,
  * collect waypoints into a ``Program``,
  * ``save`` it to disk (JSON) and ``load`` it on any machine,
  * have the controller replay it with the trapezoidal motion planner.

Works identically in simulation and on the real arm — a saved program is
just data, so a routine you teach in sim runs unchanged on hardware.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Saved programs live here by default.
PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "programs"


@dataclass
class Waypoint:
    name: str
    joints_deg: list[float]       # the 4 joint angles at this point
    pause_s: float = 0.0          # dwell time after arriving (honored on hardware)
    gripper: str | None = None    # "open" / "close" / None (reserved for later)


@dataclass
class Program:
    """An ordered list of waypoints — a taught routine."""
    name: str
    waypoints: list[Waypoint] = field(default_factory=list)
    description: str = ""

    def add(self, waypoint: Waypoint) -> "Program":
        self.waypoints.append(waypoint)
        return self

    def __len__(self) -> int:
        return len(self.waypoints)

    # --- Serialization -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "waypoints": [asdict(w) for w in self.waypoints],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Program":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            waypoints=[Waypoint(**w) for w in d.get("waypoints", [])],
        )

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else PROGRAMS_DIR / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Program":
        p = Path(path)
        if not p.exists() and not p.is_absolute():
            p = PROGRAMS_DIR / p  # allow loading by bare name
            if p.suffix != ".json":
                p = p.with_suffix(".json")
        return cls.from_dict(json.loads(p.read_text()))
