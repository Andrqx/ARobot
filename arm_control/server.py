"""Command server: drive the arm over the network with JSON over HTTP.

This is roadmap item #6 — the seam that lets *something else* command the
arm: a web UI, a phone, another script, or the future vision module on a
second machine. It wraps an :class:`ArmController` and exposes a small REST
API. Like the rest of the stack it is simulation-first: point it at a
``SimDriver`` arm today, a hardware-backed arm later, and the API is
identical.

Design notes
------------
* **Stdlib only.** Built on ``http.server`` — no Flask/FastAPI — so it runs on
  a bare Raspberry Pi with nothing extra to install.
* **A testable core.** All routing lives in :meth:`ArmService.dispatch`, a
  plain function that takes ``(method, path, body)`` and returns
  ``(status_code, response_dict)``. It never touches a socket, so tests can
  exercise every endpoint directly. The HTTP layer is a thin shell over it.
* **Errors map to status codes.** An unreachable target or a joint-limit
  violation comes back as a clean ``422`` with a message, not a 500/stacktrace.

API
---
======  ===============  ===================================================
Method  Path             Purpose
======  ===============  ===================================================
GET     /                endpoint index (this help)
GET     /state           current joint angles, tool pose, homed flag
GET     /info            link lengths, per-joint limits, reach envelope
GET     /programs        names of taught programs on disk
POST    /home            run the homing routine
POST    /move            body {x, y, z, pitch_deg?, elbow_up?} — Cartesian move
POST    /move_joints     body {joints_deg: [...]} — direct joint-space move
POST    /run_program     body {name: "..."} or {program: {...}} — replay
======  ===============  ===================================================

Run it::

    python examples/serve.py            # or: python -m arm_control.server

    curl localhost:8080/state
    curl -X POST localhost:8080/move -d '{"x":250,"y":80,"z":300,"pitch_deg":0}'
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .config import load_config
from .controller import ArmController, LimitViolation
from .kinematics import Unreachable
from .teach import Program

# The browser control panel served at GET /.
_UI_FILE = Path(__file__).resolve().parent / "web" / "index.html"

# The endpoints advertised at GET /api — also handy as a quick self-doc.
ENDPOINTS = {
    "GET /": "web control panel (HTML)",
    "GET /api": "this index (machine-readable)",
    "GET /state": "current joint angles, tool pose, homed flag",
    "GET /info": "link lengths, joint limits, reach envelope",
    "GET /programs": "names of taught programs on disk",
    "GET /meshes": "CAD mesh triangles (links + joints) in local frames",
    "POST /home": "run the homing routine",
    "POST /move": "Cartesian move — body {x, y, z, pitch_deg?, elbow_up?}",
    "POST /move_joints": "joint-space move — body {joints_deg: [...]}",
    "POST /gripper": "actuate the gripper — body {state: 'open'|'close'}",
    "POST /run_program": "replay a program — body {name} or {program}",
}


class ArmService:
    """The arm's command logic, independent of any transport.

    Wraps a single :class:`ArmController`. All public route handlers return a
    plain ``dict`` (the JSON body); :meth:`dispatch` adds the HTTP status code
    and turns the project's exceptions into error responses. A lock serializes
    motions so two overlapping requests can't interleave joint commands.
    """

    def __init__(self, controller: ArmController):
        self.arm = controller
        self._lock = threading.Lock()

    @classmethod
    def simulated(cls, cfg=None) -> "ArmService":
        """Convenience: a service backed by a fully simulated arm."""
        return cls(ArmController.simulated(cfg or load_config()))

    # --- State views ---------------------------------------------------

    def state(self) -> dict:
        angles = [round(float(a), 4) for a in self.arm.current_angles_deg()]
        x, y, z, pitch = self.arm.kin.tip_pose_deg(angles)
        return {
            "angles_deg": dict(zip(self._joint_names(), angles)),
            "pose": {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
                     "pitch_deg": round(pitch, 3)},
            "gripper": self.arm.gripper_state(),
            "homed": self.arm.homed,
        }

    def info(self) -> dict:
        cfg = self.arm.cfg
        rmin, rmax = self.arm.kin.reach()
        return {
            "links_mm": {
                "base_height": cfg.base_height_mm,
                "upper_arm": cfg.upper_arm_mm,
                "forearm": cfg.forearm_mm,
                "tool": cfg.tool_mm,
            },
            "joints": [
                {"name": j.name, "axis": j.axis,
                 "min_deg": j.min_deg, "max_deg": j.max_deg, "home_deg": j.home_deg}
                for j in cfg.joints
            ],
            "reach_mm": {"min": round(rmin, 1), "max": round(rmax, 1)},
        }

    def programs(self) -> dict:
        from .teach import PROGRAMS_DIR
        names = sorted(p.stem for p in PROGRAMS_DIR.glob("*.json"))
        return {"programs": names}

    def meshes(self) -> dict:
        """CAD triangles for the browser view: each part's verts in its own
        local frame (link shells + joint drives). The client transforms them by
        the live link/joint frames each frame. Empty list if no STLs exist."""
        from sim.visualize import load_link_meshes, load_joint_meshes
        parts = []
        for kind, loader in (("link", load_link_meshes), ("joint", load_joint_meshes)):
            for name, verts in loader().items():
                parts.append({"kind": kind, "name": name,
                              "tris": verts.round(1).reshape(-1).tolist()})
        return {"parts": parts}

    # --- Commands ------------------------------------------------------

    def home(self) -> dict:
        with self._lock:
            self.arm.home()
        return {"ok": True, "state": self.state()}

    def move(self, body: dict) -> dict:
        x, y, z = _require(body, "x"), _require(body, "y"), _require(body, "z")
        pitch = float(body.get("pitch_deg", 0.0))
        elbow_up = bool(body.get("elbow_up", True))
        with self._lock:
            traj = self.arm.move_to_pose(float(x), float(y), float(z),
                                         pitch_deg=pitch, elbow_up=elbow_up)
        return {"ok": True, "steps": len(traj.pos), "state": self.state()}

    def move_joints(self, body: dict) -> dict:
        q = _require(body, "joints_deg")
        with self._lock:
            traj = self.arm.move_to_angles_deg(q)
        return {"ok": True, "steps": len(traj.pos), "state": self.state()}

    def gripper(self, body: dict) -> dict:
        command = _require(body, "state")
        with self._lock:
            self.arm.set_gripper(str(command))
        return {"ok": True, "state": self.state()}

    def run_program(self, body: dict) -> dict:
        if "name" in body:
            program = Program.load(str(body["name"]))
        elif "program" in body:
            program = Program.from_dict(body["program"])
        else:
            raise KeyError("run_program needs 'name' or 'program'")
        visited: list[str] = []
        with self._lock:
            self.arm.run_program(
                program, on_waypoint=lambda wp, traj: visited.append(wp.name)
            )
        return {"ok": True, "waypoints": visited, "state": self.state()}

    # --- Routing -------------------------------------------------------

    def dispatch(self, method: str, path: str, body: dict | None) -> tuple[int, dict]:
        """Map an HTTP request to a handler and a status code.

        Returns ``(status_code, response_dict)``. This is the whole API in one
        place and is what the tests drive — no sockets required.
        """
        route = (method.upper(), urlparse(path).path.rstrip("/") or "/")
        try:
            if route == ("GET", "/api"):
                return 200, {"endpoints": ENDPOINTS}
            if route == ("GET", "/state"):
                return 200, self.state()
            if route == ("GET", "/info"):
                return 200, self.info()
            if route == ("GET", "/programs"):
                return 200, self.programs()
            if route == ("GET", "/meshes"):
                return 200, self.meshes()
            if route == ("POST", "/home"):
                return 200, self.home()
            if route == ("POST", "/move"):
                return 200, self.move(body or {})
            if route == ("POST", "/move_joints"):
                return 200, self.move_joints(body or {})
            if route == ("POST", "/gripper"):
                return 200, self.gripper(body or {})
            if route == ("POST", "/run_program"):
                return 200, self.run_program(body or {})
            return 404, {"error": f"no route for {method} {path}"}
        except Unreachable as e:
            return 422, {"error": "unreachable", "detail": str(e)}
        except LimitViolation as e:
            return 422, {"error": "limit_violation", "detail": str(e)}
        except (KeyError, ValueError, TypeError) as e:
            return 400, {"error": "bad_request", "detail": str(e)}
        except FileNotFoundError as e:
            return 404, {"error": "not_found", "detail": str(e)}

    def _joint_names(self) -> list[str]:
        return [j.name for j in self.arm.cfg.joints]


def _require(body: dict, key: str):
    """Fetch a required field or raise KeyError -> mapped to HTTP 400."""
    if key not in body:
        raise KeyError(f"missing required field '{key}'")
    return body[key]


# ----------------------------------------------------------------------
# HTTP layer — a thin shell that hands requests to ArmService.dispatch.
# ----------------------------------------------------------------------

def _load_ui() -> bytes:
    """Read the control-panel HTML (served at GET /)."""
    try:
        return _UI_FILE.read_bytes()
    except FileNotFoundError:
        return b"<h1>ARobot</h1><p>UI file missing; API is at /api</p>"


def _make_handler(service: ArmService):
    class Handler(BaseHTTPRequestHandler):
        # Quieter logs; override if you want request logging.
        def log_message(self, *args):  # noqa: D401
            pass

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, body: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict | None:
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return None
            raw = self.rfile.read(length)
            return json.loads(raw) if raw else None

        def do_GET(self):
            if urlparse(self.path).path in ("/", "/index.html", "/ui"):
                self._send_html(_load_ui())
                return
            status, payload = service.dispatch("GET", self.path, None)
            self._send(status, payload)

        def do_POST(self):
            try:
                body = self._read_body()
            except (json.JSONDecodeError, ValueError) as e:
                self._send(400, {"error": "bad_json", "detail": str(e)})
                return
            status, payload = service.dispatch("POST", self.path, body)
            self._send(status, payload)

    return Handler


def serve(service: ArmService | None = None, host: str = "0.0.0.0",
          port: int = 8080) -> None:
    """Start the command server and block, serving until Ctrl-C.

    With no ``service`` given, boots a fresh simulated arm.
    """
    service = service or ArmService.simulated()
    httpd = ThreadingHTTPServer((host, port), _make_handler(service))
    shown = "localhost" if host in ("0.0.0.0", "") else host
    print(f"ARobot command server on http://{host}:{port}  (Ctrl-C to stop)")
    print(f"  open the control panel at  http://{shown}:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()
