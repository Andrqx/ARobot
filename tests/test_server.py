"""Command-server tests.

Most tests drive ``ArmService.dispatch`` directly — no sockets, fast and
deterministic. One integration test boots a real HTTP server on an ephemeral
port and hits it with ``http.client`` to prove the wire layer works too.
"""

import json
import sys
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import ArmService, load_config  # noqa: E402
from arm_control.server import _make_handler  # noqa: E402


def make_service() -> ArmService:
    return ArmService.simulated(load_config())


# --- dispatch-level unit tests ----------------------------------------

def test_api_index_lists_endpoints():
    status, body = make_service().dispatch("GET", "/api", None)
    assert status == 200
    assert "GET /state" in body["endpoints"]


def test_state_reports_pose_and_angles():
    svc = make_service()
    status, body = svc.dispatch("GET", "/state", None)
    assert status == 200
    assert set(body["angles_deg"]) == {"base", "shoulder", "elbow", "wrist"}
    assert set(body["pose"]) == {"x", "y", "z", "pitch_deg"}
    assert body["homed"] is False


def test_info_exposes_links_and_reach():
    status, body = make_service().dispatch("GET", "/info", None)
    assert status == 200
    assert body["links_mm"]["upper_arm"] == 300.0
    assert body["reach_mm"]["max"] > body["reach_mm"]["min"]
    assert [j["name"] for j in body["joints"]] == \
        ["base", "shoulder", "elbow", "wrist"]


def test_home_sets_homed_flag():
    svc = make_service()
    status, body = svc.dispatch("POST", "/home", None)
    assert status == 200
    assert body["ok"] is True
    assert body["state"]["homed"] is True


def test_move_reaches_target_pose():
    svc = make_service()
    status, body = svc.dispatch(
        "POST", "/move", {"x": 250, "y": 80, "z": 300, "pitch_deg": 0}
    )
    assert status == 200
    assert body["steps"] > 0
    assert abs(body["state"]["pose"]["x"] - 250) < 1e-1
    assert abs(body["state"]["pose"]["y"] - 80) < 1e-1


def test_move_missing_field_is_400():
    status, body = make_service().dispatch("POST", "/move", {"x": 1, "y": 2})
    assert status == 400
    assert "z" in body["detail"]


def test_move_unreachable_is_422():
    status, body = make_service().dispatch(
        "POST", "/move", {"x": 5000, "y": 0, "z": 300, "pitch_deg": 0}
    )
    assert status == 422
    assert body["error"] == "unreachable"


def test_move_joints_drives_to_angles():
    svc = make_service()
    status, body = svc.dispatch(
        "POST", "/move_joints", {"joints_deg": [0, 90, -30, 10]}
    )
    assert status == 200
    angles = body["state"]["angles_deg"]
    assert abs(angles["shoulder"] - 90) < 1e-2
    assert abs(angles["elbow"] + 30) < 1e-2


def test_move_joints_out_of_limit_is_422():
    # shoulder max is 150deg; ask for 200 -> limit violation.
    status, body = make_service().dispatch(
        "POST", "/move_joints", {"joints_deg": [0, 200, 0, 0]}
    )
    assert status == 422
    assert body["error"] == "limit_violation"


def test_move_joints_wrong_count_is_400():
    status, body = make_service().dispatch(
        "POST", "/move_joints", {"joints_deg": [0, 90]}
    )
    assert status == 400


def test_run_program_by_name():
    svc = make_service()
    status, body = svc.dispatch("POST", "/run_program",
                                {"name": "pick_place_demo"})
    assert status == 200
    assert len(body["waypoints"]) == 7


def test_run_program_inline():
    svc = make_service()
    program = {
        "name": "inline",
        "waypoints": [
            {"name": "a", "joints_deg": [0, 90, 0, 0]},
            {"name": "b", "joints_deg": [20, 80, -20, 5]},
        ],
    }
    status, body = svc.dispatch("POST", "/run_program", {"program": program})
    assert status == 200
    assert body["waypoints"] == ["a", "b"]


def test_unknown_route_is_404():
    status, body = make_service().dispatch("GET", "/nope", None)
    assert status == 404


def test_ui_file_loads():
    from arm_control.server import _load_ui
    html = _load_ui()
    assert b"ARobot" in html and b"<html" in html.lower()


def test_meshes_endpoint_returns_parts():
    status, body = make_service().dispatch("GET", "/meshes", None)
    assert status == 200
    kinds = {(p["kind"], p["name"]) for p in body["parts"]}
    assert ("link", "upper") in kinds          # a link shell
    assert ("joint", "shoulder") in kinds      # a joint drive
    up = next(p for p in body["parts"] if p["name"] == "upper")
    assert up["tris"] and len(up["tris"]) % 9 == 0   # flat xyz triples, 3 per triangle


def test_trailing_slash_is_normalized():
    status, _ = make_service().dispatch("GET", "/state/", None)
    assert status == 200


# --- one real-socket integration test ---------------------------------

def test_http_round_trip():
    svc = make_service()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(svc))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)

        # GET / serves the HTML control panel.
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type").startswith("text/html")
        html = resp.read().decode()
        assert "ARobot" in html

        conn.request("GET", "/state")
        resp = conn.getresponse()
        assert resp.status == 200
        state = json.loads(resp.read())
        assert "pose" in state

        payload = json.dumps({"x": 250, "y": 80, "z": 300, "pitch_deg": 0})
        conn.request("POST", "/move", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 200
        moved = json.loads(resp.read())
        assert abs(moved["state"]["pose"]["x"] - 250) < 1e-1
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll server tests passed.")
