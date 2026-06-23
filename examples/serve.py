"""Launch the ARobot command server (simulated arm).

    python examples/serve.py [--host H] [--port P]

Then, from another terminal:

    curl localhost:8080/state
    curl localhost:8080/info
    curl -X POST localhost:8080/home
    curl -X POST localhost:8080/move -d '{"x":250,"y":80,"z":300,"pitch_deg":0}'
    curl -X POST localhost:8080/move_joints -d '{"joints_deg":[0,90,-30,10]}'
    curl -X POST localhost:8080/run_program -d '{"name":"pick_place_demo"}'

The arm is fully simulated. When the hardware exists, build the service with a
hardware-backed ArmController instead — the API does not change.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_control import ArmService, load_config, serve  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="ARobot command server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    service = ArmService.simulated(load_config())
    serve(service, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
