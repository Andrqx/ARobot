# ARobot — Control Software

Pure-Python control stack for a **4-DOF robotic arm** (base yaw + shoulder +
elbow + wrist pitch). Designed to run in **simulation today** on your laptop and
on a **Raspberry Pi 4** with closed-loop steppers when the hardware is built —
the same code, swapping only the joint driver.

## Hardware target
- **Brain:** Raspberry Pi 4 (all logic in Python — no separate firmware)
- **Actuators:** stepper motors with **encoders** (closed-loop, backlash correction)
- **Reductions:** cycloidal gearboxes (base + shoulder), GT2 belt pulleys (elbow + wrist)
- **Perception (future):** Pi camera — designed in as an optional module, not a dependency

## Layout
```
control/
├─ config/arm_config.yaml     # geometry + gearing — THE place to edit (fill in CAD numbers)
├─ arm_control/
│  ├─ config.py               # loads YAML, derives steps/deg + encoder counts/deg
│  ├─ kinematics.py           # 4-DOF forward + closed-form inverse kinematics
│  ├─ trajectory.py           # smooth (eased) point-to-point motion
│  ├─ drivers.py              # JointDriver interface: SimDriver now, StepperEncoderDriver later
│  ├─ trajectory.py           # synchronized trapezoidal motion profiles
│  ├─ teach.py                # teach-and-repeat: record/save/load/replay programs
│  ├─ robot_model.py          # robot description + URDF export (CAD/training seam)
│  └─ controller.py           # ArmController: move, home, teach, run_program
├─ config/geometry.yaml       # per-link CAD mesh + mass map (fill from SolidWorks)
├─ meshes/                    # drop exported STL files here
├─ programs/pick_place_demo.json  # an example taught routine
├─ sim/visualize.py           # 3D viewer + move animation + velocity profiles
├─ tests/                     # kinematics, trajectory, teach, robot-model (20 tests)
├─ examples/demo.py           # end-to-end simulated run
├─ examples/export_urdf.py    # write arobot.urdf for PyBullet/MuJoCo
└─ requirements.txt
```

## Quick start (on Windows / laptop)
```bash
cd control
python -m pip install -r requirements.txt
python examples/demo.py          # text demo: IK move + derived motor constants
python -m sim.visualize          # 3D animated move (renders CAD meshes if present)
pytest -q                        # prove the kinematics
```

`sim.visualize` draws the stick-figure skeleton until you export STLs and list
them in `config/geometry.yaml`; then it renders the real parts moving on the
arm — no code change. Two mesh maps: `links:` (shells placed along each link,
with optional per-part `mesh_rotation_deg`/`mesh_translation_mm` to align CAD
that wasn't authored pivot-at-origin) and `joints:` (drives like the cycloidal
housing, auto-centered on each pivot and spun about the joint axis). STEP files
convert with `examples/convert_step.py` (needs `pip install gmsh`).

## The key design idea: one seam to hardware
All motion goes through the `JointDriver` interface:
- **`SimDriver`** — works today, perfect tracking (optional backlash model).
- **`StepperEncoderDriver`** — stub for the Pi. When you build, implement it
  (GPIO step/dir + encoder read + correction loop). **Nothing above it changes.**

The gripper follows the same pattern via `GripperDriver` (`SimGripper` now,
`ServoGripper` stub later). Taught programs carry a per-waypoint `gripper`
field (`"open"`/`"close"`) that `run_program` actuates automatically — so the
bundled `pick_place_demo` actually grabs and releases in sim.

```python
from arm_control import ArmController, load_config
arm = ArmController.simulated(load_config())
arm.move_to_pose(x=250, y=80, z=300, pitch_deg=0)   # IK + smooth motion, simulated
```

## What you fill in from CAD
Open `config/arm_config.yaml` and replace every `TODO`:
- the four **link lengths** (mm),
- the **cycloidal gear ratios** (base, shoulder) and **wrist** pulley ratio
  (elbow is already 72T/16T = 4.5),
- your **encoder resolution**, and per-joint **angle limits**.

Everything downstream (steps-per-degree, reach, IK) updates automatically.

## Teach-and-repeat ("training")
```python
from arm_control import ArmController, Program, load_config
arm = ArmController.simulated(load_config())
arm.home()
arm.run_program(Program.load("pick_place_demo"))   # replay a taught routine

# teach your own:
prog = Program(name="my_routine")
arm.move_to_pose(x=250, y=80, z=300, pitch_deg=0)
prog.add(arm.record_waypoint("grab"))
prog.save()                                         # -> programs/my_routine.json
```

## Training in a physics sim
`python examples/export_urdf.py` writes `arobot.urdf` — load it in PyBullet
or MuJoCo to train. Placeholder geometry today; once CAD STLs are listed in
`config/geometry.yaml`, re-run and the URDF references your real meshes.

## What you fill in from CAD
- `config/arm_config.yaml` — final link lengths, encoder resolution, wrist
  servo torque (gear ratios + reach already set).
- `config/geometry.yaml` — point each link at its exported STL + real mass.

Everything downstream (steps-per-degree, reach, IK, URDF) updates automatically.

## Command server (drive the arm over the network)
A stdlib-only HTTP server (no Flask) wraps the controller, so a web UI, a
phone, or another machine can command the arm with JSON. Simulation-first like
everything else — same API once the hardware is real.

A self-contained **web control panel** is served at `GET /` (no external
CDNs, so it works offline on the Pi): a **live 3D view** of the arm (canvas +
in-browser forward kinematics, eased animation, drag-to-orbit) above a
joint/pose/gripper readout that polls once a second, plus forms to move, jog
joints, open/close the gripper, and run taught programs. Open
`http://localhost:8080/` in a browser. The raw JSON API index moves to
`GET /api`.

```bash
python examples/serve.py             # starts on http://0.0.0.0:8080
                                     # open http://localhost:8080/ for the panel

curl localhost:8080/state
curl localhost:8080/info
curl -X POST localhost:8080/home
curl -X POST localhost:8080/move        -d '{"x":250,"y":80,"z":300,"pitch_deg":0}'
curl -X POST localhost:8080/move_joints -d '{"joints_deg":[0,90,-30,10]}'
curl -X POST localhost:8080/run_program -d '{"name":"pick_place_demo"}'
```

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/state` | current joint angles, tool pose, homed flag |
| GET | `/info` | link lengths, joint limits, reach envelope |
| GET | `/programs` | names of taught programs on disk |
| POST | `/home` | run the homing routine |
| POST | `/move` | Cartesian move `{x, y, z, pitch_deg?, elbow_up?}` |
| POST | `/move_joints` | direct joint-space move `{joints_deg: [...]}` |
| POST | `/gripper` | open/close the gripper `{state: 'open'|'close'}` |
| POST | `/run_program` | replay `{name}` or inline `{program}` |

Out-of-reach targets and joint-limit violations come back as HTTP `422` with a
message; the routing core (`ArmService.dispatch`) is unit-tested without sockets.

## Perception (camera → target → move)
Same simulation-first seam as the drivers: a `DetectionSource` interface with
`SimDetector` (works today, scripted detections) and `ArucoCameraDetector` (the
Pi-camera stub you fill in later). A `CameraExtrinsics` transform converts what
the camera sees into base coordinates, and `PerceptionPipeline` feeds that
straight into `move_to_pose`.

```bash
python examples/perception_demo.py   # simulated camera spots a cube, arm goes to it
```

```python
from arm_control import (ArmController, PerceptionPipeline, SimDetector,
                         Detection, CameraExtrinsics, load_config)

arm = ArmController.simulated(load_config())
camera = SimDetector([Detection("cube", position_cam_mm=[300, 60, 250])])
pipe = PerceptionPipeline(arm, camera, CameraExtrinsics.identity())
pipe.pick_nearest_and_move(pitch_deg=0, hover_mm=40)   # hover 40mm above it
```

When the Pi camera exists, implement `ArucoCameraDetector` (OpenCV ArUco pose)
and swap it in for `SimDetector` — nothing downstream changes. Measure
`CameraExtrinsics` once via a hand-eye calibration.

## Roadmap
1. ✅ Kinematics + simulator + tests
2. ✅ Trapezoidal velocity/accel motion profiles
3. ✅ Teach-and-repeat + homing flag
4. ✅ Robot description + URDF export (CAD/training seam)
5. ⬜ `StepperEncoderDriver`: GPIO + encoder closed-loop on the Pi
6. ✅ Command server (so the arm takes target poses / programs over the network)
7. ✅ CAD mesh visualizer — `sim/visualize.py` renders STLs listed in
   `config/geometry.yaml` (pure-Python loader, no new deps), placed by
   `ArmKinematics.link_frames`; falls back to the skeleton until meshes exist
8. ✅ **Perception module** — camera → ArUco/blob detection → target `(x,y,z)`
   feeding straight into `move_to_pose` (sim-first; `ArucoCameraDetector` stub
   for the real Pi camera)
