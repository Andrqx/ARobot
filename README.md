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
python -m sim.visualize          # 3D animated move
pytest -q                        # prove the kinematics
```

## The key design idea: one seam to hardware
All motion goes through the `JointDriver` interface:
- **`SimDriver`** — works today, perfect tracking (optional backlash model).
- **`StepperEncoderDriver`** — stub for the Pi. When you build, implement it
  (GPIO step/dir + encoder read + correction loop). **Nothing above it changes.**

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

## Roadmap
1. ✅ Kinematics + simulator + tests
2. ✅ Trapezoidal velocity/accel motion profiles
3. ✅ Teach-and-repeat + homing flag
4. ✅ Robot description + URDF export (CAD/training seam)
5. ⬜ `StepperEncoderDriver`: GPIO + encoder closed-loop on the Pi
6. ⬜ Command server (so the arm takes target poses / programs over the network)
7. ⬜ CAD mesh visualizer (render STLs in the sim once exported)
8. ⬜ **Perception module** — Pi camera → ArUco/blob detection → target `(x,y,z)`
   feeding straight into `move_to_pose` (start with markers, not neural nets)
