# MuJoCo Table Pick-And-Place Scene

This folder contains a first-pass MuJoCo simulation scene: a robot picks up a bowl from a table and places it into a target bowl on the same table. The background includes shelves, boxes, a chair, and small tabletop clutter to make the workspace feel less empty.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements prefer the latest MuJoCo release when Python 3.10+ is available. On this Mac's default Python 3.9, they fall back to `mujoco==3.3.7`, which has a prebuilt wheel for this machine.

## Verify The Scene Loads

```bash
python sim/check_scene.py
```

## Run The Interactive Demo

On macOS, use `mjpython` for the passive MuJoCo viewer:

```bash
mjpython sim/run_pick_place_demo.py
```

On Linux or Windows, regular Python should work:

```bash
python sim/run_pick_place_demo.py
```

## Open The Static Scene Directly

```bash
python -m mujoco.viewer --mjcf=sim/worlds/table_bowl_pick_place.xml
```

## Hiwonder JetArm Prototype Demo

The JetArm version is a smaller desktop prototype scene for testing the same bowl-to-target-bowl task before retargeting to the larger FANUC arm:

```bash
mjpython sim/run_jetarm_pick_place_demo.py
```

The scene includes a JetArm-style green/black desktop setup, controller base, screen, cables, tabletop calibration mat, clutter objects, laptop/control station, tools, power strip, lab wall details, floor mat, storage boxes, and a four-light task-status panel. During the scripted demo the status lights advance through pick, carry, place, and success.

Interactive controls:

```text
1 overview   2 table   3 robot   4 status   5 top
Space pause/resume
R restart
[ slower     ] faster
- zoom out   + zoom in
Mouse drag/scroll still works in the MuJoCo window
```

Headless validation and waypoint export:

```bash
python sim/check_jetarm_task_success.py
python sim/run_jetarm_pick_place_demo.py --headless --export-waypoints
```

The exported waypoint draft is written to:

```bash
outputs/jetarm_bowl_to_plate_waypoints.json
```

The JetArm gripper is sized around the simulated bowl: the open span leaves clearance around the rim, and the closed span pinches the bowl body instead of clipping through it. The latest gripper model projects the finger rails forward so the claw faces outward like the physical JetArm end effector. The exported JSON includes both a simulated joint waypoint draft and a `cartesian_follow_plan_m` section with staged tabletop positions: ready, pre-pick, grasp height, lift, transfer, pre-place, release, and retreat. It also includes `gripper_calibration` pulse hints for the Hiwonder gripper servo.

Use those waypoints as a calibration artifact, not direct hardware-safe commands. Before moving the physical JetArm, confirm servo IDs, zero offsets, joint signs, angle limits, gripper open/close pulses, speed limits, and clearances in the real workspace.

## Dual FANUC Metal Buffing Demo

The dual-FANUC demo models a metal buffing workcell from the shop reference photos/videos:

- Left CR-7iA/L-style arm holds the sheet with a pneumatic/vacuum fixture.
- Right CR-7iA/L-style arm carries a motorized buffing wheel.
- The workcell includes a metal table, fixture rails, sheet holes/slots, stacked sheet metal, shelves, corrugated wall panels, lights, and a status panel.
- Success requires the holder to be clamped and all four buffing passes to complete.

Run the headless validation and generate a review PNG:

```bash
python sim/run_dual_fanuc_buffing_demo.py --headless --review
```

Open the interactive viewer on macOS:

```bash
mjpython sim/run_dual_fanuc_buffing_demo.py
```

Generated artifacts:

```text
outputs/generated/dual_fanuc_metal_buffing_scene.xml
outputs/generated/dual_fanuc_metal_buffing_review_sheet.png
outputs/generated/dual_fanuc_metal_buffing_snapshot.png
outputs/generated/dual_fanuc_metal_buffing_result.json
```

## JetArm Hardware Dry Run

The hardware bridge lives on the robot at `~/viha_jetarm/jetarm_hardware_bridge.py` and locally at:

```bash
sim/jetarm_hardware_bridge.py
```

Its default mode is safe: it creates a pulse plan but does not move the arm.

```bash
ssh -i ~/.ssh/viha_jetarm_ed25519 ubuntu@192.168.12.89 \
  'zsh -lc "source ~/ros2_ws/.hiwonderrc >/dev/null 2>&1; cd ~/viha_jetarm; python3 jetarm_hardware_bridge.py --output dry_run_plan.json"'
```

The latest generated plan was copied back to:

```bash
outputs/jetarm_hardware_dry_run_plan.json
```

The local rosbridge helper can inspect that plan without publishing motion:

```bash
node sim/jetarm_rosbridge_execute.js --plan outputs/jetarm_hardware_dry_run_plan.json
```

Only after physical clearance checks, the same helper can publish the pulses through the JetArm's running rosbridge server by adding `--execute`.

## Picture-To-JetArm Workflow

For a picture-driven task, use the photo spec template:

```bash
sim/photo_specs/jetarm_photo_task_spec_template.json
```

The intended flow is:

```text
scene photo + scale/calibration points
  -> object positions in table coordinates
  -> JetArm MuJoCo scene
  -> simulated success check
  -> JetArm waypoint JSON
  -> calibrated Hiwonder ROS/Python movement script
```

A single photo needs calibration. At minimum, provide four table-corner pixels or a known marker, the real table/workspace dimensions, and the JetArm base position. Without that, the system can identify relative layout but cannot safely infer real-world robot distances.

## Task Contract

The hand-authored task contract is in:

```bash
sim/tasks/bowl_to_plate_task_spec.json
```

It follows the context document's `task_spec.json` idea: object IDs, robot ID, ordered stages, success predicate, constraints, and concrete predicate definitions.
