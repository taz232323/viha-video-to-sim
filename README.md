# ViHa Robot Video-To-Simulation Prototype

ViHa is a local prototype for turning tabletop robot-task media into MuJoCo simulations. The current work focuses on Hiwonder JetArm first, with the same media pipeline designed to extend to FANUC later.

The project now supports:

- A detailed JetArm MuJoCo scene with table, gripper, target objects, and success checks.
- Photo/spec-to-simulation generation.
- Video recording through a local Chrome page.
- Video frame extraction from user-marked task windows.
- Clean-frame selection and selected-frame manifests.
- First-pass video-derived scene generation.
- A tissue-from-box JetArm task preset with a tissue box, tissue sheet, pull-clear target, and success check.
- A dual-FANUC CR-7iA/L metal buffing workcell demo with a pneumatic holder arm, buffing-tool arm, shop environment, review PNG, and success predicate.
- Review PNG sheets so a user can inspect sim stages without replaying the terminal viewer.

## Current Workflow

```text
record task video
  -> mark useful time windows
  -> extract candidate frames
  -> choose clean frames
  -> annotate calibration/object/target points
  -> generate MuJoCo scene
  -> run simulation
  -> save review PNG + result JSON
```

For the first video test, the task is intentionally simple:

```text
human moves a small object/cube onto a red target square
  -> JetArm sim recreates the same pick/place task
```

The Hiwonder robot can be visible in the video. It helps if the robot base is visible, as long as it does not block the calibration marker, object center, table corners, or target center.

## Quick Start

Create or activate the virtual environment, then install Python dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Install FFmpeg for video extraction:

```bash
brew install ffmpeg
```

Run the local Chrome recorder:

```bash
python3 -m http.server 8765 --directory tools/video_recorder
```

Open:

```text
http://localhost:8765
```

Download recordings into:

```text
inputs/videos/
```

Or use the newer all-in-one local upload UI:

```bash
.venv/bin/python tools/video_to_sim_app/server.py
```

Open:

```text
http://127.0.0.1:8777
```

The UI lets you upload a video, extract frames, click calibration/object/target points, build the JetArm sim, run it, open the live MuJoCo viewer, and view the generated review PNG from the browser.

The same UI also includes a dedicated `FANUC Buffing Demo` button. It builds a two-arm metal buffing process simulation and returns a review PNG, snapshot, generated scene XML, and result JSON.

For the original cube-to-square test video, use `Use First-Test Cube Points` after upload to reproduce the first working calibration. Keep `Open live MuJoCo viewer after build` checked when you want the interactive simulation window, not only the PNG review sheet.

Extract and select frames from the sample cube-to-square window plan:

```bash
.venv/bin/python sim/extract_video_frames.py \
  --plan inputs/video_plans/cube_to_square_window_plan_template.json \
  --force
```

Build and run a generated JetArm scene:

```bash
.venv/bin/python sim/build_jetarm_scene_from_media.py \
  --spec outputs/video_frames/cube_to_square_test_001/jetarm_video_task_spec.json \
  --run-headless
```

Render a review PNG:

```bash
.venv/bin/python sim/render_jetarm_review_sheet.py \
  --scene outputs/generated/cube_to_square_video_scene.xml \
  --output outputs/generated/cube_to_square_video_review_sheet.png
```

Run the dual-FANUC metal buffing demo directly:

```bash
.venv/bin/python sim/run_dual_fanuc_buffing_demo.py --headless --review
```

Open the interactive dual-FANUC viewer on macOS:

```bash
.venv/bin/mjpython sim/run_dual_fanuc_buffing_demo.py
```

## Main Files

- `sim/run_jetarm_pick_place_demo.py` - JetArm MuJoCo runner and task success logic.
- `sim/build_jetarm_scene_from_media.py` - JetArm wrapper around the shared media-to-scene builder.
- `sim/build_fanuc_scene_from_media.py` - shared media/photo/video scene builder.
- `sim/run_dual_fanuc_buffing_demo.py` - dual FANUC CR-7iA/L metal buffing workcell demo.
- `sim/extract_video_frames.py` - video window extraction and clean-frame selection.
- `sim/render_jetarm_review_sheet.py` - creates a labeled review PNG from a sim timeline.
- `sim/VIDEO_TO_SIM_PIPELINE.md` - detailed video-to-sim process.
- `sim/video_specs/jetarm_video_task_spec_template.json` - JetArm video scene template.
- `sim/video_specs/jetarm_tissue_pull_spec_template.json` - JetArm tissue-from-box video scene template.
- `tools/video_to_sim_app/server.py` - local upload/annotation/build web app.
- `tools/video_recorder/index.html` - local Chrome recorder.
- `inputs/video_plans/cube_to_square_window_plan_template.json` - starter video window plan.

## Documentation

Start here:

- `docs/PROJECT_HANDOFF.md` - what has been built so far and how to use it.
- `docs/AI_ASSISTED_VIDEO_TO_SIM_PLAN.md` - plan for reducing manual annotation with AI-assisted drafts.
- `sim/VIDEO_TO_SIM_PIPELINE.md` - detailed video-to-sim recording and processing pipeline.
- `sim/README.md` - simulation-specific notes.
- `sim/MEDIA_TO_FANUC_SIM.md` - shared media-to-sim notes and FANUC path.

## Notes

Generated videos, frames, scene outputs, and review PNGs are ignored by Git. They are meant to be regenerated locally from the scripts and specs.

The current video-derived simulations are first-pass approximations. Before any real robot execution, calibrate the camera/table/robot base carefully and verify reachability, gripper clearance, servo directions, speed limits, and emergency-stop behavior.
