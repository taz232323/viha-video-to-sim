# ViHa Project Handoff

This document summarizes the current state of the ViHa robot simulation/video-to-sim prototype.

## Project Goal

Build a practical workflow where a person can record a tabletop task, extract the useful visual information, create a MuJoCo simulation, and verify whether a robot can perform the same task.

The near-term target is the Hiwonder JetArm. The same media pipeline is designed to work for FANUC later by swapping the robot scene, calibration frame, and hardware execution backend.

## Work Completed So Far

### 1. MuJoCo Simulation Setup

The project uses MuJoCo to simulate tabletop pick-and-place tasks.

Current simulation capabilities:

- JetArm-style robot scene.
- FANUC-style scene support.
- Table/workspace geometry.
- Objects and targets generated from specs.
- Scripted pick, carry, place, release, and retreat stages.
- Headless success checks.
- Interactive local viewer.
- Review PNG rendering.

Important files:

- `sim/worlds/table_bowl_pick_place_jetarm.xml`
- `sim/run_jetarm_pick_place_demo.py`
- `sim/run_pick_place_demo.py`
- `sim/render_jetarm_review_sheet.py`

### 2. JetArm Model Improvements

The JetArm visual model was iterated to more closely match the real Hiwonder hardware:

- Green/black arm style.
- Base deck and electronics box.
- Display/screen-like mount.
- Cables and table details.
- More realistic gripper with black actuator block, linkage arms, visible pivots, and rubber tips.

The gripper was tuned around the orange/cube-style task geometry so it can close around the object and release above the target instead of getting stuck inside the bowl/target.

### 3. Orange-To-Bowl Simulation

Earlier task:

```text
JetArm picks up an orange and places it into a target bowl.
```

Improvements made:

- Orange is modeled as a sphere.
- Target bowl is open/hollow rather than visually filled.
- Removed unwanted black wall/success panel from the minimal generated scene.
- Gripper faces outward and is closer to the reference photos.
- Added success predicate so the simulation knows when the task is complete.
- Added fixed-wrist release behavior so the object releases above the bowl.

Known note:

The orange/bowl task is useful for proving pick/place and target containment, but it has extra complexity because bowl rim clearance matters.

### 4. Human Video-To-Robot Simulation Direction

The project direction shifted from “robot video” to “human demonstration video.”

New intended flow:

```text
human demonstrates a task on video
  -> system extracts task start/goal/environment
  -> robot simulation attempts the same task
```

This does not copy human hand motion directly. It extracts:

- Object identity/location.
- Start pose.
- Goal/target pose.
- Table and obstacles.
- Robot base/workspace calibration.

Then the robot solves the task in its own reachable way.

### 5. Simple Video Test Task

The first video test task is:

```text
move a small cube/object onto a red marked square
```

This was chosen because it is simpler than orange-to-bowl:

- No target depth/rim.
- Easy visual success condition.
- Easier frame annotation.
- Good for validating the video-to-sim pipeline.

### 6. Local Chrome Recorder

A browser-based local recorder was added:

- `tools/video_recorder/index.html`
- `tools/video_recorder/README.md`

Run:

```bash
python3 -m http.server 8765 --directory tools/video_recorder
```

Open:

```text
http://localhost:8765
```

The recorder:

- Uses Chrome camera access.
- Records locally with browser `MediaRecorder`.
- Downloads `.webm`.
- Does not upload the video anywhere.

Recordings should be saved under:

```text
inputs/videos/
```

### 7. Video Frame Extraction

The video processing script is:

- `sim/extract_video_frames.py`

It takes a window plan JSON and:

- Extracts candidate frames from each time window using FFmpeg.
- Scores frames by sharpness/contrast.
- Copies selected frames into an output folder.
- Writes `selected_frames.json`.
- Optionally writes a downstream video task spec.

Starter plan:

- `inputs/video_plans/cube_to_square_window_plan_template.json`

Run:

```bash
.venv/bin/python sim/extract_video_frames.py \
  --plan inputs/video_plans/cube_to_square_window_plan_template.json \
  --force
```

FFmpeg is required:

```bash
brew install ffmpeg
```

### 8. Video-To-Sim Spec

The JetArm video spec template is:

- `sim/video_specs/jetarm_video_task_spec_template.json`

The spec stores:

- Video path.
- Representative selected frame.
- Selected frame manifest.
- Time windows.
- Calibration points.
- Robot base location.
- Object center pixel.
- Target center pixel.
- Object shape options.
- Output paths.

The same shared media builder supports both photo and video-style specs.

### 9. Cube-To-Square First Pass

A first real video was processed:

```text
cube_to_square_test_001.webm
```

The pipeline successfully:

- Found/copy video.
- Installed FFmpeg.
- Extracted selected frames.
- Generated a selected-frame contact sheet.
- Created an annotated video source frame.
- Generated a JetArm video task spec.
- Built a MuJoCo cube-to-square scene.
- Ran the headless simulation.
- Returned `SUCCESS=True`.
- Launched the local viewer.
- Rendered a review PNG sheet.

The generated files are ignored by Git because they are local artifacts.

Important note:

The final post-place video window had a human hand still over the object. The sim could still be built as a first pass, but future recordings should include a clean 1-2 second pause after placement with the hand fully out of frame.

### 10. Review PNG Workflow

Instead of requiring the user to replay terminal commands or rerun the viewer, review PNGs are now generated with:

- `sim/render_jetarm_review_sheet.py`

Example:

```bash
.venv/bin/python sim/render_jetarm_review_sheet.py \
  --scene outputs/generated/cube_to_square_video_scene.xml \
  --output outputs/generated/cube_to_square_video_review_sheet.png
```

The review sheet shows:

- Start.
- Approach pick.
- Carry.
- Release.
- Success.

Going forward, each new video-derived sim should produce one review PNG.

### 11. FANUC Compatibility

The same video-to-sim perception process should work for FANUC:

```text
video
  -> selected frames
  -> calibration
  -> generated scene
  -> robot-specific runner/control backend
```

For FANUC, the changes are:

- Use FANUC scene/model.
- Use FANUC base-frame calibration.
- Use FANUC-compatible tool center point and safety settings.
- Eventually export FANUC-compatible waypoint/program instructions.

Relevant files:

- `sim/build_fanuc_scene_from_media.py`
- `sim/video_specs/fanuc_video_task_spec_template.json`
- `sim/MEDIA_TO_FANUC_SIM.md`

### 12. Saved For Later: ABC

`amazon-far/abc` was reviewed and saved as a future option.

Conclusion:

- Not the right first tool for frame extraction.
- Potentially useful later for behavior cloning, policy training, dataset formatting, and repeated successful robot demonstrations.

## Recording Requirements

For best results:

- Use 1080p minimum; 4K is better.
- Use 30 or 60 fps.
- Keep camera fixed.
- Use a slightly overhead view.
- Keep the full table/task area visible.
- Include robot base, object, target, and calibration marker/grid.
- Pause before touching the object.
- Pause after placing the object with hands out of frame.
- Avoid glare, shadows, and heavy motion blur.

The Hiwonder robot can be visible. This is useful, but it should not block:

- Calibration marker/table corners.
- Object center.
- Target center.
- Final object placement.

## Current Limitations

- Frame selection is currently sharpness/contrast based, not Katna yet.
- Object/target annotation is still approximate/manual.
- Homography assumes the task happens on a flat table plane.
- Generated simulation is not hardware-safe without real calibration.
- No automatic ArUco/ChArUco detection yet.
- No SAM/YOLO/CVAT object automation yet.
- No direct FANUC hardware export yet.
- No real JetArm motion execution from video-derived tasks yet.

## Recommended Next Steps

1. Add a click-to-annotate UI for table corners, object center, and target center.
2. Add ArUco/ChArUco marker detection with OpenCV.
3. Integrate Katna as an optional clean/key-frame selector.
4. Improve review PNG camera framing.
5. Add failure labels for unreachable targets and collision risks.
6. Add a one-command pipeline:

```text
video + window plan
  -> selected frames
  -> annotated spec
  -> sim
  -> review PNG
```

7. Re-record cube-to-square with a clean post-place pause.
8. Add FANUC-specific generated-scene examples.

## Shareable Repo Contents

This GitHub repo should include:

- Source code.
- MJCF scenes.
- Specs/templates.
- Recorder page.
- Documentation.
- Input plan templates.

It should not include by default:

- Raw videos.
- Generated frame folders.
- Generated MuJoCo output scenes/results.
- Local notebooks/work artifacts.

Those files can be regenerated locally.
