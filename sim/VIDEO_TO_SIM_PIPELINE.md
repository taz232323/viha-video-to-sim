# Video-To-Simulation Pipeline

This note is for turning a recorded robot/table video into a MuJoCo scene that the JetArm or FANUC demo runner can test. The first useful version should stay simple: use user-selected video windows, extract candidate frames, let a key-frame tool choose the cleanest frames, then build the sim from the best calibrated frame.

## Goal

```text
recorded video
  -> user-marked task windows
  -> candidate frame extraction
  -> clean/key-frame selection
  -> calibration + object annotations
  -> generated MuJoCo scene
  -> simulated pick/place rollout
  -> success result + waypoint draft
```

This should not try to infer everything from a raw video at first. The reliable path is to let the user identify the important time ranges, then automate the frame cleanup and scene generation around those ranges.

## Saved For Later: ABC

`amazon-far/abc` is a future behavior-cloning/training reference, not the first frame-extraction tool. Keep it in mind for packaging repeated successful robot demos into camera-video plus state/action episodes after the video-to-sim pipeline is working.

ABC-style later-stage use:

```text
many successful video-to-sim tasks
  -> calibrated sim/hardware trajectories
  -> episode dataset
  -> behavior cloning or policy training
```

## Build Process

### 1. Accept A Video And A Window Plan

The user provides one video and a small JSON plan that names the task-relevant windows.

Examples of useful windows:

- `scene_reference`: full table, robot base, orange, target bowl, and calibration markers visible.
- `pre_pick`: orange visible before the robot touches it.
- `pre_place`: target bowl visible before release.
- `release`: object above the target bowl.
- `post_place`: object resting in the target bowl.

The windows are time ranges in seconds. This keeps the system from picking a nice-looking frame that is useless for calibration.

### 2. Extract Candidate Frames From Each Window

Use FFmpeg, ffmpeg-python, OpenCV, or Decord to extract frames inside each window.

Suggested first settings:

- Extract at `3-5 fps` inside each selected window.
- Save full-resolution frames first.
- Also save a smaller preview copy for fast review.
- Keep timestamp and frame number in the filename.

Example output:

```text
outputs/video_frames/session_001/candidates/
  scene_reference_t000.50_f0015.jpg
  scene_reference_t000.75_f0022.jpg
  pre_pick_t003.20_f0096.jpg
  release_t008.40_f0252.jpg
```

### 3. Pick Clean Frames With Katna

Run Katna or a similar key-frame selector separately for each user-defined window. The goal is not just a summary of the whole video; the goal is the cleanest frame in each task window.

Selection criteria:

- Low motion blur.
- Robot/table/object visible.
- Minimal gripper occlusion over the object center.
- Calibration markers visible.
- Target bowl rim visible.

Expected output:

```text
outputs/video_frames/session_001/selected/
  scene_reference_best_001.jpg
  pre_pick_best_001.jpg
  pre_place_best_001.jpg
  release_best_001.jpg
  post_place_best_001.jpg
```

### 4. Choose The Primary Calibration Frame

Usually this should be `scene_reference_best_001.jpg`.

It must show:

- The tabletop or calibration mat.
- Four table corners, ArUco/ChArUco markers, or a printed grid.
- JetArm/FANUC base position.
- Object and target positions, if possible.

If the scene reference frame does not show the object well, use the reference frame only for calibration and use the `pre_pick` or `pre_place` frames for object center annotations.

### 5. Annotate Calibration And Objects

Minimum annotations:

- Table corner pixels, or marker detections.
- Real table/workspace dimensions.
- Robot base position in table/world coordinates.
- Pick object center pixel.
- Place target center pixel.
- Approximate object radius/height.
- Approximate target bowl radius/rim height.

The first version can be manual click-to-annotate. Later, we can add:

- OpenCV ArUco/ChArUco marker detection.
- SAM 2 object masks.
- YOLO tracking.
- CVAT-assisted manual correction.

### 6. Convert Pixels To World Coordinates

Use the current tabletop homography approach:

```text
image pixel coordinate
  -> table plane coordinate in meters
  -> MuJoCo body position
```

This works best when the task happens on one flat table plane. For anything requiring vertical shape reconstruction, we would later add depth, multi-camera, or COLMAP-style 3D reconstruction.

### 7. Generate The MuJoCo Scene

For JetArm:

```bash
.venv/bin/python sim/build_jetarm_scene_from_media.py \
  --spec sim/video_specs/jetarm_video_task_spec_template.json \
  --run-headless
```

For FANUC:

```bash
.venv/bin/python sim/build_fanuc_scene_from_media.py \
  --spec sim/video_specs/fanuc_video_task_spec_template.json \
  --run-headless
```

The builder writes:

```text
outputs/generated/<name>_scene.xml
outputs/generated/<name>_scene_metadata.json
outputs/generated/<name>_scene_snapshot.png
outputs/generated/<name>_scene_result.json
```

### 8. Validate The Simulation

Check:

- Object starts where the video shows it.
- Target bowl is hollow and correctly placed.
- Robot can reach pick, transfer, release, and retreat.
- Gripper releases above the target bowl, not inside it.
- `SUCCESS=True` only after the object is on/in the target.
- Result JSON records failure reasons if something is unreachable.

### 9. Export A Human-Reviewable Plan

Before trying hardware, export:

- Selected source frame.
- Annotated frame overlay.
- Generated sim snapshot.
- Cartesian staged plan.
- Sim joint waypoints.
- Success/failure JSON.

This lets a human compare video -> annotations -> sim before a real robot moves.

## Recording Requirements

### Camera

- Use 1080p minimum; 4K is better if available.
- Record at 30 fps or 60 fps.
- Lock focus and exposure if the camera app allows it.
- Avoid digital zoom.
- Keep the camera still; use a tripod or fixed mount.

### Viewpoint

- Capture the entire table workspace.
- The robot base, object, and target bowl should all appear in the same reference frame.
- Use a slightly overhead angle, about 30-60 degrees down toward the table.
- Avoid filming perfectly side-on because table-plane calibration becomes weaker.
- Avoid extreme wide-angle distortion if possible.

### Calibration

At least one of these is required:

- Four visible table/workspace corners with known real dimensions.
- A printed ArUco or ChArUco board taped flat to the table.
- A printed grid/calibration mat with known square size.
- Known measured positions of the robot base, object, and target on the table.

Recommended for JetArm:

- Put a flat grid or marker board on the table.
- Measure where the JetArm base sits relative to the grid origin.
- Measure the orange diameter and target bowl rim diameter.
- Keep the calibration marker visible during the `scene_reference` window.

### Object Visibility

- The object should not be hidden by the gripper in the reference/pre-pick frame.
- The target bowl rim should be visible.
- Use objects with clear color/shape contrast from the table.
- Avoid reflective, transparent, or motion-blurred objects for early tests.

### Lighting

- Use even lighting.
- Avoid strong shadows over the object or bowl.
- Avoid glare on the table or bowl.
- Do not let the robot cast a large shadow over calibration markers.

### Motion

- Move slowly enough that frames are not blurred.
- Pause briefly before pickup and before release.
- Pause briefly after the object is placed.
- Keep hands and unrelated objects out of the scene during calibration frames.

## User Workflow Down The Road

1. Record a video following the requirements above.
2. Open the video in the future video-to-sim tool.
3. Mark time windows:
   - reference
   - pre-pick
   - pre-place
   - release
   - post-place
4. Let the tool extract candidate frames.
5. Let Katna select the cleanest frame from each window.
6. Review the selected frames.
7. Click calibration points if markers are not detected automatically.
8. Click or confirm the object and target positions.
9. Generate the MuJoCo scene.
10. Run the simulation.
11. Review success, clearance, and reachability.
12. Export a waypoint/plan file only after the sim looks physically reasonable.

## First Implementation Milestone

Build a script that does this:

```text
input video + window JSON
  -> extract candidate frames
  -> write selected-frame manifest
  -> copy best frames into outputs/video_frames/<session>/selected
```

After that works, connect the best `scene_reference` frame to the existing media scene builder.

## Current Code Setup

The first code pass includes:

- `tools/video_recorder/index.html` - local Chrome recorder page.
- `tools/video_recorder/README.md` - recorder launch instructions.
- `tools/video_to_sim_app/` - local upload, annotation, sim-build, and review UI.
- `inputs/video_plans/cube_to_square_window_plan_template.json` - starter window plan.
- `sim/extract_video_frames.py` - extracts candidate frames and picks clean frames by sharpness/contrast.
- `sim/video_specs/jetarm_video_task_spec_template.json` - JetArm video-to-scene template.

### Record Locally Through Chrome

From the repo root:

```bash
python3 -m http.server 8765 --directory tools/video_recorder
```

Open this in Chrome:

```text
http://localhost:8765
```

Download the recording into:

```text
inputs/videos/
```

Chrome usually saves `.webm`, which is fine for FFmpeg.

### Install FFmpeg

The extraction script expects the `ffmpeg` command to be available.

On macOS with Homebrew:

```bash
brew install ffmpeg
```

### Extract And Select Frames

After recording `inputs/videos/cube_to_square_test_001.webm`, run:

```bash
.venv/bin/python sim/extract_video_frames.py \
  --plan inputs/video_plans/cube_to_square_window_plan_template.json \
  --force
```

This writes:

```text
outputs/video_frames/cube_to_square_test_001/candidates/
outputs/video_frames/cube_to_square_test_001/selected/
outputs/video_frames/cube_to_square_test_001/selected_frames.json
outputs/video_frames/cube_to_square_test_001/jetarm_video_task_spec.json
```

The selector currently uses sharpness/contrast scoring. Katna can be added as a second selector once the basic video path works.

### Use The Upload UI

The easiest path is now:

```bash
.venv/bin/python tools/video_to_sim_app/server.py
```

Open:

```text
http://127.0.0.1:8777
```

The UI handles:

```text
video upload
  -> frame extraction
  -> selected-frame preview
  -> point annotation
  -> spec generation
  -> MuJoCo run
  -> review PNG
```

### Robot In Frame

The Hiwonder robot can be in frame. This is helpful if the base position is visible. The important rule is that the robot must not block the calibration marker, table corners, object center, or target center in the frames used for calibration and annotation.
