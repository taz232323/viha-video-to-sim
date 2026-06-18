# AI-Assisted Video-To-Sim Plan

This plan describes how to reduce manual input in the ViHa video-to-simulation workflow. The goal is not to remove human review immediately. The goal is to have AI do the same rough work an operator is doing now: select frames, identify the table, find the object/target, infer the task, draft a scene spec, run the sim, and show a review artifact.

## Current Manual Workflow

```text
upload video
  -> extract frames
  -> user chooses/accepts representative frame
  -> user clicks table corners
  -> user clicks pick object
  -> user clicks target
  -> user chooses object/target sizes
  -> generated video spec
  -> MuJoCo scene
  -> headless sim
  -> review PNG/live viewer
```

The weak point is manual clicking. It works, but it is easy to click corners in the wrong order, choose the wrong frame, or use measurements that do not match the recording.

## Target AI-Assisted Workflow

```text
upload video
  -> AI selects useful frames
  -> AI detects calibration/table/workspace
  -> AI detects object and target
  -> AI infers task type
  -> AI drafts video spec
  -> sim builds/runs automatically
  -> user reviews overlay + review PNG
  -> user approves or corrects points
```

The user should mostly answer:

```text
Does this overlay look right?
Did the sim do the right task?
```

## Recommended Architecture

### 1. Keep The Manual UI As The Fallback

The current UI should stay. AI should fill the UI fields and points automatically, but the user should still be able to correct them.

This avoids a fragile black-box system.

### 2. Add An AI Draft Layer

Add a new backend step:

```text
/api/ai-draft
```

Input:

- Uploaded video path.
- Selected frames manifest.
- Task preset hint, if available.
- Optional text prompt from user.

Output:

```json
{
  "task_type": "cube_to_square",
  "confidence": 0.84,
  "points": {
    "front_left": [175, 1042],
    "front_right": [1605, 968],
    "back_right": [1515, 430],
    "back_left": [520, 415],
    "pick_object": [1325, 610],
    "target": [1398, 805]
  },
  "objects": [
    {
      "id": "cube_0",
      "label": "small white cube",
      "center_pixel": [1325, 610],
      "estimated_radius_m": 0.022,
      "confidence": 0.78
    }
  ],
  "target": {
    "id": "target_square_0",
    "label": "red target square",
    "center_pixel": [1398, 805],
    "estimated_radius_m": 0.055,
    "confidence": 0.91
  },
  "warnings": [
    "Post-place frame has hand occlusion."
  ]
}
```

The UI can load this draft into the existing annotation canvas.

### 3. Use Deterministic Calibration First

For real-world measurements, the best low-maintenance approach is not pure AI. Use visible markers when possible:

- ChArUco board.
- ArUco markers.
- Printed grid with known square size.
- Known table corners.

If markers are visible, OpenCV should detect them and produce table/world coordinates. AI should only fall back to estimating corners when marker detection fails.

### 4. Use AI For Objects, Targets, And Task Meaning

AI is more useful for:

- "Find the cube."
- "Find the red square."
- "Find the orange."
- "Find the bowl."
- "Which frame is before pickup?"
- "Which frame is after placement?"
- "What task did the human demonstrate?"

This is where an object detector, segmentation model, or vision-language model can reduce manual work.

## Model/Tool Options

### Frame And Window Selection

Use:

- Current FFmpeg frame extraction.
- Current sharpness/contrast selector.
- PySceneDetect or similar scene/window detection later.

The user can still provide rough windows, but AI should eventually suggest them automatically:

```text
scene_reference
pre_pick
pre_place
post_place
```

### Object Detection

Useful open-source options:

- Grounding DINO / Grounded SAM style pipeline for text-prompted object detection.
- YOLO-style detector if we train on our repeated objects.
- Vision-language model for high-level task interpretation.

Prompts can be simple:

```text
"small cube"
"red target square"
"orange"
"bowl"
"robot base"
"calibration board"
```

### Segmentation And Tracking

Useful options:

- SAM 2 for image/video segmentation.
- Grounded SAM 2 for text-prompted object detection plus video tracking.

This would let the pipeline follow an object across frames after one detection.

### Calibration

Use OpenCV ArUco/ChArUco first. This should be preferred over AI for metric geometry.

Fallback order:

```text
ChArUco / ArUco marker detection
  -> printed grid detection
  -> table corner detection
  -> AI-proposed table corners with human confirmation
```

## Confidence-Gated Automation

The system should not silently trust AI. Use levels:

### Level 1: Assisted

AI proposes:

- Best frame.
- Object center.
- Target center.
- Table corners.

User confirms or corrects.

This is the next build target.

### Level 2: Supervised Auto-Build

AI proposes and builds automatically if confidence is high, then user reviews:

```text
confidence >= 0.85
  -> build sim
  -> show review PNG
  -> user approves or edits
```

### Level 3: Automatic Batch Mode

For repeated videos recorded with the same camera/marker layout:

```text
upload video
  -> auto-draft
  -> auto-build
  -> save result
```

Only use this after the marker calibration is reliable.

## Proposed UI Changes

Add a button:

```text
AI Draft Scene
```

Button behavior:

1. Reads selected frames.
2. Runs calibration detection.
3. Runs object/target detection.
4. Fills annotation points.
5. Draws overlay.
6. Shows confidence and warnings.

Add a panel:

```text
AI Draft
  Task: cube_to_square
  Object: small white cube, 0.78 confidence
  Target: red square, 0.91 confidence
  Calibration: marker not found; using table-corner estimate
  Warning: hand occludes post-place frame
```

Then user can click:

```text
Accept Draft And Build
```

or manually adjust points.

## Proposed Backend Files

Add:

```text
sim/ai_draft_video_spec.py
```

Responsibilities:

- Load selected frames manifest.
- Detect calibration markers if available.
- Call object/target detector.
- Produce draft JSON.
- Save overlay.

Add optional adapters:

```text
sim/vision_adapters/
  opencv_calibration.py
  grounding_dino_adapter.py
  sam2_adapter.py
  simple_color_detector.py
```

Start with `simple_color_detector.py` for red target squares because it is easy and local. Then add stronger models later.

## Practical First Version

Implement this before using large models:

1. Detect red target square by color threshold.
2. Detect blue calibration mat/table bounds by color/edge threshold.
3. Detect small bright/white object by region proposal on the mat.
4. Let user confirm points.

This gets a fast "AI-like" draft without expensive model setup.

Then add:

1. ChArUco marker detection.
2. Grounding DINO/SAM 2 for general object/target prompts.
3. Vision-language task description.

## Maintenance Benefits

This reduces maintenance because:

- The user no longer needs to remember click order every time.
- The UI can show draft confidence and warnings.
- Errors become visible overlays instead of hidden bad coordinates.
- The manual path still works if AI fails.
- Repeated recording setups can be reused with saved calibration.

## Safety Rules

AI-generated specs must be treated as drafts.

Before any real robot execution:

- Verify the overlay.
- Verify the sim review PNG.
- Verify gripper clearance.
- Verify robot reachability.
- Verify table/base calibration.
- Verify speed limits and emergency stop.

The AI can make scene generation easier, but it should not directly command hardware without a reviewed simulation and safety checks.

