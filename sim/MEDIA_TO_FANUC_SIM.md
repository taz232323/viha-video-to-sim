# Internal Media-To-FANUC Simulation Pipeline

This is an internal validation/training pipeline, not a user-facing app. The goal is to turn calibrated image or video observations into repeatable MuJoCo scenes, run the FANUC pick-and-place task, and save machine-readable results for later analysis.

## Current MVP

```text
annotated image/video frame
  -> tabletop homography
  -> object positions in table/world meters
  -> generated FANUC MJCF scene
  -> headless MuJoCo rollout
  -> snapshot + result JSON
```

The first version expects manual annotations in JSON: table corner pixels, object center pixels, table/world coordinates, and object roles. Automatic detection can be layered in later without changing the simulation contract.

## Files

- `sim/photo_specs/fanuc_photo_task_spec_template.json` - photo input template.
- `sim/photo_specs/fanuc_sample_photo_task_spec.json` - sample scenario matching the default demo.
- `sim/video_specs/fanuc_video_task_spec_template.json` - video/representative-frame template.
- `sim/build_fanuc_scene_from_media.py` - builds a generated MJCF scene and metadata.
- `sim/run_pick_place_demo.py` - runs the FANUC sim; now accepts generated scenes with `--scene`.

## Generate And Run A Scenario

```bash
.venv/bin/python sim/build_fanuc_scene_from_media.py \
  --spec sim/photo_specs/fanuc_sample_photo_task_spec.json \
  --run-headless
```

This writes:

```text
outputs/generated/fanuc_sample_photo_scene.xml
outputs/generated/fanuc_sample_photo_scene_metadata.json
outputs/generated/fanuc_sample_photo_scene_snapshot.png
outputs/generated/fanuc_sample_photo_scene_result.json
```

You can also run any generated scene directly:

```bash
.venv/bin/python sim/run_pick_place_demo.py \
  --headless \
  --scene outputs/generated/fanuc_sample_photo_scene.xml \
  --snapshot outputs/generated/fanuc_sample_photo_scene_snapshot.png \
  --result-json outputs/generated/fanuc_sample_photo_scene_result.json
```

## Result Contract

The result JSON is meant for training/validation logs:

```json
{
  "scene": "outputs/generated/fanuc_sample_photo_scene.xml",
  "snapshot": "outputs/generated/fanuc_sample_photo_scene_snapshot.png",
  "sim_time_s": 7.95,
  "steps": 3975,
  "success": true,
  "bowl_final_position_m": [0.279724, 0.029793, 0.80498],
  "plate_target_position_m": [0.279724, 0.029793, 0.805],
  "xy_distance_to_target_m": 0.0,
  "xy_tolerance_m": 0.08,
  "z_within_plate_range": true,
  "z_range_m": [0.79, 0.88]
}
```

## Next Refinements

- Add a click-to-annotate helper for table corners and object centers.
- Add OpenCV/ArUco or ChArUco calibration for automatic marker detection.
- Add video frame extraction and per-frame scene snapshots.
- Add obstacle generation from annotations.
- Add collision and reachability failure labels in the result JSON.
- Export waypoint/path attempts as part of each scenario record.

## JetArm Real-Photo Variant

The same media-to-scene builder is also used for the Hiwonder/JetArm prototype. The first real-photo sample uses `IMG_4990.jpeg` and the measured board scale:

```text
5 board units = 2 inches = 0.0508 m
1 board unit = 0.01016 m
```

Run it with:

```bash
.venv/bin/python sim/build_jetarm_scene_from_media.py \
  --spec sim/photo_specs/jetarm_real_photo_task_spec.json \
  --run-headless
```

This writes:

```text
outputs/generated/jetarm_real_photo_scene.xml
outputs/generated/jetarm_real_photo_scene_metadata.json
outputs/generated/jetarm_real_photo_scene_snapshot.png
outputs/generated/jetarm_real_photo_scene_result.json
```

The current `jetarm_real_photo_task_spec.json` annotations are approximate visual picks from the photo. Treat the result as a proof that the pipeline works, not as a calibrated hardware-safe estimate.
