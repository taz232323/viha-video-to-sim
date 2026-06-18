# ViHa Video-To-Sim Upload UI

This local web app turns the manual video workflow into a browser-guided process:

1. Upload a task video.
2. Extract/select useful frames.
3. Click calibration and task points.
4. Build a JetArm MuJoCo scene.
5. Run the sim, optionally open the live MuJoCo viewer, and create a review PNG.

Run from the repo root:

```bash
.venv/bin/python tools/video_to_sim_app/server.py
```

Open:

```text
http://127.0.0.1:8777
```

## Click Order

On the representative frame, click:

1. `front_left`
2. `front_right`
3. `back_right`
4. `back_left`
5. `pick_object`
6. `target`

The app writes generated files under `outputs/`, which is ignored by Git.

## Shortcuts

- `Use First-Test Cube Points` restores the approximate point calibration used for the original cube-to-square demo video.
- `Open live MuJoCo viewer after build` launches the actual interactive simulation window after the scene is generated.

## Notes

- FFmpeg must be installed for upload processing.
- The first preset is cube/object-to-red-square.
- The Hiwonder can be visible, but it should not block the calibration points, object center, or target center in the selected frame.
- If the robot/object layout looks wrong, recheck the clicked table corner order and the table/base measurements.
