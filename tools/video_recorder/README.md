# Video-To-Sim Recorder

This is a local Chrome recorder for collecting tabletop task videos. It does not upload video anywhere; Chrome records with `MediaRecorder` and downloads a `.webm` file.

Run it from the repo root:

```bash
python3 -m http.server 8765 --directory tools/video_recorder
```

Open:

```text
http://localhost:8765
```

Save the downloaded video under:

```text
/Users/amardesai/Documents/ViHa/inputs/videos/
```

The Hiwonder robot can be in frame. It helps if the robot base is visible, but the robot should not cover the calibration marker, object center, or target center during the reference/pre-pick frames.
