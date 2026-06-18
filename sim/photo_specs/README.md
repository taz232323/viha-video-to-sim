# Photo-To-JetArm Spec

Use this folder for the bridge between a picture and robot motion.

The picture alone is not enough to move the real Hiwonder arm safely. We also need a small amount of calibration:

- Four tabletop corner pixels, or a printed ArUco/checkerboard marker.
- The real table/work area dimensions in meters.
- The JetArm base position on that table.
- The object centers in the image, either selected manually or detected by vision.
- A conservative object size and grasp/place height.

Once those are known, the flow is:

```text
photo + calibration spec
  -> object positions in table/world coordinates
  -> JetArm MuJoCo scene
  -> simulated pick/place rollout
  -> waypoint JSON
  -> calibrated Hiwonder ROS/Python command script
```

Start from:

```bash
sim/photo_specs/jetarm_photo_task_spec_template.json
```

The coordinates in the template are examples. Replace them with measurements from the actual scene photo before generating movement commands.
