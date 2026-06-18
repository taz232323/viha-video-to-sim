from pathlib import Path

import mujoco
import numpy as np

from run_pick_place_demo import (
    ON_PLATE_XY_TOL,
    ON_PLATE_Z_RANGE,
    PLATE_TARGET,
    SCENE_PATH,
    get_handles,
    make_demo_segments,
    set_arm_state,
    task_success,
    timeline_state,
    update_bowl_for_stage,
)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    handles = get_handles(model)
    segments = make_demo_segments(model, data, handles)

    end_time = sum(segment.duration for segment in segments) - 0.05
    steps = int(end_time / model.opt.timestep)
    carrying = False
    for step in range(steps):
        state, carrying, placed = timeline_state(segments, step * model.opt.timestep)
        set_arm_state(model, data, handles, state)
        update_bowl_for_stage(model, data, handles, carrying, placed)
        mujoco.mj_step(model, data)

    bowl_pos = data.qpos[handles.bowl_qpos : handles.bowl_qpos + 3].copy()
    xy_distance = float(np.linalg.norm(bowl_pos[:2] - PLATE_TARGET[:2]))
    z_ok = ON_PLATE_Z_RANGE[0] <= bowl_pos[2] <= ON_PLATE_Z_RANGE[1]
    success = task_success(data, handles, carrying)

    print(f"Task spec: {Path('sim/tasks/bowl_to_plate_task_spec.json')}")
    print(f"Bowl final position: {bowl_pos.round(4).tolist()}")
    print(f"Plate target position: {PLATE_TARGET.round(4).tolist()}")
    print(f"XY distance: {xy_distance:.4f} m")
    print(f"XY tolerance: {ON_PLATE_XY_TOL:.4f} m")
    print(f"Z within On predicate range: {z_ok}")
    print(f"SUCCESS: {success}")
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
