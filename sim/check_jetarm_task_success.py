import mujoco
import numpy as np

from run_jetarm_pick_place_demo import (
    ON_PLATE_XY_TOL,
    ON_PLATE_Z_RANGE,
    PLATE_TARGET,
    SCENE_PATH,
    build_model,
    set_arm_state,
    task_success,
    timeline_state,
    update_bowl_for_stage,
)


def main() -> None:
    model, data, handles, config, segments = build_model(SCENE_PATH)
    end_time = sum(segment.duration for segment in segments) - 0.05
    steps = int(end_time / model.opt.timestep)
    carrying = False

    for step in range(steps):
        state, carrying, placed = timeline_state(segments, step * model.opt.timestep)
        set_arm_state(model, data, handles, state)
        update_bowl_for_stage(model, data, handles, config, carrying, placed)
        mujoco.mj_step(model, data)

    bowl_pos = data.qpos[handles.bowl_qpos : handles.bowl_qpos + 3].copy()
    xy_distance = float(np.linalg.norm(bowl_pos[:2] - PLATE_TARGET[:2]))
    z_ok = ON_PLATE_Z_RANGE[0] <= bowl_pos[2] <= ON_PLATE_Z_RANGE[1]
    success = task_success(data, handles, config, carrying)

    print("Task spec: sim/tasks/jetarm_bowl_to_plate_task_spec.json")
    print(f"Bowl final position: {bowl_pos.round(4).tolist()}")
    print(f"Target bowl position: {PLATE_TARGET.round(4).tolist()}")
    print(f"XY distance: {xy_distance:.4f} m")
    print(f"XY tolerance: {ON_PLATE_XY_TOL:.4f} m")
    print(f"Z within target-bowl predicate range: {z_ok}")
    print(f"SUCCESS: {success}")
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
