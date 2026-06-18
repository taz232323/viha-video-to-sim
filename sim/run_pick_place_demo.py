from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


SCENE_PATH = Path(__file__).resolve().parent / "worlds" / "table_bowl_pick_place.xml"

ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")
ARM_ACTUATORS = (
    "j1_position",
    "j2_position",
    "j3_position",
    "j4_position",
    "j5_position",
    "j6_position",
)
GRIPPER_ACTUATORS = ("left_finger_position", "right_finger_position")

BOWL_START = np.array([-0.28, -0.03, 0.795])
PLATE_TARGET = np.array([0.28, 0.03, 0.805])
BOWL_GRASP_OFFSET = np.array([0.0, 0.0, -0.055])
ON_PLATE_XY_TOL = 0.08
ON_PLATE_Z_RANGE = (0.79, 0.88)

OPEN_GRIP = 0.0
CLOSED_GRIP = 0.035


@dataclass(frozen=True)
class DemoState:
    q: np.ndarray
    grip: float


@dataclass(frozen=True)
class Segment:
    duration: float
    start: DemoState
    end: DemoState
    carrying: bool
    placed: bool = False


@dataclass
class RobotHandles:
    arm_qpos: list[int]
    arm_dof: list[int]
    arm_ranges: np.ndarray
    arm_actuators: list[int]
    gripper_actuators: list[int]
    gripper_site: int
    bowl_qpos: int
    bowl_dof: int


@dataclass(frozen=True)
class RuntimeConfig:
    scene_path: Path
    bowl_start: np.ndarray
    plate_target: np.ndarray
    on_plate_xy_tol: float = ON_PLATE_XY_TOL
    on_plate_z_range: tuple[float, float] = ON_PLATE_Z_RANGE


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def mix_state(a: DemoState, b: DemoState, t: float) -> DemoState:
    s = smoothstep(t)
    return DemoState(q=a.q + (b.q - a.q) * s, grip=a.grip + (b.grip - a.grip) * s)


def named_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type.name}: {name}")
    return obj_id


def get_handles(model: mujoco.MjModel) -> RobotHandles:
    arm_joint_ids = [named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
    arm_actuator_ids = [named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_ACTUATORS]
    gripper_actuator_ids = [
        named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in GRIPPER_ACTUATORS
    ]
    bowl_joint_id = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "bowl_free")

    return RobotHandles(
        arm_qpos=[int(model.jnt_qposadr[joint_id]) for joint_id in arm_joint_ids],
        arm_dof=[int(model.jnt_dofadr[joint_id]) for joint_id in arm_joint_ids],
        arm_ranges=np.array([model.jnt_range[joint_id] for joint_id in arm_joint_ids]),
        arm_actuators=arm_actuator_ids,
        gripper_actuators=gripper_actuator_ids,
        gripper_site=named_id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site"),
        bowl_qpos=int(model.jnt_qposadr[bowl_joint_id]),
        bowl_dof=int(model.jnt_dofadr[bowl_joint_id]),
    )


def set_arm_state(model: mujoco.MjModel, data: mujoco.MjData, handles: RobotHandles, state: DemoState) -> None:
    for value, qpos_address, actuator_id in zip(state.q, handles.arm_qpos, handles.arm_actuators):
        data.qpos[qpos_address] = value
        data.ctrl[actuator_id] = value

    for actuator_id in handles.gripper_actuators:
        data.ctrl[actuator_id] = state.grip

    mujoco.mj_forward(model, data)


def set_bowl_pose(model: mujoco.MjModel, data: mujoco.MjData, handles: RobotHandles, pos: np.ndarray) -> None:
    data.qpos[handles.bowl_qpos : handles.bowl_qpos + 3] = pos
    data.qpos[handles.bowl_qpos + 3 : handles.bowl_qpos + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[handles.bowl_dof : handles.bowl_dof + 6] = 0.0
    mujoco.mj_forward(model, data)


def gripper_position(data: mujoco.MjData, handles: RobotHandles) -> np.ndarray:
    return data.site_xpos[handles.gripper_site].copy()


def scene_body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mujoco.mj_forward(model, data)
    return data.xpos[body_id].copy()


def runtime_config_from_scene(
    scene_path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
) -> RuntimeConfig:
    mujoco.mj_forward(model, data)
    bowl_start = data.qpos[handles.bowl_qpos : handles.bowl_qpos + 3].copy()
    plate_pos = scene_body_position(model, data, "plate")
    plate_target = np.array([plate_pos[0], plate_pos[1], PLATE_TARGET[2]], dtype=float)
    return RuntimeConfig(scene_path=scene_path, bowl_start=bowl_start, plate_target=plate_target)


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    target: np.ndarray,
    seed: np.ndarray,
    max_iters: int = 180,
) -> np.ndarray:
    q = seed.copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iters):
        for value, qpos_address in zip(q, handles.arm_qpos):
            data.qpos[qpos_address] = value
        mujoco.mj_forward(model, data)

        error = target - gripper_position(data, handles)
        if np.linalg.norm(error) < 0.006:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, handles.gripper_site)
        j = jacp[:, handles.arm_dof]
        damping = 1e-3
        dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(3), error)
        max_step = 0.06
        step_norm = np.linalg.norm(dq)
        if step_norm > max_step:
            dq *= max_step / step_norm

        q += dq
        q = np.clip(q, handles.arm_ranges[:, 0], handles.arm_ranges[:, 1])

    return q


def make_demo_segments(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    config: RuntimeConfig,
) -> list[Segment]:
    seed = np.array([0.15, 0.75, -1.05, 0.0, 0.65, 0.0])
    targets = {
        "home": np.array([-0.14, -0.18, 1.13]),
        "pick_hover": config.bowl_start + np.array([0.0, 0.0, 0.23]),
        "pick_low": config.bowl_start + np.array([0.0, 0.0, 0.10]),
        "transfer": np.array([0.0, 0.0, 1.12]),
        "place_hover": config.plate_target + np.array([0.0, 0.0, 0.23]),
        "place_low": config.plate_target + np.array([0.0, 0.0, 0.11]),
        "retreat": np.array([0.05, -0.22, 1.15]),
    }

    solved: dict[str, np.ndarray] = {}
    for name, target in targets.items():
        seed = solve_ik(model, data, handles, target, seed)
        solved[name] = seed.copy()

    home = DemoState(solved["home"], OPEN_GRIP)
    pick_hover = DemoState(solved["pick_hover"], OPEN_GRIP)
    pick_low_open = DemoState(solved["pick_low"], OPEN_GRIP)
    pick_low_closed = DemoState(solved["pick_low"], CLOSED_GRIP)
    pick_lifted = DemoState(solved["pick_hover"], CLOSED_GRIP)
    transfer = DemoState(solved["transfer"], CLOSED_GRIP)
    place_hover = DemoState(solved["place_hover"], CLOSED_GRIP)
    place_low_closed = DemoState(solved["place_low"], CLOSED_GRIP)
    place_low_open = DemoState(solved["place_low"], OPEN_GRIP)
    retreat = DemoState(solved["retreat"], OPEN_GRIP)

    return [
        Segment(1.0, home, pick_hover, carrying=False),
        Segment(0.9, pick_hover, pick_low_open, carrying=False),
        Segment(0.65, pick_low_open, pick_low_closed, carrying=False),
        Segment(0.9, pick_low_closed, pick_lifted, carrying=True),
        Segment(1.0, pick_lifted, transfer, carrying=True),
        Segment(1.0, transfer, place_hover, carrying=True),
        Segment(0.9, place_hover, place_low_closed, carrying=True),
        Segment(0.65, place_low_closed, place_low_open, carrying=False, placed=True),
        Segment(1.1, place_low_open, retreat, carrying=False, placed=True),
    ]


def timeline_state(segments: list[Segment], elapsed: float) -> tuple[DemoState, bool, bool]:
    total = sum(segment.duration for segment in segments)
    t = elapsed % total

    for segment in segments:
        if t <= segment.duration:
            state = mix_state(segment.start, segment.end, t / segment.duration)
            return state, segment.carrying, segment.placed
        t -= segment.duration

    last = segments[-1]
    return last.end, last.carrying, last.placed


def update_bowl_for_stage(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    config: RuntimeConfig,
    carrying: bool,
    placed: bool,
) -> None:
    if carrying:
        set_bowl_pose(model, data, handles, gripper_position(data, handles) + BOWL_GRASP_OFFSET)
    elif placed:
        set_bowl_pose(model, data, handles, config.plate_target)
    else:
        set_bowl_pose(model, data, handles, config.bowl_start)


def task_metrics(data: mujoco.MjData, handles: RobotHandles, config: RuntimeConfig, carrying: bool) -> dict[str, object]:
    bowl_pos = data.qpos[handles.bowl_qpos : handles.bowl_qpos + 3]
    xy_distance = float(np.linalg.norm(bowl_pos[:2] - config.plate_target[:2]))
    z_ok = config.on_plate_z_range[0] <= bowl_pos[2] <= config.on_plate_z_range[1]
    success = not carrying and xy_distance <= config.on_plate_xy_tol and z_ok
    return {
        "success": bool(success),
        "bowl_final_position_m": bowl_pos.round(6).tolist(),
        "plate_target_position_m": config.plate_target.round(6).tolist(),
        "xy_distance_to_target_m": xy_distance,
        "xy_tolerance_m": config.on_plate_xy_tol,
        "z_within_plate_range": bool(z_ok),
        "z_range_m": list(config.on_plate_z_range),
    }


def task_success(data: mujoco.MjData, handles: RobotHandles, config: RuntimeConfig, carrying: bool) -> bool:
    return bool(task_metrics(data, handles, config, carrying)["success"])


def save_snapshot(model: mujoco.MjModel, data: mujoco.MjData, output_path: Path) -> None:
    renderer = mujoco.Renderer(model, height=900, width=1400)
    renderer.update_scene(data, camera="overview")
    image = renderer.render()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    renderer.close()


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    segments: list[Segment],
    config: RuntimeConfig,
    output: Path,
    result_json: Path | None,
) -> None:
    end_time = sum(segment.duration for segment in segments) - 0.05
    steps = math.ceil(end_time / model.opt.timestep)
    for step in range(steps):
        state, carrying, placed = timeline_state(segments, step * model.opt.timestep)
        set_arm_state(model, data, handles, state)
        update_bowl_for_stage(model, data, handles, config, carrying, placed)
        mujoco.mj_step(model, data)

    save_snapshot(model, data, output)
    metrics = task_metrics(data, handles, config, carrying=False)
    print(f"Saved snapshot: {output}")
    print(f"Bowl final position: {metrics['bowl_final_position_m']}")
    print(f"Plate target position: {metrics['plate_target_position_m']}")
    print(f"XY distance to target: {metrics['xy_distance_to_target_m']:.4f} m")
    print(f"Z within plate range: {metrics['z_within_plate_range']}")
    print(f"TASK COMPLETE: SUCCESS={metrics['success']}")

    if result_json is not None:
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "scene": str(config.scene_path),
            "snapshot": str(output),
            "sim_time_s": round(end_time, 4),
            "steps": steps,
            **metrics,
        }
        result_json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved result: {result_json}")


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    segments: list[Segment],
    config: RuntimeConfig,
) -> None:
    start = time.time()
    announced_success = False
    overview_camera_id = named_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = overview_camera_id

        while viewer.is_running():
            elapsed = time.time() - start
            state, carrying, placed = timeline_state(segments, elapsed)
            set_arm_state(model, data, handles, state)
            update_bowl_for_stage(model, data, handles, config, carrying, placed)
            complete = task_success(data, handles, config, carrying)
            if complete and not announced_success:
                print("TASK COMPLETE: bowl_0 is on plate_0. SUCCESS=True")
                announced_success = True
            elif not complete and announced_success and not placed:
                announced_success = False

            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FANUC CR-7iA/L table pick-and-place demo.")
    parser.add_argument("--headless", action="store_true", help="Run without the viewer and save a snapshot.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=SCENE_PATH,
        help="MJCF scene path. Generated media scenes can be passed here.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "table_bowl_pick_place.png",
        help="Snapshot path for --headless.",
    )
    parser.add_argument("--result-json", type=Path, help="Optional machine-readable result path for --headless.")
    args = parser.parse_args()

    scene_path = args.scene.resolve()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    handles = get_handles(model)
    config = runtime_config_from_scene(scene_path, model, data, handles)
    segments = make_demo_segments(model, data, handles, config)

    if args.headless:
        run_headless(model, data, handles, segments, config, args.snapshot, args.result_json)
        return

    if sys.platform == "darwin" and "mjpython" not in Path(sys.executable).name:
        print("Tip: on macOS this interactive demo should be launched with `mjpython`.")
        print("Run: mjpython sim/run_pick_place_demo.py")

    run_viewer(model, data, handles, segments, config)


if __name__ == "__main__":
    main()
