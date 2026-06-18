from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from run_pick_place_demo import (
    ARM_ACTUATORS,
    ARM_JOINTS,
    GRIPPER_ACTUATORS,
    DemoState,
    RobotHandles,
    Segment,
    get_handles,
    gripper_position,
    mix_state,
    named_id,
    scene_body_position,
    set_arm_state,
    set_bowl_pose,
    solve_ik,
    timeline_state,
)


SCENE_PATH = Path(__file__).resolve().parent / "worlds" / "table_bowl_pick_place_jetarm.xml"
BOWL_START = np.array([-0.13, -0.04, 0.79])
PLATE_TARGET = np.array([0.12, 0.04, 0.835])
BOWL_GRASP_OFFSET = np.array([0.0, 0.0, -0.045])
PLACE_RELEASE_OFFSET = np.array([0.0, 0.0, 0.12])
FIXED_WRIST_Q = np.array([0.0, 0.8, 0.0])
ON_PLATE_XY_TOL = 0.055
ON_PLATE_Z_RANGE = (0.82, 0.9)
OPEN_GRIP = 0.0
CLOSED_GRIP = 0.032
DEFAULT_CAMERA_DISTANCE = 2.65
CAMERA_PRESETS = {
    "overview": {
        "lookat": np.array([0.02, -0.02, 0.92]),
        "distance": 2.7,
        "azimuth": 145,
        "elevation": -26,
    },
    "table": {
        "lookat": np.array([0.05, 0.02, 0.82]),
        "distance": 1.45,
        "azimuth": 130,
        "elevation": -35,
    },
    "robot": {
        "lookat": np.array([-0.25, -0.14, 0.95]),
        "distance": 1.1,
        "azimuth": 165,
        "elevation": -18,
    },
    "status": {
        "lookat": np.array([0.42, 0.22, 0.9]),
        "distance": 1.05,
        "azimuth": 120,
        "elevation": -22,
    },
    "top": {
        "lookat": np.array([0.0, 0.0, 0.78]),
        "distance": 1.95,
        "azimuth": 90,
        "elevation": -78,
    },
}
STATUS_LIGHTS = {
    "pick": "status_pick_light",
    "carry": "status_carry_light",
    "place": "status_place_light",
    "success": "status_success_light",
}
STATUS_COLORS = {
    "off": np.array([0.12, 0.13, 0.13, 1.0]),
    "pick": np.array([1.0, 0.58, 0.08, 1.0]),
    "carry": np.array([0.12, 0.48, 1.0, 1.0]),
    "place": np.array([1.0, 0.86, 0.08, 1.0]),
    "success": np.array([0.08, 1.0, 0.28, 1.0]),
}
SUCCESS_SCREEN_COLORS = {
    "off": {
        "success_screen_back": np.array([0.08, 0.1, 0.09, 0.0]),
        "success_screen_top": np.array([0.08, 0.9, 0.28, 0.0]),
        "success_screen_bottom": np.array([0.08, 0.9, 0.28, 0.0]),
    },
    "on": {
        "success_screen_back": np.array([0.08, 0.1, 0.09, 1.0]),
        "success_screen_top": np.array([0.08, 0.9, 0.28, 1.0]),
        "success_screen_bottom": np.array([0.08, 0.9, 0.28, 1.0]),
    },
}

SERVO_ID_MAP = {
    "j1": 1,
    "j2": 2,
    "j3": 3,
    "j4": 4,
    "j5": 5,
    "j6": 6,
    "gripper": 10,
}
GRIPPER_HARDWARE_HINT = {
    "servo_id": 10,
    "open_pulse": 200,
    "closed_pulse": 620,
    "sim_open_slide_m": OPEN_GRIP,
    "sim_closed_slide_m": CLOSED_GRIP,
    "approx_open_inner_gap_m": 0.146,
    "approx_closed_inner_gap_m": 0.07,
}


@dataclass
class ZoomControl:
    distance: float
    running: bool = True


@dataclass
class ViewerControls:
    sim_time: float = 0.0
    paused: bool = False
    speed: float = 1.0
    camera_preset: str = "table"
    zoom_delta: float = 0.0
    restart_requested: bool = False
    camera_dirty: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    scene_path: Path
    bowl_start: np.ndarray
    plate_target: np.ndarray
    on_plate_xy_tol: float = ON_PLATE_XY_TOL
    on_plate_z_range: tuple[float, float] = ON_PLATE_Z_RANGE
    task_type: str = "pick_and_place"
    object_name: str = "bowl_0"
    target_name: str = "target_bowl_0"


def start_zoom_slider(initial_distance: float) -> ZoomControl | None:
    control = ZoomControl(distance=initial_distance)

    try:
        import tkinter as tk
    except Exception as exc:
        print(f"Zoom slider unavailable ({exc}). Use the MuJoCo mouse wheel to zoom.")
        return None

    def run_slider() -> None:
        root = tk.Tk()
        root.title("JetArm Zoom")
        root.geometry("380x100")

        slider = tk.Scale(
            root,
            from_=1.2,
            to=4.4,
            resolution=0.05,
            orient="horizontal",
            length=330,
            label="Zoom distance",
        )
        slider.set(initial_distance)
        slider.pack(padx=18, pady=12)

        def update_distance(value: str) -> None:
            control.distance = float(value)

        def close() -> None:
            control.running = False
            root.destroy()

        slider.configure(command=update_distance)
        root.protocol("WM_DELETE_WINDOW", close)
        root.mainloop()

    threading.Thread(target=run_slider, daemon=True).start()
    return control


def print_viewer_controls() -> None:
    print("Viewer controls:")
    print("  1 overview | 2 table | 3 robot | 4 status | 5 top")
    print("  Space pause/resume | R restart | [ slower | ] faster | -/+ zoom")
    print("  Mouse drag/scroll still works in the MuJoCo window.")


def apply_camera_preset(viewer: mujoco.viewer.Handle, controls: ViewerControls) -> None:
    preset = CAMERA_PRESETS[controls.camera_preset]
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = preset["lookat"]
    viewer.cam.distance = max(0.45, preset["distance"] + controls.zoom_delta)
    viewer.cam.azimuth = preset["azimuth"]
    viewer.cam.elevation = preset["elevation"]


def make_key_callback(controls: ViewerControls):
    def on_key(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key == " ":
            controls.paused = not controls.paused
            print(f"PLAYBACK: {'paused' if controls.paused else 'running'}")
        elif key == "r":
            controls.restart_requested = True
            print("PLAYBACK: restart")
        elif key in {"1", "2", "3", "4", "5"}:
            controls.camera_preset = ["overview", "table", "robot", "status", "top"][int(key) - 1]
            controls.camera_dirty = True
            print(f"CAMERA: {controls.camera_preset}")
        elif key in {"-", "_"}:
            controls.zoom_delta = min(2.0, controls.zoom_delta + 0.15)
            controls.camera_dirty = True
            print(f"ZOOM: out ({controls.zoom_delta:+.2f})")
        elif key in {"=", "+"}:
            controls.zoom_delta = max(-0.75, controls.zoom_delta - 0.15)
            controls.camera_dirty = True
            print(f"ZOOM: in ({controls.zoom_delta:+.2f})")
        elif key == "[":
            controls.speed = max(0.25, controls.speed * 0.5)
            print(f"SPEED: {controls.speed:.2f}x")
        elif key == "]":
            controls.speed = min(4.0, controls.speed * 2.0)
            print(f"SPEED: {controls.speed:.2f}x")

    return on_key


def runtime_config_from_scene(
    scene_path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
) -> RuntimeConfig:
    mujoco.mj_forward(model, data)
    bowl_start = data.qpos[handles.bowl_qpos : handles.bowl_qpos + 3].copy()
    plate_pos = scene_body_position(model, data, "plate")
    tissue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tissue_sheet_collision")
    if tissue_id >= 0:
        return RuntimeConfig(
            scene_path=scene_path,
            bowl_start=bowl_start,
            plate_target=np.array([plate_pos[0], plate_pos[1], plate_pos[2]], dtype=float),
            on_plate_xy_tol=0.075,
            on_plate_z_range=(float(plate_pos[2]) - 0.055, float(plate_pos[2]) + 0.055),
            task_type="tissue_pull",
            object_name="tissue_0",
            target_name="pull_clear_target_0",
        )

    flat_target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_square_marker")
    if flat_target_id >= 0:
        target_z = float(bowl_start[2])
        z_range = (target_z - 0.03, target_z + 0.04)
    else:
        target_z = float(PLATE_TARGET[2])
        z_range = ON_PLATE_Z_RANGE
    plate_target = np.array([plate_pos[0], plate_pos[1], target_z], dtype=float)
    return RuntimeConfig(
        scene_path=scene_path,
        bowl_start=bowl_start,
        plate_target=plate_target,
        on_plate_z_range=z_range,
    )


def grasp_offset(config: RuntimeConfig) -> np.ndarray:
    if config.task_type == "tissue_pull":
        return np.array([0.0, 0.0, -0.018])
    return BOWL_GRASP_OFFSET


def solve_ik_locked_wrist(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    target: np.ndarray,
    seed: np.ndarray,
    max_iters: int = 240,
) -> np.ndarray:
    q = seed.copy()
    q[3:] = FIXED_WRIST_Q
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    active_arm_dof = handles.arm_dof[:3]

    for _ in range(max_iters):
        for value, qpos_address in zip(q, handles.arm_qpos):
            data.qpos[qpos_address] = value
        mujoco.mj_forward(model, data)

        error = target - gripper_position(data, handles)
        if np.linalg.norm(error) < 0.008:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, handles.gripper_site)
        j = jacp[:, active_arm_dof]
        damping = 1e-3
        dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(3), error)
        step_norm = np.linalg.norm(dq)
        if step_norm > 0.045:
            dq *= 0.045 / step_norm

        q[:3] += dq
        q = np.clip(q, handles.arm_ranges[:, 0], handles.arm_ranges[:, 1])
        q[3:] = FIXED_WRIST_Q

    return q


def make_demo_segments(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    config: RuntimeConfig,
) -> list[Segment]:
    if config.task_type == "tissue_pull":
        return make_tissue_pull_segments(model, data, handles, config)

    seed = np.array([0.2, 0.55, -0.8, *FIXED_WRIST_Q])
    safe_z = max(float(config.bowl_start[2]), float(config.plate_target[2])) + 0.18
    mid_xy = 0.5 * (config.bowl_start[:2] + config.plate_target[:2])
    targets = {
        "home": np.array([0.0, -0.08, safe_z + 0.05]),
        "pick_hover": np.array([config.bowl_start[0], config.bowl_start[1], safe_z]),
        "pick_low": config.bowl_start - grasp_offset(config) + np.array([0.0, 0.0, 0.004]),
        "transfer": np.array([mid_xy[0], mid_xy[1], safe_z + 0.02]),
        "place_hover": np.array([config.plate_target[0], config.plate_target[1], safe_z]),
        "place_release": config.plate_target + PLACE_RELEASE_OFFSET,
        "retreat": np.array([config.plate_target[0], config.plate_target[1], safe_z + 0.04]),
    }

    solved: dict[str, np.ndarray] = {}
    for name, target in targets.items():
        seed = solve_ik_locked_wrist(model, data, handles, target, seed, max_iters=260)
        solved[name] = seed.copy()

    return [
        Segment(0.85, DemoState(solved["home"], OPEN_GRIP), DemoState(solved["pick_hover"], OPEN_GRIP), carrying=False),
        Segment(0.65, DemoState(solved["pick_hover"], OPEN_GRIP), DemoState(solved["pick_low"], OPEN_GRIP), carrying=False),
        Segment(0.45, DemoState(solved["pick_low"], OPEN_GRIP), DemoState(solved["pick_low"], CLOSED_GRIP), carrying=False),
        Segment(0.65, DemoState(solved["pick_low"], CLOSED_GRIP), DemoState(solved["pick_hover"], CLOSED_GRIP), carrying=True),
        Segment(0.75, DemoState(solved["pick_hover"], CLOSED_GRIP), DemoState(solved["transfer"], CLOSED_GRIP), carrying=True),
        Segment(0.75, DemoState(solved["transfer"], CLOSED_GRIP), DemoState(solved["place_hover"], CLOSED_GRIP), carrying=True),
        Segment(0.65, DemoState(solved["place_hover"], CLOSED_GRIP), DemoState(solved["place_release"], CLOSED_GRIP), carrying=True),
        Segment(0.45, DemoState(solved["place_release"], CLOSED_GRIP), DemoState(solved["place_release"], OPEN_GRIP), carrying=True),
        Segment(0.25, DemoState(solved["place_release"], OPEN_GRIP), DemoState(solved["place_release"], OPEN_GRIP), carrying=False, placed=True),
        Segment(0.65, DemoState(solved["place_release"], OPEN_GRIP), DemoState(solved["retreat"], OPEN_GRIP), carrying=False, placed=True),
    ]


def make_tissue_pull_segments(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    config: RuntimeConfig,
) -> list[Segment]:
    seed = np.array([0.2, 0.55, -0.8, *FIXED_WRIST_Q])
    offset = grasp_offset(config)
    safe_z = max(float(config.bowl_start[2]), float(config.plate_target[2])) + 0.12
    pull_mid = 0.45 * config.bowl_start + 0.55 * config.plate_target + np.array([0.0, 0.0, 0.08])
    targets = {
        "home": np.array([-0.02, -0.12, safe_z + 0.05]),
        "pick_hover": np.array([config.bowl_start[0], config.bowl_start[1], safe_z]),
        "pick_low": config.bowl_start - offset + np.array([0.0, 0.0, 0.004]),
        "pull_up": config.bowl_start - offset + np.array([0.0, 0.0, 0.12]),
        "pull_mid": pull_mid - offset,
        "pull_clear": config.plate_target - offset,
        "retreat": config.plate_target - offset + np.array([-0.06, -0.03, 0.055]),
    }

    solved: dict[str, np.ndarray] = {}
    for name, target in targets.items():
        seed = solve_ik_locked_wrist(model, data, handles, target, seed, max_iters=300)
        solved[name] = seed.copy()

    return [
        Segment(0.85, DemoState(solved["home"], OPEN_GRIP), DemoState(solved["pick_hover"], OPEN_GRIP), carrying=False),
        Segment(0.65, DemoState(solved["pick_hover"], OPEN_GRIP), DemoState(solved["pick_low"], OPEN_GRIP), carrying=False),
        Segment(0.45, DemoState(solved["pick_low"], OPEN_GRIP), DemoState(solved["pick_low"], CLOSED_GRIP), carrying=False),
        Segment(0.7, DemoState(solved["pick_low"], CLOSED_GRIP), DemoState(solved["pull_up"], CLOSED_GRIP), carrying=True),
        Segment(0.9, DemoState(solved["pull_up"], CLOSED_GRIP), DemoState(solved["pull_mid"], CLOSED_GRIP), carrying=True),
        Segment(0.9, DemoState(solved["pull_mid"], CLOSED_GRIP), DemoState(solved["pull_clear"], CLOSED_GRIP), carrying=True),
        Segment(0.45, DemoState(solved["pull_clear"], CLOSED_GRIP), DemoState(solved["pull_clear"], CLOSED_GRIP), carrying=False, placed=True),
        Segment(0.65, DemoState(solved["pull_clear"], CLOSED_GRIP), DemoState(solved["retreat"], OPEN_GRIP), carrying=False, placed=True),
    ]


def cartesian_follow_plan(config: RuntimeConfig | None = None) -> list[dict[str, object]]:
    bowl_start = config.bowl_start if config is not None else BOWL_START
    plate_target = config.plate_target if config is not None else PLATE_TARGET
    if config is not None and config.task_type == "tissue_pull":
        offset = grasp_offset(config)
        return [
            {
                "stage": "ready",
                "target_m": [-0.02, -0.12, round(float(max(bowl_start[2], plate_target[2]) + 0.17), 4)],
                "gripper": "open",
                "duration_s": 1.0,
            },
            {
                "stage": "pre_grasp_tissue_tab",
                "target_m": (bowl_start - offset + np.array([0.0, 0.0, 0.12])).round(4).tolist(),
                "gripper": "open",
                "duration_s": 1.0,
            },
            {
                "stage": "grasp_tissue_tab",
                "target_m": (bowl_start - offset + np.array([0.0, 0.0, 0.004])).round(4).tolist(),
                "gripper": "closed",
                "duration_s": 0.75,
            },
            {
                "stage": "pull_up_from_box",
                "target_m": (bowl_start - offset + np.array([0.0, 0.0, 0.12])).round(4).tolist(),
                "gripper": "closed",
                "duration_s": 1.0,
            },
            {
                "stage": "pull_clear",
                "target_m": (plate_target - offset).round(4).tolist(),
                "gripper": "closed",
                "duration_s": 1.2,
            },
            {
                "stage": "hold_clear_success",
                "target_m": (plate_target - offset).round(4).tolist(),
                "gripper": "closed",
                "duration_s": 0.7,
            },
        ]
    return [
        {
            "stage": "ready",
            "target_m": [-0.16, -0.14, 0.99],
            "gripper": "open",
            "duration_s": 1.15,
        },
        {
            "stage": "pre_pick",
            "target_m": (bowl_start + np.array([0.0, 0.0, 0.17])).round(4).tolist(),
            "gripper": "open",
            "duration_s": 1.0,
        },
        {
            "stage": "pick_grasp_height",
            "target_m": (bowl_start - BOWL_GRASP_OFFSET + np.array([0.0, 0.0, 0.004])).round(4).tolist(),
            "gripper": "open",
            "duration_s": 1.0,
        },
        {
            "stage": "close_gripper",
            "target_m": (bowl_start - BOWL_GRASP_OFFSET + np.array([0.0, 0.0, 0.004])).round(4).tolist(),
            "gripper": "closed",
            "duration_s": 0.75,
        },
        {
            "stage": "lift",
            "target_m": (bowl_start + np.array([0.0, 0.0, 0.17])).round(4).tolist(),
            "gripper": "closed",
            "duration_s": 1.0,
        },
        {
            "stage": "transfer_clearance",
            "target_m": [0.02, 0.0, 1.02],
            "gripper": "closed",
            "duration_s": 1.15,
        },
        {
            "stage": "pre_place",
            "target_m": (plate_target + np.array([0.0, 0.0, 0.17])).round(4).tolist(),
            "gripper": "closed",
            "duration_s": 1.15,
        },
        {
            "stage": "release_above_target_bowl",
            "target_m": (plate_target + PLACE_RELEASE_OFFSET).round(4).tolist(),
            "gripper": "closed",
            "duration_s": 1.0,
        },
        {
            "stage": "release",
            "target_m": (plate_target + PLACE_RELEASE_OFFSET).round(4).tolist(),
            "gripper": "open",
            "duration_s": 0.75,
        },
        {
            "stage": "retreat",
            "target_m": [-0.1, -0.18, 1.0],
            "gripper": "open",
            "duration_s": 1.15,
        },
    ]


def update_bowl_for_stage(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    config: RuntimeConfig,
    carrying: bool,
    placed: bool,
) -> None:
    if carrying:
        set_bowl_pose(model, data, handles, gripper_position(data, handles) + grasp_offset(config))
    elif placed:
        set_bowl_pose(model, data, handles, config.plate_target)
    else:
        set_bowl_pose(model, data, handles, config.bowl_start)


def task_success(data: mujoco.MjData, handles: RobotHandles, config: RuntimeConfig, carrying: bool) -> bool:
    return bool(task_metrics(data, handles, config, carrying)["success"])


def task_metrics(
    data: mujoco.MjData,
    handles: RobotHandles,
    config: RuntimeConfig,
    carrying: bool,
) -> dict[str, object]:
    bowl_pos = data.qpos[handles.bowl_qpos : handles.bowl_qpos + 3]
    xy_distance = float(np.linalg.norm(bowl_pos[:2] - config.plate_target[:2]))
    z_ok = config.on_plate_z_range[0] <= bowl_pos[2] <= config.on_plate_z_range[1]
    success = not carrying and xy_distance <= config.on_plate_xy_tol and z_ok
    return {
        "success": bool(success),
        "task_type": config.task_type,
        "object_name": config.object_name,
        "target_name": config.target_name,
        "object_final_position_m": bowl_pos.round(6).tolist(),
        "target_position_m": config.plate_target.round(6).tolist(),
        "bowl_final_position_m": bowl_pos.round(6).tolist(),
        "target_bowl_position_m": config.plate_target.round(6).tolist(),
        "xy_distance_to_target_m": xy_distance,
        "xy_tolerance_m": config.on_plate_xy_tol,
        "z_within_plate_range": bool(z_ok),
        "z_range_m": list(config.on_plate_z_range),
    }


def task_phase(carrying: bool, placed: bool, complete: bool) -> str:
    if complete:
        return "success"
    if placed:
        return "place"
    if carrying:
        return "carry"
    return "pick"


def update_status_lights(model: mujoco.MjModel, phase: str) -> None:
    for light_phase, geom_name in STATUS_LIGHTS.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            continue
        model.geom_rgba[geom_id] = STATUS_COLORS[light_phase if light_phase == phase else "off"]


def update_success_screen(model: mujoco.MjModel, visible: bool) -> None:
    state = "on" if visible else "off"
    for geom_name, color in SUCCESS_SCREEN_COLORS[state].items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            model.geom_rgba[geom_id] = color


def save_snapshot(model: mujoco.MjModel, data: mujoco.MjData, output_path: Path) -> None:
    renderer = mujoco.Renderer(model, height=900, width=1400)
    renderer.update_scene(data, camera="overview")
    image = renderer.render()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    renderer.close()


def state_to_servo_waypoint(t: float, state: DemoState) -> dict[str, object]:
    joint_degrees = np.rad2deg(state.q).round(2).tolist()
    gripper_closed = state.grip >= (CLOSED_GRIP * 0.5)
    return {
        "t_sec": round(t, 3),
        "joint_order": list(ARM_JOINTS),
        "servo_ids": SERVO_ID_MAP,
        "joint_degrees": joint_degrees,
        "gripper": "closed" if gripper_closed else "open",
        "gripper_pulse_hint": (
            GRIPPER_HARDWARE_HINT["closed_pulse"]
            if gripper_closed
            else GRIPPER_HARDWARE_HINT["open_pulse"]
        ),
        "sim_gripper_ctrl": round(float(state.grip), 4),
    }


def export_waypoints(segments: list[Segment], output_path: Path, config: RuntimeConfig | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t = 0.0
    waypoints = [state_to_servo_waypoint(t, segments[0].start)]
    for segment in segments:
        t += segment.duration
        waypoints.append(state_to_servo_waypoint(t, segment.end))

    payload = {
        "robot": "hiwonder_jetarm",
        "source": "MuJoCo scripted IK demo",
        "use_note": "Treat these as calibration waypoints, not direct safe hardware commands. Verify servo zero offsets, joint signs, speed limits, and workspace clearance on the physical JetArm.",
        "gripper_calibration": GRIPPER_HARDWARE_HINT,
        "cartesian_follow_plan_m": cartesian_follow_plan(config),
        "waypoints": waypoints,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Exported JetArm waypoint draft: {output_path}")


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
    carrying = False
    update_success_screen(model, False)
    for step in range(steps):
        state, carrying, placed = timeline_state(segments, step * model.opt.timestep)
        set_arm_state(model, data, handles, state)
        update_bowl_for_stage(model, data, handles, config, carrying, placed)
        complete = task_success(data, handles, config, carrying)
        update_status_lights(model, task_phase(carrying, placed, complete))
        mujoco.mj_step(model, data)

    metrics = task_metrics(data, handles, config, carrying=False)
    if metrics["success"]:
        update_status_lights(model, "success")
        update_success_screen(model, True)
    else:
        update_status_lights(model, "place")
        update_success_screen(model, False)
    save_snapshot(model, data, output)
    print(f"Saved snapshot: {output}")
    print(f"Bowl final position: {metrics['bowl_final_position_m']}")
    print(f"Target bowl position: {metrics['target_bowl_position_m']}")
    print(f"XY distance to target: {metrics['xy_distance_to_target_m']:.4f} m")
    print(f"Z within target bowl range: {metrics['z_within_plate_range']}")
    if config.task_type == "tissue_pull":
        print(f"Tissue final position: {metrics['object_final_position_m']}")
        print(f"Pull-clear target position: {metrics['target_position_m']}")
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
    enable_zoom_slider: bool,
) -> None:
    announced_success = False
    last_phase = ""
    controls = ViewerControls()
    zoom_control = start_zoom_slider(DEFAULT_CAMERA_DISTANCE) if enable_zoom_slider else None
    print_viewer_controls()
    update_success_screen(model, False)

    with mujoco.viewer.launch_passive(model, data, key_callback=make_key_callback(controls)) as viewer:
        apply_camera_preset(viewer, controls)
        controls.camera_dirty = False
        last_tick = time.time()

        while viewer.is_running():
            now = time.time()
            wall_dt = now - last_tick
            last_tick = now

            if controls.restart_requested:
                controls.sim_time = 0.0
                announced_success = False
                last_phase = ""
                controls.restart_requested = False
                update_success_screen(model, False)

            if not controls.paused:
                controls.sim_time += wall_dt * controls.speed

            if zoom_control is not None:
                viewer.cam.distance = zoom_control.distance
            elif controls.camera_dirty:
                apply_camera_preset(viewer, controls)
                controls.camera_dirty = False

            state, carrying, placed = timeline_state(segments, controls.sim_time)
            set_arm_state(model, data, handles, state)
            update_bowl_for_stage(model, data, handles, config, carrying, placed)

            complete = task_success(data, handles, config, carrying)
            phase = task_phase(carrying, placed, complete)
            update_status_lights(model, phase)
            update_success_screen(model, complete)
            if phase != last_phase:
                print(f"TASK PHASE: {phase}")
                last_phase = phase

            if complete and not announced_success:
                if config.task_type == "tissue_pull":
                    print("TASK COMPLETE: tissue_0 is pulled clear of tissue_box_0. SUCCESS=True")
                else:
                    print("TASK COMPLETE: bowl_0 is in target_bowl_0. SUCCESS=True")
                announced_success = True
            elif not complete and announced_success and not placed:
                announced_success = False

            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def build_model(scene_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, RobotHandles, RuntimeConfig, list[Segment]]:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    handles = get_handles(model)
    config = runtime_config_from_scene(scene_path, model, data, handles)
    segments = make_demo_segments(model, data, handles, config)
    return model, data, handles, config, segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hiwonder JetArm table pick-and-place demo.")
    parser.add_argument("--headless", action="store_true", help="Run without the viewer and save a snapshot.")
    parser.add_argument("--export-waypoints", action="store_true", help="Export a draft servo-waypoint JSON file.")
    parser.add_argument("--zoom-slider", action="store_true", help="Open a separate Tk zoom slider window.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=SCENE_PATH,
        help="MJCF scene path. Generated media scenes can be passed here.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "jetarm_table_bowl_pick_place.png",
        help="Snapshot path for --headless.",
    )
    parser.add_argument("--result-json", type=Path, help="Optional machine-readable result path for --headless.")
    parser.add_argument(
        "--waypoints",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "jetarm_bowl_to_plate_waypoints.json",
        help="Waypoint JSON path for --export-waypoints.",
    )
    args = parser.parse_args()

    scene_path = args.scene.resolve()
    model, data, handles, config, segments = build_model(scene_path)

    if args.export_waypoints:
        export_waypoints(segments, args.waypoints, config)

    if args.headless:
        run_headless(model, data, handles, segments, config, args.snapshot, args.result_json)
        return

    if sys.platform == "darwin" and "mjpython" not in Path(sys.executable).name:
        print("Tip: on macOS this interactive demo should be launched with `mjpython`.")
        print("Run: mjpython sim/run_jetarm_pick_place_demo.py")

    run_viewer(model, data, handles, segments, config, args.zoom_slider)


if __name__ == "__main__":
    main()
