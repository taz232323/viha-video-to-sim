from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from run_jetarm_pick_place_demo import (
    build_model,
    set_arm_state,
    task_success,
    timeline_state,
    update_bowl_for_stage,
    update_status_lights,
    update_success_screen,
)


ROOT = Path(__file__).resolve().parents[1]


def cumulative_times(durations: list[float]) -> list[float]:
    total = 0.0
    times = [0.0]
    for duration in durations:
        total += duration
        times.append(total)
    return times


def review_times(durations: list[float]) -> list[tuple[str, float]]:
    edges = cumulative_times(durations)
    total = edges[-1]
    candidates = [
        ("start", 0.05),
        ("approach pick", edges[min(1, len(edges) - 1)] + 0.2),
        ("carry", edges[min(5, len(edges) - 1)] - 0.05),
        ("release", edges[min(8, len(edges) - 1)] - 0.05),
        ("success", total - 0.08),
    ]
    return [(label, max(0.0, min(total - 0.01, t))) for label, t in candidates]


def apply_timeline_state(model, data, handles, config, segments, sim_time: float) -> bool:
    state, carrying, placed = timeline_state(segments, sim_time)
    set_arm_state(model, data, handles, state)
    update_bowl_for_stage(model, data, handles, config, carrying, placed)
    complete = task_success(data, handles, config, carrying)
    update_status_lights(model, "success" if complete else "carry" if carrying else "place" if placed else "pick")
    update_success_screen(model, complete)
    mujoco.mj_forward(model, data)
    return complete


def make_camera(config, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    center = 0.5 * (config.bowl_start + config.plate_target)
    cam.lookat[:] = np.array([center[0], center[1], max(0.82, center[2] + 0.03)])
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def render_frame(model, data, camera, width: int, height: int) -> Image.Image:
    renderer = mujoco.Renderer(model, width=width, height=height)
    renderer.update_scene(data, camera=camera)
    image = Image.fromarray(renderer.render()).convert("RGB")
    renderer.close()
    return image


def label_image(image: Image.Image, label: str, complete: bool) -> Image.Image:
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    banner_h = 42
    draw.rectangle((0, 0, labeled.width, banner_h), fill=(12, 12, 12))
    suffix = "  SUCCESS" if complete else ""
    draw.text((14, 12), f"{label}{suffix}", fill=(255, 255, 255))
    return labeled


def make_sheet(frames: list[Image.Image], cols: int = 2) -> Image.Image:
    if not frames:
        raise ValueError("No frames to render")
    rows = (len(frames) + cols - 1) // cols
    width, height = frames[0].size
    sheet = Image.new("RGB", (cols * width, rows * height), (25, 25, 25))
    for index, frame in enumerate(frames):
        x = (index % cols) * width
        y = (index // cols) * height
        sheet.paste(frame, (x, y))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a labeled PNG review sheet for a JetArm MuJoCo scene.")
    parser.add_argument("--scene", type=Path, required=True, help="Scene XML to render.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--width", type=int, default=700, help="Per-frame render width.")
    parser.add_argument("--height", type=int, default=450, help="Per-frame render height.")
    parser.add_argument("--distance", type=float, default=1.35, help="Free camera distance.")
    parser.add_argument("--azimuth", type=float, default=145, help="Free camera azimuth.")
    parser.add_argument("--elevation", type=float, default=-62, help="Free camera elevation.")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "sim"))
    model, data, handles, config, segments = build_model(args.scene.resolve())
    camera = make_camera(config, args.distance, args.azimuth, args.elevation)
    durations = [segment.duration for segment in segments]

    frames = []
    for label, sim_time in review_times(durations):
        complete = apply_timeline_state(model, data, handles, config, segments, sim_time)
        rendered = render_frame(model, data, camera, args.width, args.height)
        frames.append(label_image(rendered, label, complete))

    sheet = make_sheet(frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Saved review sheet: {args.output}")


if __name__ == "__main__":
    main()
