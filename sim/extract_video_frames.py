from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "video_frames"


def resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise SystemExit(
            f"Missing required executable: {name}\n"
            "Install FFmpeg first, then rerun this command. On macOS with Homebrew:\n"
            "  brew install ffmpeg"
        )
    return executable


def ffprobe_metadata(video_path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {}

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return {}
    return json.loads(completed.stdout or "{}")


def safe_name(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "window"


def score_frame(frame_path: Path) -> dict[str, float]:
    image = Image.open(frame_path).convert("L")
    image.thumbnail((640, 640))
    arr = np.asarray(image, dtype=np.float32) / 255.0

    gy, gx = np.gradient(arr)
    sharpness = float(np.var(gx) + np.var(gy))
    contrast = float(np.std(arr))
    brightness = float(np.mean(arr))

    # Penalize frames that are very dark or blown out. The target brightness
    # is intentionally broad because lab videos can vary a lot.
    exposure_penalty = abs(brightness - 0.48) * 0.18
    score = sharpness + 0.12 * contrast - exposure_penalty

    return {
        "score": score,
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
    }


def extract_window_frames(
    ffmpeg: str,
    video_path: Path,
    window: dict[str, Any],
    candidates_dir: Path,
    default_fps: float,
    force: bool,
) -> list[Path]:
    name = safe_name(str(window["name"]))
    start_sec = float(window["start_sec"])
    end_sec = float(window["end_sec"])
    fps = float(window.get("candidate_fps", default_fps))
    if end_sec <= start_sec:
        raise ValueError(f"Window {name} end_sec must be greater than start_sec")
    if fps <= 0:
        raise ValueError(f"Window {name} candidate_fps must be positive")

    window_dir = candidates_dir / name
    if force and window_dir.exists():
        shutil.rmtree(window_dir)
    window_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(window_dir.glob("*.jpg"))
    if existing and not force:
        return existing

    output_pattern = window_dir / f"{name}_%05d.jpg"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{end_sec - start_sec:.3f}",
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)
    return sorted(window_dir.glob("*.jpg"))


def select_best_frames(
    frames: list[Path],
    selected_dir: Path,
    window_name: str,
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored = []
    for frame_path in frames:
        metrics = score_frame(frame_path)
        scored.append({"path": str(frame_path), **metrics})

    scored.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    selected_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(scored[:count], start=1):
        source = Path(item["path"])
        target = selected_dir / f"{safe_name(window_name)}_best_{index:03d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        selected.append({**item, "selected_path": str(target)})

    return scored, selected


def load_plan(plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    if "video_path" not in plan:
        raise ValueError("Plan must include video_path")
    if not plan.get("windows"):
        raise ValueError("Plan must include at least one window")
    for window in plan["windows"]:
        for key in ("name", "start_sec", "end_sec"):
            if key not in window:
                raise ValueError(f"Each window must include {key}")
    return plan


def write_video_spec(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    plan_dir: Path,
    output_spec_path: Path,
    template_path: Path | None,
) -> None:
    if template_path is None:
        template_value = plan.get("downstream_spec", {}).get("template")
        template_path = resolve_path(template_value, plan_dir) if template_value else None
    if template_path is None:
        template_path = ROOT / "sim" / "video_specs" / "jetarm_video_task_spec_template.json"

    spec = json.loads(template_path.read_text())
    representative = manifest.get("representative_frame_path")
    if representative is None:
        raise ValueError("No representative frame was selected; cannot write video spec")

    source = spec.setdefault("source", {})
    source["type"] = "video"
    source["video_path"] = manifest["video_path"]
    source["representative_frame_path"] = representative
    source["representative_frame_time_sec"] = manifest.get("representative_frame_time_sec", 0.0)
    source["selected_frames_manifest"] = manifest["manifest_path"]

    video_processing = spec.setdefault("video_processing", {})
    video_processing["windows"] = plan.get("windows", [])
    video_processing["candidate_fps"] = plan.get("candidate_fps", video_processing.get("candidate_fps", 4))
    video_processing["selected_frames_per_window"] = plan.get(
        "selected_frames_per_window",
        video_processing.get("selected_frames_per_window", 1),
    )
    video_processing["selection_tool"] = "sharpness_selector"
    video_processing["selection_note"] = "Katna can be added later; this spec was produced by sim/extract_video_frames.py."

    output_spec_path.parent.mkdir(parents=True, exist_ok=True)
    output_spec_path.write_text(json.dumps(spec, indent=2) + "\n")


def process_video(plan_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    ffmpeg = require_executable("ffmpeg")
    plan = load_plan(plan_path)
    plan_dir = plan_path.parent
    video_path = resolve_path(plan["video_path"], plan_dir)
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(f"Video not found: {plan['video_path']}")

    session_id = safe_name(str(plan.get("session_id") or video_path.stem))
    output_root = resolve_path(plan.get("output_dir"), plan_dir) or DEFAULT_OUTPUT_ROOT / session_id
    candidates_dir = output_root / "candidates"
    selected_dir = output_root / "selected"
    manifest_path = output_root / "selected_frames.json"
    output_root.mkdir(parents=True, exist_ok=True)

    default_fps = float(plan.get("candidate_fps", 4))
    default_selected_count = int(plan.get("selected_frames_per_window", 1))
    all_windows = []
    selected_by_window: dict[str, list[dict[str, Any]]] = {}

    for window in plan["windows"]:
        window_name = str(window["name"])
        frames = extract_window_frames(
            ffmpeg=ffmpeg,
            video_path=video_path,
            window=window,
            candidates_dir=candidates_dir,
            default_fps=default_fps,
            force=args.force,
        )
        selected_count = int(window.get("selected_frames", default_selected_count))
        scored, selected = select_best_frames(frames, selected_dir, window_name, selected_count)
        selected_by_window[safe_name(window_name)] = selected
        all_windows.append(
            {
                "name": window_name,
                "purpose": window.get("purpose"),
                "start_sec": float(window["start_sec"]),
                "end_sec": float(window["end_sec"]),
                "candidate_fps": float(window.get("candidate_fps", default_fps)),
                "candidate_count": len(frames),
                "candidates": scored,
                "selected": selected,
            }
        )

    representative = None
    representative_time = 0.0
    for preferred in ("scene_reference", "pre_pick", "pre_place"):
        entries = selected_by_window.get(preferred)
        if entries:
            representative = entries[0]["selected_path"]
            for window in all_windows:
                if safe_name(window["name"]) == preferred:
                    representative_time = float(window["start_sec"])
                    break
            break
    if representative is None:
        for entries in selected_by_window.values():
            if entries:
                representative = entries[0]["selected_path"]
                representative_time = 0.0
                break

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "manifest_path": str(manifest_path),
        "video_path": str(video_path),
        "video_metadata": ffprobe_metadata(video_path),
        "output_root": str(output_root),
        "representative_frame_path": representative,
        "representative_frame_time_sec": representative_time,
        "windows": all_windows,
        "notes": [
            "Hiwonder/FANUC can be visible in the frame if the robot does not block calibration points, object centers, or target centers.",
            "This selector ranks extracted frames by sharpness/contrast. Katna can be integrated later for key-frame selection.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    output_spec_arg = args.output_spec or plan.get("downstream_spec", {}).get("output_spec")
    if output_spec_arg:
        output_spec_path = resolve_path(str(output_spec_arg), plan_dir)
        template_path = resolve_path(args.spec_template, Path.cwd()) if args.spec_template else None
        if output_spec_path is None:
            raise ValueError("Could not resolve output spec path")
        write_video_spec(manifest, plan, plan_dir, output_spec_path, template_path)
        manifest["generated_video_spec"] = str(output_spec_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and select useful frames from a robot task video.")
    parser.add_argument("--plan", type=Path, required=True, help="Video window plan JSON.")
    parser.add_argument("--force", action="store_true", help="Re-extract frames even if candidate files already exist.")
    parser.add_argument("--output-spec", type=str, help="Optional generated sim video spec path.")
    parser.add_argument("--spec-template", type=str, help="Optional sim video spec template path.")
    args = parser.parse_args()

    manifest = process_video(args.plan.resolve(), args)
    print(f"Selected frames manifest: {manifest['manifest_path']}")
    if manifest.get("representative_frame_path"):
        print(f"Representative frame: {manifest['representative_frame_path']}")
    if manifest.get("generated_video_spec"):
        print(f"Generated video spec: {manifest['generated_video_spec']}")


if __name__ == "__main__":
    main()
