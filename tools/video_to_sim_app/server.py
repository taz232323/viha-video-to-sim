from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import platform
import shutil
import subprocess
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"


def safe_name(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "session"


def rel_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def path_from_root(relative_path: str) -> Path:
    candidate = (ROOT / relative_path).resolve()
    if ROOT not in candidate.parents and candidate != ROOT:
        raise ValueError("Path escapes project root")
    return candidate


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler: BaseHTTPRequestHandler, message: str, status: int = 400) -> None:
    json_response(handler, {"ok": False, "error": message}, status)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)


def launch_live_sim(scene_path: Path) -> str:
    mjpython = ROOT / ".venv" / "bin" / "mjpython"
    if not mjpython.exists():
        mjpython = PYTHON

    command = (
        f"cd {str(ROOT)!r} && "
        f"{str(mjpython)!r} sim/run_jetarm_pick_place_demo.py --scene {rel_path(scene_path)!r}"
    )
    if platform.system() == "Darwin":
        subprocess.Popen(
            [
                "osascript",
                "-e",
                f'tell application "Terminal" to do script "{command}"',
                "-e",
                'tell application "Terminal" to activate',
            ],
            cwd=ROOT,
        )
        return "Launched live MuJoCo viewer in Terminal."

    subprocess.Popen(
        [str(mjpython), "sim/run_jetarm_pick_place_demo.py", "--scene", rel_path(scene_path)],
        cwd=ROOT,
    )
    return "Launched live MuJoCo viewer."


def default_windows() -> list[dict]:
    return [
        {
            "name": "scene_reference",
            "start_sec": 0.5,
            "end_sec": 2.0,
            "purpose": "Full table, robot base, object, target, and calibration marker visible.",
        },
        {
            "name": "pre_pick",
            "start_sec": 2.0,
            "end_sec": 4.0,
            "purpose": "Object visible before the human touches it.",
        },
        {
            "name": "pre_place",
            "start_sec": 5.0,
            "end_sec": 7.0,
            "purpose": "Target visible before the object is placed.",
        },
        {
            "name": "post_place",
            "start_sec": 7.0,
            "end_sec": 9.0,
            "purpose": "Object resting on/in the target with the hand out of frame.",
        },
    ]


def parse_windows(value: str | None) -> list[dict]:
    if not value:
        return default_windows()
    windows = json.loads(value)
    if not isinstance(windows, list) or not windows:
        raise ValueError("windows must be a non-empty JSON list")
    for window in windows:
        for key in ("name", "start_sec", "end_sec"):
            if key not in window:
                raise ValueError(f"Each window needs {key}")
    return windows


def write_plan(session_id: str, video_path: Path, windows: list[dict], candidate_fps: float) -> Path:
    plan_path = ROOT / "inputs" / "video_plans" / f"{session_id}_window_plan.json"
    output_dir = ROOT / "outputs" / "video_frames" / session_id
    plan = {
        "session_id": session_id,
        "video_path": str(video_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "candidate_fps": candidate_fps,
        "selected_frames_per_window": 1,
        "windows": windows,
        "downstream_spec": {
            "template": str((ROOT / "sim" / "video_specs" / "jetarm_video_task_spec_template.json").resolve()),
            "output_spec": str((output_dir / "jetarm_video_task_spec.json").resolve()),
        },
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan_path


def selected_frames_from_manifest(manifest: dict) -> list[dict]:
    frames = []
    for window in manifest.get("windows", []):
        for selected in window.get("selected", []):
            selected_path = selected.get("selected_path")
            if selected_path:
                frames.append(
                    {
                        "window": window.get("name"),
                        "path": rel_path(Path(selected_path)),
                        "score": selected.get("score"),
                        "brightness": selected.get("brightness"),
                        "contrast": selected.get("contrast"),
                        "sharpness": selected.get("sharpness"),
                    }
                )
    return frames


def image_size(relative_path: str) -> tuple[int, int]:
    with Image.open(path_from_root(relative_path)) as image:
        return image.size


def generate_annotation_overlay(spec: dict, output_path: Path) -> Path | None:
    frame = spec.get("source", {}).get("representative_frame_path")
    if not frame:
        return None
    frame_path = Path(frame)
    if not frame_path.is_absolute():
        frame_path = path_from_root(frame)
    if not frame_path.exists():
        return None

    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    corners = spec["calibration"]["table_corners_pixels"]
    order = ["front_left", "front_right", "back_right", "back_left"]
    polygon = [tuple(corners[name]) for name in order] + [tuple(corners[order[0]])]
    draw.line(polygon, fill=(0, 255, 255), width=5)
    for name in order:
        x, y = corners[name]
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(0, 255, 255), outline=(0, 0, 0), width=2)
        draw.text((x + 12, y - 12), name, fill=(0, 255, 255))

    for item in spec.get("objects", []):
        x, y = item["center_pixel"]
        if item.get("role") == "pick_object":
            draw.ellipse((x - 28, y - 28, x + 28, y + 28), outline=(255, 255, 0), width=6)
            draw.text((x + 34, y - 18), item["id"], fill=(255, 255, 0))
        else:
            draw.rectangle((x - 50, y - 50, x + 50, y + 50), outline=(255, 0, 0), width=6)
            draw.text((x + 58, y - 18), item["id"], fill=(255, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    return output_path


def write_annotated_spec(payload: dict) -> Path:
    session_id = safe_name(payload["session_id"])
    output_root = ROOT / "outputs" / "video_frames" / session_id
    spec_path = output_root / "jetarm_video_task_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"Video spec not found: {spec_path}")

    spec = json.loads(spec_path.read_text())
    points = payload["points"]
    table = payload.get("table", {})
    robot = payload.get("robot", {})
    task_type = payload.get("task_type", "cube_to_square")

    table_width = float(table.get("width_m", 0.9))
    table_depth = float(table.get("depth_m", 0.6))
    table_surface_z = float(table.get("surface_z_m", 0.765))
    half_w = table_width / 2.0
    half_d = table_depth / 2.0

    spec["description"] = f"Generated from local video-to-sim UI session {session_id}."
    spec["calibration"]["table_surface_z"] = table_surface_z
    spec["calibration"]["table_corners_pixels"] = {
        "front_left": points["front_left"],
        "front_right": points["front_right"],
        "back_right": points["back_right"],
        "back_left": points["back_left"],
    }
    spec["calibration"]["table_corners_world"] = {
        "front_left": [-half_w, -half_d],
        "front_right": [half_w, -half_d],
        "back_right": [half_w, half_d],
        "back_left": [-half_w, half_d],
    }
    spec["calibration"]["robot_base_world_xyz"] = [
        float(robot.get("base_x_m", -0.16)),
        float(robot.get("base_y_m", -0.10)),
        float(robot.get("base_z_m", table_surface_z)),
    ]
    spec["calibration"]["base_frame_note"] = "Created by tools/video_to_sim_app. Verify measurements before hardware use."

    object_radius = float(payload.get("object_radius_m", 0.022))
    target_radius = float(payload.get("target_radius_m", 0.055))
    if task_type == "tissue_pull":
        tissue_box_center = payload.get("tissue_box_center") or [
            points["pick_object"][0],
            points["pick_object"][1] + 65,
        ]
        objects = [
            {
                "id": "tissue_0",
                "label": "tissue_sheet",
                "mjcf_body": "bowl",
                "center_pixel": points["pick_object"],
                "estimated_radius_m": object_radius,
                "body_origin_z_above_table_m": 0.105,
                "role": "pick_object",
            },
            {
                "id": "pull_clear_target_0",
                "label": "pull_clear_target",
                "mjcf_body": "plate",
                "center_pixel": points["target"],
                "estimated_radius_m": target_radius,
                "body_origin_z_above_table_m": 0.006,
                "role": "pull_target",
            },
        ]
        spec.setdefault("scene_options", {})["object_shapes"] = {
            "tissue_0": "tissue_sheet",
            "pull_clear_target_0": "pull_target",
        }
        spec["scene_options"]["tissue_box"] = {
            "center_pixel": tissue_box_center,
            "size_m": float(payload.get("tissue_box_size_m", 0.13)),
        }
        spec["task"] = {
            "type": "tissue_pull",
            "pick_object": "tissue_0",
            "pull_target": "pull_clear_target_0",
            "success": "tissue_0 moved clear of tissue_box_0",
        }
    elif task_type == "orange_to_bowl":
        objects = [
            {
                "id": "orange_0",
                "label": "orange",
                "mjcf_body": "bowl",
                "center_pixel": points["pick_object"],
                "estimated_radius_m": object_radius,
                "body_origin_z_above_table_m": object_radius,
                "role": "pick_object",
            },
            {
                "id": "target_bowl_0",
                "label": "target_bowl",
                "mjcf_body": "plate",
                "center_pixel": points["target"],
                "estimated_radius_m": target_radius,
                "body_origin_z_above_table_m": 0.005,
                "role": "place_target",
            },
        ]
        spec.setdefault("scene_options", {})["object_shapes"] = {"orange_0": "orange_sphere"}
        spec["task"] = {"type": "pick_and_place", "pick_object": "orange_0", "place_target": "target_bowl_0"}
    else:
        objects = [
            {
                "id": "cube_0",
                "label": "small_pick_object",
                "mjcf_body": "bowl",
                "center_pixel": points["pick_object"],
                "estimated_radius_m": object_radius,
                "body_origin_z_above_table_m": object_radius,
                "role": "pick_object",
            },
            {
                "id": "target_square_0",
                "label": "target_square",
                "mjcf_body": "plate",
                "center_pixel": points["target"],
                "estimated_radius_m": target_radius,
                "body_origin_z_above_table_m": 0.002,
                "role": "place_target",
            },
        ]
        spec.setdefault("scene_options", {})["object_shapes"] = {
            "cube_0": "cube",
            "target_square_0": "target_square",
        }
        spec["task"] = {"type": "pick_and_place", "pick_object": "cube_0", "place_target": "target_square_0"}

    spec["objects"] = objects
    spec["outputs"] = {
        "scene_xml": f"outputs/generated/{session_id}_scene.xml",
        "metadata_json": f"outputs/generated/{session_id}_scene_metadata.json",
        "snapshot_png": f"outputs/generated/{session_id}_scene_snapshot.png",
        "result_json": f"outputs/generated/{session_id}_scene_result.json",
    }

    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    generate_annotation_overlay(spec, output_root / "annotated_video_points.jpg")
    return spec_path


class VideoToSimHandler(BaseHTTPRequestHandler):
    server_version = "ViHaVideoToSim/0.1"

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_file(APP_DIR / "index.html")
            return
        if parsed.path == "/app.js":
            self.serve_file(APP_DIR / "app.js")
            return
        if parsed.path == "/style.css":
            self.serve_file(APP_DIR / "style.css")
            return
        if parsed.path == "/artifact":
            query = parse_qs(parsed.query)
            value = query.get("path", [""])[0]
            try:
                artifact = path_from_root(value)
            except ValueError as exc:
                error_response(self, str(exc), HTTPStatus.BAD_REQUEST)
                return
            if not artifact.exists() or artifact.is_dir():
                error_response(self, f"Artifact not found: {value}", HTTPStatus.NOT_FOUND)
                return
            self.serve_file(artifact)
            return
        error_response(self, "Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/upload":
                self.handle_upload()
                return
            if self.path == "/api/build":
                self.handle_build()
                return
            if self.path == "/api/open-sim":
                self.handle_open_sim()
                return
            error_response(self, "Not found", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            traceback.print_exc()
            error_response(self, str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_file(self, path: Path) -> None:
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_upload(self) -> None:
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
        upload = form["video"] if "video" in form else None
        if upload is None or not upload.filename:
            raise ValueError("Upload a video file")

        session_id = safe_name(form.getfirst("session_id") or Path(upload.filename).stem)
        ext = Path(upload.filename).suffix.lower() or ".webm"
        video_path = ROOT / "inputs" / "videos" / f"{session_id}{ext}"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with video_path.open("wb") as out:
            shutil.copyfileobj(upload.file, out)

        windows = parse_windows(form.getfirst("windows"))
        candidate_fps = float(form.getfirst("candidate_fps") or 4)
        plan_path = write_plan(session_id, video_path, windows, candidate_fps)

        command = [
            str(PYTHON),
            "sim/extract_video_frames.py",
            "--plan",
            rel_path(plan_path),
            "--force",
        ]
        completed = run_command(command)

        manifest_path = ROOT / "outputs" / "video_frames" / session_id / "selected_frames.json"
        manifest = json.loads(manifest_path.read_text())
        representative = rel_path(Path(manifest["representative_frame_path"]))
        width, height = image_size(representative)
        json_response(
            self,
            {
                "ok": True,
                "session_id": session_id,
                "video_path": rel_path(video_path),
                "plan_path": rel_path(plan_path),
                "manifest_path": rel_path(manifest_path),
                "spec_path": rel_path(ROOT / "outputs" / "video_frames" / session_id / "jetarm_video_task_spec.json"),
                "representative_frame": representative,
                "representative_size": {"width": width, "height": height},
                "selected_frames": selected_frames_from_manifest(manifest),
                "stdout": completed.stdout,
            },
        )

    def handle_build(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        session_id = safe_name(payload["session_id"])
        spec_path = write_annotated_spec(payload)

        build = run_command(
            [
                str(PYTHON),
                "sim/build_jetarm_scene_from_media.py",
                "--spec",
                rel_path(spec_path),
                "--run-headless",
            ]
        )

        scene_path = ROOT / "outputs" / "generated" / f"{session_id}_scene.xml"
        review_path = ROOT / "outputs" / "generated" / f"{session_id}_review_sheet.png"
        review = run_command(
            [
                str(PYTHON),
                "sim/render_jetarm_review_sheet.py",
                "--scene",
                rel_path(scene_path),
                "--output",
                rel_path(review_path),
            ]
        )

        result_path = ROOT / "outputs" / "generated" / f"{session_id}_scene_result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        viewer_status = None
        if payload.get("open_viewer"):
            viewer_status = launch_live_sim(scene_path)

        json_response(
            self,
            {
                "ok": True,
                "session_id": session_id,
                "spec_path": rel_path(spec_path),
                "scene_path": rel_path(scene_path),
                "result_path": rel_path(result_path),
                "snapshot_path": f"outputs/generated/{session_id}_scene_snapshot.png",
                "review_sheet_path": rel_path(review_path),
                "annotation_overlay_path": f"outputs/video_frames/{session_id}/annotated_video_points.jpg",
                "success": result.get("success"),
                "result": result,
                "viewer_status": viewer_status,
                "stdout": build.stdout + "\n" + review.stdout,
            },
        )

    def handle_open_sim(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        scene_value = payload.get("scene_path")
        if not scene_value:
            raise ValueError("scene_path is required")
        scene_path = path_from_root(scene_value)
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene not found: {scene_value}")
        status = launch_live_sim(scene_path)
        json_response(self, {"ok": True, "viewer_status": status, "scene_path": rel_path(scene_path)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local ViHa video-to-sim upload UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()

    if not PYTHON.exists():
        raise SystemExit(f"Missing virtualenv Python: {PYTHON}")

    server = ThreadingHTTPServer((args.host, args.port), VideoToSimHandler)
    print(f"Video-to-sim UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
