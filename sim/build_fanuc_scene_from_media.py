from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_SCENE = ROOT / "sim" / "worlds" / "table_bowl_pick_place.xml"
CORNER_ORDER = ("front_left", "front_right", "back_right", "back_left")


def resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    spec_relative = (base_dir / path).resolve()
    if spec_relative.exists():
        return spec_relative
    return (ROOT / path).resolve()


def parse_xy(value: list[float] | tuple[float, float], label: str) -> np.ndarray:
    if len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    return np.array([float(value[0]), float(value[1])], dtype=float)


def parse_xyz(value: list[float] | tuple[float, float, float], label: str) -> np.ndarray:
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)


def homography_from_points(pixel_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    if pixel_points.shape != (4, 2) or world_points.shape != (4, 2):
        raise ValueError("homography requires exactly four pixel points and four world points")

    rows = []
    values = []
    for (u, v), (x, y) in zip(pixel_points, world_points):
        rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v])
        values.append(x)
        rows.append([0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v])
        values.append(y)

    solution = np.linalg.solve(np.array(rows, dtype=float), np.array(values, dtype=float))
    return np.array(
        [
            [solution[0], solution[1], solution[2]],
            [solution[3], solution[4], solution[5]],
            [solution[6], solution[7], 1.0],
        ],
        dtype=float,
    )


def apply_homography(homography: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    projected = homography @ np.array([pixel[0], pixel[1], 1.0], dtype=float)
    if abs(projected[2]) < 1e-9:
        raise ValueError(f"homography projected {pixel.tolist()} to a point at infinity")
    return projected[:2] / projected[2]


def body_by_name(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise ValueError(f"MJCF body not found: {name}")


def find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


def find_geom(root: ET.Element, name: str) -> ET.Element | None:
    for geom in root.iter("geom"):
        if geom.get("name") == name:
            return geom
    return None


def find_camera(root: ET.Element, name: str) -> ET.Element | None:
    for camera in root.iter("camera"):
        if camera.get("name") == name:
            return camera
    return None


def find_site(root: ET.Element, name: str) -> ET.Element | None:
    for site in root.iter("site"):
        if site.get("name") == name:
            return site
    return None


def remove_body(root: ET.Element, name: str) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("name") == name:
                parent.remove(child)
                return


def replace_body_with_orange(body: ET.Element, radius_m: float) -> None:
    for child in list(body):
        if child.tag == "geom":
            body.remove(child)

    ET.SubElement(
        body,
        "geom",
        {
            "name": "orange_collision",
            "type": "sphere",
            "pos": "0 0 0",
            "size": f"{radius_m:.6f}",
            "rgba": "1 0.38 0.04 1",
            "mass": "0.09",
            "friction": "0.95 0.02 0.001",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "orange_highlight",
            "type": "sphere",
            "pos": f"{-0.28 * radius_m:.6f} {-0.34 * radius_m:.6f} {0.48 * radius_m:.6f}",
            "size": f"{0.34 * radius_m:.6f}",
            "rgba": "1 0.65 0.18 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )


def replace_body_with_cube(body: ET.Element, half_size_m: float) -> None:
    for child in list(body):
        if child.tag == "geom":
            body.remove(child)

    ET.SubElement(
        body,
        "geom",
        {
            "name": "cube_collision",
            "type": "box",
            "pos": "0 0 0",
            "size": f"{half_size_m:.6f} {half_size_m:.6f} {half_size_m:.6f}",
            "rgba": "0.92 0.94 0.96 1",
            "mass": "0.06",
            "friction": "0.95 0.02 0.001",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "cube_top_label",
            "type": "box",
            "pos": f"0 0 {half_size_m + 0.0008:.6f}",
            "size": f"{0.65 * half_size_m:.6f} {0.65 * half_size_m:.6f} 0.0008",
            "rgba": "0.05 0.2 0.8 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )


def replace_body_with_target_square(body: ET.Element, half_size_m: float) -> None:
    for child in list(body):
        if child.tag in {"geom", "site"}:
            body.remove(child)

    ET.SubElement(
        body,
        "geom",
        {
            "name": "target_square_marker",
            "type": "box",
            "pos": "0 0 0.002",
            "size": f"{half_size_m:.6f} {half_size_m:.6f} 0.002",
            "rgba": "0.82 0.06 0.12 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "target_square_collision",
            "type": "box",
            "pos": "0 0 0.001",
            "size": f"{half_size_m:.6f} {half_size_m:.6f} 0.001",
            "rgba": "0.82 0.06 0.12 0",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    ET.SubElement(
        body,
        "site",
        {
            "name": "success_target",
            "pos": "0 0 0.006",
            "size": f"{half_size_m:.6f}",
            "rgba": "0 0.8 0.2 0",
        },
    )


def replace_body_with_tissue_sheet(body: ET.Element, half_width_m: float) -> None:
    for child in list(body):
        if child.tag == "geom":
            body.remove(child)

    tab_height = max(0.055, half_width_m * 1.55)
    tab_width = max(0.035, half_width_m * 1.05)
    thickness = 0.003

    ET.SubElement(
        body,
        "geom",
        {
            "name": "tissue_sheet_collision",
            "type": "box",
            "pos": f"0 0 {tab_height * 0.42:.6f}",
            "size": f"{tab_width * 0.5:.6f} {thickness:.6f} {tab_height * 0.5:.6f}",
            "rgba": "0.97 0.98 0.96 0.92",
            "mass": "0.012",
            "friction": "1.0 0.02 0.001",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "tissue_fold_top",
            "type": "box",
            "pos": f"{-tab_width * 0.16:.6f} 0 {tab_height * 0.92:.6f}",
            "euler": "0 12 0",
            "size": f"{tab_width * 0.36:.6f} {thickness * 1.2:.6f} {tab_height * 0.28:.6f}",
            "rgba": "1 1 1 0.96",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "tissue_fold_side",
            "type": "box",
            "pos": f"{tab_width * 0.26:.6f} 0 {tab_height * 0.58:.6f}",
            "euler": "0 -18 0",
            "size": f"{tab_width * 0.24:.6f} {thickness * 1.2:.6f} {tab_height * 0.36:.6f}",
            "rgba": "0.94 0.96 0.95 0.88",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "tissue_inside_box",
            "type": "box",
            "pos": f"0 0 {-tab_height * 0.34:.6f}",
            "size": f"{tab_width * 0.44:.6f} {thickness * 1.1:.6f} {tab_height * 0.34:.6f}",
            "rgba": "0.92 0.94 0.93 0.55",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "tissue_soft_edge",
            "type": "capsule",
            "fromto": f"{-tab_width * 0.35:.6f} 0 {tab_height:.6f} {tab_width * 0.32:.6f} 0 {tab_height * 1.06:.6f}",
            "size": "0.006",
            "rgba": "0.92 0.94 0.93 0.85",
            "contype": "0",
            "conaffinity": "0",
        },
    )


def replace_body_with_pull_target(body: ET.Element, radius_m: float) -> None:
    for child in list(body):
        if child.tag in {"geom", "site"}:
            body.remove(child)

    ET.SubElement(
        body,
        "geom",
        {
            "name": "pull_target_marker",
            "type": "cylinder",
            "pos": "0 0 0",
            "size": f"{radius_m:.6f} 0.002",
            "rgba": "0.06 0.82 0.24 0.28",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        body,
        "site",
        {
            "name": "success_target",
            "pos": "0 0 0",
            "size": f"{radius_m:.6f}",
            "rgba": "0 0.8 0.2 0.18",
        },
    )


def add_tissue_box(root: ET.Element, world_xy: np.ndarray, table_surface_z: float, size_m: float = 0.12) -> None:
    remove_body(root, "tissue_box")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF missing worldbody")

    half_x = size_m * 0.5
    half_y = size_m * 0.43
    half_z = size_m * 0.36
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "tissue_box",
            "pos": fmt_vec(np.array([world_xy[0], world_xy[1], table_surface_z], dtype=float)),
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "tissue_box_shell",
            "type": "box",
            "pos": f"0 0 {half_z:.6f}",
            "size": f"{half_x:.6f} {half_y:.6f} {half_z:.6f}",
            "rgba": "0.32 0.76 0.38 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "tissue_box_opening",
            "type": "box",
            "pos": f"0 0 {2 * half_z + 0.003:.6f}",
            "size": f"{half_x * 0.46:.6f} {half_y * 0.34:.6f} 0.004",
            "rgba": "0.04 0.045 0.04 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    for index, (x, y) in enumerate(((-0.55, -0.45), (0.55, -0.12), (-0.15, 0.46), (0.28, 0.28))):
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"tissue_box_flower_{index}",
                "type": "cylinder",
                "pos": f"{x * half_x:.6f} {-half_y - 0.002:.6f} {(0.78 + 0.22 * y) * half_z:.6f}",
                "euler": "90 0 0",
                "size": f"{size_m * 0.055:.6f} 0.002",
                "rgba": "0.95 0.38 0.72 1",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "tissue_box_label",
            "type": "box",
            "pos": f"0 {-half_y - 0.003:.6f} {half_z:.6f}",
            "size": f"{half_x * 0.46:.6f} 0.002 {half_z * 0.18:.6f}",
            "rgba": "0.95 0.92 0.66 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )


def add_success_screen(root: ET.Element) -> None:
    if find_body(root, "success_screen") is not None:
        return

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF missing worldbody")

    screen = ET.SubElement(worldbody, "body", {"name": "success_screen", "pos": "0.31 -0.38 0.94"})
    ET.SubElement(
        screen,
        "geom",
        {
            "name": "success_screen_back",
            "type": "box",
            "pos": "0 0 0",
            "size": "0.13 0.01 0.06",
            "rgba": "0.08 0.1 0.09 0",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        screen,
        "geom",
        {
            "name": "success_screen_top",
            "type": "box",
            "pos": "0 0.012 0.018",
            "size": "0.085 0.006 0.01",
            "rgba": "0.08 0.9 0.28 0",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        screen,
        "geom",
        {
            "name": "success_screen_bottom",
            "type": "box",
            "pos": "0 0.012 -0.018",
            "size": "0.085 0.006 0.01",
            "rgba": "0.08 0.9 0.28 0",
            "contype": "0",
            "conaffinity": "0",
        },
    )


def apply_scene_options(root: ET.Element, spec: dict[str, Any]) -> None:
    options = spec.get("scene_options", {})
    if options.get("minimal_photo_scene"):
        for body_name in (
            "background_wall",
            "floor_lab_details",
            "jetarm_reference_mount",
            "jetson_box",
            "background_shelf",
            "background_chair",
            "tabletop_background_objects",
            "bench_side_equipment",
            "task_status_panel",
        ):
            remove_body(root, body_name)

        floor = find_geom(root, "floor")
        if floor is not None:
            floor.set("rgba", "0.015 0.016 0.017 1")
            floor.attrib.pop("material", None)

        tabletop = find_geom(root, "tabletop")
        if tabletop is not None:
            tabletop.set("rgba", "0.025 0.027 0.028 1")
            tabletop.attrib.pop("material", None)

        board = find_geom(root, "calibration_mat")
        if board is not None:
            board.set("pos", "0 0 0.767")
            board.set("size", "0.36 0.28 0.002")
            board.set("rgba", "0.0 0.62 0.86 1")

        target_marker = find_geom(root, "target_marker")
        if target_marker is not None:
            target_marker.set("rgba", "0.95 0.94 0.9 0")
            target_marker.attrib.pop("material", None)

        success_site = find_site(root, "success_target")
        if success_site is not None:
            success_site.set("rgba", "0 0 0 0")

        for geom_name in ("front_edge", "back_edge", "left_edge", "right_edge"):
            geom = find_geom(root, geom_name)
            if geom is not None:
                geom.set("rgba", "0.02 0.02 0.022 1")
                geom.attrib.pop("material", None)

        overview = find_camera(root, "overview")
        if overview is not None:
            overview.set("pos", "0 -0.02 1.65")
            overview.set("xyaxes", "1 0 0 0 1 0")
            overview.set("fovy", "43")

    object_shapes = options.get("object_shapes", {})
    for item in spec.get("objects", []):
        body_name = item.get("mjcf_body")
        if not body_name:
            continue
        if object_shapes.get(item.get("id")) == "orange_sphere":
            body = body_by_name(root, body_name)
            replace_body_with_orange(body, float(item.get("estimated_radius_m", 0.035)))
        elif object_shapes.get(item.get("id")) == "cube":
            body = body_by_name(root, body_name)
            replace_body_with_cube(body, float(item.get("estimated_radius_m", 0.025)))
        elif object_shapes.get(item.get("id")) == "target_square":
            body = body_by_name(root, body_name)
            replace_body_with_target_square(body, float(item.get("estimated_radius_m", 0.06)))
        elif object_shapes.get(item.get("id")) == "tissue_sheet":
            body = body_by_name(root, body_name)
            replace_body_with_tissue_sheet(body, float(item.get("estimated_radius_m", 0.035)))
        elif object_shapes.get(item.get("id")) == "pull_target":
            body = body_by_name(root, body_name)
            replace_body_with_pull_target(body, float(item.get("estimated_radius_m", 0.065)))


def fmt_vec(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.6f}".rstrip("0").rstrip(".") for value in values)


def source_frame_path(spec: dict[str, Any], spec_dir: Path) -> Path | None:
    source = spec.get("source", {})
    source_type = source.get("type", "image")
    if source_type == "image":
        return resolve_path(source.get("image_path"), spec_dir)
    if source_type == "video":
        frame_path = resolve_path(source.get("representative_frame_path"), spec_dir)
        if frame_path is not None:
            return frame_path
        return None
    raise ValueError(f"Unsupported source.type: {source_type}")


def calibration_points(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    calibration = spec["calibration"]
    pixel_corners = calibration["table_corners_pixels"]
    world_corners = calibration["table_corners_world"]
    pixels = np.array([parse_xy(pixel_corners[name], f"table_corners_pixels.{name}") for name in CORNER_ORDER])
    world = np.array([parse_xy(world_corners[name], f"table_corners_world.{name}") for name in CORNER_ORDER])
    return pixels, world


def update_scene_tree(
    tree: ET.ElementTree,
    spec: dict[str, Any],
    homography: np.ndarray,
) -> dict[str, Any]:
    root = tree.getroot()
    calibration = spec["calibration"]
    table_surface_z = float(calibration.get("table_surface_z", 0.76))
    generated_objects = []
    tissue_box = spec.get("scene_options", {}).get("tissue_box")
    tissue_box_pixel = None
    tissue_box_xy = None
    if tissue_box and tissue_box.get("center_pixel"):
        tissue_box_pixel = parse_xy(tissue_box["center_pixel"], "scene_options.tissue_box.center_pixel")
        tissue_box_xy = apply_homography(homography, tissue_box_pixel)

    robot = spec.get("robot", {})
    robot_base = calibration.get("robot_base_world_xyz", calibration.get("fanuc_base_world_xyz"))
    robot_base_body = robot.get("base_body", "fanuc_cr7ial")
    if robot_base is not None:
        robot_body = body_by_name(root, robot_base_body)
        robot_body.set("pos", fmt_vec(parse_xyz(robot_base, "robot_base_world_xyz")))

    for item in spec.get("objects", []):
        body_name = item.get("mjcf_body")
        if not body_name:
            continue

        center_pixel = parse_xy(item["center_pixel"], f"objects.{item.get('id', body_name)}.center_pixel")
        world_xy = apply_homography(homography, center_pixel)
        body_xy = world_xy
        if spec.get("task", {}).get("type") == "tissue_pull" and item.get("role") == "pick_object" and tissue_box_xy is not None:
            body_xy = tissue_box_xy
        z_offset = float(item.get("body_origin_z_above_table_m", 0.0))
        body_pos = np.array([body_xy[0], body_xy[1], table_surface_z + z_offset], dtype=float)

        body = body_by_name(root, body_name)
        body.set("pos", fmt_vec(body_pos))

        generated_objects.append(
            {
                "id": item.get("id", body_name),
                "label": item.get("label", body_name),
                "role": item.get("role"),
                "mjcf_body": body_name,
                "center_pixel": center_pixel.round(3).tolist(),
                "world_xy_m": world_xy.round(6).tolist(),
                "body_world_xy_m": body_xy.round(6).tolist(),
                "body_pos_m": body_pos.round(6).tolist(),
                "estimated_radius_m": item.get("estimated_radius_m"),
            }
        )

    generated_tissue_box = None
    if tissue_box is not None and tissue_box_pixel is not None and tissue_box_xy is not None:
        add_tissue_box(root, tissue_box_xy, table_surface_z, float(tissue_box.get("size_m", 0.12)))
        generated_tissue_box = {
            "center_pixel": tissue_box_pixel.round(3).tolist(),
            "world_xy_m": tissue_box_xy.round(6).tolist(),
            "size_m": float(tissue_box.get("size_m", 0.12)),
        }

    return {
        "table_surface_z": table_surface_z,
        "homography_pixel_to_world": homography.round(10).tolist(),
        "objects": generated_objects,
        "tissue_box": generated_tissue_box,
        "task": spec.get("task", {}),
    }


def write_generated_scene(spec_path: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    spec = json.loads(spec_path.read_text())
    spec_dir = spec_path.parent
    outputs = spec.get("outputs", {})
    scene_path = args.scene_xml or resolve_path(outputs.get("scene_xml"), ROOT) or ROOT / "outputs/generated/fanuc_scene.xml"
    metadata_path = (
        args.metadata_json
        or resolve_path(outputs.get("metadata_json"), ROOT)
        or ROOT / "outputs/generated/fanuc_scene_metadata.json"
    )

    base_scene = args.base_scene or DEFAULT_BASE_SCENE
    pixel_points, world_points = calibration_points(spec)
    homography = homography_from_points(pixel_points, world_points)

    tree = ET.parse(base_scene)
    metadata = update_scene_tree(tree, spec, homography)
    apply_scene_options(tree.getroot(), spec)
    metadata.update(
        {
            "source_spec": str(spec_path),
            "source_frame": str(source_frame_path(spec, spec_dir)) if source_frame_path(spec, spec_dir) else None,
            "base_scene": str(base_scene),
            "generated_scene": str(scene_path),
        }
    )

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(scene_path, encoding="unicode", xml_declaration=False)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Generated scene: {scene_path}", flush=True)
    print(f"Generated metadata: {metadata_path}", flush=True)
    for item in metadata["objects"]:
        print(f"  {item['id']}: pixel={item['center_pixel']} -> body_pos_m={item['body_pos_m']}", flush=True)
    return scene_path, metadata_path


def run_headless(scene_path: Path, spec_path: Path, args: argparse.Namespace) -> None:
    spec = json.loads(spec_path.read_text())
    outputs = spec.get("outputs", {})
    snapshot = args.snapshot or resolve_path(outputs.get("snapshot_png"), ROOT) or ROOT / "outputs/generated/fanuc_scene_snapshot.png"
    result = args.result_json or resolve_path(outputs.get("result_json"), ROOT) or ROOT / "outputs/generated/fanuc_scene_result.json"
    runner_script = args.runner_script if args.runner_script.is_absolute() else ROOT / args.runner_script
    command = [
        sys.executable,
        str(runner_script),
        "--headless",
        "--scene",
        str(scene_path),
        "--snapshot",
        str(snapshot),
        "--result-json",
        str(result),
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FANUC MuJoCo scenario from annotated image/video media.")
    parser.add_argument("--spec", type=Path, required=True, help="Photo or video scenario JSON spec.")
    parser.add_argument("--base-scene", type=Path, default=DEFAULT_BASE_SCENE, help="Base FANUC MJCF scene to modify.")
    parser.add_argument("--scene-xml", type=Path, help="Override generated MJCF output path.")
    parser.add_argument("--metadata-json", type=Path, help="Override generated metadata output path.")
    parser.add_argument("--run-headless", action="store_true", help="Run the generated scene through the FANUC sim.")
    parser.add_argument(
        "--runner-script",
        type=Path,
        default=ROOT / "sim" / "run_pick_place_demo.py",
        help="Simulation runner script used with --run-headless.",
    )
    parser.add_argument("--snapshot", type=Path, help="Override snapshot output path for --run-headless.")
    parser.add_argument("--result-json", type=Path, help="Override result JSON output path for --run-headless.")
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    scene_path, _ = write_generated_scene(spec_path, args)
    if args.run_headless:
        run_headless(scene_path, spec_path, args)


if __name__ == "__main__":
    main()
