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
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "outputs" / "generated" / "dual_fanuc_metal_buffing_scene.xml"
DEFAULT_SNAPSHOT = ROOT / "outputs" / "generated" / "dual_fanuc_metal_buffing_snapshot.png"
DEFAULT_REVIEW = ROOT / "outputs" / "generated" / "dual_fanuc_metal_buffing_review_sheet.png"
DEFAULT_RESULT = ROOT / "outputs" / "generated" / "dual_fanuc_metal_buffing_result.json"

ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")
PASS_GEOMS = ("buff_pass_1", "buff_pass_2", "buff_pass_3", "buff_pass_4")
STATUS_GEOMS = {
    "clamp": "status_clamp_light",
    "buff": "status_buff_light",
    "success": "status_success_light",
}
STATUS_OFF = np.array([0.08, 0.09, 0.085, 1.0])
STATUS_ON = {
    "clamp": np.array([0.1, 0.55, 1.0, 1.0]),
    "buff": np.array([1.0, 0.72, 0.08, 1.0]),
    "success": np.array([0.08, 0.95, 0.26, 1.0]),
}


@dataclass(frozen=True)
class RobotHandles:
    prefix: str
    qpos: list[int]
    dof: list[int]
    ranges: np.ndarray
    tool_site: int


@dataclass(frozen=True)
class WorkcellHandles:
    holder: RobotHandles
    buffer: RobotHandles


@dataclass(frozen=True)
class DemoState:
    holder_q: np.ndarray
    buffer_q: np.ndarray
    holder_clamped: bool
    buffing: bool
    completed_passes: int
    stage: str


@dataclass(frozen=True)
class Segment:
    duration: float
    start: DemoState
    end: DemoState


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def mix_bool(a: bool, b: bool, t: float) -> bool:
    return b if t >= 0.5 else a


def mix_state(a: DemoState, b: DemoState, t: float) -> DemoState:
    s = smoothstep(t)
    return DemoState(
        holder_q=a.holder_q + (b.holder_q - a.holder_q) * s,
        buffer_q=a.buffer_q + (b.buffer_q - a.buffer_q) * s,
        holder_clamped=mix_bool(a.holder_clamped, b.holder_clamped, t),
        buffing=mix_bool(a.buffing, b.buffing, t),
        completed_passes=max(a.completed_passes, int(round(a.completed_passes + (b.completed_passes - a.completed_passes) * s))),
        stage=b.stage if t >= 0.5 else a.stage,
    )


def named_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type.name}: {name}")
    return obj_id


def euler_z(yaw_deg: float) -> str:
    return f"0 0 {yaw_deg:.3f}"


def fanuc_robot_xml(prefix: str, pos: tuple[float, float, float], yaw_deg: float, tool: str) -> str:
    label_panel = (
        f'<geom name="{prefix}_fanuc_label_panel" type="box" pos="0.18 -0.052 0.02" size="0.09 0.004 0.024" rgba="0.78 0.92 0.62 1" contype="0" conaffinity="0"/>'
        f'<geom name="{prefix}_fanuc_label_red" type="box" pos="0.12 -0.057 0.02" size="0.018 0.002 0.01" rgba="0.75 0.04 0.03 1" contype="0" conaffinity="0"/>'
        f'<geom name="{prefix}_fanuc_label_dark" type="box" pos="0.19 -0.057 0.02" size="0.048 0.002 0.006" rgba="0.02 0.025 0.025 1" contype="0" conaffinity="0"/>'
    )
    if tool == "holder":
        tool_xml = f"""
                  <body name="{prefix}_pneumatic_holder" pos="0.088 0 0">
                    <geom name="{prefix}_holder_manifold" type="box" pos="0.035 0 0" size="0.05 0.07 0.022" rgba="0.08 0.1 0.11 1" contype="0" conaffinity="0"/>
                    <geom name="{prefix}_vacuum_bar" type="box" pos="0.095 0 0" size="0.018 0.12 0.014" material="metal_mat" contype="0" conaffinity="0"/>
                    <geom name="{prefix}_suction_cup_l" type="cylinder" pos="0.12 -0.062 -0.006" euler="0 90 0" size="0.032 0.011" material="rubber_mat" contype="0" conaffinity="0"/>
                    <geom name="{prefix}_suction_cup_r" type="cylinder" pos="0.12 0.062 -0.006" euler="0 90 0" size="0.032 0.011" material="rubber_mat" contype="0" conaffinity="0"/>
                    <geom name="{prefix}_pneumatic_line" type="capsule" fromto="-0.02 0.035 0.02 0.09 0.11 0.05" size="0.006" material="hose_mat" contype="0" conaffinity="0"/>
                    <site name="{prefix}_tool_site" pos="0.135 0 0" size="0.011" rgba="0.05 0.9 0.2 1"/>
                  </body>
"""
    else:
        tool_xml = f"""
                  <body name="{prefix}_buffing_tool" pos="0.088 0 0">
                    <geom name="{prefix}_buff_motor" type="cylinder" pos="0.042 0 0" euler="0 90 0" size="0.036 0.055" rgba="0.08 0.1 0.12 1" contype="0" conaffinity="0"/>
                    <geom name="{prefix}_buff_handle" type="capsule" fromto="-0.018 0 0.038 0.052 0 0.038" size="0.012" material="hose_mat" contype="0" conaffinity="0"/>
                    <geom name="{prefix}_buff_wheel" type="cylinder" pos="0.102 0 -0.032" euler="90 0 0" size="0.055 0.018" rgba="0.12 0.18 0.24 1" contype="0" conaffinity="0"/>
                    <geom name="{prefix}_buff_pad" type="cylinder" pos="0.103 0 -0.055" euler="90 0 0" size="0.062 0.009" rgba="0.88 0.82 0.64 1" contype="0" conaffinity="0"/>
                    <geom name="{prefix}_buff_motion_blur" type="cylinder" pos="0.103 0 -0.055" euler="90 0 0" size="0.072 0.002" rgba="1 0.78 0.22 0.0" contype="0" conaffinity="0"/>
                    <site name="{prefix}_tool_site" pos="0.103 0 -0.058" size="0.011" rgba="1 0.72 0.08 1"/>
                  </body>
"""

    return f"""
    <body name="{prefix}_fanuc_cr7ial" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" euler="{euler_z(yaw_deg)}">
      <geom name="{prefix}_pedestal" type="cylinder" pos="0 0 -0.22" size="0.125 0.22" rgba="0.16 0.17 0.17 1"/>
      <geom name="{prefix}_base_plate" type="cylinder" pos="0 0 0.015" size="0.17 0.03" material="fanuc_joint_gray" contype="0" conaffinity="0"/>
      <geom name="{prefix}_base_green" type="cylinder" pos="0 0 0.095" size="0.115 0.075" material="fanuc_green" contype="0" conaffinity="0"/>
      <geom name="{prefix}_base_black_ring" type="cylinder" pos="0 0 0.178" size="0.118 0.012" material="fanuc_dark" contype="0" conaffinity="0"/>
      <geom name="{prefix}_base_label" type="box" pos="0 -0.108 -0.07" size="0.055 0.004 0.035" rgba="0.96 0.82 0.14 1" contype="0" conaffinity="0"/>
      <geom name="{prefix}_mount_rail_l" type="box" pos="0 -0.205 -0.19" size="0.23 0.018 0.018" material="extrusion_mat" contype="0" conaffinity="0"/>
      <geom name="{prefix}_mount_rail_r" type="box" pos="0 0.205 -0.19" size="0.23 0.018 0.018" material="extrusion_mat" contype="0" conaffinity="0"/>

      <body name="{prefix}_joint_1" pos="0 0 0.16">
        <joint name="{prefix}_j1" type="hinge" axis="0 0 1" range="-170 170" limited="true" damping="35" armature="0.05"/>
        <geom name="{prefix}_j1_cover" type="sphere" pos="0 0 0.03" size="0.12" material="fanuc_green" contype="0" conaffinity="0"/>
        <geom name="{prefix}_j1_flat_front" type="box" pos="0.072 -0.028 0.025" size="0.034 0.008 0.052" rgba="0.75 0.82 0.72 1" contype="0" conaffinity="0"/>

        <body name="{prefix}_joint_2" pos="0 0 0.12">
          <joint name="{prefix}_j2" type="hinge" axis="0 1 0" range="-95 145" limited="true" damping="38" armature="0.05"/>
          <geom name="{prefix}_j2_cover" type="cylinder" pos="0 0 0" euler="90 0 0" size="0.105 0.08" material="fanuc_green" contype="0" conaffinity="0"/>
          <geom name="{prefix}_j2_dark_cap_l" type="cylinder" pos="0 0.086 0" euler="90 0 0" size="0.082 0.006" material="fanuc_dark" contype="0" conaffinity="0"/>
          <geom name="{prefix}_j2_dark_cap_r" type="cylinder" pos="0 -0.086 0" euler="90 0 0" size="0.082 0.006" material="fanuc_dark" contype="0" conaffinity="0"/>
          <geom name="{prefix}_upper_arm" type="capsule" fromto="0 0 0.03 0 0 0.405" size="0.058" material="fanuc_green" contype="0" conaffinity="0"/>

          <body name="{prefix}_joint_3" pos="0 0 0.405">
            <joint name="{prefix}_j3" type="hinge" axis="0 1 0" range="-160 170" limited="true" damping="35" armature="0.045"/>
            <geom name="{prefix}_j3_cover" type="sphere" pos="0 0 0" size="0.1" material="fanuc_green" contype="0" conaffinity="0"/>
            <geom name="{prefix}_forearm" type="capsule" fromto="0 0 0 0.355 0 0" size="0.05" material="fanuc_green" contype="0" conaffinity="0"/>
            <geom name="{prefix}_forearm_side_panel" type="box" pos="0.19 -0.055 0.018" size="0.15 0.009 0.034" material="fanuc_green" contype="0" conaffinity="0"/>
            {label_panel}

            <body name="{prefix}_joint_4" pos="0.355 0 0">
              <joint name="{prefix}_j4" type="hinge" axis="1 0 0" range="-190 190" limited="true" damping="20" armature="0.025"/>
              <geom name="{prefix}_j4_cover" type="cylinder" pos="0 0 0" euler="0 90 0" size="0.072 0.085" material="fanuc_green" contype="0" conaffinity="0"/>
              <geom name="{prefix}_j4_gray_band" type="cylinder" pos="0.004 0 0" euler="0 90 0" size="0.076 0.01" material="fanuc_joint_gray" contype="0" conaffinity="0"/>
              <geom name="{prefix}_wrist_link_1" type="capsule" fromto="0 0 0 0.13 0 0" size="0.04" material="fanuc_green" contype="0" conaffinity="0"/>

              <body name="{prefix}_joint_5" pos="0.13 0 0">
                <joint name="{prefix}_j5" type="hinge" axis="0 1 0" range="-140 140" limited="true" damping="20" armature="0.025"/>
                <geom name="{prefix}_j5_cover" type="sphere" pos="0 0 0" size="0.068" material="fanuc_green" contype="0" conaffinity="0"/>
                <geom name="{prefix}_wrist_link_2" type="capsule" fromto="0 0 0 0.095 0 0" size="0.035" material="fanuc_green" contype="0" conaffinity="0"/>

                <body name="{prefix}_joint_6" pos="0.095 0 0">
                  <joint name="{prefix}_j6" type="hinge" axis="1 0 0" range="-360 360" limited="true" damping="14" armature="0.02"/>
                  <geom name="{prefix}_j6_flange" type="cylinder" pos="0 0 0" euler="0 90 0" size="0.052 0.035" material="fanuc_joint_gray" contype="0" conaffinity="0"/>
{tool_xml}
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
"""


def ribbed_wall_xml() -> str:
    ribs = []
    for idx in range(13):
        z = 0.12 + idx * 0.105
        ribs.append(
            f'<geom name="wall_rib_{idx}" type="box" pos="0 -0.045 {z:.3f}" size="2.4 0.006 0.006" rgba="0.42 0.44 0.43 1" contype="0" conaffinity="0"/>'
        )
    return "\n      ".join(ribs)


def sheet_holes_xml() -> str:
    holes = []
    index = 0
    for x in (-0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24):
        for y in (-0.17, -0.09, -0.01, 0.07, 0.15):
            if abs(x) < 0.02 and abs(y) < 0.04:
                continue
            holes.append(
                f'<geom name="sheet_hole_{index}" type="cylinder" pos="{x:.3f} {y:.3f} 0.010" size="0.008 0.0015" rgba="0.03 0.035 0.04 1" contype="0" conaffinity="0"/>'
            )
            index += 1
    holes.extend(
        [
            '<geom name="sheet_slot_1" type="box" pos="-0.31 -0.16 0.011" size="0.035 0.012 0.0015" rgba="0.03 0.035 0.04 1" contype="0" conaffinity="0"/>',
            '<geom name="sheet_slot_2" type="box" pos="0.31 0.16 0.011" size="0.035 0.012 0.0015" rgba="0.03 0.035 0.04 1" contype="0" conaffinity="0"/>',
            '<geom name="sheet_edge_notch" type="box" pos="0.0 0.205 0.011" size="0.07 0.012 0.0015" rgba="0.03 0.035 0.04 1" contype="0" conaffinity="0"/>',
        ]
    )
    return "\n        ".join(holes)


def scene_xml() -> str:
    holder = fanuc_robot_xml("holder", (-0.86, -0.22, 0.765), 6.0, "holder")
    buffer = fanuc_robot_xml("buffer", (0.86, -0.22, 0.765), 174.0, "buffer")
    return f"""<mujoco model="dual_fanuc_metal_buffing">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81"/>

  <visual>
    <headlight diffuse="0.62 0.62 0.6" ambient="0.25 0.25 0.25" specular="0.18 0.18 0.18"/>
    <rgba haze="0.68 0.78 0.86 1"/>
    <global azimuth="145" elevation="-26" offwidth="1600" offheight="1000"/>
  </visual>

  <asset>
    <texture name="floor_grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.18 0.18 0.17" rgb2="0.24 0.24 0.23"/>
    <material name="floor_mat" texture="floor_grid" texrepeat="5 5" reflectance="0"/>
    <material name="fanuc_green" rgba="0.48 0.83 0.22 1" specular="0.28" shininess="0.45"/>
    <material name="fanuc_dark" rgba="0.035 0.045 0.04 1" specular="0.15" shininess="0.3"/>
    <material name="fanuc_joint_gray" rgba="0.58 0.62 0.6 1" specular="0.25" shininess="0.45"/>
    <material name="metal_mat" rgba="0.62 0.66 0.67 1" specular="0.45" shininess="0.75"/>
    <material name="brushed_metal" rgba="0.72 0.76 0.76 1" specular="0.72" shininess="0.9"/>
    <material name="polished_path" rgba="0.95 0.98 0.97 0.15" specular="0.95" shininess="1"/>
    <material name="rubber_mat" rgba="0.035 0.038 0.04 1"/>
    <material name="hose_mat" rgba="0.015 0.015 0.018 1"/>
    <material name="table_mat" rgba="0.53 0.43 0.32 1"/>
    <material name="cardboard_mat" rgba="0.64 0.43 0.22 1"/>
    <material name="pallet_mat" rgba="0.36 0.25 0.16 1"/>
    <material name="wall_mat" rgba="0.55 0.57 0.55 1"/>
    <material name="shelf_blue" rgba="0.18 0.42 0.5 1"/>
    <material name="shelf_orange" rgba="0.65 0.28 0.12 1"/>
    <material name="extrusion_mat" rgba="0.66 0.68 0.66 1" specular="0.35" shininess="0.55"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="4 4 0.05" material="floor_mat"/>
    <light name="key_light" pos="-1.8 -2.3 3.2" dir="0.55 0.55 -1" diffuse="0.95 0.92 0.84" specular="0.35 0.35 0.3"/>
    <light name="shop_fill" pos="1.6 1.6 2.4" dir="-0.5 -0.45 -1" diffuse="0.45 0.48 0.5"/>
    <camera name="overview" pos="2.6 -3.6 2.35" xyaxes="0.82 0.57 0 -0.29 0.41 0.86" fovy="52"/>
    <camera name="process_close" pos="1.1 -1.9 1.45" xyaxes="0.88 0.48 0 -0.28 0.51 0.81" fovy="44"/>

    <body name="corrugated_shop_wall" pos="0 1.55 0.68">
      <geom name="wall_panel" type="box" pos="0 0 0.42" size="2.45 0.035 0.92" material="wall_mat" contype="0" conaffinity="0"/>
      {ribbed_wall_xml()}
      <geom name="steel_beam_low" type="box" pos="0 -0.055 1.02" size="2.45 0.035 0.035" rgba="0.33 0.19 0.13 1" contype="0" conaffinity="0"/>
      <geom name="steel_beam_high" type="box" pos="0 -0.055 1.42" size="2.45 0.04 0.045" rgba="0.27 0.16 0.11 1" contype="0" conaffinity="0"/>
      <geom name="overhead_light" type="box" pos="-0.55 -0.09 1.62" size="0.22 0.012 0.035" rgba="1 0.96 0.78 1" contype="0" conaffinity="0"/>
    </body>

    <body name="worktable" pos="0 0 0">
      <geom name="tabletop" type="box" pos="0 0 0.735" size="0.9 0.55 0.035" material="table_mat" friction="0.9 0.02 0.001"/>
      <geom name="table_leg_fl" type="box" pos="-0.78 -0.45 0.36" size="0.035 0.035 0.36" rgba="0.36 0.36 0.34 1"/>
      <geom name="table_leg_fr" type="box" pos="0.78 -0.45 0.36" size="0.035 0.035 0.36" rgba="0.36 0.36 0.34 1"/>
      <geom name="table_leg_bl" type="box" pos="-0.78 0.45 0.36" size="0.035 0.035 0.36" rgba="0.36 0.36 0.34 1"/>
      <geom name="table_leg_br" type="box" pos="0.78 0.45 0.36" size="0.035 0.035 0.36" rgba="0.36 0.36 0.34 1"/>
      <geom name="fixture_rail_front" type="box" pos="0 -0.31 0.795" size="0.62 0.018 0.018" material="extrusion_mat" contype="0" conaffinity="0"/>
      <geom name="fixture_rail_back" type="box" pos="0 0.31 0.795" size="0.62 0.018 0.018" material="extrusion_mat" contype="0" conaffinity="0"/>
      <geom name="fixture_stop_l" type="box" pos="-0.42 0 0.82" size="0.016 0.2 0.045" material="extrusion_mat" contype="0" conaffinity="0"/>
      <geom name="fixture_stop_r" type="box" pos="0.42 0 0.82" size="0.016 0.2 0.045" material="extrusion_mat" contype="0" conaffinity="0"/>

      <body name="held_metal_sheet" pos="0 0 0.815">
        <geom name="sheet_plate" type="box" pos="0 0 0" size="0.46 0.23 0.006" material="brushed_metal" friction="0.65 0.02 0.001"/>
        {sheet_holes_xml()}
        <geom name="buff_pass_1" type="box" pos="-0.08 -0.145 0.0115" size="0.31 0.018 0.0015" material="polished_path" contype="0" conaffinity="0"/>
        <geom name="buff_pass_2" type="box" pos="0.08 -0.055 0.012" size="0.31 0.018 0.0015" material="polished_path" contype="0" conaffinity="0"/>
        <geom name="buff_pass_3" type="box" pos="-0.08 0.035 0.0125" size="0.31 0.018 0.0015" material="polished_path" contype="0" conaffinity="0"/>
        <geom name="buff_pass_4" type="box" pos="0.08 0.125 0.013" size="0.31 0.018 0.0015" material="polished_path" contype="0" conaffinity="0"/>
      </body>
    </body>

    <body name="sheet_stack_left" pos="-1.45 0.35 0">
      <geom name="pallet_left" type="box" pos="0 0 0.08" size="0.45 0.32 0.08" material="pallet_mat" contype="0" conaffinity="0"/>
      <geom name="stacked_sheet_l_1" type="box" pos="0 0 0.18" size="0.42 0.29 0.012" material="metal_mat" contype="0" conaffinity="0"/>
      <geom name="stacked_sheet_l_2" type="box" pos="0.015 -0.01 0.205" size="0.42 0.29 0.012" material="metal_mat" contype="0" conaffinity="0"/>
      <geom name="stacked_sheet_l_3" type="box" pos="-0.012 0.012 0.23" size="0.42 0.29 0.012" material="metal_mat" contype="0" conaffinity="0"/>
      <geom name="orange_job_tag_left" type="box" pos="0.12 -0.09 0.245" size="0.07 0.045 0.002" rgba="1.0 0.48 0.12 1" contype="0" conaffinity="0"/>
    </body>

    <body name="background_shelving" pos="1.45 0.7 0">
      <geom name="shelf_post_l" type="box" pos="-0.38 0 0.56" size="0.025 0.025 0.56" material="shelf_blue" contype="0" conaffinity="0"/>
      <geom name="shelf_post_r" type="box" pos="0.38 0 0.56" size="0.025 0.025 0.56" material="shelf_blue" contype="0" conaffinity="0"/>
      <geom name="shelf_beam_1" type="box" pos="0 0 0.25" size="0.42 0.03 0.025" material="shelf_orange" contype="0" conaffinity="0"/>
      <geom name="shelf_beam_2" type="box" pos="0 0 0.72" size="0.42 0.03 0.025" material="shelf_orange" contype="0" conaffinity="0"/>
      <geom name="shelf_beam_3" type="box" pos="0 0 1.1" size="0.42 0.03 0.025" material="shelf_orange" contype="0" conaffinity="0"/>
      <geom name="shop_box" type="box" pos="-0.12 -0.03 0.34" size="0.12 0.1 0.07" material="cardboard_mat" contype="0" conaffinity="0"/>
      <geom name="bucket" type="cylinder" pos="0.2 -0.02 0.36" size="0.055 0.08" rgba="0.16 0.18 0.5 1" contype="0" conaffinity="0"/>
    </body>

    <body name="status_panel" pos="0 -0.72 0.92">
      <geom name="status_panel_back" type="box" pos="0 0 0" size="0.28 0.015 0.07" rgba="0.06 0.065 0.06 1" contype="0" conaffinity="0"/>
      <geom name="status_clamp_light" type="sphere" pos="-0.12 -0.02 0.015" size="0.025" rgba="0.08 0.09 0.085 1" contype="0" conaffinity="0"/>
      <geom name="status_buff_light" type="sphere" pos="0 -0.02 0.015" size="0.025" rgba="0.08 0.09 0.085 1" contype="0" conaffinity="0"/>
      <geom name="status_success_light" type="sphere" pos="0.12 -0.02 0.015" size="0.025" rgba="0.08 0.09 0.085 1" contype="0" conaffinity="0"/>
    </body>

{holder}
{buffer}
  </worldbody>
</mujoco>
"""


def write_scene(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scene_xml())
    return path


def robot_handles(model: mujoco.MjModel, prefix: str) -> RobotHandles:
    joint_ids = [named_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{name}") for name in ARM_JOINTS]
    return RobotHandles(
        prefix=prefix,
        qpos=[int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids],
        dof=[int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids],
        ranges=np.array([model.jnt_range[joint_id] for joint_id in joint_ids]),
        tool_site=named_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_tool_site"),
    )


def get_handles(model: mujoco.MjModel) -> WorkcellHandles:
    return WorkcellHandles(holder=robot_handles(model, "holder"), buffer=robot_handles(model, "buffer"))


def set_robot_q(data: mujoco.MjData, handles: RobotHandles, q: np.ndarray) -> None:
    for value, qpos in zip(q, handles.qpos):
        data.qpos[qpos] = value


def set_state(model: mujoco.MjModel, data: mujoco.MjData, handles: WorkcellHandles, state: DemoState, sim_time: float) -> None:
    holder_q = state.holder_q.copy()
    buffer_q = state.buffer_q.copy()
    if state.buffing:
        buffer_q[5] = ((buffer_q[5] + sim_time * 18.0 + math.pi) % (2.0 * math.pi)) - math.pi

    set_robot_q(data, handles.holder, holder_q)
    set_robot_q(data, handles.buffer, buffer_q)
    mujoco.mj_forward(model, data)
    update_visual_state(model, state)


def tool_pos(data: mujoco.MjData, handles: RobotHandles) -> np.ndarray:
    return data.site_xpos[handles.tool_site].copy()


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    target: np.ndarray,
    seed: np.ndarray,
    max_iters: int = 260,
) -> np.ndarray:
    q = seed.copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(max_iters):
        set_robot_q(data, handles, q)
        mujoco.mj_forward(model, data)
        error = target - tool_pos(data, handles)
        if np.linalg.norm(error) < 0.008:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, handles.tool_site)
        j = jacp[:, handles.dof]
        damping = 1e-3
        dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(3), error)
        step_norm = np.linalg.norm(dq)
        if step_norm > 0.05:
            dq *= 0.05 / step_norm
        q += dq
        q = np.clip(q, handles.ranges[:, 0], handles.ranges[:, 1])
    return q


def solve_waypoints(model: mujoco.MjModel, data: mujoco.MjData, handles: WorkcellHandles) -> dict[str, dict[str, np.ndarray]]:
    holder_seed = np.array([0.10, 0.82, -1.04, 0.0, 0.48, 0.0])
    buffer_seed = np.array([-0.10, 0.82, -1.04, 0.0, 0.48, 0.0])
    holder_targets = {
        "home": np.array([-0.55, -0.42, 1.18]),
        "approach": np.array([-0.28, -0.16, 0.98]),
        "clamp": np.array([-0.31, -0.02, 0.835]),
        "hold": np.array([-0.31, -0.02, 0.835]),
    }
    buffer_targets = {
        "home": np.array([0.55, -0.42, 1.18]),
        "approach": np.array([0.32, -0.22, 1.0]),
        "pass_1_start": np.array([0.28, -0.145, 0.866]),
        "pass_1_end": np.array([-0.26, -0.145, 0.866]),
        "pass_2_end": np.array([0.26, -0.055, 0.866]),
        "pass_3_end": np.array([-0.26, 0.035, 0.866]),
        "pass_4_end": np.array([0.26, 0.125, 0.866]),
        "retract": np.array([0.34, 0.22, 1.02]),
    }
    solved_holder: dict[str, np.ndarray] = {}
    for name, target in holder_targets.items():
        holder_seed = solve_ik(model, data, handles.holder, target, holder_seed)
        solved_holder[name] = holder_seed.copy()

    solved_buffer: dict[str, np.ndarray] = {}
    for name, target in buffer_targets.items():
        buffer_seed = solve_ik(model, data, handles.buffer, target, buffer_seed)
        solved_buffer[name] = buffer_seed.copy()

    return {"holder": solved_holder, "buffer": solved_buffer}


def make_segments(model: mujoco.MjModel, data: mujoco.MjData, handles: WorkcellHandles) -> list[Segment]:
    wp = solve_waypoints(model, data, handles)
    h = wp["holder"]
    b = wp["buffer"]

    def state(holder: str, buffer: str, clamped: bool, buffing: bool, passes: int, stage: str) -> DemoState:
        return DemoState(h[holder], b[buffer], clamped, buffing, passes, stage)

    return [
        Segment(1.0, state("home", "home", False, False, 0, "ready"), state("approach", "home", False, False, 0, "holder_approach")),
        Segment(0.8, state("approach", "home", False, False, 0, "holder_approach"), state("clamp", "home", False, False, 0, "holder_clamp_pose")),
        Segment(0.55, state("clamp", "home", False, False, 0, "holder_clamp_pose"), state("hold", "home", True, False, 0, "vacuum_clamp_engaged")),
        Segment(0.9, state("hold", "home", True, False, 0, "vacuum_clamp_engaged"), state("hold", "approach", True, False, 0, "buffer_approach")),
        Segment(0.7, state("hold", "approach", True, False, 0, "buffer_approach"), state("hold", "pass_1_start", True, False, 0, "tool_at_start")),
        Segment(1.0, state("hold", "pass_1_start", True, True, 0, "buff_pass_1"), state("hold", "pass_1_end", True, True, 1, "buff_pass_1")),
        Segment(1.0, state("hold", "pass_1_end", True, True, 1, "buff_pass_2"), state("hold", "pass_2_end", True, True, 2, "buff_pass_2")),
        Segment(1.0, state("hold", "pass_2_end", True, True, 2, "buff_pass_3"), state("hold", "pass_3_end", True, True, 3, "buff_pass_3")),
        Segment(1.0, state("hold", "pass_3_end", True, True, 3, "buff_pass_4"), state("hold", "pass_4_end", True, True, 4, "buff_pass_4")),
        Segment(0.8, state("hold", "pass_4_end", True, False, 4, "buffing_complete"), state("hold", "retract", True, False, 4, "tool_retract")),
        Segment(0.7, state("hold", "retract", True, False, 4, "tool_retract"), state("hold", "retract", True, False, 4, "success")),
    ]


def timeline_state(segments: list[Segment], elapsed: float) -> DemoState:
    total = sum(segment.duration for segment in segments)
    t = min(elapsed, total - 1e-6)
    for segment in segments:
        if t <= segment.duration:
            return mix_state(segment.start, segment.end, t / segment.duration)
        t -= segment.duration
    return segments[-1].end


def update_visual_state(model: mujoco.MjModel, state: DemoState) -> None:
    for index, geom_name in enumerate(PASS_GEOMS, start=1):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            alpha = 0.82 if index <= state.completed_passes else 0.12
            model.geom_rgba[geom_id] = np.array([0.94, 0.98, 0.97, alpha])

    blur_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "buffer_buff_motion_blur")
    if blur_id >= 0:
        model.geom_rgba[blur_id] = np.array([1.0, 0.78, 0.22, 0.32 if state.buffing else 0.0])

    for key, geom_name in STATUS_GEOMS.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            enabled = (
                (key == "clamp" and state.holder_clamped)
                or (key == "buff" and state.buffing)
                or (key == "success" and task_success(state))
            )
            model.geom_rgba[geom_id] = STATUS_ON[key] if enabled else STATUS_OFF


def task_metrics(state: DemoState) -> dict[str, object]:
    coverage = state.completed_passes / len(PASS_GEOMS)
    return {
        "success": bool(task_success(state)),
        "holder_clamped": bool(state.holder_clamped),
        "completed_buff_passes": int(state.completed_passes),
        "required_buff_passes": len(PASS_GEOMS),
        "coverage_fraction": round(float(coverage), 3),
        "stage": state.stage,
    }


def task_success(state: DemoState) -> bool:
    return bool(state.holder_clamped and state.completed_passes >= len(PASS_GEOMS))


def save_snapshot(model: mujoco.MjModel, data: mujoco.MjData, output_path: Path, camera: str = "overview") -> None:
    renderer = mujoco.Renderer(model, height=1000, width=1600)
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    renderer.close()


def render_frame(model: mujoco.MjModel, data: mujoco.MjData, label: str, complete: bool, width: int, height: int) -> Image.Image:
    renderer = mujoco.Renderer(model, width=width, height=height)
    renderer.update_scene(data, camera="overview")
    image = Image.fromarray(renderer.render()).convert("RGB")
    renderer.close()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 42), fill=(12, 12, 12))
    draw.text((14, 12), f"{label}{'  SUCCESS' if complete else ''}", fill=(255, 255, 255))
    return image


def render_review_sheet(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: WorkcellHandles,
    segments: list[Segment],
    output_path: Path,
    width: int = 700,
    height: int = 450,
) -> None:
    total = sum(segment.duration for segment in segments)
    samples = [
        ("start", 0.05),
        ("clamp", 2.25),
        ("tool approach", 3.6),
        ("buffing pass", 5.7),
        ("success", total - 0.05),
    ]
    frames = []
    for label, sim_time in samples:
        state = timeline_state(segments, sim_time)
        set_state(model, data, handles, state, sim_time)
        frames.append(render_frame(model, data, label, task_success(state), width, height))

    cols = 2
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * width, rows * height), (28, 28, 28))
    for index, frame in enumerate(frames):
        sheet.paste(frame, ((index % cols) * width, (index // cols) * height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"Saved review sheet: {output_path}")


def build_model(scene_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, WorkcellHandles, list[Segment]]:
    if not scene_path.exists():
        write_scene(scene_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    handles = get_handles(model)
    segments = make_segments(model, data, handles)
    return model, data, handles, segments


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: WorkcellHandles,
    segments: list[Segment],
    scene_path: Path,
    snapshot: Path,
    result_json: Path | None,
) -> None:
    end_time = sum(segment.duration for segment in segments) - 0.05
    state = timeline_state(segments, end_time)
    set_state(model, data, handles, state, end_time)
    mujoco.mj_forward(model, data)
    save_snapshot(model, data, snapshot)
    metrics = task_metrics(state)
    print(f"Saved snapshot: {snapshot}")
    print(f"Holder clamped: {metrics['holder_clamped']}")
    print(f"Completed buff passes: {metrics['completed_buff_passes']}/{metrics['required_buff_passes']}")
    print(f"Coverage fraction: {metrics['coverage_fraction']}")
    print(f"TASK COMPLETE: dual_fanuc_metal_buffing SUCCESS={metrics['success']}")
    if result_json is not None:
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(
            json.dumps(
                {
                    "scene": str(scene_path),
                    "snapshot": str(snapshot),
                    "sim_time_s": round(end_time, 4),
                    **metrics,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Saved result: {result_json}")


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, handles: WorkcellHandles, segments: list[Segment]) -> None:
    start = time.time()
    announced = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = named_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        while viewer.is_running():
            elapsed = (time.time() - start) % sum(segment.duration for segment in segments)
            state = timeline_state(segments, elapsed)
            set_state(model, data, handles, state, elapsed)
            if task_success(state) and not announced:
                print("TASK COMPLETE: dual_fanuc_metal_buffing SUCCESS=True")
                announced = True
            elif not task_success(state):
                announced = False
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dual FANUC CR-7iA/L metal buffing workcell demo.")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="Generated MJCF scene path.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer and save final snapshot/result.")
    parser.add_argument("--review", action="store_true", help="Render a multi-stage review PNG.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--write-scene-only", action="store_true", help="Write the generated MJCF and exit.")
    args = parser.parse_args()

    scene_path = args.scene.resolve()
    write_scene(scene_path)
    if args.write_scene_only:
        print(f"Generated scene: {scene_path}")
        return

    model, data, handles, segments = build_model(scene_path)

    if args.review:
        render_review_sheet(model, data, handles, segments, args.review_output)

    if args.headless:
        run_headless(model, data, handles, segments, scene_path, args.snapshot, args.result_json)
        return

    if sys.platform == "darwin" and "mjpython" not in Path(sys.executable).name:
        print("Tip: on macOS this interactive demo should be launched with `mjpython`.")
        print("Run: mjpython sim/run_dual_fanuc_buffing_demo.py")

    run_viewer(model, data, handles, segments)


if __name__ == "__main__":
    main()
