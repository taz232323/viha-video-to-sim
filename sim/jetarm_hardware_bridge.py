#!/usr/bin/env python3
"""Plan or execute JetArm tabletop pick/place motions through Hiwonder ROS2.

Default behavior is dry-run only. It calls the JetArm IK service and writes the
servo pulse plan, but does not publish motion unless --execute is provided.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from kinematics_msgs.srv import SetRobotPose
from servo_controller_msgs.msg import ServoPosition, ServosPosition


DEFAULT_TASK = {
    "name": "bowl_to_plate_tabletop_demo",
    "notes": "Conservative starter coordinates in JetArm base frame, meters.",
    "waypoints": [
        {
            "name": "ready",
            "kind": "servo",
            "duration": 1.0,
            "servos": [[1, 500], [2, 560], [3, 130], [4, 115], [5, 500], [10, 200]],
        },
        {
            "name": "pre_pick",
            "kind": "pose",
            "duration": 1.0,
            "position": [0.18, 0.06, 0.24],
            "pitch": 0.0,
            "pitch_range": [-90.0, 90.0],
            "resolution": 1.0,
            "gripper": 200,
        },
        {
            "name": "pick",
            "kind": "pose",
            "duration": 1.0,
            "position": [0.18, 0.06, 0.20],
            "pitch": 0.0,
            "pitch_range": [-90.0, 90.0],
            "resolution": 1.0,
            "gripper": 200,
        },
        {
            "name": "grip",
            "kind": "servo",
            "duration": 0.6,
            "servos": [[10, 600]],
        },
        {
            "name": "lift",
            "kind": "pose",
            "duration": 1.0,
            "position": [0.18, 0.06, 0.27],
            "pitch": 0.0,
            "pitch_range": [-90.0, 90.0],
            "resolution": 1.0,
            "gripper": 600,
        },
        {
            "name": "pre_place",
            "kind": "pose",
            "duration": 1.0,
            "position": [0.18, -0.06, 0.27],
            "pitch": 0.0,
            "pitch_range": [-90.0, 90.0],
            "resolution": 1.0,
            "gripper": 600,
        },
        {
            "name": "place",
            "kind": "pose",
            "duration": 1.0,
            "position": [0.18, -0.06, 0.20],
            "pitch": 0.0,
            "pitch_range": [-90.0, 90.0],
            "resolution": 1.0,
            "gripper": 600,
        },
        {
            "name": "release",
            "kind": "servo",
            "duration": 0.6,
            "servos": [[10, 200]],
        },
        {
            "name": "retreat",
            "kind": "pose",
            "duration": 1.0,
            "position": [0.18, -0.06, 0.27],
            "pitch": 0.0,
            "pitch_range": [-90.0, 90.0],
            "resolution": 1.0,
            "gripper": 200,
        },
    ],
}


def load_task(path: str | None) -> dict:
    if not path:
        return DEFAULT_TASK
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class JetArmBridge(Node):
    def __init__(self) -> None:
        super().__init__("viha_jetarm_bridge")
        self.ik_client = self.create_client(SetRobotPose, "/kinematics/set_pose_target")
        self.servo_pub = self.create_publisher(ServosPosition, "/servo_controller", 1)

    def wait_for_ik(self, timeout_sec: float) -> None:
        if not self.ik_client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("Timed out waiting for /kinematics/set_pose_target")

    def solve_pose(self, waypoint: dict, timeout_sec: float) -> dict:
        req = SetRobotPose.Request()
        req.position = [float(v) for v in waypoint["position"]]
        req.pitch = float(waypoint.get("pitch", 0.0))
        req.pitch_range = [float(v) for v in waypoint.get("pitch_range", [-90.0, 90.0])]
        req.resolution = float(waypoint.get("resolution", 1.0))
        req.duration = float(waypoint.get("duration", 1.0))

        future = self.ik_client.call_async(req)
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if future.done():
                break
        if not future.done():
            raise RuntimeError(f"IK timeout for waypoint {waypoint.get('name', '<unnamed>')}")
        result = future.result()
        if result is None:
            raise RuntimeError(f"IK call failed for waypoint {waypoint.get('name', '<unnamed>')}")

        pulses = [int(v) for v in result.pulse]
        current_pulse = [int(v) for v in result.current_pulse]
        servo_pairs = [[i + 1, pulse] for i, pulse in enumerate(pulses[:5])]
        if "gripper" in waypoint:
            servo_pairs.append([10, int(waypoint["gripper"])])

        return {
            "ik_success": bool(result.success),
            "pulse": pulses,
            "current_pulse": current_pulse,
            "rpy": [float(v) for v in result.rpy],
            "min_variation": float(result.min_variation),
            "servos": servo_pairs,
        }

    def publish_servos(self, duration: float, servos: list[list[int]]) -> None:
        msg = ServosPosition()
        msg.duration = float(duration)
        msg.position_unit = "pulse"
        msg.position = []
        for servo_id, pulse in servos:
            position = ServoPosition()
            position.id = int(servo_id)
            position.position = float(pulse)
            msg.position.append(position)
        self.servo_pub.publish(msg)


class OfflineJetArmPlanner:
    """Use Hiwonder's kinematics library without joining the live ROS graph."""

    def solve_pose(self, waypoint: dict, timeout_sec: float) -> dict:
        del timeout_sec
        from kinematics.inverse_kinematics import get_ik
        from kinematics import transform

        position = [float(v) for v in waypoint["position"]]
        pitch = float(waypoint.get("pitch", 0.0))
        pitch_range = [float(v) for v in waypoint.get("pitch_range", [-90.0, 90.0])]
        resolution = float(waypoint.get("resolution", 1.0))

        solutions = get_ik(position, pitch, pitch_range, resolution)
        if not solutions:
            return {
                "ik_success": False,
                "pulse": [],
                "current_pulse": [],
                "rpy": [],
                "min_variation": 0.0,
                "servos": [],
            }

        pulse_solutions = transform.angle2pulse(solutions[0][0], True)
        pulses = [int(v) for v in pulse_solutions[0]] if pulse_solutions else []
        servo_pairs = [[i + 1, pulse] for i, pulse in enumerate(pulses[:5])]
        if "gripper" in waypoint:
            servo_pairs.append([10, int(waypoint["gripper"])])

        return {
            "ik_success": bool(pulses),
            "pulse": pulses,
            "current_pulse": [],
            "rpy": [float(v) for v in solutions[0][1]],
            "min_variation": 0.0,
            "servos": servo_pairs,
            "planner": "offline_hiwonder_ik",
        }


def build_plan(bridge: JetArmBridge, task: dict, timeout_sec: float) -> dict:
    plan = {"task": task.get("name", "unnamed"), "waypoints": []}
    for waypoint in task.get("waypoints", []):
        planned = {
            "name": waypoint.get("name", "unnamed"),
            "kind": waypoint.get("kind", "pose"),
            "duration": float(waypoint.get("duration", 1.0)),
        }
        if planned["kind"] == "pose":
            planned["position"] = waypoint["position"]
            planned.update(bridge.solve_pose(waypoint, timeout_sec=timeout_sec))
        elif planned["kind"] == "servo":
            planned["servos"] = [[int(sid), int(pulse)] for sid, pulse in waypoint["servos"]]
        else:
            raise ValueError(f"Unsupported waypoint kind: {planned['kind']}")
        plan["waypoints"].append(planned)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="JSON file with cartesian/servo waypoints")
    parser.add_argument("--output", default="jetarm_hardware_plan.json")
    parser.add_argument("--ik-timeout", type=float, default=8.0)
    parser.add_argument("--execute", action="store_true", help="Publish servo commands to the real arm")
    args = parser.parse_args()

    task = load_task(args.task)
    rclpy.init()
    bridge = JetArmBridge()
    try:
        try:
            bridge.wait_for_ik(timeout_sec=args.ik_timeout)
            planner = bridge
        except RuntimeError:
            if args.execute:
                raise RuntimeError(
                    "Live ROS IK service is not reachable from this process; refusing --execute."
                )
            print("Live ROS IK service not reachable; using offline Hiwonder IK for dry-run planning.")
            planner = OfflineJetArmPlanner()

        plan = build_plan(planner, task, timeout_sec=args.ik_timeout)
        Path(args.output).write_text(json.dumps(plan, indent=2), encoding="utf-8")

        print(json.dumps(plan, indent=2))
        print(f"\nWrote plan: {args.output}")
        if not args.execute:
            print("DRY RUN: no servo commands were published. Add --execute only after physical clearance checks.")
            return 0

        for waypoint in plan["waypoints"]:
            servos = waypoint.get("servos", [])
            if not servos:
                raise RuntimeError(f"No servo pulses for waypoint {waypoint['name']}")
            print(f"EXECUTE {waypoint['name']}: {servos}")
            bridge.publish_servos(waypoint["duration"], servos)
            time.sleep(max(0.1, waypoint["duration"]))
        print("EXECUTION COMPLETE")
        return 0
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
