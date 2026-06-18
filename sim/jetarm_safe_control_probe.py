#!/usr/bin/env python3
"""Safer JetArm control probe built from the downloaded test scripts.

Default mode is dry-run. Use --execute to actually send bytes/ROS messages.
This script is intended to run on the JetArm Ubuntu machine.
"""

from __future__ import annotations

import argparse
import sys
import time


SDK_DIR = "/home/ubuntu/factory_utils/arm_pc/"


def parse_pair(text: str) -> tuple[int, int]:
    try:
        servo_id_text, position_text = text.split(":", 1)
        servo_id = int(servo_id_text)
        position = int(position_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected SERVO_ID:POSITION, for example 6:500") from exc
    if not (0 <= position <= 1000):
        raise argparse.ArgumentTypeError("Position must be 0..1000")
    if not (1 <= servo_id <= 254):
        raise argparse.ArgumentTypeError("Servo ID must be 1..254")
    return servo_id, position


def run_sdk_board(port: str, baud: int, positions: list[tuple[int, int]], duration: float, execute: bool) -> None:
    print(f"SDK board protocol: port={port} baud={baud} duration={duration}s positions={positions}")
    if not execute:
        print("DRY RUN: not opening SDK board port.")
        return

    sys.path.insert(0, SDK_DIR)
    from ros_robot_controller_sdk import Board

    board = Board(port, baudrate=baud)
    for servo_id, _ in positions:
        if hasattr(board, "bus_servo_enable_torque"):
            board.bus_servo_enable_torque(servo_id, 1)
            time.sleep(0.03)
    board.bus_servo_set_position(duration, [[servo_id, position] for servo_id, position in positions])
    time.sleep(duration + 0.2)
    print("SDK board command sent.")


def run_raw_controller(port: str, baud: int, positions: list[tuple[int, int]], duration: float, execute: bool) -> None:
    duration_ms = int(duration * 1000)
    print(f"Raw controller protocol: port={port} baud={baud} duration={duration_ms}ms positions={positions}")
    # Hiwonder/Lobot controller packet: 55 55 LEN CMD COUNT TIME_L TIME_H ...
    # LEN is 5 + 3 * servo_count. The downloaded direct_move.py uses 0x08
    # for one servo, matching this formula.
    packet = bytearray([0x55, 0x55, 5 + 3 * len(positions), 0x03, len(positions)])
    packet.append(duration_ms & 0xFF)
    packet.append((duration_ms >> 8) & 0xFF)
    for servo_id, position in positions:
        packet.append(servo_id)
        packet.append(position & 0xFF)
        packet.append((position >> 8) & 0xFF)
    print("packet_hex:", packet.hex(" "))
    if not execute:
        print("DRY RUN: not opening raw serial port.")
        return

    import serial

    ser = serial.Serial(port, baud, timeout=1)
    ser.write(packet)
    ser.flush()
    ser.close()
    time.sleep(duration + 0.2)
    print("Raw controller command sent.")


def run_ros_bus(positions: list[tuple[int, int]], duration: float, execute: bool) -> None:
    print(f"ROS bus topic: /ros_robot_controller/bus_servo/set_position duration={duration}s positions={positions}")
    if not execute:
        print("DRY RUN: not publishing ROS message.")
        return

    import rclpy
    from rclpy.node import Node
    from ros_robot_controller_msgs.msg import ServoPosition, ServosPosition

    rclpy.init()
    node = Node("viha_jetarm_safe_control_probe")
    pub = node.create_publisher(ServosPosition, "/ros_robot_controller/bus_servo/set_position", 10)
    time.sleep(0.7)
    msg = ServosPosition()
    msg.duration = float(duration)
    for servo_id, position in positions:
        servo = ServoPosition()
        servo.id = int(servo_id)
        servo.position = int(position)
        msg.position.append(servo)
    pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.1)
    time.sleep(duration + 0.2)
    node.destroy_node()
    rclpy.shutdown()
    print("ROS bus command published.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sdk-board", "raw-controller", "ros-bus"],
        default="sdk-board",
    )
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--position", action="append", type=parse_pair, default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    positions = args.position or [(6, 500)]
    if args.mode == "sdk-board":
        run_sdk_board(args.port or "/dev/ttyCH341USB1", args.baud or 1000000, positions, args.duration, args.execute)
    elif args.mode == "raw-controller":
        run_raw_controller(args.port or "/dev/ttyTHS1", args.baud or 115200, positions, args.duration, args.execute)
    elif args.mode == "ros-bus":
        run_ros_bus(positions, args.duration, args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
