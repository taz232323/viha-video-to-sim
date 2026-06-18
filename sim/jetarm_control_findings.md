# JetArm Control Findings

This summarizes the control scripts from `/Users/amardesai/Downloads`.

## Main Control Paths Found

1. Hiwonder board SDK on `/dev/ttyCH341USB1` at `1000000` baud
   - Used by `arm_commander.py`, `force_move.py`, `id_one_test.py`, `brute_force_arm.py`, `torque_wake.py`.
   - Imports `/home/ubuntu/factory_utils/arm_pc/ros_robot_controller_sdk.py`.
   - Sends framed packets beginning with `0xAA 0x55` through `Board.bus_servo_set_position(...)`.
   - This talks to a controller board, not directly to a raw bus servo.

2. ROS2 topic control
   - Used by `jetarm_basic_movements.py`, `jetarm_test.py`, `simple_move.py`.
   - Publishes to `/servo_controller` or `/ros_robot_controller/bus_servo/set_position`.
   - This only works if the `ros_robot_controller` bridge is running and connected to the correct serial port.
   - On this robot, that bridge initially failed because it looked for `/dev/ttyUSB0` while the controller appeared as `/dev/ttyCH341USB1`.

3. Raw serial on `/dev/ttyTHS1` at `115200` baud
   - Used by `direct_move.py`, `hard_reset_test.py`, `jetarm_victory_test.py`.
   - `direct_move.py` sends a Hiwonder bus-servo-controller style packet: `55 55 len cmd count time_l time_h id pos_l pos_h`.
   - `hard_reset_test.py` sends a different single-servo packet shape and omits a checksum, so it is likely incomplete for direct LX-style servo control.

## Servo ID Mappings In The Scripts

Two mappings appear:

- ROS / Hiwonder controller mapping:
  - Arm joints: IDs `1..5`
  - Gripper: virtual ID `10`

- Raw `/dev/ttyTHS1` mapping comments:
  - `6`: base
  - `5`: shoulder
  - `4`: elbow
  - `3`: wrist
  - `2`: wrist rotate
  - `1`: gripper

The mismatch explains why sending `ID 10` works in ROS examples but would not make sense for raw direct serial scripts.

## Bad Or Risky Assumptions Found

- `brute_force_arm.py` runs `sudo killall -9 python3`, which can kill the robot stack. Do not use as-is.
- `torque_wake.py` calls `bus_servo_set_mode`, but the installed SDK does not define that method.
- Several scripts print success after writing bytes or publishing ROS messages, but they do not verify that a motor actually received/responded.
- `final_hw_check.py` only proves the serial port can open/write; it does not prove the servo bus is moving.

## Most Plausible Fix Path

1. Start/fix `ros_robot_controller` so it opens the real board port.
2. Verify servo state reads from `/ros_robot_controller/bus_servo/get_state`.
3. Only after reads work, send a small command to one servo.
4. Then use `/servo_controller` or `/ros_robot_controller/bus_servo/set_position` for normal motion.

