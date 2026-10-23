#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoothly reset all 16 motion joints and both grippers to zero."""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, Iterable, Optional, Sequence

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


LIFT_JOINTS = ["base_elevator_joint"]
WAIST_JOINTS = ["elevator_pitch_joint"]
LEFT_ROOT_JOINTS = ["pitch_arm_left1_joint"]
RIGHT_ROOT_JOINTS = ["pitch_arm_right1_joint"]
LEFT_ARM_JOINTS = [
    "arm_left1_arm_left2_joint",
    "arm_left2_arm_left3_joint",
    "arm_left3_arm_left4_joint",
    "arm_left4_arm_left5_joint",
    "arm_left5_arm_left6_joint",
    "arm_left6_arm_left7_joint",
]
RIGHT_ARM_JOINTS = [
    "arm_right1_arm_right2_joint",
    "arm_right2_arm_right3_joint",
    "arm_right3_arm_right4_joint",
    "arm_right4_arm_right5_joint",
    "arm_right5_arm_right6_joint",
    "arm_right6_arm_right7_joint",
]
ALL_MOTION_JOINTS = (
    LIFT_JOINTS
    + WAIST_JOINTS
    + LEFT_ROOT_JOINTS
    + RIGHT_ROOT_JOINTS
    + LEFT_ARM_JOINTS
    + RIGHT_ARM_JOINTS
)


class JointZeroReset(Node):
    def __init__(self, cli: argparse.Namespace) -> None:
        super().__init__("box_all_joints_zero")
        self.cli = cli
        self.positions: Dict[str, float] = {}
        self.active_handles = []
        self.create_subscription(JointState, "/joint_states", self.on_state, 20)
        self.completed_pub = self.create_publisher(
            Bool,
            "/box_joint_reset/completed",
            10,
        )
        self.groups = {
            "lift": (
                ActionClient(
                    self,
                    FollowJointTrajectory,
                    "/lift_controller/follow_joint_trajectory",
                ),
                LIFT_JOINTS,
            ),
            "waist": (
                ActionClient(
                    self,
                    FollowJointTrajectory,
                    "/waist_controller/follow_joint_trajectory",
                ),
                WAIST_JOINTS,
            ),
            "left_root": (
                ActionClient(
                    self,
                    FollowJointTrajectory,
                    "/left_arm_root_controller/follow_joint_trajectory",
                ),
                LEFT_ROOT_JOINTS,
            ),
            "right_root": (
                ActionClient(
                    self,
                    FollowJointTrajectory,
                    "/right_arm_root_controller/follow_joint_trajectory",
                ),
                RIGHT_ROOT_JOINTS,
            ),
            "left_arm": (
                ActionClient(
                    self,
                    FollowJointTrajectory,
                    "/left_arm_controller/follow_joint_trajectory",
                ),
                LEFT_ARM_JOINTS,
            ),
            "right_arm": (
                ActionClient(
                    self,
                    FollowJointTrajectory,
                    "/right_arm_controller/follow_joint_trajectory",
                ),
                RIGHT_ARM_JOINTS,
            ),
        }
        self.grippers = {
            "left_gripper": ActionClient(
                self,
                GripperCommand,
                "/left_gripper_controller/gripper_cmd",
            ),
            "right_gripper": ActionClient(
                self,
                GripperCommand,
                "/right_gripper_controller/gripper_cmd",
            ),
        }

    def on_state(self, message: JointState) -> None:
        for name, position in zip(message.name, message.position):
            if name in ALL_MOTION_JOINTS:
                self.positions[name] = float(position)

    def wait_futures(self, futures: Iterable, timeout: float) -> bool:
        pending = list(futures)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if all(future.done() for future in pending):
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return all(future.done() for future in pending)

    def wait_ready(self) -> bool:
        ready = True
        for name, (client, _) in self.groups.items():
            if client.wait_for_server(timeout_sec=self.cli.server_timeout):
                self.get_logger().info(f"Controller ready: {name}")
            else:
                self.get_logger().error(f"Controller unavailable: {name}")
                ready = False
        for name, client in self.grippers.items():
            if client.wait_for_server(timeout_sec=self.cli.server_timeout):
                self.get_logger().info(f"Controller ready: {name}")
            else:
                self.get_logger().error(f"Controller unavailable: {name}")
                ready = False
        return ready

    def wait_joint_state(self) -> bool:
        deadline = time.monotonic() + self.cli.state_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            missing = [name for name in ALL_MOTION_JOINTS if name not in self.positions]
            if not missing:
                self.get_logger().info("Received all 16 motion joint positions")
                return True
        missing = [name for name in ALL_MOTION_JOINTS if name not in self.positions]
        self.get_logger().error("/joint_states is missing: " + ", ".join(missing))
        return False

    def trajectory_goal(
        self,
        joint_names: Sequence[str],
    ) -> FollowJointTrajectory.Goal:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(joint_names)
        for index in range(1, self.cli.steps + 1):
            ratio = index / self.cli.steps
            weight = ratio * ratio * (3.0 - 2.0 * ratio)
            point = JointTrajectoryPoint()
            point.positions = [
                self.positions[name] * (1.0 - weight)
                for name in joint_names
            ]
            timestamp = self.cli.duration * ratio
            seconds = int(math.floor(timestamp))
            point.time_from_start.sec = seconds
            point.time_from_start.nanosec = int(
                round((timestamp - seconds) * 1_000_000_000)
            )
            if point.time_from_start.nanosec >= 1_000_000_000:
                point.time_from_start.sec += 1
                point.time_from_start.nanosec -= 1_000_000_000
            goal.trajectory.points.append(point)
        return goal

    def cancel_active(self) -> None:
        for handle in self.active_handles:
            try:
                handle.cancel_goal_async()
            except Exception:
                pass
        self.active_handles = []

    def reset_motion_joints(self) -> bool:
        send_futures = {
            name: client.send_goal_async(self.trajectory_goal(joint_names))
            for name, (client, joint_names) in self.groups.items()
        }
        if not self.wait_futures(send_futures.values(), self.cli.server_timeout):
            self.get_logger().error("Timed out while sending zero trajectories")
            return False

        handles = {}
        for name, future in send_futures.items():
            handle = future.result()
            if handle is None or not handle.accepted:
                self.get_logger().error(f"Zero trajectory rejected: {name}")
            else:
                handles[name] = handle
        if len(handles) != len(send_futures):
            self.active_handles = list(handles.values())
            self.cancel_active()
            return False

        self.active_handles = list(handles.values())
        result_futures = {
            name: handle.get_result_async()
            for name, handle in handles.items()
        }
        timeout = self.cli.duration + self.cli.result_margin
        if not self.wait_futures(result_futures.values(), timeout):
            self.get_logger().error("Joint reset execution timed out")
            self.cancel_active()
            return False

        success = True
        for name, future in result_futures.items():
            wrapped = future.result()
            result_ok = (
                wrapped is not None
                and wrapped.status == GoalStatus.STATUS_SUCCEEDED
                and wrapped.result.error_code
                == FollowJointTrajectory.Result.SUCCESSFUL
            )
            if result_ok:
                self.get_logger().info(f"Joint group reset completed: {name}")
            else:
                self.get_logger().error(f"Joint group reset failed: {name}")
                success = False
        self.active_handles = []
        return success

    def reset_grippers(self) -> bool:
        if self.cli.leave_grippers:
            return True
        send_futures = {}
        for name, client in self.grippers.items():
            goal = GripperCommand.Goal()
            goal.command.position = 0.0
            goal.command.max_effort = self.cli.gripper_effort
            send_futures[name] = client.send_goal_async(goal)
        if not self.wait_futures(send_futures.values(), self.cli.server_timeout):
            self.get_logger().error("Timed out while sending gripper reset goals")
            return False
        handles = {}
        for name, future in send_futures.items():
            handle = future.result()
            if handle is None or not handle.accepted:
                self.get_logger().error(f"Gripper reset rejected: {name}")
            else:
                handles[name] = handle
        if len(handles) != len(send_futures):
            return False
        results = [handle.get_result_async() for handle in handles.values()]
        if not self.wait_futures(results, self.cli.result_margin):
            self.get_logger().error("Gripper reset execution timed out")
            return False
        success = all(
            future.result() is not None
            and future.result().status == GoalStatus.STATUS_SUCCEEDED
            for future in results
        )
        if not success:
            self.get_logger().error("One or both grippers failed to reach zero")
        return success

    def execute(self) -> bool:
        self.completed_pub.publish(Bool(data=False))
        if not self.wait_ready() or not self.wait_joint_state():
            return False
        max_error = max(abs(self.positions[name]) for name in ALL_MOTION_JOINTS)
        self.get_logger().info(
            f"Resetting all motion joints to zero: duration={self.cli.duration:.1f}s, "
            f"largest initial magnitude={max_error:.3f}rad/m"
        )
        if not self.cli.execute:
            self.get_logger().warning("Dry run only; add --execute to move the robot")
            return True
        if not self.reset_motion_joints():
            return False
        if not self.reset_grippers():
            return False
        self.completed_pub.publish(Bool(data=True))
        self.get_logger().info("All motion joints and grippers are at zero")
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset all robot joints to zero")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--server-timeout", type=float, default=12.0)
    parser.add_argument("--state-timeout", type=float, default=12.0)
    parser.add_argument("--result-margin", type=float, default=20.0)
    parser.add_argument("--gripper-effort", type=float, default=30.0)
    parser.add_argument("--leave-grippers", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be greater than zero")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    for name in ("server_timeout", "state_timeout", "result_margin", "gripper_effort"):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    return args


def main() -> int:
    cli = parse_args()
    rclpy.init(args=[])
    node: Optional[JointZeroReset] = None
    try:
        node = JointZeroReset(cli)
        return 0 if node.execute() else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Joint reset exception: {type(exc).__name__}: {exc}")
        else:
            print(f"Joint reset creation failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if node is not None:
            node.cancel_active()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
