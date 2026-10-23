#!/usr/bin/env python3

"""Start Robot A patrol, stop after its first NavigateToPose goal succeeds."""

import argparse
import os
import signal
import sys
import time
from typing import Dict, Optional, Set, Tuple

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_srvs.srv import Trigger


def goal_id_bytes(status) -> bytes:
    return bytes(status.goal_info.goal_id.uuid)


class FirstPatrolWaypointController(Node):
    def __init__(self, patrol_pid: int, timeout: float):
        super().__init__("first_patrol_waypoint_controller")
        self.patrol_pid = patrol_pid
        self.timeout = timeout
        self.start_time = time.monotonic()
        self.start_requested = False
        self.start_future = None
        self.result: Optional[bool] = None
        self.failure_reason = ""

        self.known_before_start: Dict[str, Set[bytes]] = {
            "navigate_to_pose": set(),
            "navigate_through_poses": set(),
        }
        self.first_goal: Optional[Tuple[str, bytes]] = None

        self.create_subscription(
            GoalStatusArray,
            "/navigate_to_pose/_action/status",
            lambda msg: self.status_callback("navigate_to_pose", msg),
            10,
        )
        self.create_subscription(
            GoalStatusArray,
            "/navigate_through_poses/_action/status",
            lambda msg: self.status_callback("navigate_through_poses", msg),
            10,
        )

        self.start_client = self.create_client(Trigger, "/robot_a/patrol/start")
        self.stop_client = self.create_client(Trigger, "/robot_a/patrol/stop")
        self.cancel_pose_client = self.create_client(
            CancelGoal, "/navigate_to_pose/_action/cancel_goal"
        )
        self.cancel_poses_client = self.create_client(
            CancelGoal, "/navigate_through_poses/_action/cancel_goal"
        )

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.controller_cmd_pub = self.create_publisher(
            Twist, "/mecanum_drive_controller/reference_unstamped", 10
        )
        self.timer = self.create_timer(0.1, self.control_loop)

    def status_callback(self, action_name: str, msg: GoalStatusArray):
        if not self.start_requested:
            self.known_before_start[action_name].update(
                goal_id_bytes(status) for status in msg.status_list
            )
            return

        if self.first_goal is None:
            candidates = [
                status
                for status in msg.status_list
                if goal_id_bytes(status) not in self.known_before_start[action_name]
                and status.status
                in (
                    GoalStatus.STATUS_ACCEPTED,
                    GoalStatus.STATUS_EXECUTING,
                    GoalStatus.STATUS_CANCELING,
                    GoalStatus.STATUS_SUCCEEDED,
                    GoalStatus.STATUS_CANCELED,
                    GoalStatus.STATUS_ABORTED,
                )
            ]
            if candidates:
                selected = candidates[-1]
                self.first_goal = (action_name, goal_id_bytes(selected))
                self.get_logger().info(
                    f"Captured first patrol goal from /{action_name}."
                )

        if self.first_goal is None or self.first_goal[0] != action_name:
            return

        tracked_id = self.first_goal[1]
        for status in msg.status_list:
            if goal_id_bytes(status) != tracked_id:
                continue
            if status.status == GoalStatus.STATUS_SUCCEEDED:
                if action_name == "navigate_through_poses":
                    self.fail(
                        "Patrol sent all waypoints as one NavigateThroughPoses goal; "
                        "the first waypoint has no separate completion event."
                    )
                else:
                    self.succeed()
            elif status.status in (
                GoalStatus.STATUS_ABORTED,
                GoalStatus.STATUS_CANCELED,
            ):
                self.fail(
                    f"First patrol goal ended with Nav2 status {status.status}."
                )
            break

    def control_loop(self):
        elapsed = time.monotonic() - self.start_time
        if elapsed > self.timeout and self.result is None:
            self.fail(f"Timed out after {self.timeout:.0f} seconds.")
            return

        if not self.start_requested:
            if not self.start_client.service_is_ready():
                return
            self.get_logger().info("Calling /robot_a/patrol/start ...")
            self.start_future = self.start_client.call_async(Trigger.Request())
            self.start_requested = True
            return

        if self.start_future is not None and self.start_future.done():
            try:
                response = self.start_future.result()
            except Exception as exc:
                self.fail(f"Patrol start service failed: {exc}")
                return
            self.start_future = None
            if not response.success:
                self.fail(f"Patrol refused to start: {response.message}")
            else:
                self.get_logger().info(
                    f"Patrol started: {response.message or 'OK'}; waiting for first goal."
                )

    def stop_patrol_and_navigation(self):
        if self.stop_client.service_is_ready():
            self.stop_client.call_async(Trigger.Request())

        # Stop the launch process group so it cannot send waypoint #2.
        try:
            os.killpg(self.patrol_pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(self.patrol_pid, signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                pass

        # Cancel any next goal that the patrol state machine may already have sent.
        cancel_request = CancelGoal.Request()  # zero UUID + zero stamp = cancel all
        if self.cancel_pose_client.service_is_ready():
            self.cancel_pose_client.call_async(cancel_request)
        if self.cancel_poses_client.service_is_ready():
            self.cancel_poses_client.call_async(cancel_request)

        zero = Twist()
        for _ in range(10):
            self.cmd_vel_pub.publish(zero)
            self.controller_cmd_pub.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.03)
            time.sleep(0.03)

    def succeed(self):
        if self.result is not None:
            return
        self.get_logger().info(
            "First patrol waypoint reached. Stopping patrol before waypoint #2."
        )
        self.stop_patrol_and_navigation()
        self.result = True

    def fail(self, reason: str):
        if self.result is not None:
            return
        self.failure_reason = reason
        self.get_logger().error(reason)
        self.stop_patrol_and_navigation()
        self.result = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patrol-pid", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    rclpy.init()
    node = FirstPatrolWaypointController(args.patrol_pid, args.timeout)
    try:
        while rclpy.ok() and node.result is None:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.fail("Interrupted by user.")
    finally:
        success = node.result is True
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

