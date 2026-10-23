#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 YAML 顺序导航，并在指定路点执行抓取、抬升和释放任务。

本节点直接控制 /navigate_to_pose，运行时不要同时启动
real_robot_patrol.launch.py，也不要调用 /robot_a/patrol/start。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from lifecycle_msgs.srv import GetState
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener


@dataclass(frozen=True)
class Waypoint:
    object_id: str
    x: float
    y: float
    yaw: float
    dwell_seconds: float


def _recovery_env_float(
    name: str,
    default: float,
    minimum: float = 0.0,
) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是数值，当前为 {raw!r}") from exc
    if not math.isfinite(value) or value < minimum:
        raise RuntimeError(f"{name} 必须为有限数且不小于 {minimum}")
    return value


def _recovery_env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是整数，当前为 {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} 必须不小于 {minimum}")
    return value


def load_waypoints(path: Path) -> list[Waypoint]:
    if not path.is_file():
        raise RuntimeError(f"路点文件不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("format_version") != 1:
        raise RuntimeError("只支持 format_version: 1")
    if str(data.get("frame_id", "")) != "map":
        raise RuntimeError("路点 YAML 必须使用 frame_id: map")

    result: list[Waypoint] = []
    seen: set[str] = set()
    for index, item in enumerate(data.get("waypoints", []), start=1):
        object_id = str(item.get("object_id", "")).strip()
        pose = item.get("dock_pose", [])
        dwell = float(item.get("dwell_seconds", 5.0))
        if not object_id or object_id in seen:
            raise RuntimeError(f"第 {index} 个路点 ID 为空或重复")
        if not isinstance(pose, list) or len(pose) != 3:
            raise RuntimeError(
                f"路点 {object_id} 的 dock_pose 必须是 [x, y, yaw]"
            )
        x, y, yaw = (float(value) for value in pose)
        if not all(math.isfinite(value) for value in (x, y, yaw, dwell)):
            raise RuntimeError(f"路点 {object_id} 包含非法数值")
        if dwell < 0.0:
            raise RuntimeError(f"路点 {object_id} 停留时间不能小于零")
        seen.add(object_id)
        result.append(Waypoint(object_id, x, y, yaw, dwell))
    if not result:
        raise RuntimeError("YAML 中没有路点")
    return result


class BoxWaypointMission(Node):
    def __init__(self, cli: argparse.Namespace) -> None:
        super().__init__("box_waypoint_mission")
        self.cli = cli
        self.waypoints = load_waypoints(Path(cli.waypoints).resolve())
        self.nav = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bt_state_client = self.create_client(
            GetState,
            "/bt_navigator/get_state",
        )
        self.costmap_clients = (
            (
                "local",
                self.create_client(
                    ClearEntireCostmap,
                    self.cli.local_costmap_clear_service,
                ),
            ),
            (
                "global",
                self.create_client(
                    ClearEntireCostmap,
                    self.cli.global_costmap_clear_service,
                ),
            ),
        )
        self.manual_cmd_vel_topic = os.environ.get(
            "BOX_MISSION_MANUAL_CMD_VEL_TOPIC",
            "/box_mission/manual_cmd_vel",
        ).strip()
        self.stop_pub = self.create_publisher(
            Twist,
            self.manual_cmd_vel_topic,
            10,
        )
        self.state_pub = self.create_publisher(
            String, "/robot_a/box_mission/state", 10
        )
        self.point_pub = self.create_publisher(
            String, "/robot_a/box_mission/current_waypoint", 10
        )
        self.done_pub = self.create_publisher(
            Bool, "/robot_a/box_mission/completed", 10
        )
        self.create_subscription(
            Bool,
            "/box_grasp/completed",
            self._on_grasp_completed,
            10,
        )

        self.commands = {
            key: self._required_env(f"BOX_MISSION_{key.upper()}_CMD")
            for key in (
                "vision",
                "grasp",
                "lift_1",
                "lift_2",
                "release",
                "reset",
            )
        }
        self.vision_setup = self._required_env("BOX_MISSION_VISION_SETUP")
        if not Path(self.vision_setup).is_file():
            raise RuntimeError(
                f"视觉环境脚本不存在: {self.vision_setup}"
            )
        ids = {waypoint.object_id for waypoint in self.waypoints}
        self.grasp_waypoint_ids = {
            cli.grasp_waypoint_1,
            cli.grasp_waypoint_2,
        }
        self.lift_command_by_waypoint = {
            cli.lift_waypoint_1: "lift_1",
            cli.lift_waypoint_2: "lift_2",
        }
        self.release_waypoint_ids = {
            cli.release_waypoint_1,
            cli.release_waypoint_2,
        }
        self.reset_waypoint_ids = (
            {cli.reset_waypoint} if cli.reset_waypoint else set()
        )
        for label, height in (
            ("第一轮", cli.lift_height_1),
            ("第二轮", cli.lift_height_2),
        ):
            if not math.isfinite(height) or height <= 0.0:
                raise RuntimeError(
                    f"{label} TCP中点目标高度必须为正的有限数，当前={height}"
                )
        task_sequence = (
            cli.grasp_waypoint_1,
            cli.lift_waypoint_1,
            cli.release_waypoint_1,
            cli.grasp_waypoint_2,
            cli.lift_waypoint_2,
            cli.release_waypoint_2,
        )
        if len(set(task_sequence)) != len(task_sequence):
            raise RuntimeError("两轮任务的六个操作路点必须互不重复")
        required = set(task_sequence) | {cli.final_waypoint}
        required.update(self.reset_waypoint_ids)
        missing = sorted(required - ids)
        if missing:
            raise RuntimeError("YAML 缺少任务路点: " + ", ".join(missing))
        waypoint_order = {
            waypoint.object_id: index
            for index, waypoint in enumerate(self.waypoints)
        }
        if any(
            waypoint_order[first] >= waypoint_order[second]
            for first, second in zip(task_sequence, task_sequence[1:])
        ):
            raise RuntimeError(
                "YAML 两轮任务顺序必须为："
                + " -> ".join(task_sequence)
            )

        self.grasp_done = threading.Event()
        self.shutdown_event = threading.Event()
        self.active_goal = None
        self.processes: list[subprocess.Popen] = []

    @staticmethod
    def _required_env(name: str) -> str:
        value = os.environ.get(name, "").strip()
        if not value:
            raise RuntimeError(f"环境变量 {name} 未设置")
        return value

    def state(self, name: str, detail: str = "") -> None:
        message = name if not detail else f"{name}: {detail}"
        self.state_pub.publish(String(data=message))
        self.get_logger().info(message)

    def stop_robot(self) -> None:
        for _ in range(3):
            self.stop_pub.publish(Twist())

    def _on_grasp_completed(self, message: Bool) -> None:
        if message.data:
            self.grasp_done.set()
            self.get_logger().info("收到 /box_grasp/completed=true")

    def wait_future(self, future, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not future.done():
            if self.shutdown_event.is_set() or not rclpy.ok():
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def sleep(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if self.shutdown_event.is_set() or not rclpy.ok():
                return False
            time.sleep(min(0.1, deadline - time.monotonic()))
        return True

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def current_map_pose(self) -> Optional[tuple[float, float, float]]:
        """Return base pose (x, y, yaw) in map frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                self.cli.base_frame,
                Time(),
            )
        except Exception:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = self.yaw_from_quaternion(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )
        return float(translation.x), float(translation.y), float(yaw)

    def arrival_tolerances(self, point: Waypoint) -> tuple[float, float, str]:
        if point.object_id in self.grasp_waypoint_ids:
            return (
                self.cli.grasp_arrival_position_tolerance,
                self.cli.grasp_arrival_yaw_tolerance,
                "grasp",
            )
        task_point_ids = (
            set(self.lift_command_by_waypoint)
            | self.release_waypoint_ids
        )
        if point.object_id in task_point_ids:
            return (
                self.cli.task_arrival_position_tolerance,
                self.cli.task_arrival_yaw_tolerance,
                "task",
            )
        return (
            self.cli.arrival_position_tolerance,
            self.cli.arrival_yaw_tolerance,
            "normal",
        )

    def requires_precision_alignment(self, point: Waypoint) -> bool:
        return point.object_id in {
            *self.grasp_waypoint_ids,
            *self.lift_command_by_waypoint,
            *self.release_waypoint_ids,
        }

    def finalize_arrival(self, point: Waypoint, reason: str) -> bool:
        """Enforce TF-based fine alignment before any manipulation task."""
        if self.requires_precision_alignment(point):
            self.get_logger().info(
                f"Nav2 arrival for {point.object_id} is not accepted yet; "
                "starting mandatory low-speed pose refinement"
            )
            if not self.fine_align(point):
                self.get_logger().error(
                    f"Mandatory pose refinement failed for {point.object_id}; "
                    "the manipulation task will not start"
                )
                return False
        else:
            self.stop_robot()
        self.state("ARRIVED", f"{point.object_id} ({reason})")
        return True

    def waypoint_error(self, point: Waypoint) -> Optional[tuple[float, float]]:
        """Return actual map-frame position/yaw error, or None without TF."""
        current = self.current_map_pose()
        if current is None:
            self.get_logger().warning(
                f"Cannot check actual arrival for {point.object_id}: "
                f"map->{self.cli.base_frame} TF unavailable"
            )
            return None
        current_x, current_y, current_yaw = current
        position_error = math.hypot(
            current_x - point.x,
            current_y - point.y,
        )
        yaw_error = abs(self.normalize_angle(current_yaw - point.yaw))
        return position_error, yaw_error

    def waypoint_reached(self, point: Waypoint, reason: str) -> bool:
        error = self.waypoint_error(point)
        if error is None:
            return False

        position_error, yaw_error = error
        position_tolerance, yaw_tolerance, tolerance_kind = (
            self.arrival_tolerances(point)
        )
        reached = (
            position_error <= position_tolerance
            and yaw_error <= yaw_tolerance
        )
        message = (
            f"Arrival check ({reason}) for {point.object_id}: "
            f"distance={position_error:.3f}m/"
            f"{position_tolerance:.3f}m, "
            f"yaw={math.degrees(yaw_error):.1f}deg/"
            f"{math.degrees(yaw_tolerance):.1f}deg, "
            f"kind={tolerance_kind}, reached={reached}"
        )
        # Keep INFO and WARNING on different Python call sites. Some rclpy
        # versions cache a call site's severity and raise ValueError when the
        # same call site changes from WARNING to INFO after reaching a goal.
        if reached:
            self.get_logger().info(message)
        else:
            self.get_logger().warning(message)
        return reached

    def fine_align(self, point: Waypoint) -> bool:
        """Low-speed holonomic final alignment for a nearby waypoint."""
        if not self.cli.enable_fine_alignment:
            return False

        current = self.current_map_pose()
        if current is None:
            return False
        current_x, current_y, _ = current
        initial_distance = math.hypot(point.x - current_x, point.y - current_y)
        if initial_distance > self.cli.fine_alignment_start_radius:
            self.get_logger().warning(
                f"Fine alignment skipped for {point.object_id}: "
                f"distance={initial_distance:.3f}m exceeds "
                f"{self.cli.fine_alignment_start_radius:.3f}m"
            )
            return False

        position_tolerance, yaw_tolerance, _ = self.arrival_tolerances(point)
        self.get_logger().warning(
            f"Starting low-speed fine alignment for {point.object_id}: "
            f"distance={initial_distance:.3f}m"
        )
        self.stop_robot()
        if not self.sleep(0.2):
            return False
        deadline = time.monotonic() + self.cli.fine_alignment_timeout
        stable_samples = 0
        last_log = 0.0
        try:
            while time.monotonic() < deadline:
                if self.shutdown_event.is_set() or not rclpy.ok():
                    return False
                current = self.current_map_pose()
                if current is None:
                    self.stop_robot()
                    if not self.sleep(0.1):
                        return False
                    continue
                current_x, current_y, current_yaw = current
                dx_map = point.x - current_x
                dy_map = point.y - current_y
                position_error = math.hypot(dx_map, dy_map)
                yaw_error_signed = self.normalize_angle(point.yaw - current_yaw)
                yaw_error = abs(yaw_error_signed)

                if (
                    position_error <= position_tolerance
                    and yaw_error <= yaw_tolerance
                ):
                    stable_samples += 1
                    self.stop_robot()
                    if stable_samples >= self.cli.fine_alignment_stable_samples:
                        self.get_logger().info(
                            f"Fine alignment completed for {point.object_id}: "
                            f"distance={position_error:.3f}m, "
                            f"yaw={math.degrees(yaw_error):.1f}deg"
                        )
                        return True
                else:
                    stable_samples = 0
                    cos_yaw = math.cos(current_yaw)
                    sin_yaw = math.sin(current_yaw)
                    dx_base = cos_yaw * dx_map + sin_yaw * dy_map
                    dy_base = -sin_yaw * dx_map + cos_yaw * dy_map
                    command = Twist()
                    command.linear.x = self.clamp(
                        self.cli.fine_alignment_linear_kp * dx_base,
                        -self.cli.fine_alignment_max_linear_speed,
                        self.cli.fine_alignment_max_linear_speed,
                    )
                    command.linear.y = self.clamp(
                        self.cli.fine_alignment_linear_kp * dy_base,
                        -self.cli.fine_alignment_max_linear_speed,
                        self.cli.fine_alignment_max_linear_speed,
                    )
                    planar_speed = math.hypot(command.linear.x, command.linear.y)
                    if planar_speed > self.cli.fine_alignment_max_linear_speed:
                        scale = self.cli.fine_alignment_max_linear_speed / planar_speed
                        command.linear.x *= scale
                        command.linear.y *= scale
                    angular_command = self.clamp(
                        self.cli.fine_alignment_angular_kp * yaw_error_signed,
                        -self.cli.fine_alignment_max_angular_speed,
                        self.cli.fine_alignment_max_angular_speed,
                    )
                    if (
                        yaw_error > yaw_tolerance
                        and 0.0 < abs(angular_command)
                        < self.cli.fine_alignment_min_angular_speed
                    ):
                        angular_command = math.copysign(
                            self.cli.fine_alignment_min_angular_speed,
                            angular_command,
                        )
                    command.angular.z = angular_command
                    self.stop_pub.publish(command)

                now = time.monotonic()
                if now - last_log >= 1.0:
                    self.get_logger().info(
                        f"Fine alignment {point.object_id}: "
                        f"distance={position_error:.3f}m, "
                        f"yaw={math.degrees(yaw_error):.1f}deg"
                    )
                    last_log = now
                if not self.sleep(0.05):
                    return False
        finally:
            self.stop_robot()

        self.get_logger().error(
            f"Fine alignment timed out for {point.object_id} after "
            f"{self.cli.fine_alignment_timeout:.1f}s"
        )
        return False

    def confirm_waypoint_reached(
        self,
        point: Waypoint,
        reason: str,
        timeout: float,
    ) -> bool:
        """Retry the TF arrival check to tolerate short localization/TF gaps."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if self.waypoint_reached(point, reason):
                return True
            if time.monotonic() >= deadline:
                return False
            if not self.sleep(min(0.5, deadline - time.monotonic())):
                return False

    def clear_navigation_costmaps(self, reason: str) -> bool:
        """Clear stale obstacle data without disabling Nav2 collision checks."""
        all_cleared = True
        self.stop_robot()
        for label, client in self.costmap_clients:
            if not client.wait_for_service(
                timeout_sec=self.cli.costmap_service_timeout
            ):
                self.get_logger().warning(
                    f"Cannot clear {label} costmap ({reason}): service unavailable"
                )
                all_cleared = False
                continue
            future = client.call_async(ClearEntireCostmap.Request())
            if not self.wait_future(future, self.cli.costmap_service_timeout):
                self.get_logger().warning(
                    f"Timed out clearing {label} costmap ({reason})"
                )
                all_cleared = False
                continue
            try:
                future.result()
            except Exception as exc:
                self.get_logger().warning(
                    f"Failed to clear {label} costmap ({reason}): {exc}"
                )
                all_cleared = False
                continue
            self.get_logger().info(f"Cleared {label} costmap ({reason})")
        if not self.sleep(self.cli.costmap_repopulate_delay):
            return False
        return all_cleared

    def get_bt_navigator_state(self, timeout: float = 3.0) -> Optional[str]:
        if not self.bt_state_client.wait_for_service(timeout_sec=timeout):
            return None
        future = self.bt_state_client.call_async(GetState.Request())
        if not self.wait_future(future, timeout):
            return None
        try:
            return str(future.result().current_state.label).strip().lower()
        except Exception:
            return None

    def ensure_navigation_active(self) -> bool:
        """Wait for Nav2's own lifecycle manager; never race it with commands."""
        deadline = time.monotonic() + self.cli.nav_lifecycle_timeout
        last_state: Optional[str] = None
        while time.monotonic() < deadline:
            state = self.get_bt_navigator_state(timeout=1.0)
            if state == "active":
                self.get_logger().info("bt_navigator is active; navigation may start")
                return True
            if state != last_state:
                self.get_logger().warning(
                    "Waiting for bt_navigator to become active; "
                    f"current state={state or 'unavailable'}"
                )
                last_state = state
            if not self.sleep(0.5):
                return False

        self.get_logger().error(
            "Nav2 did not become active within "
            f"{self.cli.nav_lifecycle_timeout:.1f}s"
        )
        return False

    def wait_for_map_tf(self) -> bool:
        """Wait until localization publishes map -> base frame TF."""
        deadline = time.monotonic() + self.cli.tf_ready_timeout
        last_error = "unknown"
        while time.monotonic() < deadline:
            try:
                self.tf_buffer.lookup_transform(
                    "map",
                    self.cli.base_frame,
                    Time(),
                )
                self.get_logger().info(
                    f"TF is ready: map -> {self.cli.base_frame}"
                )
                return True
            except Exception as exc:
                last_error = str(exc)
            if not self.sleep(0.5):
                return False
        self.get_logger().error(
            f"TF map->{self.cli.base_frame} was not ready within "
            f"{self.cli.tf_ready_timeout:.1f}s: {last_error}"
        )
        return False

    def _wait_navigation_result_or_stuck(self, result, point: Waypoint) -> str:
        """Wait for Nav2 while independently checking progress toward the goal."""
        started = time.monotonic()
        deadline = started + self.cli.goal_timeout
        last_progress = started
        progress_distance: Optional[float] = None
        progress_yaw_error: Optional[float] = None
        arrival_position_tolerance, _, _ = self.arrival_tolerances(point)
        last_log = 0.0

        while not result.done():
            if self.shutdown_event.is_set() or not rclpy.ok():
                return "shutdown"
            now = time.monotonic()
            current = self.current_map_pose()
            if current is not None:
                remaining = math.hypot(
                    point.x - current[0],
                    point.y - current[1],
                )
                yaw_error = abs(
                    self.normalize_angle(point.yaw - current[2])
                )
                if progress_distance is None:
                    progress_distance = remaining
                    progress_yaw_error = yaw_error
                    last_progress = now
                elif (
                    progress_distance - remaining
                    >= self.nav_stuck_min_translation
                ):
                    progress_distance = remaining
                    progress_yaw_error = yaw_error
                    last_progress = now
                elif (
                    remaining <= max(0.15, 2.0 * arrival_position_tolerance)
                    and progress_yaw_error is not None
                    and progress_yaw_error - yaw_error >= self.nav_stuck_min_yaw
                ):
                    progress_yaw_error = yaw_error
                    last_progress = now

                if now - last_log >= 2.0:
                    self.get_logger().info(
                        "[NAV_PROGRESS_CHECK] "
                        f"waypoint={point.object_id}, remaining={remaining:.3f}m, "
                        f"yaw_error={math.degrees(yaw_error):.1f}deg, "
                        f"no_progress={now-last_progress:.1f}s/"
                        f"{self.nav_stuck_timeout:.1f}s"
                    )
                    last_log = now

            if (
                now - started >= self.nav_stuck_grace
                and now - last_progress >= self.nav_stuck_timeout
            ):
                return "stuck"
            if now >= deadline:
                return "timeout"
            time.sleep(0.10)
        return "result"

    def _cancel_navigation_for_recovery(self, handle, result, reason: str) -> bool:
        # Remove Nav2 velocity ownership before cancellation so late controller
        # output cannot race the direct escape command.
        self.active_goal = None
        self.stop_robot()
        self.state("NAV_CANCEL_FOR_RECOVERY", reason)
        cancel_future = handle.cancel_goal_async()
        if not self.wait_future(cancel_future, 5.0):
            self.get_logger().error("等待 Nav2 取消确认超时")
            return False
        try:
            cancel_response = cancel_future.result()
            canceling = list(cancel_response.goals_canceling)
        except Exception as exc:
            self.get_logger().error(f"读取 Nav2 取消结果失败: {exc}")
            return False
        if not canceling and not result.done():
            self.get_logger().error("Nav2 未确认取消待脱困目标")
            return False
        if not result.done() and not self.wait_future(result, 5.0):
            self.get_logger().error("Nav2 已接受取消，但目标未在 5 秒内终止")
            return False
        self.stop_robot()
        self.state("NAV_CANCEL_CONFIRMED", reason)
        return True

    def navigate(self, point: Waypoint) -> bool:
        self.point_pub.publish(String(data=point.object_id))

        if not self.ensure_navigation_active():
            self.get_logger().error("Nav2 is not active")
            return False

        if not self.wait_for_map_tf():
            self.get_logger().error("Localization TF is not ready")
            return False

        # Do not ask Nav2 to navigate to a pose the robot has already reached.
        # Some Nav2 configurations return ABORTED for such a zero-length goal.
        if self.waypoint_reached(point, "before sending goal"):
            if self.finalize_arrival(point, "confirmed by TF"):
                return True

        if not self.nav.wait_for_server(timeout_sec=self.cli.nav_ready_timeout):
            self.get_logger().error("/navigate_to_pose 未就绪")
            return False

        for attempt in range(self.cli.nav_retries + 1):
            if attempt > 0:
                if self.confirm_waypoint_reached(
                    point,
                    "before retry",
                    self.cli.arrival_check_timeout,
                ):
                    if self.finalize_arrival(point, "confirmed before retry"):
                        return True
                if not self.ensure_navigation_active():
                    self.get_logger().error("Nav2 became inactive before retry")
                    return False
            self.state(
                "NAVIGATING",
                f"{point.object_id} attempt={attempt + 1}/"
                f"{self.cli.nav_retries + 1}",
            )
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.orientation.z = math.sin(point.yaw * 0.5)
            pose.pose.orientation.w = math.cos(point.yaw * 0.5)
            goal = NavigateToPose.Goal()
            goal.pose = pose

            sent = self.nav.send_goal_async(goal)
            if not self.wait_future(sent, 15.0):
                self.get_logger().error("发送导航目标超时")
                continue
            try:
                handle = sent.result()
            except Exception as exc:
                self.get_logger().error(f"发送目标异常: {exc}")
                continue
            if handle is None or not handle.accepted:
                self.get_logger().error("Nav2 拒绝目标")
                continue

            self.active_goal = handle
            result = handle.get_result_async()
            wait_outcome = self._wait_navigation_result_or_stuck(result, point)
            if wait_outcome != "result":
                if wait_outcome == "shutdown":
                    return False
                state_name = (
                    "NAV_STUCK_DETECTED"
                    if wait_outcome == "stuck"
                    else "NAV_GOAL_TIMEOUT"
                )
                reason = (
                    f"{point.object_id}: no qualifying goal progress for "
                    f"{self.nav_stuck_timeout:.1f}s"
                    if wait_outcome == "stuck"
                    else f"{point.object_id}: goal timeout={self.cli.goal_timeout:.1f}s"
                )
                self.state(state_name, reason)
                if not self._cancel_navigation_for_recovery(
                    handle, result, reason
                ):
                    return False
                if self.confirm_waypoint_reached(
                    point,
                    "after stuck/timeout cancellation",
                    self.cli.arrival_check_timeout,
                ):
                    if self.finalize_arrival(
                        point,
                        "TF confirmed arrival before escape",
                    ):
                        return True
                if attempt < self.cli.nav_retries:
                    if not self._apply_navigation_escape(
                        attempt + 1,
                        reason,
                    ):
                        return False
                continue
            self.active_goal = None
            try:
                wrapped_result = result.result()
                status = wrapped_result.status
                navigation_result = wrapped_result.result
            except Exception as exc:
                self.get_logger().error(f"读取导航结果异常: {exc}")
                self.stop_robot()
                continue
            # Always zero the base immediately when the action finishes.  In
            # particular, an ABORTED result must not leave the last controller
            # command active during TF arrival checks.
            self.stop_robot()
            nav_error_code = getattr(navigation_result, "error_code", None)
            nav_error_msg = str(
                getattr(navigation_result, "error_msg", "") or ""
            ).strip()
            if status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().error(
                    f"Nav2 result for {point.object_id}: status={status}, "
                    f"error_code={nav_error_code}, "
                    f"error_msg={nav_error_msg or 'not provided by this Nav2 build'}"
                )
            if status == GoalStatus.STATUS_SUCCEEDED:
                if self.finalize_arrival(point, "Nav2 succeeded + refined"):
                    return True
                self.get_logger().warning(
                    f"Nav2 reported success for {point.object_id}, but mandatory "
                    "pose refinement did not pass; retrying instead of starting task"
                )
                continue

            # Nav2 may report ABORTED after the controller has physically
            # reached the goal (or when the submitted goal is effectively the
            # current pose). Confirm against map->base_link before declaring
            # mission failure.
            if self.sleep(self.cli.abort_pose_check_delay) and self.confirm_waypoint_reached(
                point,
                f"after Nav2 status={status}",
                self.cli.arrival_check_timeout,
            ):
                self.get_logger().warning(
                    f"Nav2 returned status={status}, but TF confirms that "
                    f"{point.object_id} is near the target; refining pose."
                )
                if self.finalize_arrival(
                    point,
                    f"TF fallback after status={status} + refined",
                ):
                    return True
                if self.requires_precision_alignment(point):
                    self.get_logger().warning(
                        f"TF says {point.object_id} is near the target, but the "
                        "mandatory refinement failed; retrying Nav2"
                    )
                    continue
            if self.requires_precision_alignment(point):
                if self.finalize_arrival(
                    point,
                    f"low-speed fine alignment after status={status}",
                ):
                    return True
            elif self.fine_align(point):
                self.state(
                    "ARRIVED",
                    f"{point.object_id} (low-speed fine alignment after status={status})",
                )
                return True
            self.get_logger().error(
                f"导航 {point.object_id} 失败，status={status}"
            )
            self.stop_robot()
            if attempt < self.cli.nav_retries:
                if not self._apply_navigation_escape(
                    attempt + 1,
                    f"Nav2 terminal failure status={status}, "
                    f"error_code={nav_error_code}",
                ):
                    return False
        return False

    def start_process(
        self,
        label: str,
        command: str,
        setup_script: Optional[str] = None,
    ) -> subprocess.Popen:
        argv = shlex.split(command)
        if not argv:
            raise RuntimeError(f"{label} 命令为空")
        shown_command = command
        if setup_script:
            shell_command = (
                f"source {shlex.quote(setup_script)} && "
                f"exec {shlex.join(argv)}"
            )
            argv = ["bash", "-lc", shell_command]
            shown_command = f"source {setup_script} && {command}"
        self.state("TASK_START", f"{label}: {shown_command}")
        process = subprocess.Popen(
            argv,
            cwd=self.cli.box_dir,
            start_new_session=True,
        )
        self.processes.append(process)
        return process

    def stop_process(self, process: Optional[subprocess.Popen]) -> None:
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        if process in self.processes:
            self.processes.remove(process)

    def simple_task(self, label: str, key: str) -> bool:
        process = self.start_process(label, self.commands[key])
        deadline = time.monotonic() + self.cli.task_timeout
        while process.poll() is None:
            if self.shutdown_event.is_set() or not rclpy.ok():
                self.stop_process(process)
                return False
            if time.monotonic() >= deadline:
                self.get_logger().error(f"{label} 超时")
                self.stop_process(process)
                return False
            time.sleep(0.1)
        code = int(process.returncode)
        if process in self.processes:
            self.processes.remove(process)
        if code != 0:
            self.get_logger().error(f"{label} 失败，退出码={code}")
            return False
        self.state("TASK_SUCCEEDED", label)
        return True

    def grasp_task(self) -> bool:
        self.grasp_done.clear()
        vision = None
        grasp = None
        try:
            vision = self.start_process(
                "视觉识别",
                self.commands["vision"],
                setup_script=self.vision_setup,
            )
            if not self.sleep(self.cli.vision_warmup):
                return False
            if vision.poll() is not None:
                self.get_logger().error("视觉节点提前退出")
                return False
            grasp = self.start_process("箱体抓取", self.commands["grasp"])
            deadline = time.monotonic() + self.cli.task_timeout
            while time.monotonic() < deadline:
                if self.grasp_done.is_set():
                    self.state("TASK_SUCCEEDED", "视觉抓取")
                    return True
                if self.shutdown_event.is_set() or not rclpy.ok():
                    return False
                if vision.poll() is not None:
                    self.get_logger().error("抓取期间视觉节点退出")
                    return False
                if grasp.poll() is not None:
                    self.get_logger().error(
                        "抓取程序已退出，但没有收到 "
                        f"/box_grasp/completed=true；退出码={grasp.returncode}"
                    )
                    return False
                time.sleep(0.1)
            self.get_logger().error("视觉抓取任务超时")
            return False
        finally:
            self.stop_process(grasp)
            self.stop_process(vision)

    def task_at(self, object_id: str) -> bool:
        if object_id in self.grasp_waypoint_ids and not self.grasp_task():
            return False
        if object_id in self.lift_command_by_waypoint:
            command_key = self.lift_command_by_waypoint[object_id]
            target_height = (
                self.cli.lift_height_1
                if object_id == self.cli.lift_waypoint_1
                else self.cli.lift_height_2
            )
            if not self.simple_task(
                f"{object_id} 调整TCP中点高度至 {target_height:.2f}m",
                command_key,
            ):
                return False
            self.get_logger().info(
                f"{object_id} height adjustment completed; "
                "clearing stale costmap observations before transport"
            )
            self.clear_navigation_costmaps("after lift before transport")
        if object_id in self.release_waypoint_ids and not self.simple_task(
            "放置并释放", "release"
        ):
            return False
        if object_id in self.reset_waypoint_ids and not self.simple_task(
            "全部关节归零", "reset"
        ):
            return False
        return True

    def run_mission(self) -> bool:
        self.done_pub.publish(Bool(data=False))
        self.stop_robot()
        self.state(
            "STARTUP_RESET_BEGIN",
            "run all-joints zero reset once before the first waypoint",
        )
        if not self.simple_task("startup all-joints zero reset", "reset"):
            self.stop_robot()
            self.state(
                "STARTUP_RESET_FAILED",
                "reset command failed; waypoint navigation will not start",
            )
            return False
        self.state(
            "STARTUP_RESET_COMPLETE",
            "startup reset succeeded; waypoint navigation may start",
        )
        self.state(
            "READY",
            f"{len(self.waypoints)} points; round1="
            f"{self.cli.grasp_waypoint_1}->{self.cli.lift_waypoint_1}"
            f"({self.cli.lift_height_1:.2f}m)->"
            f"{self.cli.release_waypoint_1}; round2="
            f"{self.cli.grasp_waypoint_2}->{self.cli.lift_waypoint_2}"
            f"({self.cli.lift_height_2:.2f}m)->"
            f"{self.cli.release_waypoint_2}",
        )
        task_points = (
            self.grasp_waypoint_ids
            | set(self.lift_command_by_waypoint)
            | self.release_waypoint_ids
            | self.reset_waypoint_ids
        )
        for index, point in enumerate(self.waypoints, start=1):
            self.state("WAYPOINT", f"{index}/{len(self.waypoints)} {point.object_id}")
            if not self.navigate(point):
                self.state("FAILED", f"无法到达 {point.object_id}")
                return False
            settle = max(
                point.dwell_seconds,
                self.cli.arrival_settle if point.object_id in task_points else 0.0,
            )
            if not self.sleep(settle):
                return False
            if not self.task_at(point.object_id):
                self.stop_robot()
                self.state(
                    "FAILED",
                    f"{point.object_id} 任务失败，不继续导航",
                )
                return False

        if self.waypoints[-1].object_id != self.cli.final_waypoint:
            self.get_logger().warning(
                f"YAML 末点为 {self.waypoints[-1].object_id}，"
                f"不是 {self.cli.final_waypoint}"
            )
        self.stop_robot()
        self.done_pub.publish(Bool(data=True))
        self.state("COMPLETED", "全部路点和任务完成")
        return True

    def shutdown(self) -> None:
        self.shutdown_event.set()
        if self.active_goal is not None:
            try:
                self.active_goal.cancel_goal_async()
            except Exception:
                pass
        for process in list(self.processes):
            self.stop_process(process)
        self.stop_robot()


class RecoveryBoxWaypointMission(BoxWaypointMission):
    """原状态机的增量视觉确认与非 Nav2 底盘扫描纠错层。"""

    def __init__(self, cli: argparse.Namespace) -> None:
        super().__init__(cli)
        self.nav_cmd_vel_input_topic = os.environ.get(
            "BOX_MISSION_NAV_CMD_VEL_INPUT_TOPIC",
            "/box_mission/nav_cmd_vel",
        ).strip()
        self.base_cmd_vel_output_topic = os.environ.get(
            "BOX_MISSION_BASE_CMD_VEL_OUTPUT_TOPIC",
            "/mecanum_drive_controller/reference_unstamped",
        ).strip()
        if not self.nav_cmd_vel_input_topic or not self.base_cmd_vel_output_topic:
            raise RuntimeError("velocity arbiter topics must not be empty")
        if self.manual_cmd_vel_topic == self.base_cmd_vel_output_topic:
            raise RuntimeError("manual cmd_vel input must differ from base output")
        if self.nav_cmd_vel_input_topic == self.base_cmd_vel_output_topic:
            raise RuntimeError("Nav2 cmd_vel input must differ from base output")

        self._velocity_arbiter_lock = threading.Lock()
        self._velocity_arbiter_last_owner = ""
        self._velocity_arbiter_last_reject_log = 0.0
        self._velocity_arbiter_last_forward_log = 0.0
        self._manual_nonzero_received = 0
        self._base_nonzero_published = 0
        self.base_cmd_vel_pub = self.create_publisher(
            Twist,
            self.base_cmd_vel_output_topic,
            10,
        )
        self.create_subscription(
            Twist,
            self.nav_cmd_vel_input_topic,
            self._on_nav_cmd_vel,
            20,
        )
        self.create_subscription(
            Twist,
            self.manual_cmd_vel_topic,
            self._on_manual_cmd_vel,
            20,
        )
        self.vision_detection_timeout = _recovery_env_float(
            "BOX_MISSION_VISION_DETECTION_TIMEOUT", 25.0, 0.1
        )
        self.vision_stable_samples = _recovery_env_int(
            "BOX_MISSION_VISION_STABLE_SAMPLES", 1, 1
        )
        self.vision_sample_max_age = _recovery_env_float(
            "BOX_MISSION_VISION_SAMPLE_MAX_AGE", 5.0, 0.05
        )
        self.vision_point_sync_tolerance = _recovery_env_float(
            "BOX_MISSION_VISION_POINT_SYNC_TOLERANCE", 0.50, 0.0
        )
        self.vision_coord_stable_samples = _recovery_env_int(
            "BOX_MISSION_VISION_COORD_STABLE_SAMPLES", 1, 1
        )
        self.vision_coord_max_spread = _recovery_env_float(
            "BOX_MISSION_VISION_COORD_MAX_SPREAD", 0.080, 0.001
        )
        self.vision_recovery_cycles = _recovery_env_int(
            "BOX_MISSION_VISION_RECOVERY_CYCLES", 2, 0
        )
        self.vision_restart_delay = _recovery_env_float(
            "BOX_MISSION_VISION_RESTART_DELAY", 1.0, 0.0
        )
        self.vision_scan_speed = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_SPEED", 0.04, 0.005
        )
        self.vision_scan_min_speed = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_MIN_SPEED", 0.025, 0.0
        )
        self.vision_scan_yaw_tolerance = math.radians(
            _recovery_env_float(
                "BOX_MISSION_VISION_SCAN_YAW_TOLERANCE_DEG", 2.0, 0.1
            )
        )
        self.vision_scan_move_timeout = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_MOVE_TIMEOUT", 8.0, 0.5
        )
        self.vision_scan_dwell = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_DWELL", 3.0, 0.1
        )
        self.vision_scan_max_position_drift = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_MAX_POSITION_DRIFT", 0.04, 0.005
        )
        self.vision_scan_translation_distance = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_TRANSLATION_DISTANCE", 0.0, 0.0
        )
        self.vision_scan_linear_speed = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_LINEAR_SPEED", 0.025, 0.005
        )
        self.vision_scan_min_linear_speed = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_MIN_LINEAR_SPEED", 0.012, 0.0
        )
        self.vision_scan_position_tolerance = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_POSITION_TOLERANCE", 0.02, 0.002
        )
        self.vision_scan_lateral_drift = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_LATERAL_DRIFT", 0.05, 0.005
        )
        self.vision_scan_translation_timeout = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_TRANSLATION_TIMEOUT", 8.0, 0.5
        )
        if self.vision_scan_min_speed > self.vision_scan_speed:
            raise RuntimeError("视觉扫描最小角速度不能大于最大角速度")
        if self.vision_scan_min_linear_speed > self.vision_scan_linear_speed:
            raise RuntimeError("视觉扫描最小线速度不能大于最大线速度")

        self._vision_lock = threading.Lock()
        self._vision_epoch = time.monotonic()
        self._vision_capture_epoch = (
            self.get_clock().now().nanoseconds * 1e-9
        )
        self._vision_valid = False
        self._vision_valid_streak = 0
        self._vision_valid_received = 0.0
        self._vision_left: Optional[PointStamped] = None
        self._vision_right: Optional[PointStamped] = None
        self._vision_left_received = 0.0
        self._vision_right_received = 0.0
        self._vision_left_sequence = 0
        self._vision_right_sequence = 0
        self._vision_last_window_sequence = (0, 0)
        self._vision_coordinate_window = deque(
            maxlen=self.vision_coord_stable_samples
        )
        self._vision_center_2d_received = 0.0
        self._vision_left_2d_received = 0.0
        self._vision_right_2d_received = 0.0
        self._last_grasp_status = ""
        self._base_recovery_event = threading.Event()
        self._base_recovery_recommendation = None
        self.base_recovery_cycles = _recovery_env_int(
            "BOX_MISSION_BASE_RECOVERY_CYCLES", 3, 0
        )
        self.base_recovery_position_tolerance = _recovery_env_float(
            "BOX_MISSION_BASE_RECOVERY_POSITION_TOLERANCE", 0.02, 0.002
        )
        self.base_recovery_timeout = _recovery_env_float(
            "BOX_MISSION_BASE_RECOVERY_TIMEOUT", 20.0, 1.0
        )
        self.base_recovery_tf_stable_samples = _recovery_env_int(
            "BOX_MISSION_BASE_RECOVERY_TF_STABLE_SAMPLES", 3, 2
        )
        self.base_recovery_tf_stable_position = _recovery_env_float(
            "BOX_MISSION_BASE_RECOVERY_TF_STABLE_POSITION", 0.010, 0.0005
        )
        self.base_recovery_tf_stable_yaw = math.radians(
            _recovery_env_float(
                "BOX_MISSION_BASE_RECOVERY_TF_STABLE_YAW_DEG", 1.0, 0.05
            )
        )
        self.base_recovery_tf_stable_timeout = _recovery_env_float(
            "BOX_MISSION_BASE_RECOVERY_TF_STABLE_TIMEOUT", 10.0, 0.5
        )
        self.planning_recovery_cycles = _recovery_env_int(
            "BOX_MISSION_PLANNING_RECOVERY_CYCLES", 2, 0
        )
        self.planning_recovery_forward_step = _recovery_env_float(
            "BOX_MISSION_PLANNING_RECOVERY_FORWARD_STEP", 0.05, 0.005
        )
        self.planning_recovery_target_x = _recovery_env_float(
            "BOX_MISSION_PLANNING_RECOVERY_TARGET_X", 0.75, 0.50
        )
        self.planning_recovery_x_deadband = _recovery_env_float(
            "BOX_MISSION_PLANNING_RECOVERY_X_DEADBAND", 0.03, 0.0
        )
        self.planning_recovery_min_result_x = _recovery_env_float(
            "BOX_MISSION_PLANNING_RECOVERY_MIN_RESULT_X", 0.50, 0.0
        )
        self.planning_recovery_max_total = _recovery_env_float(
            "BOX_MISSION_PLANNING_RECOVERY_MAX_TOTAL", 0.10, 0.01
        )
        self.planning_recovery_linear_speed = _recovery_env_float(
            "BOX_MISSION_PLANNING_RECOVERY_LINEAR_SPEED", 0.015, 0.005
        )
        self.planning_recovery_position_tolerance = _recovery_env_float(
            "BOX_MISSION_PLANNING_RECOVERY_POSITION_TOLERANCE", 0.005, 0.001
        )
        self.planning_recovery_timeout = _recovery_env_float(
            "BOX_MISSION_PLANNING_RECOVERY_TIMEOUT", 45.0, 5.0
        )
        self.centerline_recovery_margin = _recovery_env_float(
            "BOX_MISSION_CENTERLINE_RECOVERY_MARGIN", 0.005, 0.0
        )
        self.centerline_recovery_min_step = _recovery_env_float(
            "BOX_MISSION_CENTERLINE_RECOVERY_MIN_STEP", 0.05, 0.005
        )
        self.centerline_recovery_max_step = _recovery_env_float(
            "BOX_MISSION_CENTERLINE_RECOVERY_MAX_STEP", 0.08, 0.01
        )
        self.centerline_recovery_max_total = _recovery_env_float(
            "BOX_MISSION_CENTERLINE_RECOVERY_MAX_TOTAL", 0.20, 0.01
        )
        self.centerline_recovery_linear_speed = _recovery_env_float(
            "BOX_MISSION_CENTERLINE_RECOVERY_LINEAR_SPEED", 0.02, 0.005
        )
        self.centerline_recovery_position_tolerance = _recovery_env_float(
            "BOX_MISSION_CENTERLINE_RECOVERY_POSITION_TOLERANCE", 0.005, 0.001
        )
        if self.planning_recovery_linear_speed > 0.03:
            raise RuntimeError("规划失败前后纠错速度不得超过 0.03m/s")
        if (
            self.planning_recovery_position_tolerance
            >= self.planning_recovery_forward_step
        ):
            raise RuntimeError("规划失败前后纠错到位容差必须小于单步距离")
        if self.centerline_recovery_linear_speed > 0.03:
            raise RuntimeError("中心线横移纠错速度不得超过 0.03m/s")
        if (
            self.centerline_recovery_min_step
            > self.centerline_recovery_max_step
        ):
            raise RuntimeError("中心线横移最小步长不能大于单步上限")
        if (
            self.centerline_recovery_max_step
            > self.centerline_recovery_max_total
        ):
            raise RuntimeError("中心线横移单步上限不能大于累计上限")
        if (
            self.centerline_recovery_position_tolerance
            >= self.centerline_recovery_min_step
        ):
            raise RuntimeError("中心线横移到位容差必须小于最小步长")
        if (
            self.planning_recovery_cycles
            * self.planning_recovery_forward_step
            > self.planning_recovery_max_total + 1e-9
        ):
            raise RuntimeError(
                "规划失败前后纠错次数×单步距离超过累计距离上限"
            )
        self.release_retreat_distance = _recovery_env_float(
            "BOX_MISSION_RELEASE_RETREAT_DISTANCE", 0.50, 0.0
        )
        self.release_retreat_speed = _recovery_env_float(
            "BOX_MISSION_RELEASE_RETREAT_SPEED", 0.05, 0.005
        )
        self.release_retreat_timeout = _recovery_env_float(
            "BOX_MISSION_RELEASE_RETREAT_TIMEOUT", 20.0, 1.0
        )
        self.grasp_retreat_distance = _recovery_env_float(
            "BOX_MISSION_GRASP_RETREAT_DISTANCE", 0.50, 0.0
        )
        self.grasp_retreat_speed = _recovery_env_float(
            "BOX_MISSION_GRASP_RETREAT_SPEED", 0.05, 0.005
        )
        self.grasp_retreat_timeout = _recovery_env_float(
            "BOX_MISSION_GRASP_RETREAT_TIMEOUT", 20.0, 1.0
        )
        self.nav_stuck_grace = _recovery_env_float(
            "BOX_MISSION_NAV_STUCK_GRACE", 8.0, 0.0
        )
        self.nav_stuck_timeout = _recovery_env_float(
            "BOX_MISSION_NAV_STUCK_TIMEOUT", 20.0, 1.0
        )
        self.nav_stuck_min_translation = _recovery_env_float(
            "BOX_MISSION_NAV_STUCK_MIN_TRANSLATION", 0.03, 0.001
        )
        self.nav_stuck_min_yaw = math.radians(
            _recovery_env_float(
                "BOX_MISSION_NAV_STUCK_MIN_YAW_DEG", 3.0, 0.1
            )
        )
        self.nav_escape_distance = _recovery_env_float(
            "BOX_MISSION_NAV_ESCAPE_DISTANCE", 0.15, 0.01
        )
        self.nav_escape_speed = _recovery_env_float(
            "BOX_MISSION_NAV_ESCAPE_SPEED", 0.035, 0.005
        )
        self.nav_escape_timeout = _recovery_env_float(
            "BOX_MISSION_NAV_ESCAPE_TIMEOUT", 12.0, 1.0
        )
        if self.release_retreat_speed > 0.08:
            raise RuntimeError("释放后退速度不得超过 0.08m/s")
        if self.grasp_retreat_speed > 0.08:
            raise RuntimeError("抓取后退速度不得超过 0.08m/s")
        if self.nav_escape_speed > 0.06:
            raise RuntimeError("导航脱困速度不得超过 0.06m/s")
        self._grasp_planning_failed_event = threading.Event()
        self._grasp_planning_entered_event = threading.Event()
        self._grasp_planning_passed_event = threading.Event()
        self._grasp_execution_started_event = threading.Event()
        self._grasp_planning_failure_detail = ""
        self._planning_recovery_left_x = None
        self._planning_recovery_right_x = None
        self._planning_recovery_requested = False
        self._centerline_recovery_event = threading.Event()
        self._centerline_recovery_requested = False
        self._centerline_recovery_detail = ""
        self._centerline_recovery_dy = 0.0

        self.create_subscription(
            Bool,
            "/box_grasp_point/valid",
            self._on_vision_valid,
            10,
        )
        self.create_subscription(
            PointStamped,
            "/box_grasp_point/short_edge_1",
            self._on_vision_left,
            10,
        )
        self.create_subscription(
            PointStamped,
            "/box_grasp_point/short_edge_2",
            self._on_vision_right,
            10,
        )
        self.create_subscription(
            String,
            "/box_grasp/status",
            self._on_grasp_status,
            10,
        )
        self.create_subscription(
            PointStamped,
            "/box_detection/center_pixel",
            self._on_vision_center_2d,
            10,
        )
        self.create_subscription(
            PointStamped,
            "/box_detection/short_edge_midpoint_1_pixel",
            self._on_vision_left_2d,
            10,
        )
        self.create_subscription(
            PointStamped,
            "/box_detection/short_edge_midpoint_2_pixel",
            self._on_vision_right_2d,
            10,
        )
        self.get_logger().warning(
            "视觉纠错已启用：稳定帧="
            f"{self.vision_stable_samples}，识别超时="
            f"{self.vision_detection_timeout:.1f}s，左右旋转扫描=禁用，前后距离="
            f"{self.vision_scan_translation_distance:.3f}m，三维稳定窗口="
            f"{self.vision_coord_stable_samples}帧/"
            f"{self.vision_coord_max_spread * 1000.0:.1f}mm"
        )
        self.get_logger().warning(
            "速度仲裁已启用："
            f"Nav2={self.nav_cmd_vel_input_topic}, "
            f"manual={self.manual_cmd_vel_topic}, "
            f"base={self.base_cmd_vel_output_topic}; "
            "active_goal存在时只放行Nav2，否则只放行手动纠错，零速度始终放行"
        )
        self.get_logger().warning(
            "规划失败前后自适应纠错已启用："
            f"step={self.planning_recovery_forward_step:.3f}m，"
            f"target_x={self.planning_recovery_target_x:.3f}m±"
            f"{self.planning_recovery_x_deadband:.3f}m，"
            f"min_result_x={self.planning_recovery_min_result_x:.3f}m，"
            f"speed={self.planning_recovery_linear_speed:.3f}m/s，"
            f"cycles={self.planning_recovery_cycles}，"
            f"max_total={self.planning_recovery_max_total:.3f}m，"
            f"position_tolerance="
            f"{self.planning_recovery_position_tolerance:.3f}m，"
            f"planning_timeout={self.planning_recovery_timeout:.1f}s"
        )
        self.get_logger().warning(
            "抓取点中心线纠错已启用：左右点未跨中心线时只横移、不前进；"
            f"margin={self.centerline_recovery_margin:.3f}m，"
            f"min_step={self.centerline_recovery_min_step:.3f}m，"
            f"max_step={self.centerline_recovery_max_step:.3f}m，"
            f"max_total={self.centerline_recovery_max_total:.3f}m，"
            f"speed={self.centerline_recovery_linear_speed:.3f}m/s，"
            f"position_tolerance="
            f"{self.centerline_recovery_position_tolerance:.3f}m"
        )
        self.get_logger().warning(
            "释放后退与导航脱困已启用："
            f"release_retreat={self.release_retreat_distance:.2f}m@"
            f"{self.release_retreat_speed:.3f}m/s；"
            f"grasp_retreat={self.grasp_retreat_distance:.2f}m@"
            f"{self.grasp_retreat_speed:.3f}m/s；"
            f"stuck={self.nav_stuck_timeout:.1f}s，"
            f"escape={self.nav_escape_distance:.2f}m@"
            f"{self.nav_escape_speed:.3f}m/s"
        )

    @staticmethod
    def _twist_is_zero(message: Twist, epsilon: float = 1e-6) -> bool:
        return all(
            abs(value) <= epsilon
            for value in (
                message.linear.x,
                message.linear.y,
                message.linear.z,
                message.angular.x,
                message.angular.y,
                message.angular.z,
            )
        )

    def _publish_arbitrated_velocity(self, owner: str, message: Twist) -> None:
        with self._velocity_arbiter_lock:
            if owner != self._velocity_arbiter_last_owner:
                self._velocity_arbiter_last_owner = owner
                self.state("VELOCITY_OWNER", owner)
            self.base_cmd_vel_pub.publish(message)
            if not self._twist_is_zero(message):
                self._base_nonzero_published += 1
                now = time.monotonic()
                if now - self._velocity_arbiter_last_forward_log >= 1.0:
                    self._velocity_arbiter_last_forward_log = now
                    self.get_logger().info(
                        "[VELOCITY_ARBITER_FORWARD] "
                        f"owner={owner}, "
                        f"cmd=(x={message.linear.x:+.3f},"
                        f"y={message.linear.y:+.3f},"
                        f"w={message.angular.z:+.3f}), "
                        f"manual_rx={self._manual_nonzero_received}, "
                        f"base_tx={self._base_nonzero_published}, "
                        f"base_subscribers="
                        f"{self.base_cmd_vel_pub.get_subscription_count()}"
                    )

    def _on_nav_cmd_vel(self, message: Twist) -> None:
        # Nav2 is allowed to drive only while this mission owns an accepted,
        # non-terminal NavigateToPose goal. Idle controller/smoother output is
        # therefore unable to overwrite manipulation recovery commands.
        if self.active_goal is not None:
            self._publish_arbitrated_velocity("NAV2", message)
            return
        if self._twist_is_zero(message):
            return
        now = time.monotonic()
        if now - self._velocity_arbiter_last_reject_log >= 2.0:
            self._velocity_arbiter_last_reject_log = now
            self.get_logger().warning(
                "[VELOCITY_ARBITER] ignored non-zero Nav2 cmd_vel without active goal"
            )

    def _on_manual_cmd_vel(self, message: Twist) -> None:
        # A zero command is always safe and must be able to stop the chassis
        # during cancellation/shutdown. Non-zero manual motion is rejected
        # while Nav2 still owns an action goal.
        if self._twist_is_zero(message):
            self._publish_arbitrated_velocity("STOP", message)
            return
        if self.active_goal is None:
            self._manual_nonzero_received += 1
            self._publish_arbitrated_velocity("MANUAL", message)
            return
        now = time.monotonic()
        if now - self._velocity_arbiter_last_reject_log >= 2.0:
            self._velocity_arbiter_last_reject_log = now
            self.get_logger().error(
                "[VELOCITY_ARBITER] rejected manual motion while Nav2 goal is active"
            )

    @staticmethod
    def _point_stamp_seconds(message: PointStamped) -> float:
        return (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1e-9
        )

    @staticmethod
    def _point_is_finite(message: PointStamped) -> bool:
        return all(
            math.isfinite(value)
            for value in (
                message.point.x,
                message.point.y,
                message.point.z,
            )
        )

    def _reset_vision_observation(self) -> None:
        with self._vision_lock:
            self._vision_epoch = time.monotonic()
            self._vision_capture_epoch = (
                self.get_clock().now().nanoseconds * 1e-9
            )
            self._vision_valid = False
            self._vision_valid_streak = 0
            self._vision_valid_received = 0.0
            self._vision_left = None
            self._vision_right = None
            self._vision_left_received = 0.0
            self._vision_right_received = 0.0
            self._vision_left_sequence = 0
            self._vision_right_sequence = 0
            self._vision_last_window_sequence = (0, 0)
            self._vision_coordinate_window.clear()
            self._vision_center_2d_received = 0.0
            self._vision_left_2d_received = 0.0
            self._vision_right_2d_received = 0.0

    def _on_vision_valid(self, message: Bool) -> None:
        with self._vision_lock:
            self._vision_valid = bool(message.data)
            self._vision_valid_received = time.monotonic()
            if message.data:
                self._vision_valid_streak += 1
            else:
                self._vision_valid_streak = 0

    def _on_vision_left(self, message: PointStamped) -> None:
        with self._vision_lock:
            self._vision_left = message
            self._vision_left_received = time.monotonic()
            self._vision_left_sequence += 1

    def _on_vision_right(self, message: PointStamped) -> None:
        with self._vision_lock:
            self._vision_right = message
            self._vision_right_received = time.monotonic()
            self._vision_right_sequence += 1

    def _on_vision_center_2d(self, _message: PointStamped) -> None:
        with self._vision_lock:
            self._vision_center_2d_received = time.monotonic()

    def _on_vision_left_2d(self, _message: PointStamped) -> None:
        with self._vision_lock:
            self._vision_left_2d_received = time.monotonic()

    def _on_vision_right_2d(self, _message: PointStamped) -> None:
        with self._vision_lock:
            self._vision_right_2d_received = time.monotonic()

    def _on_grasp_status(self, message: String) -> None:
        status = str(message.data).strip()
        if status and status != self._last_grasp_status:
            self._last_grasp_status = status
            self.state("GRASP_STATUS", status)
        prefix = "BASE_RECOVERY_RECOMMENDATION "
        if status.startswith(prefix):
            try:
                value = json.loads(status[len(prefix):])
                dx = float(value["dx"])
                dy = float(value["dy"])
                yaw = float(value["yaw"])
                result_min_x = float(value["result_min_x"])
                if not all(math.isfinite(v) for v in (dx, dy, yaw, result_min_x)):
                    raise ValueError("non-finite recovery value")
                if result_min_x < 0.50 - 1e-9:
                    raise ValueError(f"result_min_x={result_min_x:.3f}m < 0.50m")
                self._base_recovery_recommendation = value
                self._base_recovery_event.set()
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.get_logger().error(f"invalid base recovery recommendation: {exc}")

    @staticmethod
    def _vision_coordinate_spread(window) -> float:
        """Return the largest same-side 3D displacement in the window."""
        samples = list(window)
        maximum = 0.0
        for side in (0, 1):
            points = [sample[side] for sample in samples]
            for index, first in enumerate(points):
                for second in points[index + 1:]:
                    distance = math.sqrt(
                        sum(
                            (float(a) - float(b)) ** 2
                            for a, b in zip(first, second)
                        )
                    )
                    maximum = max(maximum, distance)
        return maximum

    def _vision_diagnostic(self) -> tuple[str, str]:
        now = time.monotonic()
        with self._vision_lock:
            valid = self._vision_valid
            streak = self._vision_valid_streak
            valid_received = self._vision_valid_received
            center_2d = self._vision_center_2d_received
            left_2d = self._vision_left_2d_received
            right_2d = self._vision_right_2d_received
            left_3d = self._vision_left_received
            right_3d = self._vision_right_received
            coordinate_window = list(self._vision_coordinate_window)

        def age_text(received: float) -> str:
            return "never" if received <= 0.0 else f"{now - received:.2f}s"

        recent_limit = self.vision_sample_max_age
        center_recent = center_2d > 0.0 and now - center_2d <= recent_limit
        edges_2d_recent = (
            left_2d > 0.0
            and right_2d > 0.0
            and now - left_2d <= recent_limit
            and now - right_2d <= recent_limit
        )
        points_3d_recent = (
            left_3d > 0.0
            and right_3d > 0.0
            and now - left_3d <= recent_limit
            and now - right_3d <= recent_limit
        )
        coordinate_spread = self._vision_coordinate_spread(
            coordinate_window
        )
        if not center_recent:
            stage = "NO_RECENT_2D_TARGET"
        elif not edges_2d_recent:
            stage = "2D_CENTER_WITHOUT_EDGE_MIDPOINTS"
        elif not points_3d_recent:
            stage = "2D_OK_BUT_3D_NOT_READY"
        elif not valid:
            stage = "3D_POINTS_PRESENT_BUT_VALID_FALSE"
        elif streak < self.vision_stable_samples:
            stage = "VALID_TRUE_NOT_YET_STABLE"
        elif len(coordinate_window) < self.vision_coord_stable_samples:
            stage = "COLLECTING_3D_STABILITY_WINDOW"
        elif coordinate_spread > self.vision_coord_max_spread:
            stage = "3D_POINTS_UNSTABLE"
        else:
            stage = "VISUAL_PIPELINE_READY"
        detail = (
            f"stage={stage}, valid={valid}, "
            f"streak={streak}/{self.vision_stable_samples}, "
            f"valid_age={age_text(valid_received)}, "
            f"center2d_age={age_text(center_2d)}, "
            f"edge2d_age=({age_text(left_2d)},{age_text(right_2d)}), "
            f"point3d_age=({age_text(left_3d)},{age_text(right_3d)}), "
            f"coord_window={len(coordinate_window)}/"
            f"{self.vision_coord_stable_samples}, "
            f"coord_spread={coordinate_spread * 1000.0:.1f}mm/"
            f"{self.vision_coord_max_spread * 1000.0:.1f}mm"
        )
        return stage, detail

    def _recent_visual_recognition(self) -> tuple[bool, str]:
        """True when this recovery round has seen any target evidence.

        Scanning is reserved for the complete absence of recognition. A 2D
        target without usable 3D points is reported and retried in place.
        """
        now = time.monotonic()
        with self._vision_lock:
            epoch = self._vision_epoch
            center_2d = self._vision_center_2d_received
            left_2d = self._vision_left_2d_received
            right_2d = self._vision_right_2d_received
            left_3d = self._vision_left_received
            right_3d = self._vision_right_received
        def seen_this_round(received: float) -> bool:
            return received >= epoch

        def age_text(received: float) -> str:
            return "never" if received <= 0.0 else f"{now - received:.2f}s"

        center_seen = seen_this_round(center_2d)
        edges_2d_seen = (
            seen_this_round(left_2d) or seen_this_round(right_2d)
        )
        points_3d_seen = (
            seen_this_round(left_3d) or seen_this_round(right_3d)
        )
        recognized = center_seen or edges_2d_seen or points_3d_seen
        detail = (
            f"center2d={center_seen}, edge2d={edges_2d_seen}, "
            f"point3d={points_3d_seen}, "
            f"ages=center:{age_text(center_2d)},"
            f"edges:({age_text(left_2d)},{age_text(right_2d)}),"
            f"points:({age_text(left_3d)},{age_text(right_3d)})"
        )
        return recognized, detail

    def _vision_target_ready(self) -> tuple[bool, str]:
        now = time.monotonic()
        with self._vision_lock:
            epoch = self._vision_epoch
            capture_epoch = self._vision_capture_epoch
            valid = self._vision_valid
            streak = self._vision_valid_streak
            valid_received = self._vision_valid_received
            left = self._vision_left
            right = self._vision_right
            left_received = self._vision_left_received
            right_received = self._vision_right_received
            left_sequence = self._vision_left_sequence
            right_sequence = self._vision_right_sequence
            center_2d_received = self._vision_center_2d_received
            left_2d_received = self._vision_left_2d_received
            right_2d_received = self._vision_right_2d_received

        if left is None or right is None:
            return False, "等待左右三维抓取点"
        receipt_times = (left_received, right_received)
        if any(received < epoch for received in receipt_times):
            return False, "等待本轮视觉的新数据"
        if valid:
            if valid_received < epoch:
                return False, "等待本轮视觉的新valid=true"
            if streak < self.vision_stable_samples:
                return False, f"稳定帧={streak}/{self.vision_stable_samples}"
            gate_mode = "valid=true"
        else:
            soft_2d_ready = (
                center_2d_received >= epoch
                and left_2d_received >= epoch
                and right_2d_received >= epoch
                and now - center_2d_received <= self.vision_sample_max_age
                and now - left_2d_received <= self.vision_sample_max_age
                and now - right_2d_received <= self.vision_sample_max_age
            )
            if not soft_2d_ready:
                return False, "valid=false，等待本轮完整二维证据和左右三维点"
            gate_mode = "valid=false-soft-3d"
        # Header stamps are camera capture time. Reject frames captured before
        # this recovery round even when slow processing delivers them later.
        for label, message in (("left", left), ("right", right)):
            capture_time = self._point_stamp_seconds(message)
            if capture_time > 1e-6 and capture_time < capture_epoch:
                return False, (
                    f"丢弃本轮开始前拍摄的{label}积压帧："
                    f"{capture_epoch - capture_time:.3f}s"
                )
        freshness_times = receipt_times + ((valid_received,) if valid else ())
        oldest_age = max(now - received for received in freshness_times)
        if oldest_age > self.vision_sample_max_age:
            return False, f"视觉数据已过期：{oldest_age:.2f}s"
        if not self._point_is_finite(left) or not self._point_is_finite(right):
            return False, "三维抓取点包含非有限数"
        if not left.header.frame_id or left.header.frame_id != right.header.frame_id:
            return False, "左右抓取点坐标系不一致"
        stamp_delta = abs(
            self._point_stamp_seconds(left)
            - self._point_stamp_seconds(right)
        )
        if stamp_delta > self.vision_point_sync_tolerance:
            return False, f"左右时间差={stamp_delta:.3f}s"

        pair_sequence = (left_sequence, right_sequence)
        with self._vision_lock:
            # Do not count repeated polling of one ROS pair as multiple stable
            # samples. Both point topics must advance before a new pair enters
            # the coordinate window.
            if pair_sequence != (
                self._vision_left_sequence,
                self._vision_right_sequence,
            ):
                return False, "左右三维点正在更新，等待完整新数据对"
            last_sequence = self._vision_last_window_sequence
            if pair_sequence != last_sequence:
                if (
                    last_sequence != (0, 0)
                    and (
                        pair_sequence[0] == last_sequence[0]
                        or pair_sequence[1] == last_sequence[1]
                    )
                ):
                    return False, "只更新了一侧三维点，等待另一侧"
                left_xyz = (
                    float(left.point.x),
                    float(left.point.y),
                    float(left.point.z),
                )
                right_xyz = (
                    float(right.point.x),
                    float(right.point.y),
                    float(right.point.z),
                )
                self._vision_coordinate_window.append(
                    (left_xyz, right_xyz)
                )
                self._vision_last_window_sequence = pair_sequence
            coordinate_window = list(self._vision_coordinate_window)

        coordinate_count = len(coordinate_window)
        coordinate_spread = self._vision_coordinate_spread(
            coordinate_window
        )
        if coordinate_count < self.vision_coord_stable_samples:
            return False, (
                f"累计三维稳定点={coordinate_count}/"
                f"{self.vision_coord_stable_samples}"
            )
        if coordinate_spread > self.vision_coord_max_spread:
            return False, (
                f"三维点仍在抖动：{coordinate_spread * 1000.0:.1f}mm > "
                f"{self.vision_coord_max_spread * 1000.0:.1f}mm"
            )
        return True, (
            f"gate_mode={gate_mode}，frame={left.header.frame_id}，"
            f"稳定帧={streak}，"
            f"左右时间差={stamp_delta:.3f}s，三维稳定窗口="
            f"{coordinate_count}帧/{coordinate_spread * 1000.0:.1f}mm"
        )

    def _wait_for_vision_target(
        self,
        vision: subprocess.Popen,
        timeout: float,
        phase: str,
    ) -> bool:
        deadline = time.monotonic() + timeout
        last_log = 0.0
        self.state("WAITING_FOR_VISION_TARGET", phase)
        while time.monotonic() < deadline:
            if self.shutdown_event.is_set() or not rclpy.ok():
                return False
            if vision.poll() is not None:
                self.get_logger().error(
                    f"视觉进程在 {phase} 期间退出，退出码={vision.returncode}"
                )
                return False
            ready, detail = self._vision_target_ready()
            if ready:
                self.stop_robot()
                self.state("VISION_TARGET_CONFIRMED", f"{phase}: {detail}")
                return True
            now = time.monotonic()
            if now - last_log >= 1.0:
                stage, diagnostic = self._vision_diagnostic()
                self.get_logger().info(
                    f"[VISION_DIAGNOSTIC] pid={vision.pid}, "
                    f"phase={phase}, process_alive={vision.poll() is None}, "
                    f"gate={detail}, {diagnostic}"
                )
                last_log = now
            if not self.sleep(0.05):
                return False
        stage, diagnostic = self._vision_diagnostic()
        self.get_logger().warning(
            f"{phase} 在 {timeout:.1f}s 内未获得稳定视觉目标；"
            f"final_stage={stage}；{diagnostic}"
        )
        return False

    def _rotate_without_nav(
        self,
        reference_pose: tuple[float, float, float],
        yaw_offset: float,
        vision: Optional[subprocess.Popen] = None,
    ) -> bool:
        """只发 cmd_vel；不发送、取消或重试任何 Nav2 目标。"""
        if self.active_goal is not None:
            self.get_logger().error("仍有活动 Nav2 目标，拒绝视觉扫描")
            return False
        reference_x, reference_y, reference_yaw = reference_pose
        target_yaw = self.normalize_angle(reference_yaw + yaw_offset)
        deadline = time.monotonic() + self.vision_scan_move_timeout
        self.state(
            "VISION_SCAN_MOVING",
            f"cmd_vel 航向偏移={math.degrees(yaw_offset):+.1f}deg",
        )
        last_log = 0.0
        try:
            while time.monotonic() < deadline:
                if self.shutdown_event.is_set() or not rclpy.ok():
                    return False
                if vision is not None and vision.poll() is not None:
                    return False
                current = self.current_map_pose()
                if current is None:
                    self.stop_robot()
                    if not self.sleep(0.1):
                        return False
                    continue
                current_x, current_y, current_yaw = current
                drift = math.hypot(
                    current_x - reference_x,
                    current_y - reference_y,
                )
                if drift > self.vision_scan_max_position_drift:
                    self.get_logger().error(
                        f"视觉扫描平移漂移 {drift:.3f}m 超过 "
                        f"{self.vision_scan_max_position_drift:.3f}m，停止"
                    )
                    return False
                yaw_error = self.normalize_angle(target_yaw - current_yaw)
                now = time.monotonic()
                if now - last_log >= 1.0:
                    self.get_logger().info(
                        "[VISION_SCAN] mode=yaw, "
                        f"target_offset={math.degrees(yaw_offset):+.1f}deg, "
                        f"remaining={math.degrees(yaw_error):+.1f}deg, "
                        f"position_drift={drift:.3f}m, "
                        "nav_goal_active=false"
                    )
                    last_log = now
                if abs(yaw_error) <= self.vision_scan_yaw_tolerance:
                    return True
                angular = self.clamp(
                    0.8 * yaw_error,
                    -self.vision_scan_speed,
                    self.vision_scan_speed,
                )
                if 0.0 < abs(angular) < self.vision_scan_min_speed:
                    angular = math.copysign(
                        self.vision_scan_min_speed,
                        angular,
                    )
                command = Twist()
                command.angular.z = angular
                self.stop_pub.publish(command)
                if not self.sleep(0.05):
                    return False
        finally:
            self.stop_robot()
        self.get_logger().warning(
            f"视觉扫描转动超时：{math.degrees(yaw_offset):+.1f}deg"
        )
        return False

    def _translate_without_nav(
        self,
        reference_pose: tuple[float, float, float],
        longitudinal_offset: float,
        vision: Optional[subprocess.Popen] = None,
        position_tolerance: Optional[float] = None,
    ) -> bool:
        """沿进入纠错时的车头方向前后移动，不调用任何 Nav2 接口。"""
        if self.active_goal is not None:
            self.get_logger().error("仍有活动 Nav2 目标，拒绝前后视觉扫描")
            return False
        reference_x, reference_y, reference_yaw = reference_pose
        cos_yaw = math.cos(reference_yaw)
        sin_yaw = math.sin(reference_yaw)
        effective_position_tolerance = (
            self.vision_scan_position_tolerance
            if position_tolerance is None
            else float(position_tolerance)
        )
        deadline = time.monotonic() + self.vision_scan_translation_timeout
        self.state(
            "VISION_SCAN_TRANSLATING",
            f"cmd_vel 纵向偏移={longitudinal_offset:+.3f}m",
        )
        last_log = 0.0
        try:
            while time.monotonic() < deadline:
                if self.shutdown_event.is_set() or not rclpy.ok():
                    return False
                if vision is not None and vision.poll() is not None:
                    return False
                current = self.current_map_pose()
                if current is None:
                    self.stop_robot()
                    if not self.sleep(0.1):
                        return False
                    continue
                current_x, current_y, current_yaw = current
                dx = current_x - reference_x
                dy = current_y - reference_y
                longitudinal = cos_yaw * dx + sin_yaw * dy
                lateral = -sin_yaw * dx + cos_yaw * dy
                position_error = longitudinal_offset - longitudinal
                yaw_error = self.normalize_angle(reference_yaw - current_yaw)

                now = time.monotonic()
                if now - last_log >= 1.0:
                    self.get_logger().info(
                        "[VISION_SCAN] mode=translation, "
                        f"target={longitudinal_offset:+.3f}m, "
                        f"current={longitudinal:+.3f}m, "
                        f"remaining={position_error:+.3f}m, "
                        f"tolerance={effective_position_tolerance:.3f}m, "
                        f"lateral={lateral:+.3f}m, "
                        f"yaw_error={math.degrees(yaw_error):+.1f}deg, "
                        "nav_goal_active=false"
                    )
                    last_log = now

                if abs(lateral) > self.vision_scan_lateral_drift:
                    self.get_logger().error(
                        f"前后扫描横向漂移 {lateral:+.3f}m 超过 "
                        f"{self.vision_scan_lateral_drift:.3f}m，停止"
                    )
                    return False
                allowed_longitudinal = (
                    max(
                        self.vision_scan_translation_distance,
                        abs(longitudinal_offset),
                    )
                    + 0.04
                )
                if abs(longitudinal) > allowed_longitudinal:
                    self.get_logger().error(
                        f"前后扫描纵向位移 {longitudinal:+.3f}m 越界，停止"
                    )
                    return False
                if (
                    abs(position_error)
                    <= effective_position_tolerance
                    and abs(yaw_error)
                    <= self.vision_scan_yaw_tolerance
                ):
                    return True

                # One-way displacement: after crossing the requested target,
                # stop instead of reversing to chase a small TF overshoot.
                target_crossed = (
                    longitudinal_offset > 0.0
                    and longitudinal
                    >= longitudinal_offset - effective_position_tolerance
                ) or (
                    longitudinal_offset < 0.0
                    and longitudinal
                    <= longitudinal_offset + effective_position_tolerance
                )
                if target_crossed:
                    self.state(
                        "DIRECT_TRANSLATION_TARGET_REACHED",
                        f"target={longitudinal_offset:+.3f}m, "
                        f"actual={longitudinal:+.3f}m；禁止反向追偿",
                    )
                    return True

                linear = self.clamp(
                    0.8 * position_error,
                    -self.vision_scan_linear_speed,
                    self.vision_scan_linear_speed,
                )
                if (
                    abs(position_error) > effective_position_tolerance
                    and 0.0 < abs(linear) < self.vision_scan_min_linear_speed
                ):
                    linear = math.copysign(
                        self.vision_scan_min_linear_speed,
                        linear,
                    )
                angular = self.clamp(
                    0.8 * yaw_error,
                    -self.vision_scan_speed,
                    self.vision_scan_speed,
                )
                if (
                    abs(yaw_error) > self.vision_scan_yaw_tolerance
                    and 0.0 < abs(angular) < self.vision_scan_min_speed
                ):
                    angular = math.copysign(
                        self.vision_scan_min_speed,
                        angular,
                    )
                command = Twist()
                command.linear.x = linear
                command.angular.z = angular
                self.stop_pub.publish(command)
                if not self.sleep(0.05):
                    return False
        finally:
            self.stop_robot()
        self.get_logger().warning(
            f"前后视觉扫描移动超时：{longitudinal_offset:+.3f}m"
        )
        return False

    def _scan_for_vision_target(self, vision: subprocess.Popen) -> bool:
        reference_pose = self.current_map_pose()
        if reference_pose is None:
            self.get_logger().error("没有 map->base TF，不能开始视觉扫描")
            return False
        if self.vision_scan_translation_distance <= 0.0:
            self.state(
                "VISION_TRANSLATION_RECOVERY_DISABLED",
                "前后视觉扫描已禁用；不会执行先前进再后退",
            )
            self.stop_robot()
            return False
        self.state(
            "VISION_TRANSLATION_RECOVERY",
            "rotational and reverse scans disabled; try one forward observation only",
        )
        if (
            vision.poll() is None
            and self.vision_scan_translation_distance > 0.0
        ):
            for offset in (self.vision_scan_translation_distance,):
                if not self._translate_without_nav(
                    reference_pose,
                    offset,
                    vision,
                ):
                    break
                if not self.sleep(0.4):
                    return False
                self._reset_vision_observation()
                if self._wait_for_vision_target(
                    vision,
                    self.vision_scan_dwell,
                    f"单向前观察位 {offset:+.3f}m",
                ):
                    return True
        self.state(
            "VISION_SCAN_NO_REVERSE",
            "单向观察未识别到目标；保持当前位置，不反向返回扫描起点",
        )
        self.stop_robot()
        return False

    def _translate_xy_without_nav(
        self,
        reference_pose: tuple[float, float, float],
        dx: float,
        dy: float,
        *,
        max_linear_speed: Optional[float] = None,
        move_timeout: Optional[float] = None,
        position_tolerance: Optional[float] = None,
        state_name: str = "BASE_RECOVERY_TRANSLATING",
        log_tag: str = "BASE_RECOVERY_MOVE",
    ) -> bool:
        """Closed-loop mecanum translation in the starting base frame."""
        if self.active_goal is not None:
            self.get_logger().error("active Nav2 goal exists; base recovery refused")
            return False
        reference_x, reference_y, reference_yaw = reference_pose
        target_map_x = reference_x + math.cos(reference_yaw) * dx - math.sin(reference_yaw) * dy
        target_map_y = reference_y + math.sin(reference_yaw) * dx + math.cos(reference_yaw) * dy
        linear_speed = (
            self.vision_scan_linear_speed
            if max_linear_speed is None
            else float(max_linear_speed)
        )
        timeout = (
            self.base_recovery_timeout
            if move_timeout is None
            else float(move_timeout)
        )
        effective_position_tolerance = (
            self.base_recovery_position_tolerance
            if position_tolerance is None
            else float(position_tolerance)
        )
        minimum_speed = min(self.vision_scan_min_linear_speed, linear_speed)
        deadline = time.monotonic() + timeout
        self.state(state_name, f"dx={dx:+.3f}m dy={dy:+.3f}m")
        last_log = 0.0
        try:
            while time.monotonic() < deadline:
                if self.shutdown_event.is_set() or not rclpy.ok():
                    return False
                current = self.current_map_pose()
                if current is None:
                    if not self.sleep(0.1):
                        return False
                    continue
                current_x, current_y, current_yaw = current
                error_map_x = target_map_x - current_x
                error_map_y = target_map_y - current_y
                error_x = math.cos(current_yaw) * error_map_x + math.sin(current_yaw) * error_map_y
                error_y = -math.sin(current_yaw) * error_map_x + math.cos(current_yaw) * error_map_y
                yaw_error = self.normalize_angle(reference_yaw - current_yaw)
                distance = math.hypot(error_map_x, error_map_y)
                now = time.monotonic()
                should_log = now - last_log >= 1.0
                if distance <= effective_position_tolerance and abs(yaw_error) <= self.vision_scan_yaw_tolerance:
                    traveled = math.hypot(
                        current_x - reference_x,
                        current_y - reference_y,
                    )
                    self.state(
                        "DIRECT_XY_TARGET_WITHIN_TOLERANCE",
                        f"requested=(dx={dx:+.3f},dy={dy:+.3f})m, "
                        f"actual_distance={traveled:.3f}m, "
                        f"remaining={distance:.3f}m/"
                        f"{effective_position_tolerance:.3f}m",
                    )
                    return True
                requested_distance = math.hypot(dx, dy)
                if requested_distance > 1e-9:
                    traveled_map_x = current_x - reference_x
                    traveled_map_y = current_y - reference_y
                    requested_map_x = target_map_x - reference_x
                    requested_map_y = target_map_y - reference_y
                    along = (
                        traveled_map_x * requested_map_x
                        + traveled_map_y * requested_map_y
                    ) / requested_distance
                    if along >= requested_distance - effective_position_tolerance:
                        self.state(
                            "DIRECT_XY_TARGET_REACHED",
                            f"requested={requested_distance:.3f}m, "
                            f"along={along:.3f}m；禁止越点后反向追偿",
                        )
                        return True
                command = Twist()
                command.linear.x = self.clamp(0.8 * error_x, -linear_speed, linear_speed)
                command.linear.y = self.clamp(0.8 * error_y, -linear_speed, linear_speed)
                if abs(error_x) > effective_position_tolerance and 0.0 < abs(command.linear.x) < self.vision_scan_min_linear_speed:
                    command.linear.x = math.copysign(minimum_speed, command.linear.x)
                if abs(error_y) > effective_position_tolerance and 0.0 < abs(command.linear.y) < self.vision_scan_min_linear_speed:
                    command.linear.y = math.copysign(minimum_speed, command.linear.y)
                command.angular.z = self.clamp(0.8 * yaw_error, -self.vision_scan_speed, self.vision_scan_speed)
                if should_log:
                    self.get_logger().info(
                        f"[{log_tag}] "
                        f"remaining_x={error_x:+.3f}m, "
                        f"remaining_y={error_y:+.3f}m, "
                        f"distance={distance:.3f}m, "
                        f"tolerance={effective_position_tolerance:.3f}m, "
                        f"yaw_error={math.degrees(yaw_error):+.1f}deg, "
                        f"cmd=(x={command.linear.x:+.3f},"
                        f"y={command.linear.y:+.3f},"
                        f"w={command.angular.z:+.3f}), "
                        f"manual_subscribers="
                        f"{self.stop_pub.get_subscription_count()}, "
                        "nav_goal_active=false"
                    )
                    last_log = now
                self.stop_pub.publish(command)
                if not self.sleep(0.05):
                    return False
        finally:
            self.stop_robot()
        self.get_logger().error(
            f"[{log_tag}] translation timed out after {timeout:.1f}s"
        )
        return False

    def task_at(self, object_id: str) -> bool:
        if not super().task_at(object_id):
            return False
        if object_id in self.grasp_waypoint_ids:
            return self._retreat_after_grasp()
        if object_id in self.release_waypoint_ids:
            return self._retreat_after_release()
        return True

    def _retreat_after_grasp(self) -> bool:
        if self.grasp_retreat_distance <= 0.0:
            self.state("GRASP_RETREAT_SKIPPED", "configured distance is zero")
            return True
        reference = self.current_map_pose()
        if reference is None:
            self.state(
                "GRASP_RETREAT_FAILED",
                "map->base_link unavailable; navigation remains blocked",
            )
            return False
        self.state(
            "GRASP_RETREAT_BEGIN",
            f"grasp completed; retreat={self.grasp_retreat_distance:.3f}m "
            f"speed<={self.grasp_retreat_speed:.3f}m/s; arm pose retained",
        )
        if not self._translate_xy_without_nav(
            reference,
            -self.grasp_retreat_distance,
            0.0,
            max_linear_speed=self.grasp_retreat_speed,
            move_timeout=self.grasp_retreat_timeout,
            state_name="GRASP_RETREAT_MOVING",
            log_tag="GRASP_RETREAT_PROGRESS",
        ):
            self.state(
                "GRASP_RETREAT_FAILED",
                f"{self.grasp_retreat_distance:.3f}m retreat did not reach "
                "its TF-verified target",
            )
            return False
        if not self._wait_for_recovery_tf_stability():
            self.state(
                "GRASP_RETREAT_FAILED",
                "base did not settle after retreat; navigation remains blocked",
            )
            return False
        if not self.clear_navigation_costmaps("after grasp retreat"):
            self.get_logger().warning(
                "抓取后退已完成，但一个或多个 Nav2 代价地图清理失败"
            )
        self.state(
            "GRASP_RETREAT_COMPLETE",
            f"TF confirmed {self.grasp_retreat_distance:.3f}m backward "
            "retreat; next navigation may start",
        )
        return True

    def _retreat_after_release(self) -> bool:
        if self.release_retreat_distance <= 0.0:
            self.state(
                "RELEASE_RETREAT_SKIPPED",
                "后退已禁用；直接执行释放后的关节复位",
            )
            if not self.simple_task("post-release all-joints zero reset", "reset"):
                self.stop_robot()
                self.state(
                    "POST_RELEASE_RESET_FAILED",
                    "reset command failed; navigation remains blocked",
                )
                return False
            if not self.clear_navigation_costmaps("after release without retreat"):
                self.get_logger().warning(
                    "释放后复位已完成，但一个或多个 Nav2 代价地图清理失败"
                )
            self.state(
                "POST_RELEASE_RESET_COMPLETE",
                "未后退；全关节复位成功，下一导航可开始",
            )
            return True
        reference = self.current_map_pose()
        if reference is None:
            self.state(
                "RELEASE_RETREAT_FAILED",
                "map->base_link unavailable; navigation remains blocked",
            )
            return False
        self.state(
            "RELEASE_RETREAT_BEGIN",
            f"release succeeded; retreat={self.release_retreat_distance:.3f}m "
            f"speed<={self.release_retreat_speed:.3f}m/s",
        )
        if not self._translate_xy_without_nav(
            reference,
            -self.release_retreat_distance,
            0.0,
            max_linear_speed=self.release_retreat_speed,
            move_timeout=self.release_retreat_timeout,
            state_name="RELEASE_RETREAT_MOVING",
            log_tag="RELEASE_RETREAT_PROGRESS",
        ):
            self.state(
                "RELEASE_RETREAT_FAILED",
                f"{self.release_retreat_distance:.3f}m retreat did not reach "
                "its TF-verified target",
            )
            return False
        if not self._wait_for_recovery_tf_stability():
            self.state(
                "RELEASE_RETREAT_FAILED",
                "base did not settle after retreat; navigation remains blocked",
            )
            return False
        self.state(
            "POST_RETREAT_RESET_BEGIN",
            f"{self.release_retreat_distance:.3f}m retreat completed; "
            "run all-joints zero reset before navigation",
        )
        if not self.simple_task("post-retreat all-joints zero reset", "reset"):
            self.stop_robot()
            self.state(
                "POST_RETREAT_RESET_FAILED",
                "reset command failed; navigation remains blocked",
            )
            return False
        self.state(
            "POST_RETREAT_RESET_COMPLETE",
            "all-joints zero reset succeeded; preparing Nav2",
        )
        if not self.clear_navigation_costmaps("after release retreat"):
            self.get_logger().warning(
                "释放后退已完成，但一个或多个 Nav2 代价地图清理失败"
            )
        self.state(
            "RELEASE_RETREAT_COMPLETE",
            f"TF confirmed {self.release_retreat_distance:.3f}m retreat and "
            "reset succeeded; next navigation may start",
        )
        return True

    def _apply_navigation_escape(self, cycle: int, reason: str) -> bool:
        # One bounded direct move per Nav2 retry. Reverse escape is disabled;
        # only mecanum lateral motion or a forward move may be selected.
        candidates = (
            ("left", 0.0, self.nav_escape_distance),
            ("right", 0.0, -self.nav_escape_distance),
            ("forward", self.nav_escape_distance, 0.0),
        )
        label, dx, dy = candidates[(max(1, cycle) - 1) % len(candidates)]
        if self.active_goal is not None:
            self.state(
                "NAV_ESCAPE_BLOCKED",
                "Nav2 goal still active; direct chassis motion refused",
            )
            return False
        reference = self.current_map_pose()
        if reference is None:
            self.state("NAV_ESCAPE_FAILED", "map->base_link unavailable")
            return False
        self.state(
            "NAV_ESCAPE_SELECTED",
            f"cycle={cycle}, direction={label}, dx={dx:+.3f}m, "
            f"dy={dy:+.3f}m, reason={reason}",
        )
        if not self._translate_xy_without_nav(
            reference,
            dx,
            dy,
            max_linear_speed=self.nav_escape_speed,
            move_timeout=self.nav_escape_timeout,
            state_name="NAV_ESCAPE_MOVING",
            log_tag="NAV_ESCAPE_PROGRESS",
        ):
            self.state(
                "NAV_ESCAPE_FAILED",
                f"cycle={cycle}, direction={label}",
            )
            return False
        if not self._wait_for_recovery_tf_stability():
            self.state("NAV_ESCAPE_FAILED", "map->base_link did not settle")
            return False
        if not self.clear_navigation_costmaps(
            f"after navigation escape cycle {cycle} ({label})"
        ):
            self.get_logger().warning(
                "导航脱困移动已完成，但一个或多个代价地图清理失败"
            )
        self.state(
            "NAV_ESCAPE_COMPLETE",
            f"cycle={cycle}, direction={label}; Nav2 retry may start",
        )
        return True

    def _apply_base_recovery(self, recommendation: dict) -> bool:
        reference = self.current_map_pose()
        if reference is None:
            self.get_logger().error("map->base_link unavailable; cannot recover base")
            return False
        dx = float(recommendation["dx"])
        dy = float(recommendation["dy"])
        yaw = float(recommendation["yaw"])
        self.state(
            "BASE_RECOVERY_SELECTED",
            f"dx={dx:+.3f}m dy={dy:+.3f}m yaw={math.degrees(yaw):+.1f}deg "
            f"result_min_x={float(recommendation['result_min_x']):.3f}m",
        )
        if not self._translate_xy_without_nav(reference, dx, dy):
            return False
        translated = self.current_map_pose()
        if translated is None or not self._rotate_without_nav(translated, yaw):
            return False
        self.stop_robot()
        if not self._wait_for_recovery_tf_stability():
            self.state("BASE_RECOVERY_TF_UNSTABLE", "map->base_link did not settle")
            return False
        # A direct chassis correction changes the robot pose without a Nav2
        # action. Refresh costmaps before ownership can return to navigation.
        if not self.clear_navigation_costmaps("after direct base recovery"):
            self.get_logger().warning(
                "one or more costmaps were not cleared after recovery; "
                "the next navigation goal will still re-check lifecycle and TF"
            )
        self.state("BASE_RECOVERY_APPLIED", "new vision points are now required")
        return True

    def _wait_for_recovery_tf_stability(self) -> bool:
        deadline = time.monotonic() + self.base_recovery_tf_stable_timeout
        previous = None
        stable = 0
        while time.monotonic() < deadline:
            current = self.current_map_pose()
            if current is None:
                stable = 0
                previous = None
            elif previous is None:
                previous = current
                stable = 1
            else:
                position_delta = math.hypot(
                    current[0] - previous[0],
                    current[1] - previous[1],
                )
                yaw_delta = abs(self.normalize_angle(current[2] - previous[2]))
                if (
                    position_delta <= self.base_recovery_tf_stable_position
                    and yaw_delta <= self.base_recovery_tf_stable_yaw
                ):
                    stable += 1
                else:
                    stable = 1
                previous = current
                if stable >= self.base_recovery_tf_stable_samples:
                    self.state(
                        "BASE_RECOVERY_TF_STABLE",
                        f"samples={stable}, pose=({current[0]:+.3f},"
                        f"{current[1]:+.3f},{math.degrees(current[2]):+.1f}deg)",
                    )
                    return True
            if not self.sleep(0.1):
                return False
        return False

    def _monitor_grasp_output(self, process: subprocess.Popen) -> None:
        """Forward V7 output and convert explicit log markers into events."""
        stream = process.stdout
        if stream is None:
            self.get_logger().error(
                "[GRASP_MONITOR] 抓取进程没有可读取的输出流"
            )
            return
        try:
            for raw_line in stream:
                line = raw_line.rstrip()
                if not line:
                    continue
                self.get_logger().info(f"[GRASP_PROCESS_OUTPUT] {line}")
                if "[GRASP_PLANNING] ENTER" in line:
                    target_x = self._parse_v7_planning_target_x(line)
                    if target_x is not None:
                        left_x, right_x = target_x
                        self._planning_recovery_left_x = left_x
                        self._planning_recovery_right_x = right_x
                        self.state(
                            "GRASP_PLANNING_TARGET_X",
                            f"left_x={left_x:.4f}m, right_x={right_x:.4f}m, "
                            f"center_x={0.5 * (left_x + right_x):.4f}m, "
                            f"min_x={min(left_x, right_x):.4f}m",
                        )
                    self._grasp_planning_entered_event.set()
                    self.state("GRASP_PLANNING_ENTERED", line[-500:])
                if (
                    "[GRASP_PLANNING] PASSED" in line
                    and not self._grasp_execution_started_event.is_set()
                ):
                    self._grasp_planning_passed_event.set()
                    self.state("GRASP_PLANNING_PASSED", line[-500:])
                centerline = self._parse_centerline_rejection(line)
                if (
                    centerline is not None
                    and not self._grasp_execution_started_event.is_set()
                    and not self._centerline_recovery_event.is_set()
                ):
                    dy, detail = centerline
                    self._centerline_recovery_dy = dy
                    self._centerline_recovery_detail = detail
                    self._centerline_recovery_event.set()
                    self.state(
                        "GRASP_CENTERLINE_REJECTION_DETECTED",
                        detail,
                    )
                if any(
                    marker in line
                    for marker in (
                        "REAL ASYMMETRIC DUAL-ARM STAGED BOX GRASP START",
                        "REAL DUAL-ARM BOX GRASP START",
                        "[1/8] lift",
                    )
                ):
                    if not self._grasp_execution_started_event.is_set():
                        self.state(
                            "GRASP_REAL_EXECUTION_STARTED",
                            "检测到V7真实动作开始标志；禁止底盘规划纠错",
                        )
                    self._grasp_execution_started_event.set()
                if (
                    ("规划失败" in line or "[GRASP_PLANNING] FAILED" in line)
                    and not self._grasp_execution_started_event.is_set()
                ):
                    self._grasp_planning_failure_detail = line
                    self._grasp_planning_failed_event.set()
                    self.state(
                        "GRASP_PLANNING_FAILURE_DETECTED",
                        line[-500:],
                    )
        except Exception as exc:
            self.get_logger().error(
                "[GRASP_MONITOR] 读取抓取输出失败："
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            try:
                stream.close()
            except Exception:
                pass
            self.get_logger().info(
                "[GRASP_MONITOR] 抓取进程输出流已结束"
            )

    @staticmethod
    def _parse_v7_planning_target_x(
        line: str,
    ) -> Optional[tuple[float, float]]:
        """Read base_link X values from V7's structured planning entry log."""
        number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        match = re.search(
            rf"left_raw=\[\s*({number}).*?right_raw=\[\s*({number})",
            line,
        )
        if match is None:
            return None
        left_x = float(match.group(1))
        right_x = float(match.group(2))
        if not math.isfinite(left_x) or not math.isfinite(right_x):
            return None
        return left_x, right_x

    def _select_planning_recovery_step(self) -> tuple[float, str]:
        """Choose forward (+X) or backward (-X) from the latest base targets."""
        magnitude = self.planning_recovery_forward_step
        left_x = self._planning_recovery_left_x
        right_x = self._planning_recovery_right_x
        if left_x is None or right_x is None:
            return magnitude, (
                "没有解析到V7的base_link抓取点X；使用兼容兜底方向=前进"
            )

        center_x = 0.5 * (left_x + right_x)
        min_x = min(left_x, right_x)
        lower = (
            self.planning_recovery_target_x
            - self.planning_recovery_x_deadband
        )
        upper = (
            self.planning_recovery_target_x
            + self.planning_recovery_x_deadband
        )
        predicted_forward_min_x = min_x - magnitude

        if predicted_forward_min_x < self.planning_recovery_min_result_x:
            signed_step = -magnitude
            reason = (
                "前进会突破最小X安全门槛，强制选择后退"
            )
        elif center_x > upper:
            signed_step = magnitude
            reason = "目标中心偏远，选择前进靠近"
        elif center_x < lower:
            signed_step = -magnitude
            reason = "目标中心偏近，选择后退拉开"
        else:
            signed_step = magnitude
            reason = "目标中心位于工作带内；规划失败时优先小步前进改善可达性"

        detail = (
            f"{reason}；left_x={left_x:.4f}m, right_x={right_x:.4f}m, "
            f"center_x={center_x:.4f}m, target_band=[{lower:.4f},{upper:.4f}]m, "
            f"predicted_forward_min_x={predicted_forward_min_x:.4f}m, "
            f"min_result_x={self.planning_recovery_min_result_x:.4f}m"
        )
        return signed_step, detail

    def _parse_centerline_rejection(
        self,
        line: str,
    ) -> Optional[tuple[float, str]]:
        """Turn the V7 straddle safety rejection into a lateral base move."""
        if "视觉抓取点被安全检查拒绝" not in line:
            return None
        patterns = (
            (
                "left",
                r"左抓取点 Y=([+-]?\d+(?:\.\d+)?)m.*?要求 >= ([+-]?\d+(?:\.\d+)?)m",
            ),
            (
                "right",
                r"右抓取点 Y=([+-]?\d+(?:\.\d+)?)m.*?要求 <= ([+-]?\d+(?:\.\d+)?)m",
            ),
        )
        for side, pattern in patterns:
            match = re.search(pattern, line)
            if match is None:
                continue
            measured_y = float(match.group(1))
            threshold_y = float(match.group(2))
            desired_y = (
                threshold_y + self.centerline_recovery_margin
                if side == "left"
                else threshold_y - self.centerline_recovery_margin
            )
            # In the starting base frame, object_y_new = object_y_old - base_dy.
            requested_dy = measured_y - desired_y
            bounded_dy = math.copysign(
                self.clamp(
                    abs(requested_dy),
                    self.centerline_recovery_min_step,
                    self.centerline_recovery_max_step,
                ),
                requested_dy,
            )
            detail = (
                f"side={side}, measured_y={measured_y:+.4f}m, "
                f"threshold={threshold_y:+.4f}m, "
                f"desired_with_margin={desired_y:+.4f}m, "
                f"computed_base_dy={requested_dy:+.4f}m, "
                f"bounded_base_dy={bounded_dy:+.4f}m；"
                "本次只横移，不进入规划失败前后纠错"
            )
            return bounded_dy, detail
        return None

    def _start_monitored_grasp_process(self) -> subprocess.Popen:
        command = self.commands["grasp"]
        argv = shlex.split(command)
        if not argv:
            raise RuntimeError("箱体抓取命令为空")
        self._grasp_planning_failed_event.clear()
        self._grasp_planning_entered_event.clear()
        self._grasp_planning_passed_event.clear()
        self._grasp_execution_started_event.clear()
        self._centerline_recovery_event.clear()
        self._grasp_planning_failure_detail = ""
        self._planning_recovery_left_x = None
        self._planning_recovery_right_x = None
        self._planning_recovery_requested = False
        self._centerline_recovery_requested = False
        self._centerline_recovery_detail = ""
        self._centerline_recovery_dy = 0.0
        self.state("TASK_START", f"箱体抓取（监控模式）: {command}")
        process = subprocess.Popen(
            argv,
            cwd=self.cli.box_dir,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        self.processes.append(process)
        monitor = threading.Thread(
            target=self._monitor_grasp_output,
            args=(process,),
            daemon=True,
            name=f"grasp-output-{process.pid}",
        )
        monitor.start()
        return process

    def _apply_planning_failure_translation(
        self,
        cycle: int,
        cumulative_before: float,
    ) -> bool:
        if self.active_goal is not None:
            self.state(
                "PLANNING_RECOVERY_BLOCKED",
                "Nav2目标仍处于活动状态，禁止底盘前后纠错",
            )
            return False
        step, selection_detail = self._select_planning_recovery_step()
        direction = "forward" if step > 0.0 else "backward"
        cumulative_after = cumulative_before + abs(step)
        if cumulative_after > self.planning_recovery_max_total + 1e-9:
            self.state(
                "PLANNING_RECOVERY_LIMIT",
                f"请求累计={cumulative_after:.3f}m > "
                f"上限={self.planning_recovery_max_total:.3f}m",
            )
            return False
        reference = self.current_map_pose()
        if reference is None:
            self.state(
                "PLANNING_RECOVERY_NO_TF",
                "缺少map->base_link，不能安全执行前后纠错",
            )
            return False
        self.state(
            "PLANNING_RECOVERY_DIRECTION_SELECTED",
            f"cycle={cycle}/{self.planning_recovery_cycles}, "
            f"direction={direction}, step={step:+.3f}m, speed<="
            f"{self.planning_recovery_linear_speed:.3f}m/s, "
            f"cumulative={cumulative_after:.3f}/"
            f"{self.planning_recovery_max_total:.3f}m, "
            f"nav_goal_active=false；{selection_detail}",
        )
        old_speed = self.vision_scan_linear_speed
        old_min_speed = self.vision_scan_min_linear_speed
        try:
            self.vision_scan_linear_speed = (
                self.planning_recovery_linear_speed
            )
            self.vision_scan_min_linear_speed = min(
                old_min_speed,
                self.planning_recovery_linear_speed,
            )
            moved = self._translate_without_nav(
                reference,
                step,
                vision=None,
                position_tolerance=self.planning_recovery_position_tolerance,
            )
        finally:
            self.vision_scan_linear_speed = old_speed
            self.vision_scan_min_linear_speed = old_min_speed
            self.stop_robot()
        if not moved:
            self.state(
                "PLANNING_RECOVERY_MOVE_FAILED",
                f"cycle={cycle}, direction={direction}, "
                f"requested_step={step:+.3f}m",
            )
            return False
        if not self._wait_for_recovery_tf_stability():
            self.state(
                "PLANNING_RECOVERY_TF_UNSTABLE",
                f"cycle={cycle}",
            )
            return False
        if not self.clear_navigation_costmaps(
            f"after planning-failure {direction} translation"
        ):
            self.get_logger().warning(
                "[PLANNING_RECOVERY] 代价地图未全部清理；"
                "后续导航仍会重新检查Nav2和TF"
            )
        self.state(
            "PLANNING_RECOVERY_APPLIED",
            f"cycle={cycle}, direction={direction}, step={step:+.3f}m, "
            f"cumulative={cumulative_after:.3f}m；"
            "下一轮必须重新获取视觉并重新规划",
        )
        return True

    def _apply_centerline_lateral_recovery(
        self,
        cycle: int,
        cumulative_before: float,
    ) -> bool:
        if self.active_goal is not None:
            self.state(
                "CENTERLINE_RECOVERY_BLOCKED",
                "Nav2目标仍处于活动状态，禁止底盘横移",
            )
            return False
        dy = self._centerline_recovery_dy
        cumulative_after = cumulative_before + abs(dy)
        if cumulative_after > self.centerline_recovery_max_total + 1e-9:
            self.state(
                "CENTERLINE_RECOVERY_LIMIT",
                f"请求累计={cumulative_after:.3f}m > "
                f"上限={self.centerline_recovery_max_total:.3f}m",
            )
            return False
        reference = self.current_map_pose()
        if reference is None:
            self.state(
                "CENTERLINE_RECOVERY_NO_TF",
                "缺少map->base_link，不能安全横移",
            )
            return False
        self.state(
            "CENTERLINE_RECOVERY_LATERAL_MOVE",
            f"cycle={cycle}/{self.base_recovery_cycles}, dx=+0.000m, "
            f"dy={dy:+.3f}m, speed<="
            f"{self.centerline_recovery_linear_speed:.3f}m/s, "
            f"cumulative={cumulative_after:.3f}/"
            f"{self.centerline_recovery_max_total:.3f}m；"
            "forward_approach=false, nav_goal_active=false",
        )
        if not self._translate_xy_without_nav(
            reference,
            0.0,
            dy,
            max_linear_speed=self.centerline_recovery_linear_speed,
            move_timeout=self.base_recovery_timeout,
            position_tolerance=self.centerline_recovery_position_tolerance,
            state_name="CENTERLINE_RECOVERY_MOVING",
            log_tag="CENTERLINE_RECOVERY_PROGRESS",
        ):
            self.state(
                "CENTERLINE_RECOVERY_MOVE_FAILED",
                f"cycle={cycle}, requested_dy={dy:+.3f}m",
            )
            return False
        if not self._wait_for_recovery_tf_stability():
            self.state("CENTERLINE_RECOVERY_TF_UNSTABLE", f"cycle={cycle}")
            return False
        if not self.clear_navigation_costmaps("after centerline lateral recovery"):
            self.get_logger().warning(
                "[CENTERLINE_RECOVERY] 横移完成，但代价地图未全部清理"
            )
        self.state(
            "CENTERLINE_RECOVERY_APPLIED",
            f"cycle={cycle}, dy={dy:+.3f}m, "
            f"cumulative={cumulative_after:.3f}m；重新获取视觉并重新安全检查",
        )
        return True

    def _execute_confirmed_grasp(
        self,
        vision: subprocess.Popen,
    ) -> Optional[bool]:
        grasp = None
        try:
            self._base_recovery_event.clear()
            self._base_recovery_recommendation = None
            grasp = self._start_monitored_grasp_process()
            self.state(
                "GRASP_PROCESS_STARTED",
                f"pid={grasp.pid}; 等待抓取节点订阅左右三维点",
            )
            deadline = time.monotonic() + self.cli.task_timeout
            handoff_started = time.monotonic()
            last_handoff_log = 0.0
            last_phase_log = 0.0
            while time.monotonic() < deadline:
                now = time.monotonic()
                if now - last_phase_log >= 2.0:
                    if self._grasp_execution_started_event.is_set():
                        phase = "REAL_EXECUTION_STARTED"
                    elif self._grasp_planning_passed_event.is_set():
                        phase = "PLANNING_PASSED_WAITING_EXECUTION"
                    elif self._grasp_planning_entered_event.is_set():
                        phase = "PLANNING_IN_PROGRESS"
                    else:
                        phase = "WAITING_CONTROLLER_OR_TARGET_HANDOFF"
                    self.get_logger().info(
                        "[GRASP_PHASE_DIAGNOSTIC] "
                        f"phase={phase}, elapsed={now-handoff_started:.1f}s, "
                        f"planning_entered="
                        f"{self._grasp_planning_entered_event.is_set()}, "
                        f"planning_passed="
                        f"{self._grasp_planning_passed_event.is_set()}, "
                        f"execution_started="
                        f"{self._grasp_execution_started_event.is_set()}, "
                        f"planning_failed="
                        f"{self._grasp_planning_failed_event.is_set()}"
                    )
                    last_phase_log = now
                if (
                    now - handoff_started <= 20.0
                    and now - last_handoff_log >= 1.0
                ):
                    ready, gate = self._vision_target_ready()
                    point_1_topic = "/box_grasp_point/short_edge_1"
                    point_2_topic = "/box_grasp_point/short_edge_2"
                    valid_topic = "/box_grasp_point/valid"
                    point_1_publishers = self.count_publishers(point_1_topic)
                    point_2_publishers = self.count_publishers(point_2_topic)
                    point_1_subscribers = self.count_subscribers(point_1_topic)
                    point_2_subscribers = self.count_subscribers(point_2_topic)
                    valid_publishers = self.count_publishers(valid_topic)
                    # 左右点各至少应有两个订阅者：任务门控节点和抓取节点。
                    link_stage = (
                        "LINK_READY"
                        if (
                            point_1_publishers >= 1
                            and point_2_publishers >= 1
                            and point_1_subscribers >= 2
                            and point_2_subscribers >= 2
                        )
                        else "WAITING_FOR_PUBLISHER_OR_GRASP_SUBSCRIBER"
                    )
                    self.get_logger().warning(
                        "[VISION_TO_GRASP] "
                        f"stage={link_stage}, "
                        f"elapsed={now - handoff_started:.1f}s, "
                        f"vision_gate={ready}({gate}), "
                        f"edge1=pub{point_1_publishers}/sub{point_1_subscribers}, "
                        f"edge2=pub{point_2_publishers}/sub{point_2_subscribers}, "
                        f"valid=pub{valid_publishers}"
                    )
                    last_handoff_log = now
                if self._base_recovery_event.is_set():
                    self.state(
                        "GRASP_STOPPING_FOR_BASE_RECOVERY",
                        "planner supplied a checked SE(2) correction",
                    )
                    return None
                if self._centerline_recovery_event.is_set():
                    if self._grasp_execution_started_event.is_set():
                        self.state(
                            "CENTERLINE_RECOVERY_REJECTED",
                            "机械臂已经开始真实执行，禁止横移底盘",
                        )
                        return False
                    self._centerline_recovery_requested = True
                    self.state(
                        "GRASP_STOPPING_FOR_CENTERLINE_RECOVERY",
                        self._centerline_recovery_detail,
                    )
                    return None
                if self._grasp_planning_failed_event.is_set():
                    if self._grasp_execution_started_event.is_set():
                        self.state(
                            "PLANNING_RECOVERY_REJECTED",
                            "检测到机械臂已经开始真实执行；"
                            "即使后续日志包含规划失败也禁止移动底盘",
                        )
                        return False
                    self._planning_recovery_requested = True
                    self.state(
                        "GRASP_STOPPING_FOR_SLOW_APPROACH",
                        self._grasp_planning_failure_detail[-500:]
                        or "V7明确报告规划失败",
                    )
                    return None
                if (
                    not self._grasp_execution_started_event.is_set()
                    and now - handoff_started
                    >= self.planning_recovery_timeout
                ):
                    self._grasp_planning_failure_detail = (
                        "抓取进程在 "
                        f"{self.planning_recovery_timeout:.1f}s 内"
                        "既未开始真实执行，也未完成；按规划等待超时处理"
                    )
                    self._planning_recovery_requested = True
                    self.state(
                        "GRASP_PLANNING_TIMEOUT",
                        self._grasp_planning_failure_detail,
                    )
                    return None
                if self.grasp_done.is_set():
                    self.state("TASK_SUCCEEDED", "视觉抓取")
                    return True
                if self.shutdown_event.is_set() or not rclpy.ok():
                    return False
                if vision.poll() is not None:
                    self.get_logger().error("抓取期间视觉节点退出")
                    return False
                if grasp.poll() is not None:
                    self.get_logger().error(
                        "抓取程序已退出，但没有收到 "
                        "/box_grasp/completed=true；"
                        f"退出码={grasp.returncode}"
                    )
                    return False
                time.sleep(0.1)
            self.get_logger().error("视觉抓取任务超时")
            return False
        finally:
            self.stop_process(grasp)

    def grasp_task(self) -> bool:
        """Prefer planning recovery; scan only when vision sees nothing."""
        self.grasp_done.clear()
        self._last_grasp_status = ""
        total_attempts = (
            self.vision_recovery_cycles
            + 2 * self.base_recovery_cycles
            + self.planning_recovery_cycles
            + 1
        )
        base_recovery_used = 0
        planning_recovery_used = 0
        planning_recovery_total = 0.0
        centerline_recovery_used = 0
        centerline_recovery_total = 0.0
        vision_failures = 0

        for attempt in range(total_attempts):
            vision = None
            try:
                self._reset_vision_observation()
                self.state(
                    "VISION_RECOVERY_ATTEMPT",
                    f"{attempt + 1}/{total_attempts}",
                )
                vision = self.start_process(
                    "视觉识别",
                    self.commands["vision"],
                    setup_script=self.vision_setup,
                )
                if not self.sleep(self.cli.vision_warmup):
                    return False
                if vision.poll() is not None:
                    self.state(
                        "VISION_PROCESS_EXITED",
                        f"视觉节点提前退出，退出码={vision.returncode}"
                    )
                    vision_ready = False
                else:
                    vision_ready = self._wait_for_vision_target(
                        vision,
                        self.vision_detection_timeout,
                        f"静止识别第 {attempt + 1} 轮",
                    )

                if not vision_ready and vision.poll() is None:
                    recognized, recognition_detail = (
                        self._recent_visual_recognition()
                    )
                    if recognized:
                        self.state(
                            "VISION_RECOGNIZED_NO_SCAN",
                            "已看到目标证据，但尚未形成可用三维稳定点；"
                            f"{recognition_detail}；不移动底盘扫描",
                        )
                    else:
                        self.state(
                            "VISION_COMPLETELY_MISSING_SCAN",
                            f"{recognition_detail}；仅此情况启动视觉扫描",
                        )
                        vision_ready = self._scan_for_vision_target(
                            vision
                        )

                if vision_ready:
                    outcome = self._execute_confirmed_grasp(vision)
                    if outcome is not None:
                        return outcome

                    # The grasp process must be stopped before any direct base
                    # motion. The finally block is not late enough because the
                    # correction is applied in this branch.
                    self.stop_process(vision)
                    vision = None

                    if self._centerline_recovery_requested:
                        if centerline_recovery_used >= self.base_recovery_cycles:
                            self.state(
                                "CENTERLINE_RECOVERY_EXHAUSTED",
                                f"cycles={centerline_recovery_used}, "
                                f"cumulative={centerline_recovery_total:.3f}m, "
                                f"last_error={self._centerline_recovery_detail[-300:]}",
                            )
                            return False
                        centerline_recovery_used += 1
                        step = abs(self._centerline_recovery_dy)
                        if not self._apply_centerline_lateral_recovery(
                            centerline_recovery_used,
                            centerline_recovery_total,
                        ):
                            return False
                        centerline_recovery_total += step
                        continue

                    if self._planning_recovery_requested:
                        if (
                            planning_recovery_used
                            >= self.planning_recovery_cycles
                        ):
                            self.state(
                                "PLANNING_RECOVERY_EXHAUSTED",
                                f"cycles={planning_recovery_used}, "
                                f"cumulative={planning_recovery_total:.3f}m, "
                                f"last_error="
                                f"{self._grasp_planning_failure_detail[-300:]}",
                            )
                            return False
                        planning_recovery_used += 1
                        if not self._apply_planning_failure_translation(
                            planning_recovery_used,
                            planning_recovery_total,
                        ):
                            return False
                        planning_recovery_total += (
                            self.planning_recovery_forward_step
                        )
                        continue

                    recommendation = self._base_recovery_recommendation
                    if recommendation is None:
                        self.state(
                            "GRASP_RECOVERY_SIGNAL_MISSING",
                            "抓取进程要求恢复，但既没有规划失败标志，"
                            "也没有结构化底盘纠错建议",
                        )
                        return False
                    if base_recovery_used >= self.base_recovery_cycles:
                        self.state(
                            "BASE_RECOVERY_EXHAUSTED",
                            f"cycles={base_recovery_used}",
                        )
                        return False
                    base_recovery_used += 1
                    if not self._apply_base_recovery(recommendation):
                        return False
                    continue

                vision_failures += 1
                self.state(
                    "VISION_ROUND_FAILED",
                    f"failures={vision_failures}/"
                    f"{self.vision_recovery_cycles + 1}",
                )
                if vision_failures > self.vision_recovery_cycles:
                    self.state(
                        "VISION_RECOVERY_EXHAUSTED",
                        "视觉未在有限等待内形成可用三维点；"
                        "没有继续占用时间或盲目启动抓取",
                    )
                    return False
                if attempt + 1 < total_attempts:
                    self.stop_process(vision)
                    vision = None
                    self.state(
                        "VISION_RECOVERY_RESTART",
                        "短暂重启视觉；不执行底盘扫描或靠近",
                    )
                    if not self.sleep(self.vision_restart_delay):
                        return False
            finally:
                self.stop_process(vision)
                self.stop_robot()

        self.state(
            "GRASP_ATTEMPTS_EXHAUSTED",
            f"attempts={total_attempts}, "
            f"planning_recovery={planning_recovery_used}/"
            f"{self.planning_recovery_cycles}, "
            f"centerline_recovery={centerline_recovery_used}/"
            f"{self.base_recovery_cycles}, "
            f"base_recovery={base_recovery_used}/"
            f"{self.base_recovery_cycles}, "
            f"vision_failures={vision_failures}",
        )
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="箱体多点导航任务")
    parser.add_argument("--waypoints", required=True)
    parser.add_argument("--box-dir", required=True)
    parser.add_argument("--grasp-waypoint-1", default="grab1")
    parser.add_argument("--lift-waypoint-1", default="mid1")
    parser.add_argument("--release-waypoint-1", default="drop1")
    parser.add_argument("--lift-height-1", type=float, default=0.90)
    parser.add_argument("--grasp-waypoint-2", default="grab3")
    parser.add_argument("--lift-waypoint-2", default="mid3")
    parser.add_argument("--release-waypoint-2", default="drop3")
    parser.add_argument("--lift-height-2", type=float, default=0.50)
    parser.add_argument(
        "--reset-waypoint",
        default="",
        help="可选额外复位点；两次drop后的复位不依赖此参数",
    )
    parser.add_argument("--final-waypoint", default="end")
    parser.add_argument("--nav-ready-timeout", type=float, default=120.0)
    parser.add_argument("--goal-timeout", type=float, default=240.0)
    parser.add_argument("--nav-retries", type=int, default=1)
    parser.add_argument("--task-timeout", type=float, default=360.0)
    parser.add_argument("--vision-warmup", type=float, default=3.0)
    parser.add_argument("--arrival-settle", type=float, default=2.0)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument(
        "--arrival-position-tolerance",
        type=float,
        default=0.40,
        help="TF fallback position tolerance in metres",
    )
    parser.add_argument(
        "--arrival-yaw-tolerance-deg",
        type=float,
        default=35.0,
        help="TF fallback yaw tolerance in degrees",
    )
    parser.add_argument(
        "--task-arrival-position-tolerance",
        type=float,
        default=0.15,
        help="Position tolerance for grasp/lift/release waypoints in metres",
    )
    parser.add_argument(
        "--task-arrival-yaw-tolerance-deg",
        type=float,
        default=15.0,
        help="Yaw tolerance for grasp/lift/release waypoints in degrees",
    )
    parser.add_argument(
        "--grasp-arrival-position-tolerance",
        type=float,
        default=0.08,
        help="Mandatory final position tolerance at grasp waypoint in metres",
    )
    parser.add_argument(
        "--grasp-arrival-yaw-tolerance-deg",
        type=float,
        default=8.0,
        help="Mandatory final yaw tolerance at grasp waypoint in degrees",
    )
    parser.add_argument("--abort-pose-check-delay", type=float, default=1.0)
    parser.add_argument("--arrival-check-timeout", type=float, default=6.0)
    parser.add_argument("--nav-lifecycle-timeout", type=float, default=30.0)
    parser.add_argument("--tf-ready-timeout", type=float, default=30.0)
    parser.add_argument(
        "--local-costmap-clear-service",
        default="/local_costmap/clear_entirely_local_costmap",
    )
    parser.add_argument(
        "--global-costmap-clear-service",
        default="/global_costmap/clear_entirely_global_costmap",
    )
    parser.add_argument("--costmap-service-timeout", type=float, default=4.0)
    parser.add_argument("--costmap-repopulate-delay", type=float, default=1.5)
    parser.add_argument("--enable-fine-alignment", action="store_true")
    parser.add_argument("--fine-alignment-start-radius", type=float, default=0.35)
    parser.add_argument("--fine-alignment-timeout", type=float, default=40.0)
    parser.add_argument("--fine-alignment-linear-kp", type=float, default=0.60)
    parser.add_argument("--fine-alignment-angular-kp", type=float, default=0.80)
    parser.add_argument(
        "--fine-alignment-max-linear-speed",
        type=float,
        default=0.035,
    )
    parser.add_argument(
        "--fine-alignment-max-angular-speed",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--fine-alignment-min-angular-speed",
        type=float,
        default=0.025,
        help="Minimum angular command outside yaw tolerance, to overcome base deadband",
    )
    parser.add_argument("--fine-alignment-stable-samples", type=int, default=2)
    args = parser.parse_args()
    args.arrival_yaw_tolerance = math.radians(args.arrival_yaw_tolerance_deg)
    args.task_arrival_yaw_tolerance = math.radians(
        args.task_arrival_yaw_tolerance_deg
    )
    args.grasp_arrival_yaw_tolerance = math.radians(
        args.grasp_arrival_yaw_tolerance_deg
    )
    if args.nav_retries < 0:
        parser.error("--nav-retries 不能小于零")
    for name in (
        "nav_ready_timeout",
        "goal_timeout",
        "task_timeout",
        "vision_warmup",
        "arrival_settle",
        "arrival_position_tolerance",
        "arrival_yaw_tolerance",
        "task_arrival_position_tolerance",
        "task_arrival_yaw_tolerance",
        "grasp_arrival_position_tolerance",
        "grasp_arrival_yaw_tolerance",
        "abort_pose_check_delay",
        "arrival_check_timeout",
        "nav_lifecycle_timeout",
        "tf_ready_timeout",
        "costmap_service_timeout",
        "costmap_repopulate_delay",
        "fine_alignment_start_radius",
        "fine_alignment_timeout",
        "fine_alignment_linear_kp",
        "fine_alignment_angular_kp",
        "fine_alignment_max_linear_speed",
        "fine_alignment_max_angular_speed",
        "fine_alignment_min_angular_speed",
    ):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"{name} 不能小于零")
    if args.fine_alignment_stable_samples < 1:
        parser.error("--fine-alignment-stable-samples 必须至少为 1")
    return args


def main() -> int:
    cli = parse_args()
    # 本程序的 CLI 参数已经由 argparse 处理，不再交给 ROS 参数解析器，
    # 避免 --waypoints 等普通参数被不同 ROS2 发行版误判。
    rclpy.init(args=[])
    node: Optional[BoxWaypointMission] = None
    executor: Optional[MultiThreadedExecutor] = None
    thread: Optional[threading.Thread] = None
    try:
        node = RecoveryBoxWaypointMission(cli)
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        return 0 if node.run_mission() else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"任务异常: {type(exc).__name__}: {exc}")
        else:
            print(f"任务创建失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.shutdown()
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if thread is not None:
            thread.join(timeout=2.0)
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
