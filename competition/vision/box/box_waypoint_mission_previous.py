#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 YAML 顺序导航，并在指定路点执行抓取、抬升和释放任务。

本节点直接控制 /navigate_to_pose，运行时不要同时启动
real_robot_patrol.launch.py，也不要调用 /robot_a/patrol/start。
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
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
        self.stop_pub = self.create_publisher(Twist, "/cmd_vel", 10)
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
            for key in ("vision", "grasp", "lift", "release", "reset")
        }
        self.vision_setup = self._required_env("BOX_MISSION_VISION_SETUP")
        if not Path(self.vision_setup).is_file():
            raise RuntimeError(
                f"视觉环境脚本不存在: {self.vision_setup}"
            )
        ids = {waypoint.object_id for waypoint in self.waypoints}
        required = {
            cli.grasp_waypoint,
            cli.lift_waypoint,
            cli.release_waypoint,
            cli.final_waypoint,
            cli.reset_waypoint,
        }
        missing = sorted(required - ids)
        if missing:
            raise RuntimeError("YAML 缺少任务路点: " + ", ".join(missing))

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
        if point.object_id == self.cli.grasp_waypoint:
            return (
                self.cli.grasp_arrival_position_tolerance,
                self.cli.grasp_arrival_yaw_tolerance,
                "grasp",
            )
        task_point_ids = {
            self.cli.lift_waypoint,
            self.cli.release_waypoint,
        }
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
            self.cli.grasp_waypoint,
            self.cli.lift_waypoint,
            self.cli.release_waypoint,
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
            if not self.wait_future(result, self.cli.goal_timeout):
                self.get_logger().error(f"导航 {point.object_id} 超时")
                handle.cancel_goal_async()
                self.active_goal = None
                self.stop_robot()
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
                self.clear_navigation_costmaps(
                    f"before retry {attempt + 2} for {point.object_id}"
                )
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
        if object_id == self.cli.grasp_waypoint and not self.grasp_task():
            return False
        if object_id == self.cli.lift_waypoint:
            if not self.simple_task("抬升到运输高度", "lift"):
                return False
            self.get_logger().info(
                "Lift completed; clearing stale costmap observations before transport"
            )
            self.clear_navigation_costmaps("after lift before transport")
        if object_id == self.cli.release_waypoint and not self.simple_task(
            "放置并释放", "release"
        ):
            return False
        if object_id == self.cli.reset_waypoint and not self.simple_task(
            "全部关节归零", "reset"
        ):
            return False
        return True

    def run_mission(self) -> bool:
        self.done_pub.publish(Bool(data=False))
        self.state(
            "READY",
            f"{len(self.waypoints)} points; tasks="
            f"{self.cli.grasp_waypoint}/{self.cli.lift_waypoint}/"
            f"{self.cli.release_waypoint}/{self.cli.reset_waypoint}",
        )
        task_points = {
            self.cli.grasp_waypoint,
            self.cli.lift_waypoint,
            self.cli.release_waypoint,
            self.cli.reset_waypoint,
        }
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
        self.vision_detection_timeout = _recovery_env_float(
            "BOX_MISSION_VISION_DETECTION_TIMEOUT", 10.0, 0.1
        )
        self.vision_stable_samples = _recovery_env_int(
            "BOX_MISSION_VISION_STABLE_SAMPLES", 1, 1
        )
        self.vision_sample_max_age = _recovery_env_float(
            "BOX_MISSION_VISION_SAMPLE_MAX_AGE", 2.0, 0.05
        )
        self.vision_point_sync_tolerance = _recovery_env_float(
            "BOX_MISSION_VISION_POINT_SYNC_TOLERANCE", 0.20, 0.0
        )
        self.vision_recovery_cycles = _recovery_env_int(
            "BOX_MISSION_VISION_RECOVERY_CYCLES", 2, 0
        )
        self.vision_restart_delay = _recovery_env_float(
            "BOX_MISSION_VISION_RESTART_DELAY", 1.0, 0.0
        )
        self.vision_scan_angle = math.radians(
            _recovery_env_float(
                "BOX_MISSION_VISION_SCAN_ANGLE_DEG", 8.0, 0.0
            )
        )
        self.vision_scan_speed = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_SPEED", 0.04, 0.005
        )
        self.vision_scan_min_speed = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_MIN_SPEED", 0.025, 0.0
        )
        self.vision_scan_yaw_tolerance = math.radians(
            _recovery_env_float(
                "BOX_MISSION_VISION_SCAN_YAW_TOLERANCE_DEG", 1.0, 0.1
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
            "BOX_MISSION_VISION_SCAN_TRANSLATION_DISTANCE", 0.06, 0.0
        )
        self.vision_scan_linear_speed = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_LINEAR_SPEED", 0.025, 0.005
        )
        self.vision_scan_min_linear_speed = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_MIN_LINEAR_SPEED", 0.012, 0.0
        )
        self.vision_scan_position_tolerance = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_POSITION_TOLERANCE", 0.01, 0.002
        )
        self.vision_scan_lateral_drift = _recovery_env_float(
            "BOX_MISSION_VISION_SCAN_LATERAL_DRIFT", 0.03, 0.005
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
        self._vision_valid = False
        self._vision_valid_streak = 0
        self._vision_valid_received = 0.0
        self._vision_left: Optional[PointStamped] = None
        self._vision_right: Optional[PointStamped] = None
        self._vision_left_received = 0.0
        self._vision_right_received = 0.0
        self._vision_center_2d_received = 0.0
        self._vision_left_2d_received = 0.0
        self._vision_right_2d_received = 0.0
        self._last_grasp_status = ""

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
            f"{self.vision_detection_timeout:.1f}s，扫描角="
            f"{math.degrees(self.vision_scan_angle):.1f}deg，前后距离="
            f"{self.vision_scan_translation_distance:.3f}m"
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
            self._vision_valid = False
            self._vision_valid_streak = 0
            self._vision_valid_received = 0.0
            self._vision_left = None
            self._vision_right = None
            self._vision_left_received = 0.0
            self._vision_right_received = 0.0
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

    def _on_vision_right(self, message: PointStamped) -> None:
        with self._vision_lock:
            self._vision_right = message
            self._vision_right_received = time.monotonic()

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
        else:
            stage = "VISUAL_PIPELINE_READY"
        detail = (
            f"stage={stage}, valid={valid}, "
            f"streak={streak}/{self.vision_stable_samples}, "
            f"valid_age={age_text(valid_received)}, "
            f"center2d_age={age_text(center_2d)}, "
            f"edge2d_age=({age_text(left_2d)},{age_text(right_2d)}), "
            f"point3d_age=({age_text(left_3d)},{age_text(right_3d)})"
        )
        return stage, detail

    def _vision_target_ready(self) -> tuple[bool, str]:
        now = time.monotonic()
        with self._vision_lock:
            epoch = self._vision_epoch
            valid = self._vision_valid
            streak = self._vision_valid_streak
            valid_received = self._vision_valid_received
            left = self._vision_left
            right = self._vision_right
            left_received = self._vision_left_received
            right_received = self._vision_right_received

        if not valid:
            return False, "valid=false"
        if streak < self.vision_stable_samples:
            return False, f"稳定帧={streak}/{self.vision_stable_samples}"
        if left is None or right is None:
            return False, "等待左右三维抓取点"
        receipt_times = (valid_received, left_received, right_received)
        if any(received < epoch for received in receipt_times):
            return False, "等待本轮视觉的新数据"
        oldest_age = max(now - received for received in receipt_times)
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
        return True, (
            f"frame={left.header.frame_id}，稳定帧={streak}，"
            f"左右时间差={stamp_delta:.3f}s"
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
    ) -> bool:
        """沿进入纠错时的车头方向前后移动，不调用任何 Nav2 接口。"""
        if self.active_goal is not None:
            self.get_logger().error("仍有活动 Nav2 目标，拒绝前后视觉扫描")
            return False
        reference_x, reference_y, reference_yaw = reference_pose
        cos_yaw = math.cos(reference_yaw)
        sin_yaw = math.sin(reference_yaw)
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
                    <= self.vision_scan_position_tolerance
                    and abs(yaw_error)
                    <= self.vision_scan_yaw_tolerance
                ):
                    return True

                linear = self.clamp(
                    0.8 * position_error,
                    -self.vision_scan_linear_speed,
                    self.vision_scan_linear_speed,
                )
                if (
                    abs(position_error) > self.vision_scan_position_tolerance
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
        if self.vision_scan_angle <= 0.0:
            return False

        for offset in (self.vision_scan_angle, -self.vision_scan_angle):
            if not self._rotate_without_nav(reference_pose, offset, vision):
                break
            if not self.sleep(0.4):
                return False
            self._reset_vision_observation()
            if self._wait_for_vision_target(
                vision,
                self.vision_scan_dwell,
                f"扫描观察位 {math.degrees(offset):+.1f}deg",
            ):
                return True

        # Translation observations start from the original yaw. This remains
        # direct cmd_vel control and never creates a Nav2 goal.
        self._rotate_without_nav(reference_pose, 0.0)
        if (
            vision.poll() is None
            and self.vision_scan_translation_distance > 0.0
        ):
            for offset in (
                self.vision_scan_translation_distance,
                -self.vision_scan_translation_distance,
            ):
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
                    f"前后观察位 {offset:+.3f}m",
                ):
                    return True

        position_restored = self._translate_without_nav(
            reference_pose,
            0.0,
        )
        yaw_restored = self._rotate_without_nav(reference_pose, 0.0)
        if not position_restored or not yaw_restored:
            self.state(
                "VISION_SCAN_RESTORE_FAILED",
                "底盘已停止，但没有完全恢复进入扫描前的位置/航向",
            )
        self.stop_robot()
        return False

    def _execute_confirmed_grasp(
        self,
        vision: subprocess.Popen,
    ) -> bool:
        grasp = None
        try:
            grasp = self.start_process("箱体抓取", self.commands["grasp"])
            self.state("GRASP_PROCESS_STARTED", f"pid={grasp.pid}")
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
        """确认稳定的新视觉点后，仅启动一次抓取进程。"""
        self.grasp_done.clear()
        self._last_grasp_status = ""
        total_attempts = self.vision_recovery_cycles + 1

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
                    self.get_logger().error(
                        f"视觉节点提前退出，退出码={vision.returncode}"
                    )
                elif self._wait_for_vision_target(
                    vision,
                    self.vision_detection_timeout,
                    f"静止识别第 {attempt + 1} 轮",
                ):
                    return self._execute_confirmed_grasp(vision)
                elif self._scan_for_vision_target(vision):
                    return self._execute_confirmed_grasp(vision)
            finally:
                self.stop_process(vision)
                self.stop_robot()

            if attempt + 1 < total_attempts:
                self.state(
                    "VISION_RECOVERY_RESTART",
                    f"准备第 {attempt + 2}/{total_attempts} 轮",
                )
                if not self.sleep(self.vision_restart_delay):
                    return False

        self.state(
            "VISION_TARGET_NOT_FOUND",
            f"{total_attempts} 轮均失败；未启动机械臂抓取",
        )
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="箱体多点导航任务")
    parser.add_argument("--waypoints", required=True)
    parser.add_argument("--box-dir", required=True)
    parser.add_argument("--grasp-waypoint", default="start")
    parser.add_argument("--lift-waypoint", default="mid4")
    parser.add_argument("--release-waypoint", default="drop")
    parser.add_argument("--reset-waypoint", default="end")
    parser.add_argument("--final-waypoint", default="final")
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
        default=0.10,
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
        default=0.03,
        help="Mandatory final position tolerance at grasp waypoint in metres",
    )
    parser.add_argument(
        "--grasp-arrival-yaw-tolerance-deg",
        type=float,
        default=2.0,
        help="Mandatory final yaw tolerance at grasp waypoint in degrees",
    )
    parser.add_argument("--abort-pose-check-delay", type=float, default=1.0)
    parser.add_argument("--arrival-check-timeout", type=float, default=3.0)
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
    parser.add_argument("--fine-alignment-start-radius", type=float, default=0.25)
    parser.add_argument("--fine-alignment-timeout", type=float, default=25.0)
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
    parser.add_argument("--fine-alignment-stable-samples", type=int, default=5)
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

