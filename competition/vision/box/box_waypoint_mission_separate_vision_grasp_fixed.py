#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Separate-Fixed 版：视觉常驻、抓取独立等待触发，并可靠续跑后续路点。

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


@dataclass(frozen=True)
class VisionPointSample:
    xyz: tuple[float, float, float]
    capture_time_s: float
    receive_time_s: float
    sequence: int


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
        self.create_subscription(
            PointStamped,
            "/box_grasp_point/short_edge_1",
            self._on_vision_point_1,
            10,
        )
        self.create_subscription(
            PointStamped,
            "/box_grasp_point/short_edge_2",
            self._on_vision_point_2,
            10,
        )

        # 在 separate 版本中，视觉和抓取由 m1 的独立终端长期管理。
        # 本任务节点只负责：到达 start -> 验证视觉稳定 -> 发触发 -> 等抓取完成。
        self.commands = {
            key: self._required_env(f"BOX_MISSION_{key.upper()}_CMD")
            for key in ("lift", "release", "reset")
        }
        self.grasp_trigger_file = Path(
            self._required_env("BOX_MISSION_GRASP_TRIGGER_FILE")
        )
        self.grasp_ack_file = Path(
            self._required_env("BOX_MISSION_GRASP_ACK_FILE")
        )
        self.grasp_result_file = Path(
            self._required_env("BOX_MISSION_GRASP_RESULT_FILE")
        )
        for control_file in (
            self.grasp_trigger_file,
            self.grasp_ack_file,
            self.grasp_result_file,
        ):
            control_file.parent.mkdir(parents=True, exist_ok=True)

        self._vision_lock = threading.Lock()
        self._vision_sequence = [0, 0]
        self._vision_latest: list[Optional[VisionPointSample]] = [None, None]
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

    @staticmethod
    def _stamp_seconds(message: PointStamped) -> float:
        return (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1e-9
        )

    def _store_vision_point(self, index: int, message: PointStamped) -> None:
        xyz = (
            float(message.point.x),
            float(message.point.y),
            float(message.point.z),
        )
        if not all(math.isfinite(value) for value in xyz):
            return
        capture_time_s = self._stamp_seconds(message)
        receive_time_s = time.monotonic()
        with self._vision_lock:
            self._vision_sequence[index] += 1
            self._vision_latest[index] = VisionPointSample(
                xyz=xyz,
                capture_time_s=capture_time_s,
                receive_time_s=receive_time_s,
                sequence=self._vision_sequence[index],
            )

    def _on_vision_point_1(self, message: PointStamped) -> None:
        self._store_vision_point(0, message)

    def _on_vision_point_2(self, message: PointStamped) -> None:
        self._store_vision_point(1, message)

    @staticmethod
    def _point_distance(
        a: tuple[float, float, float],
        b: tuple[float, float, float],
    ) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @classmethod
    def _window_spread(
        cls,
        samples: list[
            tuple[VisionPointSample, VisionPointSample]
        ],
    ) -> float:
        if len(samples) <= 1:
            return 0.0
        maximum = 0.0
        for side in (0, 1):
            points = [pair[side].xyz for pair in samples]
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    maximum = max(
                        maximum,
                        cls._point_distance(points[i], points[j]),
                    )
        return maximum

    def wait_for_vision_ready(self) -> bool:
        # 只允许“到达抓取路点以后拍摄”的视觉结果参与门控。
        gate_capture_time_s = self.get_clock().now().nanoseconds * 1e-9
        gate_receive_time_s = time.monotonic()

        with self._vision_lock:
            start_sequence = tuple(self._vision_sequence)

        window: deque[
            tuple[VisionPointSample, VisionPointSample]
        ] = deque(maxlen=max(1, int(self.cli.vision_ready_stable_samples)))

        last_used = start_sequence
        discarded_pre_arrival = 0
        deadline = time.monotonic() + self.cli.vision_ready_timeout
        next_log = 0.0
        last_reason = "等待两路抓取点的新消息"

        self.state(
            "VISION_WAIT",
            "视觉已在独立终端常驻；清空导航途中样本，"
            "只统计到达抓取点之后收到的新识别结果",
        )

        while time.monotonic() < deadline:
            if self.shutdown_event.is_set() or not rclpy.ok():
                return False

            now = time.monotonic()
            with self._vision_lock:
                left = self._vision_latest[0]
                right = self._vision_latest[1]
                current_sequence = tuple(self._vision_sequence)

            if left is None or right is None:
                last_reason = "尚未同时收到左右两路抓取点"
            elif current_sequence == last_used:
                left_age = now - left.receive_time_s
                right_age = now - right.receive_time_s
                oldest_age = max(left_age, right_age)
                if oldest_age > self.cli.vision_ready_max_age:
                    last_reason = (
                        "视觉当前没有持续刷新，"
                        f"最新接收间隔={oldest_age:.2f}s"
                    )
                else:
                    last_reason = (
                        f"等待下一组视觉消息；当前窗口 "
                        f"{len(window)}/{self.cli.vision_ready_stable_samples}"
                    )
            elif (
                left.sequence == last_used[0]
                or right.sequence == last_used[1]
            ):
                # 一侧已经更新，另一侧还没到。慢推理时不清空稳定窗口。
                last_reason = "左右两路尚未形成新的一对消息"
            else:
                last_used = current_sequence

                capture_is_valid = True
                # header.stamp 为 0 时退回到真实接收时间；否则严格拒绝
                # 导航途中拍摄、到达 start 后才处理完的积压帧。
                for sample in (left, right):
                    if sample.capture_time_s > 1e-6:
                        if sample.capture_time_s < gate_capture_time_s:
                            capture_is_valid = False
                    elif sample.receive_time_s < gate_receive_time_s:
                        capture_is_valid = False

                if not capture_is_valid:
                    discarded_pre_arrival += 1
                    last_reason = (
                        "丢弃到达抓取点之前拍摄的积压识别结果；"
                        f"累计={discarded_pre_arrival}"
                    )
                else:
                    receive_pair_dt = abs(
                        left.receive_time_s - right.receive_time_s
                    )
                    capture_pair_dt = 0.0
                    if (
                        left.capture_time_s > 1e-6
                        and right.capture_time_s > 1e-6
                    ):
                        capture_pair_dt = abs(
                            left.capture_time_s - right.capture_time_s
                        )
                    pair_dt = max(receive_pair_dt, capture_pair_dt)

                    if pair_dt > self.cli.vision_ready_pair_dt:
                        last_reason = (
                            "左右抓取点接收/拍摄不同步 "
                            f"dt={pair_dt:.2f}s/"
                            f"{self.cli.vision_ready_pair_dt:.2f}s"
                        )
                    else:
                        window.append((left, right))
                        spread = self._window_spread(list(window))
                        if len(window) < self.cli.vision_ready_stable_samples:
                            last_reason = (
                                "正在累计稳定抓取点 "
                                f"{len(window)}/"
                                f"{self.cli.vision_ready_stable_samples}"
                            )
                        elif spread > self.cli.vision_ready_max_spread:
                            # 滑动窗口：不清零，等下一组消息替换最旧样本。
                            last_reason = (
                                "抓取点仍在抖动 "
                                f"spread={spread * 1000.0:.1f}mm/"
                                f"{self.cli.vision_ready_max_spread * 1000.0:.1f}mm；"
                                "继续等待新数据"
                            )
                        else:
                            latest_age = max(
                                now - left.receive_time_s,
                                now - right.receive_time_s,
                            )
                            if latest_age > self.cli.vision_ready_max_age:
                                last_reason = (
                                    "稳定窗口满足，但最新视觉消息过旧 "
                                    f"age={latest_age:.2f}s"
                                )
                            else:
                                first_rx = min(
                                    pair[0].receive_time_s
                                    for pair in window
                                )
                                first_rx = min(
                                    first_rx,
                                    min(
                                        pair[1].receive_time_s
                                        for pair in window
                                    ),
                                )
                                capture_window = max(
                                    0.0,
                                    max(
                                        left.receive_time_s,
                                        right.receive_time_s,
                                    )
                                    - first_rx,
                                )
                                self.state(
                                    "VISION_READY",
                                    f"samples={len(window)}, "
                                    f"spread={spread * 1000.0:.1f}mm, "
                                    f"pair_dt={pair_dt:.2f}s, "
                                    f"receive_age={latest_age:.2f}s, "
                                    f"capture_window={capture_window:.1f}s；"
                                    "允许触发独立抓取终端",
                                )
                                return True

            if now >= next_log:
                self.get_logger().warning(f"等待视觉就绪：{last_reason}")
                next_log = now + self.cli.vision_ready_log_period

            time.sleep(0.05)

        self.get_logger().error(
            "等待视觉稳定超时；最后状态：" + last_reason
        )
        return False

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)

    def trigger_grasp_terminal(self) -> bool:
        for path in (
            self.grasp_trigger_file,
            self.grasp_ack_file,
            self.grasp_result_file,
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        token = f"GO {time.time():.6f}\n"
        self._atomic_write(self.grasp_trigger_file, token)
        self.state(
            "GRASP_TRIGGERED",
            "视觉已稳定，已向独立抓取终端发送持久触发文件",
        )

        deadline = time.monotonic() + self.cli.grasp_trigger_ack_timeout
        while time.monotonic() < deadline:
            if self.shutdown_event.is_set() or not rclpy.ok():
                return False
            if self.grasp_ack_file.is_file():
                detail = self.grasp_ack_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).strip()
                self.state(
                    "GRASP_STARTED",
                    detail or "独立抓取终端已确认启动",
                )
                return True
            time.sleep(0.05)

        self.get_logger().error(
            "独立抓取终端没有在规定时间内确认触发；"
            f"trigger={self.grasp_trigger_file}"
        )
        return False

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

        if not self.wait_for_vision_ready():
            return False
        if not self.trigger_grasp_terminal():
            return False

        deadline = time.monotonic() + self.cli.task_timeout
        while time.monotonic() < deadline:
            if self.grasp_done.is_set():
                self.state("TASK_SUCCEEDED", "视觉抓取")
                return True
            if self.shutdown_event.is_set() or not rclpy.ok():
                return False

            if self.grasp_result_file.is_file():
                result_text = self.grasp_result_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).strip()
                try:
                    result_code = int(result_text.split()[0])
                except (ValueError, IndexError):
                    result_code = None

                # separate 架构里 06 抓取终端会在抓取程序退出后写 result。
                # 退出码 0 本身就是可靠的“整段抓取程序成功完成”信号。
                # 原版本把 0 也当成失败，导致箱子已经抓住后任务节点仍在
                # start 点返回 False，从而永远不会导航到下一个路点。
                if result_code == 0:
                    self.state(
                        "TASK_SUCCEEDED",
                        "视觉抓取（独立抓取进程正常退出，code=0）",
                    )
                    self.get_logger().info(
                        "抓取完成握手确认：result=0；"
                        "即使 /box_grasp/completed=true 因时序未被收到，"
                        "也允许继续执行后续导航。"
                    )
                    return True

                if result_code is not None:
                    self.get_logger().error(
                        "抓取程序异常退出；"
                        f"退出码={result_code}，停止后续导航"
                    )
                    return False

            time.sleep(0.1)

        self.get_logger().error("视觉抓取任务超时")
        return False

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

            # 抓取成功后明确宣布下一步，便于现场一眼确认状态机已经
            # 从 start 退出，而不是仍卡在抓取等待阶段。
            if (
                point.object_id == self.cli.grasp_waypoint
                and index < len(self.waypoints)
            ):
                next_point = self.waypoints[index]
                self.state(
                    "GRASP_COMPLETE_CONTINUE",
                    f"抓取确认成功，继续导航到下一个路点 {next_point.object_id}",
                )

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
    parser.add_argument("--vision-ready-timeout", type=float, default=120.0)
    parser.add_argument("--vision-ready-stable-samples", type=int, default=3)
    parser.add_argument("--vision-ready-max-spread", type=float, default=0.025)
    parser.add_argument("--vision-ready-max-age", type=float, default=8.0)
    parser.add_argument("--vision-ready-pair-dt", type=float, default=2.0)
    parser.add_argument("--vision-ready-log-period", type=float, default=2.0)
    parser.add_argument("--grasp-trigger-ack-timeout", type=float, default=15.0)
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
        "vision_ready_timeout",
        "vision_ready_max_spread",
        "vision_ready_max_age",
        "vision_ready_pair_dt",
        "vision_ready_log_period",
        "grasp_trigger_ack_timeout",
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
    if args.vision_ready_stable_samples < 2:
        parser.error("--vision-ready-stable-samples 必须至少为 2")
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
        node = BoxWaypointMission(cli)
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
