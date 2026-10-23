#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
screw_grasp_bridge.py
=====================

V11：视觉抓取闭环；右夹爪闭合后验证抓取结果，失败则 Lift 回安全高度并重新识别/规划。

数据链:
  YOLO center_pixel (RGB image timestamp)
      +
  /camera_head/depth/points (实际坐标系以 header.frame_id 为准)
      +
  /camera_head/color/camera_info
      |
      v
  1) PointCloud frame -> color optical (TF2；若已是 color optical 则不重复变换)
  2) 在 YOLO 中心附近 ROI 选择 3D 点，做鲁棒中位数
  3) PointCloud frame -> camera_head_link (TF2，相机内部固定 TF)
  4) camera_head_link -> head_pitch_link (显式读取右手眼 .calib)
  5) head_pitch_link -> base_link (TF2，使用点云时刻的实时机器人 TF)
      |
      v
  /screw_bag_detection/point_base
  并持续打印 base_link XYZ；稳定后锁定一次并进入 IK/人工确认流程:
      target_x = Y_base
      target_y = X_base
      target_z = Z_base

重要:
- 默认 enable_motion=false 时绝不运动；实机执行需要显式参数 + 终端人工确认。
- 故意不使用 TF 树里的 head_pitch_link <-> camera_head_link，
  避免当前多个 handeye/static publisher 带来的冲突。
- 当前 screw_ros2.py 只发布 YOLO 中心像素，没有 bbox，因此本版本
  使用中心圆形 ROI。后续可升级为 bbox 内点云筛选。
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, PointCloud2
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener

# 复用现有抓取脚本中的真实 IK / FK / URDF 限位逻辑。
# 本文件只调用 calculate_plan()，并用 DryRun 子类硬性封锁所有运动接口。
from ik_grasp_dual_mirror_real_v9_grasp_verify_retry import (
    IKGraspRealController,
    load_urdf_limits,
)

try:
    from sensor_msgs_py import point_cloud2
except ImportError as exc:
    raise RuntimeError(
        "缺少 sensor_msgs_py。请确认 ROS 2 环境已 source，并安装 sensor_msgs_py。"
    ) from exc

try:
    import yaml
except ImportError as exc:
    raise RuntimeError(
        "缺少 PyYAML。可先运行: python3 -m pip install PyYAML"
    ) from exc


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("四元数长度为 0")
    x, y, z, w = q / norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_stamped_to_matrix(msg: TransformStamped) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    T[:3, 3] = [float(t.x), float(t.y), float(t.z)]
    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nx3 点批量做刚体变换。"""
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    return T[:3, :3] @ np.asarray(point, dtype=np.float64) + T[:3, 3]


def load_eye_on_base_calibration(calib_path: str):
    """
    读取 easy_handeye 风格 calib 文件。

    对 eye_on_base:
      robot_base_frame = head_pitch_link
      tracking_base_frame = camera_head_link

    文件中的 transform 按这里的系统约定解释为:
      P_robot_base = T_robotBase_trackingBase @ P_tracking_base

    即本项目中:
      P_head_pitch = T_headPitch_cameraHead @ P_camera_head
    """
    path = Path(calib_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"手眼标定文件不存在: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = data.get("parameters", {})
    tf_data = data.get("transform", {})
    trans = tf_data.get("translation", {})
    rot = tf_data.get("rotation", {})

    calibration_type = str(params.get("calibration_type", ""))
    robot_base_frame = str(params.get("robot_base_frame", ""))
    tracking_base_frame = str(params.get("tracking_base_frame", ""))

    if calibration_type != "eye_on_base":
        raise RuntimeError(
            f"期望 eye_on_base 标定，但文件中 calibration_type={calibration_type!r}"
        )

    required_t = ("x", "y", "z")
    required_q = ("x", "y", "z", "w")
    if any(k not in trans for k in required_t) or any(k not in rot for k in required_q):
        raise RuntimeError("calib 文件 transform 字段不完整")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_rotation_matrix(
        float(rot["x"]),
        float(rot["y"]),
        float(rot["z"]),
        float(rot["w"]),
    )
    T[:3, 3] = [
        float(trans["x"]),
        float(trans["y"]),
        float(trans["z"]),
    ]

    return T, robot_base_frame, tracking_base_frame


def pointcloud2_xyz(msg: PointCloud2) -> np.ndarray:
    """
    将 PointCloud2 读成 Nx3 float64。
    优先使用 read_points_numpy；兼容旧版 sensor_msgs_py。
    """
    if hasattr(point_cloud2, "read_points_numpy"):
        try:
            arr = point_cloud2.read_points_numpy(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
            )
            arr = np.asarray(arr)

            if arr.dtype.names:
                pts = np.column_stack(
                    [arr["x"], arr["y"], arr["z"]]
                )
            else:
                pts = arr.reshape(-1, 3)

            return np.asarray(pts, dtype=np.float64)
        except Exception:
            pass

    arr = point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )

    if isinstance(arr, np.ndarray):
        if arr.dtype.names:
            pts = np.column_stack(
                [arr["x"], arr["y"], arr["z"]]
            )
        else:
            pts = np.asarray(arr).reshape(-1, 3)
    else:
        pts = np.asarray(list(arr), dtype=np.float64).reshape(-1, 3)

    return np.asarray(pts, dtype=np.float64)


def project_plumb_bob(
    points_color: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 color optical frame 中的 3D 点投影到原始 RGB 图像。

    支持常见 plumb_bob / OpenCV Brown-Conrady:
      k1, k2, p1, p2, k3[, k4, k5, k6]
    当存在 k4-k6 时使用 OpenCV rational denominator。
    """
    X = points_color[:, 0]
    Y = points_color[:, 1]
    Z = points_color[:, 2]

    x = X / Z
    y = Y / Z

    d = np.zeros(8, dtype=np.float64)
    n = min(len(D), 8)
    if n > 0:
        d[:n] = D[:n]

    k1, k2, p1, p2, k3, k4, k5, k6 = d

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    radial_num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    radial_den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6

    # 防止极端情况下除 0。
    radial_den = np.where(
        np.abs(radial_den) < 1e-12,
        1.0,
        radial_den,
    )

    radial = radial_num / radial_den

    x_dist = (
        x * radial
        + 2.0 * p1 * x * y
        + p2 * (r2 + 2.0 * x * x)
    )
    y_dist = (
        y * radial
        + p1 * (r2 + 2.0 * y * y)
        + 2.0 * p2 * x * y
    )

    fx = float(K[0, 0])
    skew = float(K[0, 1])
    cx = float(K[0, 2])
    fy = float(K[1, 1])
    cy = float(K[1, 2])

    u = fx * x_dist + skew * y_dist + cx
    v = fy * y_dist + cy

    return u, v


class GuardedIKGraspController(IKGraspRealController):
    """
    真实 IKGraspRealController + 软件运动保险。

    默认 DISARMED：
      execute_plan / send_motion / command_gripper / open_both_grippers
      都会拒绝执行。

    只有用户在终端明确确认后，bridge 才调用 arm_motion()。
    execute_plan() 结束（成功或异常）后会立即 disarm。
    """

    def __init__(self, *args, **kwargs):
        self._motion_armed = False
        super().__init__(*args, **kwargs)

    def arm_motion(self) -> None:
        self._motion_armed = True

    def disarm_motion(self) -> None:
        self._motion_armed = False

    def _require_armed(self) -> None:
        if not self._motion_armed:
            raise RuntimeError(
                "MOTION SAFETY BLOCK: 运动接口未解锁，拒绝发送机器人命令"
            )

    def execute_plan(self, *args, **kwargs):
        self._require_armed()
        return super().execute_plan(*args, **kwargs)

    def send_motion(self, *args, **kwargs):
        self._require_armed()
        return super().send_motion(*args, **kwargs)

    def command_gripper(self, *args, **kwargs):
        self._require_armed()
        return super().command_gripper(*args, **kwargs)

    def open_both_grippers(self, *args, **kwargs):
        self._require_armed()
        return super().open_both_grippers(*args, **kwargs)


class ScrewGraspBridge(Node):
    def __init__(self) -> None:
        super().__init__("screw_grasp_bridge")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter(
            "center_topic",
            "/screw_bag_detection/center_pixel",
        )
        self.declare_parameter(
            "valid_topic",
            "/screw_bag_detection/valid",
        )
        self.declare_parameter(
            "points_topic",
            "/camera_head/depth/points",
        )
        self.declare_parameter(
            "color_info_topic",
            "/camera_head/color/camera_info",
        )
        self.declare_parameter(
            "output_point_topic",
            "/screw_bag_detection/point_base",
        )
        self.declare_parameter(
            "output_valid_topic",
            "/screw_bag_detection/point_base_valid",
        )

        self.declare_parameter(
            "calib_path",
            "/home/niic/Desktop/competition/vision/transformation matrix/eob_calib_right.calib",
        )

        self.declare_parameter(
            "depth_frame",
            "camera_head_depth_optical_frame",
        )
        self.declare_parameter(
            "color_frame",
            "camera_head_color_optical_frame",
        )
        self.declare_parameter(
            "camera_head_frame",
            "camera_head_link",
        )
        self.declare_parameter(
            "head_pitch_frame",
            "head_pitch_link",
        )
        self.declare_parameter(
            "base_frame",
            "base_link",
        )

        # 当前视觉节点只有中心像素，因此用中心圆形 ROI。
        self.declare_parameter("roi_radius_px", 28.0)

        # RGB 检测 stamp 与点云 stamp 最大允许差。
        self.declare_parameter("max_sync_dt_s", 0.08)

        # 点云粗过滤范围。
        self.declare_parameter("min_depth_m", 0.15)
        self.declare_parameter("max_depth_m", 2.00)

        # ROI 内至少需要多少个 3D 点。
        self.declare_parameter("min_roi_points", 20)

        # ROI 点的 Z 鲁棒过滤：
        # keep |Z - median(Z)| <= max(min_band, sigma*MAD)
        self.declare_parameter("depth_mad_sigma", 3.0)
        self.declare_parameter("depth_min_band_m", 0.015)
        self.declare_parameter("depth_max_band_m", 0.060)

        # 保存最近几帧点云，用检测图像时间戳找最近点云。
        self.declare_parameter("cloud_buffer_size", 6)

        # 避免每个 YOLO 帧都处理整幅点云。
        self.declare_parameter("max_process_rate_hz", 3.0)

        # 仅定位阶段默认不加抓取补偿；后续可在 base_link 中加偏移。
        self.declare_parameter("offset_base_x_m", 0.0)
        self.declare_parameter("offset_base_y_m", 0.0)
        self.declare_parameter("offset_base_z_m", 0.0)

        # TF 精确时间查询超时。
        self.declare_parameter("tf_timeout_s", 0.20)

        # 只定位/机器人静止阶段建议 True：
        # robot_state_publisher 在关节不变化时可能没有“点云同一时刻”的动态 TF，
        # 此时按点云精确时间查询会产生 ExtrapolationException。
        # True = 使用 TF buffer 中最新的 base_link <- head_pitch_link。
        # 后续若需要机器人运动过程中持续视觉定位，再改为 False，并解决时间同步。
        self.declare_parameter("robot_tf_use_latest", True)

        # ------------------------------------------------------------------
        # IK dry-run planning
        # ------------------------------------------------------------------
        self.declare_parameter(
            "ik_script_path",
            "/home/niic/Desktop/competition/vision/ik_grasp_dual_mirror_real_v4.py",
        )
        self.declare_parameter(
            "urdf_path",
            "/home/niic/project/20260623/20260116/test_robot_A/"
            "src/robot_A/robot_a_description/urdf/robot_a.urdf",
        )
        self.declare_parameter("velocity_scale", 0.30)
        self.declare_parameter("gripper_effort", 50.0)
        self.declare_parameter("max_fk_error_cm", 5.0)
        self.declare_parameter("max_final_xy_error_cm", 5.0)

        # 连续若干帧稳定后才调用 calculate_plan，避免每帧反复解 IK。
        self.declare_parameter("plan_stability_frames", 5)
        self.declare_parameter("plan_stability_xy_m", 0.010)
        self.declare_parameter("plan_stability_z_m", 0.015)

        # 同一稳定目标只规划一次；目标移动超过该阈值后重新允许规划。
        self.declare_parameter("plan_rearm_distance_m", 0.030)

        # ------------------------------------------------------------------
        # V5 one-shot real execution gate
        # ------------------------------------------------------------------
        # 默认 False：即使代码运行，也只会锁定目标 + IK 检查，不会运动。
        # 真正实机测试必须显式:
        #   --ros-args -p enable_motion:=true
        self.declare_parameter("enable_motion", False)

        # 保留兼容参数；本版本不再读取终端确认词。
        self.declare_parameter("execute_confirmation", "EXECUTE")

        # 本版本为正式抓取版本：最终必须闭合右夹爪。
        self.declare_parameter("close_gripper", True)

        # ------------------------------------------------------------------
        # V11 抓取结果验证 + 自动重抓
        # ------------------------------------------------------------------
        # 与单独右夹爪测试程序一致的默认阈值。
        self.declare_parameter(
            "grasp_verify_position_threshold_mm",
            1.5,
        )
        self.declare_parameter(
            "grasp_verify_effort_threshold",
            5.0,
        )
        # 默认只接受 GRASP_LIKELY。
        self.declare_parameter(
            "grasp_accept_possible",
            False,
        )
        # 失败后最多再重试 2 次，即总尝试次数最多 3 次。
        self.declare_parameter(
            "max_grasp_retries",
            2,
        )
        # 抓取失败后只抬 Lift 回这个安全高度。
        self.declare_parameter(
            "retry_lift_m",
            0.20,
        )
        # Lift 到位后先忽略一小段时间的积压视觉帧。
        self.declare_parameter(
            "retry_vision_settle_s",
            1.0,
        )

        self.center_topic = str(self.get_parameter("center_topic").value)
        self.valid_topic = str(self.get_parameter("valid_topic").value)
        self.points_topic = str(self.get_parameter("points_topic").value)
        self.color_info_topic = str(self.get_parameter("color_info_topic").value)
        self.output_point_topic = str(self.get_parameter("output_point_topic").value)
        self.output_valid_topic = str(self.get_parameter("output_valid_topic").value)

        self.calib_path = str(self.get_parameter("calib_path").value)

        self.depth_frame = str(self.get_parameter("depth_frame").value)
        self.color_frame = str(self.get_parameter("color_frame").value)
        self.camera_head_frame = str(self.get_parameter("camera_head_frame").value)
        self.head_pitch_frame = str(self.get_parameter("head_pitch_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        self.roi_radius_px = float(self.get_parameter("roi_radius_px").value)
        self.max_sync_dt_s = float(self.get_parameter("max_sync_dt_s").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.min_roi_points = int(self.get_parameter("min_roi_points").value)
        self.depth_mad_sigma = float(self.get_parameter("depth_mad_sigma").value)
        self.depth_min_band_m = float(self.get_parameter("depth_min_band_m").value)
        self.depth_max_band_m = float(self.get_parameter("depth_max_band_m").value)
        self.cloud_buffer_size = max(
            2,
            int(self.get_parameter("cloud_buffer_size").value),
        )
        self.max_process_rate_hz = max(
            0.1,
            float(self.get_parameter("max_process_rate_hz").value),
        )
        self.tf_timeout_s = float(self.get_parameter("tf_timeout_s").value)
        self.robot_tf_use_latest = bool(
            self.get_parameter("robot_tf_use_latest").value
        )

        self.urdf_path = str(self.get_parameter("urdf_path").value)
        self.velocity_scale = float(
            self.get_parameter("velocity_scale").value
        )
        self.gripper_effort = float(
            self.get_parameter("gripper_effort").value
        )
        self.max_fk_error_m = float(
            self.get_parameter("max_fk_error_cm").value
        ) / 100.0
        self.max_final_xy_error_m = float(
            self.get_parameter("max_final_xy_error_cm").value
        ) / 100.0

        self.plan_stability_frames = max(
            2,
            int(self.get_parameter("plan_stability_frames").value),
        )
        self.plan_stability_xy_m = float(
            self.get_parameter("plan_stability_xy_m").value
        )
        self.plan_stability_z_m = float(
            self.get_parameter("plan_stability_z_m").value
        )
        self.plan_rearm_distance_m = float(
            self.get_parameter("plan_rearm_distance_m").value
        )

        self.enable_motion = bool(
            self.get_parameter("enable_motion").value
        )
        self.execute_confirmation = str(
            self.get_parameter("execute_confirmation").value
        )

        self.close_gripper = bool(
            self.get_parameter("close_gripper").value
        )

        self.grasp_verify_position_threshold_m = (
            float(
                self.get_parameter(
                    "grasp_verify_position_threshold_mm"
                ).value
            )
            / 1000.0
        )
        self.grasp_verify_effort_threshold = float(
            self.get_parameter(
                "grasp_verify_effort_threshold"
            ).value
        )
        self.grasp_accept_possible = bool(
            self.get_parameter(
                "grasp_accept_possible"
            ).value
        )
        self.max_grasp_retries = max(
            0,
            int(
                self.get_parameter(
                    "max_grasp_retries"
                ).value
            ),
        )
        self.retry_lift_m = float(
            self.get_parameter(
                "retry_lift_m"
            ).value
        )
        self.retry_vision_settle_s = max(
            0.0,
            float(
                self.get_parameter(
                    "retry_vision_settle_s"
                ).value
            ),
        )

        if not self.close_gripper:
            raise RuntimeError(
                "本版本是正式抓取版本：close_gripper 必须为 true。"
            )

        self.offset_base = np.array(
            [
                float(self.get_parameter("offset_base_x_m").value),
                float(self.get_parameter("offset_base_y_m").value),
                float(self.get_parameter("offset_base_z_m").value),
            ],
            dtype=np.float64,
        )

        # ------------------------------------------------------------------
        # Hand-eye calibration
        # ------------------------------------------------------------------
        (
            self.T_headPitch_cameraHead,
            calib_robot_base,
            calib_tracking_base,
        ) = load_eye_on_base_calibration(self.calib_path)

        if calib_robot_base != self.head_pitch_frame:
            raise RuntimeError(
                "手眼标定 robot_base_frame 与参数不一致: "
                f"calib={calib_robot_base!r}, expected={self.head_pitch_frame!r}"
            )

        if calib_tracking_base != self.camera_head_frame:
            raise RuntimeError(
                "手眼标定 tracking_base_frame 与参数不一致: "
                f"calib={calib_tracking_base!r}, expected={self.camera_head_frame!r}"
            )

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.color_K: Optional[np.ndarray] = None
        self.color_D: Optional[np.ndarray] = None
        self.color_width: Optional[int] = None
        self.color_height: Optional[int] = None
        self.color_info_frame: Optional[str] = None

        self.detection_valid = False
        self.latest_center: Optional[PointStamped] = None

        self.cloud_buffer: Deque[PointCloud2] = deque(
            maxlen=self.cloud_buffer_size
        )

        self.last_process_monotonic = 0.0
        self.last_processed_detection_stamp: Optional[float] = None

        # IK dry-run 状态。
        self.plan_points: Deque[np.ndarray] = deque(
            maxlen=self.plan_stability_frames
        )
        self.last_planned_target_base: Optional[np.ndarray] = None
        self.ik_controller: Optional[GuardedIKGraspController] = None

        # V5: 只锁定一次，只允许执行一次。
        self.target_locked = False
        self.locked_target_base: Optional[np.ndarray] = None
        self.locked_plan: Optional[dict] = None
        self.execution_started = False
        self.execution_finished = False

        # V11 自动重抓状态。
        self.grasp_attempt_count = 0
        self.retry_vision_not_before = 0.0

        # ------------------------------------------------------------------
        # ROS I/O
        # ------------------------------------------------------------------
        self.center_sub = self.create_subscription(
            PointStamped,
            self.center_topic,
            self._on_center,
            10,
        )
        self.valid_sub = self.create_subscription(
            Bool,
            self.valid_topic,
            self._on_valid,
            10,
        )
        self.points_sub = self.create_subscription(
            PointCloud2,
            self.points_topic,
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.color_info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
        )

        self.point_pub = self.create_publisher(
            PointStamped,
            self.output_point_topic,
            10,
        )
        self.point_valid_pub = self.create_publisher(
            Bool,
            self.output_valid_topic,
            10,
        )

        # ------------------------------------------------------------------
        # 初始化现有抓取类（仅用于 calculate_plan）
        # ------------------------------------------------------------------
        # 构造器会连接已有控制器 Action server 并同步 /joint_states，
        # 但不会发送任何 goal。所有运动接口又被 DryRun 子类硬封锁。
        self.get_logger().info("正在初始化 IK dry-run 控制器...")
        limits = load_urdf_limits(self.urdf_path)
        self.ik_controller = GuardedIKGraspController(
            limits,
            velocity_scale=self.velocity_scale,
            gripper_effort=self.gripper_effort,
        )
        self.get_logger().info(
            "IK 控制器就绪：当前 DISARMED；只有人工确认后才临时解锁运动"
        )

        self.get_logger().info("====================================================")
        self.get_logger().info("screw_grasp_bridge V11 已启动：抓取验证 + 失败自动重新识别/规划")
        self.get_logger().info(f"YOLO center : {self.center_topic}")
        self.get_logger().info(f"YOLO valid  : {self.valid_topic}")
        self.get_logger().info(f"PointCloud  : {self.points_topic}")
        self.get_logger().info(f"ColorInfo   : {self.color_info_topic}")
        self.get_logger().info(f"Output      : {self.output_point_topic}")
        self.get_logger().info(f"Hand-eye    : {self.calib_path}")
        self.get_logger().info(
            "坐标链: PointCloud.header.frame_id -> "
            f"{self.camera_head_frame} [TF] -> "
            f"{self.head_pitch_frame} [CALIB] -> "
            f"{self.base_frame} [TF]"
        )
        self.get_logger().info(
            "点云 frame 将动态识别："
            f"{self.depth_frame} 或 {self.color_frame} 均可"
        )
        self.get_logger().info(
            "注意：不会查询 "
            f"{self.head_pitch_frame} <-> {self.camera_head_frame} "
            "这条可能冲突的 TF。"
        )
        self.get_logger().info(
            "机器人 TF 时间模式: "
            + (
                "LATEST（适合当前只定位、机器人静止验证）"
                if self.robot_tf_use_latest
                else "EXACT STAMP（要求机器人 TF 与相机时间严格同步）"
            )
        )
        self.get_logger().info(
            "V11 模式: 稳定视觉目标 -> 两次 IK 安全检查 -> "
            "Lift/XY/Z 抓取 -> 右夹爪反馈验证；"
            "失败则 Lift 回安全高度并重新识别/规划"
        )
        self.get_logger().info(
            "运动总开关: "
            + (
                "ENABLED（目标稳定且两次 IK 检查通过后自动执行）"
                if self.enable_motion
                else "DISABLED（本次运行绝不会运动）"
            )
        )
        if self.enable_motion:
            self.get_logger().warning(
                "⚠️ AUTO EXECUTE 模式已启用："
                "本次运行不会等待任何终端输入；"
                "视觉目标稳定并通过两次 IK/URDF 安全检查后将自动执行真实运动。"
            )
        self.get_logger().info(
            "IK 安全阈值: "
            f"FK<={self.max_fk_error_m*100.0:.1f}cm, "
            f"final XY<={self.max_final_xy_error_m*100.0:.1f}cm"
        )
        self.get_logger().info(
            "抓取验证阈值: "
            f"position>{self.grasp_verify_position_threshold_m*1000.0:.2f}mm, "
            f"|effort|>={self.grasp_verify_effort_threshold:.3f}, "
            f"accept_possible={self.grasp_accept_possible}"
        )
        self.get_logger().info(
            "自动重抓: "
            f"max_retries={self.max_grasp_retries}, "
            f"retry_lift={self.retry_lift_m:.3f}m, "
            f"vision_settle={self.retry_vision_settle_s:.2f}s"
        )
        self.get_logger().info("====================================================")

    # ------------------------------------------------------------------
    # V11 retry helpers
    # ------------------------------------------------------------------
    def _vision_retry_settling(self) -> bool:
        return (
            time.monotonic()
            < float(self.retry_vision_not_before)
        )

    def _reset_for_new_visual_attempt(self) -> None:
        """
        Lift 已经回到安全高度以后才调用。

        清掉旧锁定目标和旧视觉缓存，然后重新允许 YOLO +
        PointCloud 收集稳定目标。不会在这里发送任何运动命令。
        """
        self.plan_points.clear()
        self.last_planned_target_base = None

        self.target_locked = False
        self.locked_target_base = None
        self.locked_plan = None

        self.execution_started = False
        self.execution_finished = False

        self.detection_valid = False
        self.latest_center = None
        self.cloud_buffer.clear()

        self.last_process_monotonic = 0.0
        self.last_processed_detection_stamp = None

        self.retry_vision_not_before = (
            time.monotonic()
            + self.retry_vision_settle_s
        )

        self._publish_output_valid(False)

        self.get_logger().warning(
            "🔄 已解除旧目标锁定；"
            f"先忽略 {self.retry_vision_settle_s:.2f}s 积压视觉帧，"
            "随后重新识别弹簧并重新规划抓取。"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.color_K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.color_D = np.asarray(msg.d, dtype=np.float64)
        self.color_width = int(msg.width)
        self.color_height = int(msg.height)
        self.color_info_frame = str(msg.header.frame_id)

        if self.color_info_frame and self.color_info_frame != self.color_frame:
            self.get_logger().warning(
                "CameraInfo frame 与 color_frame 参数不同: "
                f"{self.color_info_frame!r} != {self.color_frame!r}"
            )

        # 内参只需读取一次即可。
        if self.camera_info_sub is not None:
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None

        self.get_logger().info(
            "已读取 RGB CameraInfo: "
            f"{self.color_width}x{self.color_height}, "
            f"fx={self.color_K[0,0]:.3f}, "
            f"fy={self.color_K[1,1]:.3f}, "
            f"cx={self.color_K[0,2]:.3f}, "
            f"cy={self.color_K[1,2]:.3f}, "
            f"frame={self.color_info_frame}"
        )

    def _on_valid(self, msg: Bool) -> None:
        self.detection_valid = bool(msg.data)
        if not self.detection_valid:
            self.latest_center = None
            self._publish_output_valid(False)

    def _on_center(self, msg: PointStamped) -> None:
        # 一旦锁定目标，不再接受新的视觉目标，避免运动前/运动中目标漂移。
        if self.target_locked:
            return

        # 自动重抓时丢弃运动期间积压的旧视觉帧。
        if self._vision_retry_settling():
            return

        # screw_ros2.py 只在检测有效时发布 center，所以收到 center 即可视为有效。
        self.detection_valid = True
        self.latest_center = msg

        self._try_process_latest_detection()

    def _on_cloud(self, msg: PointCloud2) -> None:
        # 锁定后冻结视觉，不再更新点云目标。
        if self.target_locked:
            return

        if self._vision_retry_settling():
            return

        cloud_frame = str(msg.header.frame_id).strip()

        if not cloud_frame:
            self.get_logger().warning(
                "PointCloud2 header.frame_id 为空，忽略该帧"
            )
            return

        # /depth/points 可能由驱动发布在 depth optical，
        # 也可能已经对齐到 color optical。后续始终以消息自己的
        # header.frame_id 为准，不再强行假定为 depth_frame。
        if cloud_frame not in (self.depth_frame, self.color_frame):
            self.get_logger().warning(
                "PointCloud 使用非预期 frame，但将按消息 frame 动态 TF: "
                f"{cloud_frame!r}"
            )

        self.cloud_buffer.append(msg)

        # 如果中心消息先到了但当时没有匹配点云，这里再尝试一次。
        if self.detection_valid and self.latest_center is not None:
            self._try_process_latest_detection()

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------
    def _find_nearest_cloud(
        self,
        detection_stamp_s: float,
    ) -> Tuple[Optional[PointCloud2], float]:
        if not self.cloud_buffer:
            return None, float("inf")

        best_msg = None
        best_dt = float("inf")

        for cloud in self.cloud_buffer:
            cloud_t = stamp_to_seconds(cloud.header.stamp)
            dt = abs(cloud_t - detection_stamp_s)
            if dt < best_dt:
                best_dt = dt
                best_msg = cloud

        return best_msg, best_dt

    def _try_process_latest_detection(self) -> None:
        if self._vision_retry_settling():
            return

        if not self.detection_valid or self.latest_center is None:
            return

        if self.color_K is None or self.color_D is None:
            return

        now_mono = time.monotonic()
        min_interval = 1.0 / self.max_process_rate_hz
        if now_mono - self.last_process_monotonic < min_interval:
            return

        detection_stamp_s = stamp_to_seconds(
            self.latest_center.header.stamp
        )

        # 同一个检测时间戳不重复做重计算。
        if (
            self.last_processed_detection_stamp is not None
            and abs(
                detection_stamp_s - self.last_processed_detection_stamp
            ) < 1e-9
        ):
            return

        cloud, sync_dt = self._find_nearest_cloud(detection_stamp_s)

        if cloud is None:
            return

        if sync_dt > self.max_sync_dt_s:
            # 点云缓冲区还可能等到更接近的一帧，所以这里不标记 processed。
            self._publish_output_valid(False)
            return

        self.last_process_monotonic = now_mono

        try:
            ok = self._localize(
                self.latest_center,
                cloud,
                sync_dt,
            )
        except Exception as exc:
            self.get_logger().error(
                f"定位失败: {type(exc).__name__}: {exc}"
            )
            self._publish_output_valid(False)
            return

        if ok:
            self.last_processed_detection_stamp = detection_stamp_s

    # ------------------------------------------------------------------
    # TF helpers
    # ------------------------------------------------------------------
    def _lookup_matrix(
        self,
        target_frame: str,
        source_frame: str,
        stamp,
        *,
        allow_latest_fallback: bool = False,
    ) -> np.ndarray:
        query_time = Time.from_msg(stamp)

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                query_time,
                timeout=Duration(seconds=self.tf_timeout_s),
            )
            return transform_stamped_to_matrix(tf_msg)

        except TransformException:
            if not allow_latest_fallback:
                raise

        # 只建议用于相机内部静态 TF。
        tf_msg = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
            timeout=Duration(seconds=self.tf_timeout_s),
        )
        return transform_stamped_to_matrix(tf_msg)

    # ------------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------------
    def _localize(
        self,
        center_msg: PointStamped,
        cloud_msg: PointCloud2,
        sync_dt: float,
    ) -> bool:
        center_u = float(center_msg.point.x)
        center_v = float(center_msg.point.y)

        # --------------------------------------------------------------
        # 1) PointCloud2 -> Nx3 in depth optical frame
        # --------------------------------------------------------------
        points_depth = pointcloud2_xyz(cloud_msg)

        if points_depth.size == 0:
            self.get_logger().warning("点云为空")
            self._publish_output_valid(False)
            return False

        finite = np.isfinite(points_depth).all(axis=1)
        z_ok = (
            (points_depth[:, 2] >= self.min_depth_m)
            & (points_depth[:, 2] <= self.max_depth_m)
        )
        points_depth = points_depth[finite & z_ok]

        if len(points_depth) < self.min_roi_points:
            self.get_logger().warning(
                f"有效点云过少: {len(points_depth)}"
            )
            self._publish_output_valid(False)
            return False

        # --------------------------------------------------------------
        # 2) PointCloud 实际 frame -> color optical，仅用于 RGB 投影匹配 YOLO
        # --------------------------------------------------------------
        cloud_frame = str(cloud_msg.header.frame_id).strip()
        if not cloud_frame:
            self.get_logger().warning("PointCloud frame_id 为空")
            self._publish_output_valid(False)
            return False

        if cloud_frame == self.color_frame:
            # 点云已经在 RGB optical frame，不再重复做 depth->color 外参。
            T_color_cloud = np.eye(4, dtype=np.float64)
        else:
            T_color_cloud = self._lookup_matrix(
                self.color_frame,
                cloud_frame,
                cloud_msg.header.stamp,
                allow_latest_fallback=True,  # 相机内部通常是静态 TF
            )

        points_color = transform_points(
            T_color_cloud,
            points_depth,
        )

        valid_front = (
            np.isfinite(points_color).all(axis=1)
            & (points_color[:, 2] > 1e-6)
        )
        points_depth = points_depth[valid_front]
        points_color = points_color[valid_front]

        u, v = project_plumb_bob(
            points_color,
            self.color_K,
            self.color_D,
        )

        in_image = (
            np.isfinite(u)
            & np.isfinite(v)
            & (u >= 0.0)
            & (v >= 0.0)
            & (u < float(self.color_width))
            & (v < float(self.color_height))
        )

        # 当前只拿到了 YOLO center，所以使用圆形 ROI。
        du = u - center_u
        dv = v - center_v
        in_roi = (
            in_image
            & ((du * du + dv * dv) <= self.roi_radius_px ** 2)
        )

        roi_depth = points_depth[in_roi]
        roi_color = points_color[in_roi]

        if len(roi_depth) < self.min_roi_points:
            self.get_logger().warning(
                "YOLO 中心 ROI 内 3D 点不足: "
                f"{len(roi_depth)} < {self.min_roi_points}, "
                f"center=({center_u:.1f},{center_v:.1f}), "
                f"radius={self.roi_radius_px:.1f}px, "
                f"sync_dt={sync_dt*1000.0:.1f}ms"
            )
            self._publish_output_valid(False)
            return False

        # --------------------------------------------------------------
        # 3) 对 ROI 深度做 MAD 鲁棒过滤，减少背景/飞点
        # --------------------------------------------------------------
        z_color = roi_color[:, 2]
        z_med = float(np.median(z_color))
        mad = float(np.median(np.abs(z_color - z_med)))
        robust_sigma = 1.4826 * mad

        z_band = max(
            self.depth_min_band_m,
            self.depth_mad_sigma * robust_sigma,
        )
        z_band = min(
            z_band,
            self.depth_max_band_m,
        )

        cluster_mask = np.abs(z_color - z_med) <= z_band
        cluster_depth = roi_depth[cluster_mask]
        cluster_color = roi_color[cluster_mask]

        if len(cluster_depth) < self.min_roi_points:
            # 如果 MAD 过滤过强，退回全部 ROI 点，不直接丢失目标。
            cluster_depth = roi_depth
            cluster_color = roi_color

        target_depth = np.median(
            cluster_depth,
            axis=0,
        ).astype(np.float64)

        target_color = np.median(
            cluster_color,
            axis=0,
        ).astype(np.float64)

        # --------------------------------------------------------------
        # 4) PointCloud 实际 frame -> camera_head_link
        # --------------------------------------------------------------
        if cloud_frame == self.camera_head_frame:
            T_cameraHead_cloud = np.eye(4, dtype=np.float64)
        else:
            T_cameraHead_cloud = self._lookup_matrix(
                self.camera_head_frame,
                cloud_frame,
                cloud_msg.header.stamp,
                allow_latest_fallback=True,  # 相机内部固定 TF
            )

        target_camera_head = transform_point(
            T_cameraHead_cloud,
            target_depth,
        )

        # --------------------------------------------------------------
        # 5) camera_head_link -> head_pitch_link
        #    ★ 显式使用右手眼标定，不查询冲突 TF
        # --------------------------------------------------------------
        target_head_pitch = transform_point(
            self.T_headPitch_cameraHead,
            target_camera_head,
        )

        # --------------------------------------------------------------
        # 6) head_pitch_link -> base_link
        #    使用点云时间戳的实时机器人 TF
        # --------------------------------------------------------------
        # 当前阶段机器人保持静止，因此默认取最新可用机器人 TF。
        # 这样可避免 robot_state_publisher 在关节静止时不刷新动态 TF，
        # 导致点云时间戳比最新 TF 更新而触发 future extrapolation。
        if self.robot_tf_use_latest:
            tf_base_head = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.head_pitch_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_s),
            )
            T_base_headPitch = transform_stamped_to_matrix(tf_base_head)
        else:
            T_base_headPitch = self._lookup_matrix(
                self.base_frame,
                self.head_pitch_frame,
                cloud_msg.header.stamp,
                allow_latest_fallback=False,
            )

        target_base_raw = transform_point(
            T_base_headPitch,
            target_head_pitch,
        )

        target_base = target_base_raw + self.offset_base

        # --------------------------------------------------------------
        # 7) 发布 base_link 点
        # --------------------------------------------------------------
        out = PointStamped()
        out.header.stamp = cloud_msg.header.stamp
        out.header.frame_id = self.base_frame
        out.point.x = float(target_base[0])
        out.point.y = float(target_base[1])
        out.point.z = float(target_base[2])

        self.point_pub.publish(out)
        self._publish_output_valid(True)

        # --------------------------------------------------------------
        # 8) 只打印未来 IK 参数映射，不执行
        # --------------------------------------------------------------
        # 现有 ik_grasp_dual_mirror_real_v4.py:
        #   ROS X = 输入 Y
        #   ROS Y = 输入 X
        # 所以:
        ik_x = float(target_base[1])
        ik_y = float(target_base[0])
        ik_z = float(target_base[2])

        spread_xyz = np.std(cluster_depth, axis=0)

        self.get_logger().info(
            "\n"
            "================ 视觉定位结果 ================\n"
            f"RGB center        : ({center_u:.1f}, {center_v:.1f}) px\n"
            f"RGB/Cloud Δt      : {sync_dt*1000.0:.1f} ms\n"
            f"ROI points        : {len(roi_depth)}\n"
            f"Cluster points    : {len(cluster_depth)}\n"
            f"Depth Z median    : {z_med:.4f} m\n"
            f"Depth Z band      : ±{z_band:.4f} m\n"
            f"Depth spread XYZ  : "
            f"({spread_xyz[0]:.4f}, {spread_xyz[1]:.4f}, {spread_xyz[2]:.4f}) m\n"
            f"PointCloud frame   : {cloud_frame}\n"
            f"P_cloud            : "
            f"({target_depth[0]:+.4f}, {target_depth[1]:+.4f}, {target_depth[2]:+.4f}) m\n"
            f"P_color optical   : "
            f"({target_color[0]:+.4f}, {target_color[1]:+.4f}, {target_color[2]:+.4f}) m\n"
            f"P_camera_head     : "
            f"({target_camera_head[0]:+.4f}, {target_camera_head[1]:+.4f}, {target_camera_head[2]:+.4f}) m\n"
            f"P_head_pitch      : "
            f"({target_head_pitch[0]:+.4f}, {target_head_pitch[1]:+.4f}, {target_head_pitch[2]:+.4f}) m\n"
            f"P_base RAW        : "
            f"({target_base_raw[0]:+.4f}, {target_base_raw[1]:+.4f}, {target_base_raw[2]:+.4f}) m\n"
            f"P_base FINAL      : "
            f"({target_base[0]:+.4f}, {target_base[1]:+.4f}, {target_base[2]:+.4f}) m\n"
            f"未来 IK 参数      : "
            f"-x {ik_x:+.4f} -y {ik_y:+.4f} -z {ik_z:+.4f}\n"
            "⚠️ 当前仅完成视觉定位；是否运动由 V5 双重确认门控制\n"
            "================================================"
        )

        # --------------------------------------------------------------
        # 9) 稳定目标 -> 现有 calculate_plan()，仍然只 dry-run
        # --------------------------------------------------------------
        self._maybe_lock_and_execute(target_base)

        return True

    def _maybe_lock_and_execute(
        self,
        target_base: np.ndarray,
    ) -> None:
        """
        V11 单次尝试流程：
          1) 连续稳定视觉目标
          2) 锁定 base_link 坐标
          3) 第一次 calculate_plan() 安全检查
          4) enable_motion=False 时停在 dry-run
          5) enable_motion=True 时用冻结目标做第二次 IK 检查
          6) 临时 arm_motion()
          7) execute_plan(..., close_gripper=True, verify_grasp=True)
          8) 抓取成功：保持夹持并结束
          9) 抓取失败：只抬 Lift 到 retry_lift_m
         10) 未超过重试次数时清空旧视觉状态，重新识别并重新规划
         11) 每次运动结束后立即 disarm

        每次抓取运动期间目标保持锁定；
        只有抓取检测失败且 Lift 已回安全高度后才重新开放视觉。
        """
        if self.ik_controller is None:
            return

        if self.target_locked or self.execution_started:
            return

        p = np.asarray(target_base, dtype=np.float64).reshape(3)
        self.plan_points.append(p.copy())

        if len(self.plan_points) < self.plan_stability_frames:
            self.get_logger().info(
                "V11 等待稳定目标: "
                f"{len(self.plan_points)}/{self.plan_stability_frames}"
            )
            return

        samples = np.asarray(self.plan_points, dtype=np.float64)
        mean_base = np.mean(samples, axis=0)
        std_base = np.std(samples, axis=0)

        stable = (
            std_base[0] <= self.plan_stability_xy_m
            and std_base[1] <= self.plan_stability_xy_m
            and std_base[2] <= self.plan_stability_z_m
        )

        if not stable:
            self.get_logger().warning(
                "V11 暂不锁定：目标尚未稳定，"
                f"std=({std_base[0]*1000.0:.1f}, "
                f"{std_base[1]*1000.0:.1f}, "
                f"{std_base[2]*1000.0:.1f}) mm"
            )
            return

        target_x = float(mean_base[1])
        target_y = float(mean_base[0])
        target_z = float(mean_base[2])

        self.get_logger().info(
            "\n"
            "================ V11 TARGET CANDIDATE ===============\n"
            f"稳定 P_base mean : "
            f"({mean_base[0]:+.4f}, {mean_base[1]:+.4f}, {mean_base[2]:+.4f}) m\n"
            f"稳定 P_base std  : "
            f"({std_base[0]*1000.0:.2f}, "
            f"{std_base[1]*1000.0:.2f}, "
            f"{std_base[2]*1000.0:.2f}) mm\n"
            f"calculate_plan 输入: "
            f"x={target_x:+.4f}, y={target_y:+.4f}, z={target_z:+.4f}\n"
            "先执行 IK/FK/URDF 安全检查..."
        )

        try:
            plan = self.ik_controller.calculate_plan(
                target_x,
                target_y,
                target_z,
                max_fk_error_m=self.max_fk_error_m,
                max_final_xy_error_m=self.max_final_xy_error_m,
            )
        except Exception as exc:
            self.get_logger().error(
                "\n"
                "❌ V5 目标未通过 IK 安全检查，未锁定、未运动\n"
                f"原因: {type(exc).__name__}: {exc}"
            )
            # 清空窗口，允许重新稳定后再尝试。
            self.plan_points.clear()
            return

        # IK 通过以后才正式冻结目标。
        self.target_locked = True
        self.locked_target_base = mean_base.copy()
        self.locked_plan = plan

        self._log_plan(
            plan,
            mean_base,
            std_base,
            title="✅ V11 目标已锁定 / 第一次 IK 检查通过",
        )

        self.get_logger().warning(
            "\n"
            "🔒 视觉目标现已冻结；后续 YOLO/PointCloud 更新将被忽略。\n"
            f"锁定 base_link = "
            f"({mean_base[0]:+.4f}, {mean_base[1]:+.4f}, {mean_base[2]:+.4f}) m\n"
            "本版本执行时会先打开左右夹爪，并真实移动双臂与 lift；"
            "最终 Z 定位完成后右夹爪会闭合。"
        )

        if not self.enable_motion:
            self.get_logger().warning(
                "\n"
                "🛑 enable_motion=false：运动总开关关闭。\n"
                "本次运行到此结束，不会发送任何运动命令。\n"
                "确认现场安全后，如需实机路径测试请重新运行：\n"
                "  python3 screw_grasp_bridge_v6_tcp_center_execute_close_right.py "
                "--ros-args -p enable_motion:=true"
            )
            return

        # --------------------------------------------------------------
        # enable_motion=true 时不再等待任何终端输入。
        # 目标保持冻结，随后使用冻结坐标 + 最新 joint_states
        # 做第二次最终 IK 检查，通过后自动执行。
        # --------------------------------------------------------------
        # --------------------------------------------------------------
        # 确认后，用冻结坐标 + 当前 joint_states 再算一次。
        # --------------------------------------------------------------
        self.get_logger().warning(
            "✅ 已收到人工确认。正在用冻结目标和当前关节状态做最后一次 IK 检查..."
        )

        try:
            final_plan = self.ik_controller.calculate_plan(
                target_x,
                target_y,
                target_z,
                max_fk_error_m=self.max_fk_error_m,
                max_final_xy_error_m=self.max_final_xy_error_m,
            )
        except Exception as exc:
            self.get_logger().error(
                "❌ 最终 IK 检查失败，取消运动: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        self._log_plan(
            final_plan,
            mean_base,
            std_base,
            title="✅ 最终 IK 检查通过，即将执行",
        )

        self.execution_started = True
        self.grasp_attempt_count += 1
        current_attempt = self.grasp_attempt_count
        total_allowed = 1 + self.max_grasp_retries

        self.ik_controller.arm_motion()

        try:
            self.get_logger().warning(
                "\n"
                "🚨 REAL MOTION START\n"
                f"抓取尝试 {current_attempt}/{total_allowed}\n"
                "执行：Lift安全高度 → XY → Z → 右夹爪闭合 → 抓取反馈验证。"
            )

            execution_result = (
                self.ik_controller.execute_plan(
                    final_plan,
                    close_gripper=True,
                    verify_grasp=True,
                    grasp_position_threshold_m=
                        self.grasp_verify_position_threshold_m,
                    grasp_effort_threshold=
                        self.grasp_verify_effort_threshold,
                    grasp_accept_possible=
                        self.grasp_accept_possible,
                )
            )

            verification = execution_result.get(
                "grasp_verification"
            )
            if verification is None:
                raise RuntimeError(
                    "抓取流程没有返回 grasp_verification"
                )

            verified = bool(
                verification.get("verified", False)
            )
            verdict = str(
                verification.get("verdict", "UNKNOWN")
            )
            reason = str(
                verification.get("reason", "")
            )

            if verified:
                self.execution_finished = True
                self.get_logger().info(
                    "\n"
                    "✅ SPRING GRASP VERIFIED\n"
                    f"attempt={current_attempt}/{total_allowed}\n"
                    f"verdict={verdict}\n"
                    f"reason={reason}\n"
                    "弹簧抓取确认成功；保持当前夹持状态，停止自动重抓。"
                )
                return

            # ----------------------------------------------------------
            # 抓取验证失败：先只抬 Lift，再决定是否重新开放视觉。
            # ----------------------------------------------------------
            self.get_logger().error(
                "\n"
                "❌ SPRING GRASP NOT VERIFIED\n"
                f"attempt={current_attempt}/{total_allowed}\n"
                f"verdict={verdict}\n"
                f"reason={reason}\n"
                "现在先只把 Lift 抬回安全高度。"
            )

            if not self.ik_controller.raise_lift_for_grasp_retry(
                self.retry_lift_m
            ):
                raise RuntimeError(
                    "抓取失败后 Lift 回安全高度失败；"
                    "停止自动重抓"
                )

            if current_attempt <= self.max_grasp_retries:
                self._reset_for_new_visual_attempt()
                self.get_logger().warning(
                    f"♻️ 将进行下一次抓取尝试 "
                    f"{current_attempt + 1}/{total_allowed}："
                    "等待新的 YOLO + PointCloud 稳定目标。"
                )
                return

            # 已达到重试次数上限：Lift 已经安全抬起，但不再自动运动。
            self.execution_finished = True
            self.execution_started = True
            self.get_logger().error(
                "\n"
                "🛑 已达到最大自动重抓次数。\n"
                f"总尝试次数={current_attempt}；"
                "Lift 已回安全高度，程序不再自动执行抓取。"
            )

        except KeyboardInterrupt:
            self.get_logger().warning(
                "收到 Ctrl+C，正在取消活动 Action..."
            )
            try:
                self.ik_controller.cancel_active_goals()
            except Exception:
                pass
            raise

        except Exception as exc:
            self.get_logger().error(
                "\n"
                "❌ V5 实机执行失败: "
                f"{type(exc).__name__}: {exc}\n"
                "正在请求取消活动 Action。"
            )
            try:
                self.ik_controller.cancel_active_goals()
            except Exception:
                pass

        finally:
            self.ik_controller.disarm_motion()
            self.get_logger().warning(
                "🔒 运动接口已重新 DISARMED。"
            )

    def _log_plan(
        self,
        plan: dict,
        target_base: np.ndarray,
        std_base: np.ndarray,
        *,
        title: str,
    ) -> None:
        ik_q = np.asarray(plan["ik_q"], dtype=np.float64)
        pregrasp_q = np.asarray(plan["pregrasp_q"], dtype=np.float64)
        grasp_q = np.asarray(plan["grasp_q"], dtype=np.float64)
        left_pregrasp_q = np.asarray(
            plan["left_pregrasp_q"],
            dtype=np.float64,
        )
        left_grasp_q = np.asarray(
            plan["left_grasp_q"],
            dtype=np.float64,
        )

        mirror_pre = float(
            np.max(np.abs(pregrasp_q + left_pregrasp_q))
        )
        mirror_grasp = float(
            np.max(np.abs(grasp_q + left_grasp_q))
        )

        self.get_logger().info(
            "\n"
            + title
            + "\n"
            f"target base_link            = "
            f"({target_base[0]:+.4f}, "
            f"{target_base[1]:+.4f}, "
            f"{target_base[2]:+.4f}) m\n"
            f"target std                  = "
            f"({std_base[0]*1000.0:.2f}, "
            f"{std_base[1]*1000.0:.2f}, "
            f"{std_base[2]*1000.0:.2f}) mm\n"
            f"ik_q(deg)                   = {np.round(np.degrees(ik_q), 2)}\n"
            f"pregrasp_q(deg)             = {np.round(np.degrees(pregrasp_q), 2)}\n"
            f"grasp_q(deg)                = {np.round(np.degrees(grasp_q), 2)}\n"
            f"left_pregrasp_q(deg)        = {np.round(np.degrees(left_pregrasp_q), 2)}\n"
            f"left_grasp_q(deg)           = {np.round(np.degrees(left_grasp_q), 2)}\n"
            f"IK FK error                 = {plan['ik_fk_error_m']*100.0:.3f} cm\n"
            f"pregrasp XY offset          = {plan['pregrasp_xy_offset_m']*100.0:.3f} cm\n"
            f"final XY error              = {plan['grasp_xy_error_m']*100.0:.3f} cm\n"
            f"grasp TCP hand offset       = "
            f"{np.round(np.asarray(plan['grasp_tcp_offset_hand']), 6)} m\n"
            f"TCP downward offset         = "
            f"{plan['grasp_tcp_down_offset_m']*1000.0:.1f} mm\n"
            f"current lift                = {plan['current_lift']:+.6f} m\n"
            f"XY fixed lift               = {plan['xy_lift']:+.6f} m\n"
            f"lift -> XY delta            = {plan['lift_to_xy_delta']:+.6f} m\n"
            f"target lift (final Z)       = {plan['target_lift']:+.6f} m\n"
            f"XY -> Z lift delta          = {plan['lift_delta']:+.6f} m\n"
            f"predicted final Z           = {plan['predicted_final_z']:+.6f} m\n"
            f"final Z error               = {plan['final_z_error_m']*100.0:.3f} cm\n"
            f"mirror error pregrasp       = {mirror_pre:.6f} rad\n"
            f"mirror error grasp          = {mirror_grasp:.6f} rad"
        )

    def _publish_output_valid(self, valid: bool) -> None:
        msg = Bool()
        msg.data = bool(valid)
        self.point_valid_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[ScrewGraspBridge] = None

    try:
        node = ScrewGraspBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(
                f"节点异常退出: {type(exc).__name__}: {exc}"
            )
        else:
            print(
                f"节点初始化失败: {type(exc).__name__}: {exc}"
            )
        raise
    finally:
        if node is not None:
            controller = getattr(node, "ik_controller", None)
            if controller is not None:
                try:
                    controller.shutdown()
                except Exception:
                    pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
