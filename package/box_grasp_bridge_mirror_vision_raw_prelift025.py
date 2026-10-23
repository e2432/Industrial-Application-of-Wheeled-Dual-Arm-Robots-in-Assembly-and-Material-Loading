#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO Pose 3D抓取点 -> base_link -> 纯视觉镜像目标 -> 新模板精确镜像双臂 TCP IK -> 实机抓箱

视觉输入（来自 box_ros2_single_box.py）:
  /box_grasp_point/short_edge_1   geometry_msgs/PointStamped
  /box_grasp_point/short_edge_2   geometry_msgs/PointStamped

每个点已经由 YOLO Pose + 对齐深度计算为 3D 点，通常位于
camera_head_color_optical_frame。

坐标链:
  P_source
    -> camera_head_link            [TF2，相机内部固定 TF]
    -> head_pitch_link             [eob_calib_right.calib]
    -> base_link                   [TF2，机器人实时 TF]

随后：
  base_link 中 Y 较大的点 -> 左臂抓取点
  base_link 中 Y 较小的点 -> 右臂抓取点
  不对视觉 XYZ 做固定加减；仅将左右目标关于 base_link Y=0 镜像化。

抓取核心:
  box_dual_arm_grasp_v3_exact_mirror.py

安全:
- 默认 enable_motion=false，绝不运动。
- 即使 enable_motion=true，也必须在终端精确输入 EXECUTE。
- 目标稳定后冻结，只执行一次。
- 执行前用冻结目标和最新 joint_states 再规划一次。
- 最终左右夹爪都会闭合。
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

try:
    import yaml
except ImportError as exc:
    raise RuntimeError(
        "缺少 PyYAML。请安装 PyYAML 后再运行。"
    ) from exc

from box_dual_arm_grasp_v9_template10_shoulder007_prelift025 import (
    DualArmGraspRealController,
    load_urdf_limits,
    DEFAULT_LEFT_TEMPLATE_DEG,
    DEFAULT_RIGHT_TEMPLATE_DEG,
    LEFT_GRASP_TCP_OFFSET_HAND,
    RIGHT_GRASP_TCP_OFFSET_HAND,
    GRASP_TCP_DOWN_OFFSET_M,
)


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quaternion_to_rotation_matrix(
    x: float,
    y: float,
    z: float,
    w: float,
) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("四元数长度为 0")
    x, y, z, w = q / norm

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def transform_stamped_to_matrix(
    msg: TransformStamped,
) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_rotation_matrix(
        q.x,
        q.y,
        q.z,
        q.w,
    )
    T[:3, 3] = [
        float(t.x),
        float(t.y),
        float(t.z),
    ]
    return T


def transform_point(
    T: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    return (
        T[:3, :3]
        @ np.asarray(point, dtype=np.float64)
        + T[:3, 3]
    )


def load_eye_on_base_calibration(
    calib_path: str,
):
    path = Path(calib_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"手眼标定文件不存在: {path}"
        )

    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    params = data.get("parameters", {})
    tf_data = data.get("transform", {})
    trans = tf_data.get("translation", {})
    rot = tf_data.get("rotation", {})

    calibration_type = str(
        params.get("calibration_type", "")
    )
    robot_base_frame = str(
        params.get("robot_base_frame", "")
    )
    tracking_base_frame = str(
        params.get("tracking_base_frame", "")
    )

    if calibration_type != "eye_on_base":
        raise RuntimeError(
            "期望 eye_on_base 标定，"
            f"实际={calibration_type!r}"
        )

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

    return (
        T,
        robot_base_frame,
        tracking_base_frame,
    )


class GuardedDualArmGraspController(
    DualArmGraspRealController
):
    """默认锁死所有真实运动接口。"""

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
                "MOTION SAFETY BLOCK: "
                "双臂运动接口未解锁"
            )

    def execute(self, *args, **kwargs):
        self._require_armed()
        return super().execute(*args, **kwargs)

    def animate(self, *args, **kwargs):
        self._require_armed()
        return super().animate(*args, **kwargs)

    def command_grippers(self, *args, **kwargs):
        self._require_armed()
        return super().command_grippers(
            *args,
            **kwargs,
        )


class BoxPoseGraspBridge(Node):
    def __init__(self) -> None:
        super().__init__("box_pose_grasp_bridge")

        # --------------------------------------------------------------
        # 视觉输入
        # --------------------------------------------------------------
        self.declare_parameter(
            "grasp_point_1_topic",
            "/box_grasp_point/short_edge_1",
        )
        self.declare_parameter(
            "grasp_point_2_topic",
            "/box_grasp_point/short_edge_2",
        )
        self.declare_parameter(
            "max_pair_dt_s",
            0.15,
        )

        # --------------------------------------------------------------
        # 坐标系 / 标定
        # --------------------------------------------------------------
        self.declare_parameter(
            "calib_path",
            "/home/niic/Desktop/competition/vision/"
            "transformation matrix/eob_calib_right.calib",
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
        self.declare_parameter(
            "tf_timeout_s",
            0.20,
        )
        self.declare_parameter(
            "robot_tf_use_latest",
            True,
        )

        # 可选整体位置补偿，默认 0。
        self.declare_parameter(
            "offset_base_x_m",
            0.0,
        )
        self.declare_parameter(
            "offset_base_y_m",
            0.0,
        )
        self.declare_parameter(
            "offset_base_z_m",
            0.0,
        )

        # --------------------------------------------------------------
        # 兼容旧 launch 的视觉补偿参数（当前版本保留声明但不参与目标计算）
        # --------------------------------------------------------------
        # 这些参数表示“要加到视觉测量值上的修正量”，不是测量误差本身。
        #
        # 例：
        #   视觉 X 比真实抓取点大 0.05m -> visual_correction_x_m = -0.05
        #   视觉 Z 比真实抓取点小 0.18m -> visual_correction_z_m = +0.18
        #
        # Y 被拆成两个量：
        #   visual_correction_center_y_m:
        #       整对抓取点共同左右平移；对最终强制镜像目标的半间距无影响。
        #   visual_half_span_correction_m:
        #       修改左右目标到中心线的距离。
        #       若视觉 |Y| 比真实抓取点大 0.15m，则设为 -0.15。
        self.declare_parameter(
            "visual_correction_x_m",
            0.0,
        )
        self.declare_parameter(
            "visual_correction_center_y_m",
            0.0,
        )
        self.declare_parameter(
            "visual_correction_z_m",
            0.0,
        )
        self.declare_parameter(
            "visual_half_span_correction_m",
            0.0,
        )

        # 防止实机参数误输入，例如把 0.15m 写成 15m。
        self.declare_parameter(
            "max_abs_visual_correction_m",
            0.30,
        )

        # 两个视觉点至少应有一定横向分离，避免错误配对。
        self.declare_parameter(
            "min_pair_separation_y_m",
            0.05,
        )

        # --------------------------------------------------------------
        # 精确镜像目标处理
        # --------------------------------------------------------------
        # 0.0 = 使用视觉检测到的两短边3D中点间距的一半；
        # >0  = 强制使用固定半间距，例如 0.20 表示左右目标 Y=±0.20m。
        self.declare_parameter(
            "mirror_half_span_override_m",
            0.0,
        )

        # --------------------------------------------------------------
        # 兼容旧 launch 的固定抓取参数（当前版本保留声明但不参与目标计算）
        # --------------------------------------------------------------
        # Isaac / ROS(base_link) 对应关系:
        #   Isaac X = -ROS Y
        #   Isaac Y =  ROS X
        #   Isaac Z =  ROS Z
        #
        # 用户要求:
        #   左/右 Isaac X 固定为 -/+0.19m
        #   Isaac Y 在视觉结果基础上减小 0.13m
        #   Isaac Z 在视觉结果基础上增大 0.22m
        #
        # 换算到 base_link:
        #   左/右 ROS Y 固定为 +/-0.19m
        #   ROS X = visual common X - 0.13m
        #   ROS Z = visual common Z + 0.22m
        self.declare_parameter(
            "fixed_isaac_half_x_m",
            0.19,
        )
        self.declare_parameter(
            "isaac_y_offset_m",
            -0.13,
        )
        self.declare_parameter(
            "isaac_z_offset_m",
            0.22,
        )
        self.declare_parameter(
            "enforce_fixed_grasp_limits",
            True,
        )

        # 对称化会把原始两点的平均 Y 平移到 base_link Y=0。
        # 如果箱体本身偏离机器人中心线过多，则拒绝抓取，避免为了镜像
        # 把抓取点从真实箱体位置横向“拉回”太远。
        self.declare_parameter(
            "max_discarded_center_y_m",
            0.10,
        )

        # --------------------------------------------------------------
        # 稳定性
        # --------------------------------------------------------------
        self.declare_parameter(
            "stability_frames",
            1,
        )
        self.declare_parameter(
            "stability_xy_m",
            0.010,
        )
        self.declare_parameter(
            "stability_z_m",
            0.015,
        )

        # --------------------------------------------------------------
        # 双臂规划参数
        # --------------------------------------------------------------
        self.declare_parameter(
            "urdf_path",
            "/home/niic/project/20260623/20260116/"
            "test_robot_A/src/robot_A/"
            "robot_a_description/urdf/robot_a.urdf",
        )
        self.declare_parameter(
            "velocity_scale",
            0.50,
        )
        self.declare_parameter(
            "gripper_effort",
            50.0,
        )

        self.declare_parameter(
            "max_xy_error_cm",
            1.0,
        )
        self.declare_parameter(
            "max_z_error_cm",
            1.0,
        )
        self.declare_parameter(
            "max_shared_lift_mismatch_cm",
            1.0,
        )
        self.declare_parameter(
            "max_orientation_error_deg",
            5.0,
        )
        self.declare_parameter(
            "max_forearm_tilt_deg",
            3.0,
        )
        self.declare_parameter(
            "max_gripper_vertical_error_deg",
            5.0,
        )
        self.declare_parameter(
            "max_template_rmse_deg",
            20.0,
        )
        self.declare_parameter(
            "lateral_weight",
            4.0,
        )
        self.declare_parameter(
            "max_lateral_error_cm",
            0.5,
        )
        self.declare_parameter(
            "max_lateral_joint_correction_deg",
            20.0,
        )
        self.declare_parameter(
            "shoulder_clearance",
            0.07,
        )

        # --------------------------------------------------------------
        # 真实执行保险
        # --------------------------------------------------------------
        self.declare_parameter(
            "enable_motion",
            False,
        )
        self.declare_parameter(
            "execute_confirmation",
            "EXECUTE",
        )
        # 仿照之前抓弹簧版本：
        # False = enable_motion=true 后，两次规划通过即自动执行；
        # True  = 仍要求终端输入 EXECUTE。
        self.declare_parameter(
            "require_execute_confirmation",
            False,
        )

        # --------------------------------------------------------------
        # 读取参数
        # --------------------------------------------------------------
        self.point1_topic = str(
            self.get_parameter(
                "grasp_point_1_topic"
            ).value
        )
        self.point2_topic = str(
            self.get_parameter(
                "grasp_point_2_topic"
            ).value
        )
        self.max_pair_dt_s = float(
            self.get_parameter(
                "max_pair_dt_s"
            ).value
        )

        self.calib_path = str(
            self.get_parameter(
                "calib_path"
            ).value
        )
        self.camera_head_frame = str(
            self.get_parameter(
                "camera_head_frame"
            ).value
        )
        self.head_pitch_frame = str(
            self.get_parameter(
                "head_pitch_frame"
            ).value
        )
        self.base_frame = str(
            self.get_parameter(
                "base_frame"
            ).value
        )
        self.tf_timeout_s = float(
            self.get_parameter(
                "tf_timeout_s"
            ).value
        )
        self.robot_tf_use_latest = bool(
            self.get_parameter(
                "robot_tf_use_latest"
            ).value
        )

        self.offset_base = np.array(
            [
                float(
                    self.get_parameter(
                        "offset_base_x_m"
                    ).value
                ),
                float(
                    self.get_parameter(
                        "offset_base_y_m"
                    ).value
                ),
                float(
                    self.get_parameter(
                        "offset_base_z_m"
                    ).value
                ),
            ],
            dtype=np.float64,
        )

        self.visual_correction = np.array(
            [
                float(
                    self.get_parameter(
                        "visual_correction_x_m"
                    ).value
                ),
                float(
                    self.get_parameter(
                        "visual_correction_center_y_m"
                    ).value
                ),
                float(
                    self.get_parameter(
                        "visual_correction_z_m"
                    ).value
                ),
            ],
            dtype=np.float64,
        )
        self.visual_half_span_correction_m = float(
            self.get_parameter(
                "visual_half_span_correction_m"
            ).value
        )
        self.max_abs_visual_correction_m = float(
            self.get_parameter(
                "max_abs_visual_correction_m"
            ).value
        )

        if self.max_abs_visual_correction_m <= 0.0:
            raise RuntimeError(
                "max_abs_visual_correction_m 必须 > 0"
            )

        correction_values = np.concatenate(
            [
                self.visual_correction,
                np.array(
                    [
                        self.visual_half_span_correction_m,
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        if np.max(
            np.abs(correction_values)
        ) > self.max_abs_visual_correction_m:
            raise RuntimeError(
                "视觉补偿量过大，拒绝启动: "
                f"{np.round(correction_values,4)} m；"
                f"允许最大绝对值="
                f"{self.max_abs_visual_correction_m:.3f}m"
            )

        self.min_pair_separation_y_m = float(
            self.get_parameter(
                "min_pair_separation_y_m"
            ).value
        )

        self.mirror_half_span_override_m = float(
            self.get_parameter(
                "mirror_half_span_override_m"
            ).value
        )

        self.fixed_isaac_half_x_m = float(
            self.get_parameter(
                "fixed_isaac_half_x_m"
            ).value
        )
        self.isaac_y_offset_m = float(
            self.get_parameter(
                "isaac_y_offset_m"
            ).value
        )
        self.isaac_z_offset_m = float(
            self.get_parameter(
                "isaac_z_offset_m"
            ).value
        )
        self.enforce_fixed_grasp_limits = bool(
            self.get_parameter(
                "enforce_fixed_grasp_limits"
            ).value
        )

        if self.fixed_isaac_half_x_m <= 0.0:
            raise RuntimeError(
                "fixed_isaac_half_x_m 必须 > 0"
            )
        self.max_discarded_center_y_m = float(
            self.get_parameter(
                "max_discarded_center_y_m"
            ).value
        )

        if self.mirror_half_span_override_m < 0.0:
            raise RuntimeError(
                "mirror_half_span_override_m 必须 >= 0"
            )
        if self.max_discarded_center_y_m <= 0.0:
            raise RuntimeError(
                "max_discarded_center_y_m 必须 > 0"
            )

        self.stability_frames = max(
            1,
            int(
                self.get_parameter(
                    "stability_frames"
                ).value
            ),
        )
        self.stability_xy_m = float(
            self.get_parameter(
                "stability_xy_m"
            ).value
        )
        self.stability_z_m = float(
            self.get_parameter(
                "stability_z_m"
            ).value
        )

        self.urdf_path = str(
            self.get_parameter(
                "urdf_path"
            ).value
        )
        self.velocity_scale = float(
            self.get_parameter(
                "velocity_scale"
            ).value
        )
        self.gripper_effort = float(
            self.get_parameter(
                "gripper_effort"
            ).value
        )

        self.max_xy_error_m = (
            float(
                self.get_parameter(
                    "max_xy_error_cm"
                ).value
            )
            / 100.0
        )
        self.max_z_error_m = (
            float(
                self.get_parameter(
                    "max_z_error_cm"
                ).value
            )
            / 100.0
        )
        self.max_shared_lift_mismatch_m = (
            float(
                self.get_parameter(
                    "max_shared_lift_mismatch_cm"
                ).value
            )
            / 100.0
        )
        self.max_orientation_error_rad = math.radians(
            float(
                self.get_parameter(
                    "max_orientation_error_deg"
                ).value
            )
        )
        self.max_forearm_tilt_rad = math.radians(
            float(
                self.get_parameter(
                    "max_forearm_tilt_deg"
                ).value
            )
        )
        self.max_gripper_vertical_error_rad = math.radians(
            float(
                self.get_parameter(
                    "max_gripper_vertical_error_deg"
                ).value
            )
        )
        self.max_template_rmse_rad = math.radians(
            float(
                self.get_parameter(
                    "max_template_rmse_deg"
                ).value
            )
        )
        self.lateral_weight = float(
            self.get_parameter(
                "lateral_weight"
            ).value
        )
        self.max_lateral_error_m = (
            float(
                self.get_parameter(
                    "max_lateral_error_cm"
                ).value
            )
            / 100.0
        )
        self.max_lateral_joint_correction_rad = (
            math.radians(
                float(
                    self.get_parameter(
                        "max_lateral_joint_correction_deg"
                    ).value
                )
            )
        )
        self.shoulder_clearance = float(
            self.get_parameter(
                "shoulder_clearance"
            ).value
        )
        self.get_logger().info(
            "肩部初始避障抬举参数 shoulder_clearance="
            f"{self.shoulder_clearance:.3f} rad"
        )

        self.enable_motion = bool(
            self.get_parameter(
                "enable_motion"
            ).value
        )
        self.execute_confirmation = str(
            self.get_parameter(
                "execute_confirmation"
            ).value
        )
        self.require_execute_confirmation = bool(
            self.get_parameter(
                "require_execute_confirmation"
            ).value
        )

        self.left_template_q = np.radians(
            np.asarray(
                DEFAULT_LEFT_TEMPLATE_DEG,
                dtype=np.float64,
            )
        )
        self.right_template_q = np.radians(
            np.asarray(
                DEFAULT_RIGHT_TEMPLATE_DEG,
                dtype=np.float64,
            )
        )

        # --------------------------------------------------------------
        # 手眼标定
        # --------------------------------------------------------------
        (
            self.T_headPitch_cameraHead,
            calib_robot_base,
            calib_tracking_base,
        ) = load_eye_on_base_calibration(
            self.calib_path
        )

        if calib_robot_base != self.head_pitch_frame:
            raise RuntimeError(
                "手眼标定 robot_base_frame 不一致: "
                f"{calib_robot_base!r} != "
                f"{self.head_pitch_frame!r}"
            )
        if (
            calib_tracking_base
            != self.camera_head_frame
        ):
            raise RuntimeError(
                "手眼标定 tracking_base_frame 不一致: "
                f"{calib_tracking_base!r} != "
                f"{self.camera_head_frame!r}"
            )

        # --------------------------------------------------------------
        # TF
        # --------------------------------------------------------------
        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=10.0)
        )
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        # --------------------------------------------------------------
        # 抓取控制器
        # --------------------------------------------------------------
        limits = load_urdf_limits(
            self.urdf_path
        )
        self.grasp_controller = (
            GuardedDualArmGraspController(
                limits=limits,
                velocity_scale=self.velocity_scale,
                gripper_effort=self.gripper_effort,
            )
        )

        if not self.grasp_controller.wait_for_servers():
            raise RuntimeError(
                "部分实机 Action 控制器未就绪"
            )
        if not self.grasp_controller.wait_for_state():
            raise RuntimeError(
                "无法获得完整 /joint_states"
            )

        # --------------------------------------------------------------
        # 状态
        # --------------------------------------------------------------
        self.latest_p1: Optional[PointStamped] = None
        self.latest_p2: Optional[PointStamped] = None
        self.last_pair_key: Optional[
            Tuple[int, int]
        ] = None

        self.pair_samples: Deque[np.ndarray] = deque(
            maxlen=self.stability_frames
        )

        self.target_locked = False
        self.execution_started = False
        self.execution_finished = False

        self.locked_left_base: Optional[
            np.ndarray
        ] = None
        self.locked_right_base: Optional[
            np.ndarray
        ] = None

        # --------------------------------------------------------------
        # I/O
        # --------------------------------------------------------------
        self.left_base_pub = self.create_publisher(
            PointStamped,
            "/box_grasp_point_base/left",
            10,
        )
        self.right_base_pub = self.create_publisher(
            PointStamped,
            "/box_grasp_point_base/right",
            10,
        )

        self.sub1 = self.create_subscription(
            PointStamped,
            self.point1_topic,
            self._on_point1,
            10,
        )
        self.sub2 = self.create_subscription(
            PointStamped,
            self.point2_topic,
            self._on_point2,
            10,
        )

        self.get_logger().info(
            "===================================================="
        )
        self.get_logger().info(
            "Box Pose 双臂抓取 bridge 已启动"
        )
        self.get_logger().info(
            f"视觉点1: {self.point1_topic}"
        )
        self.get_logger().info(
            f"视觉点2: {self.point2_topic}"
        )
        self.get_logger().info(
            "坐标链: point frame -> "
            f"{self.camera_head_frame} [TF] -> "
            f"{self.head_pitch_frame} [CALIB] -> "
            f"{self.base_frame} [TF]"
        )
        self.get_logger().info(
            "TCP offset: "
            f"left={np.round(LEFT_GRASP_TCP_OFFSET_HAND,6)}, "
            f"right={np.round(RIGHT_GRASP_TCP_OFFSET_HAND,6)}, "
            f"down={GRASP_TCP_DOWN_OFFSET_M*1000.0:.1f}mm"
        )
        self.get_logger().warning(
            "运动总开关: "
            + (
                (
                    "ENABLED；仍需终端输入 EXECUTE"
                    if self.require_execute_confirmation
                    else "ENABLED；两次规划通过后自动执行一次"
                )
                if self.enable_motion
                else "DISABLED；本次绝不会运动"
            )
        )
        self.get_logger().info(
            "纯视觉镜像模式: 不做XYZ加减/固定；最终目标强制 "
            "XL=XR, YL=-YR, ZL=ZR；"
            "核心强制 q_left=-q_right，waist=0"
        )
        self.get_logger().info(
            "mirror_half_span_override_m="
            f"{self.mirror_half_span_override_m:.3f} "
            "(0=使用视觉两短边间距/2)"
        )
        self.get_logger().info(
            "===================================================="
        )

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _on_point1(
        self,
        msg: PointStamped,
    ) -> None:
        if self.target_locked:
            return
        self.latest_p1 = msg
        self._try_process_pair()

    def _on_point2(
        self,
        msg: PointStamped,
    ) -> None:
        if self.target_locked:
            return
        self.latest_p2 = msg
        self._try_process_pair()

    def _try_process_pair(self) -> None:
        if (
            self.latest_p1 is None
            or self.latest_p2 is None
            or self.target_locked
        ):
            return

        t1 = stamp_to_seconds(
            self.latest_p1.header.stamp
        )
        t2 = stamp_to_seconds(
            self.latest_p2.header.stamp
        )

        if abs(t1 - t2) > self.max_pair_dt_s:
            return

        key = (
            int(t1 * 1e9),
            int(t2 * 1e9),
        )
        if key == self.last_pair_key:
            return
        self.last_pair_key = key

        try:
            p1_base = self._point_to_base(
                self.latest_p1
            )
            p2_base = self._point_to_base(
                self.latest_p2
            )
        except Exception as exc:
            self.get_logger().error(
                "3D 抓取点坐标转换失败: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        # ROS base_link: +Y 为机器人左侧。
        # 先保留视觉得到的真实两短边中点。
        if p1_base[1] >= p2_base[1]:
            raw_left_base = np.asarray(
                p1_base,
                dtype=np.float64,
            )
            raw_right_base = np.asarray(
                p2_base,
                dtype=np.float64,
            )
        else:
            raw_left_base = np.asarray(
                p2_base,
                dtype=np.float64,
            )
            raw_right_base = np.asarray(
                p1_base,
                dtype=np.float64,
            )

        # 保存“纯视觉+TF”的 base_link 原始值，方便比较校准前后。
        measured_left_base = raw_left_base.copy()
        measured_right_base = raw_right_base.copy()

        # 不对视觉 XYZ 做任何固定加减或标定补偿。
        # 后续只做左右镜像化：保留视觉 common X / common Z / 半间距。

        raw_y_separation = float(
            raw_left_base[1]
            - raw_right_base[1]
        )
        if (
            raw_y_separation
            < self.min_pair_separation_y_m
        ):
            self.get_logger().warning(
                "两个短边3D中点横向距离过小，"
                f"ΔY={raw_y_separation:.4f}m，"
                "本帧不用于抓取"
            )
            self.pair_samples.clear()
            return

        # ----------------------------------------------------------
        # ★ 最终抓取坐标强制关于机器人中心线镜像
        # ----------------------------------------------------------
        # 视觉原始点:
        #   Lraw=[XL,+YL,ZL], Rraw=[XR,-YR,ZR]
        #
        # 镜像目标:
        #   common_X = (XL+XR)/2
        #   common_Z = (ZL+ZR)/2
        #   half_Y   = |YL-YR|/2
        #
        #   L=[common_X,+half_Y,common_Z]
        #   R=[common_X,-half_Y,common_Z]
        #
        # 因而:
        #   X_L == X_R
        #   Y_L == -Y_R
        #   Z_L == Z_R
        #
        # 再配合 mirror core：
        #   q_left == -q_right
        raw_center_y = 0.5 * float(
            raw_left_base[1]
            + raw_right_base[1]
        )

        if (
            abs(raw_center_y)
            > self.max_discarded_center_y_m
        ):
            self.get_logger().warning(
                "箱体短边中点整体偏离机器人中心线过大，"
                "为保证镜像不强行平移抓取目标："
                f"raw_center_y={raw_center_y:+.4f}m > "
                f"{self.max_discarded_center_y_m:.4f}m"
            )
            self.pair_samples.clear()
            return

        visual_common_x = 0.5 * float(
            raw_left_base[0]
            + raw_right_base[0]
        )
        visual_common_z = 0.5 * float(
            raw_left_base[2]
            + raw_right_base[2]
        )
        vision_half_span = 0.5 * abs(
            raw_y_separation
        )

        # ------------------------------------------------------
        # 纯视觉镜像目标：不做任何 XYZ 固定值、加减偏移或半间距覆盖。
        #
        # 原始视觉点:
        #   Lraw=[XL, YL, ZL], Rraw=[XR, YR, ZR]
        #
        # 仅做镜像化:
        #   common_X = (XL + XR) / 2
        #   common_Z = (ZL + ZR) / 2
        #   half_Y   = |YL - YR| / 2
        #
        # 最终:
        #   L=[common_X, +half_Y, common_Z]
        #   R=[common_X, -half_Y, common_Z]
        #
        # 因此左右仍严格关于 base_link Y=0 镜像，但坐标尺度完全来自视觉。
        # ------------------------------------------------------
        common_x = visual_common_x
        common_z = visual_common_z
        half_span = vision_half_span
        half_span_source = "vision_only"

        if half_span < 0.5 * self.min_pair_separation_y_m:
            self.get_logger().warning(
                "镜像半间距过小，拒绝抓取: "
                f"vision={vision_half_span:.4f}m, "
                f"final={half_span:.4f}m, "
                f"source={half_span_source}"
            )
            self.pair_samples.clear()
            return

        left_base = np.array(
            [
                common_x,
                +half_span,
                common_z,
            ],
            dtype=np.float64,
        )
        right_base = np.array(
            [
                common_x,
                -half_span,
                common_z,
            ],
            dtype=np.float64,
        )

        # 数值级对称断言。
        symmetry_error = max(
            abs(left_base[0] - right_base[0]),
            abs(left_base[1] + right_base[1]),
            abs(left_base[2] - right_base[2]),
        )
        if symmetry_error > 1e-12:
            raise RuntimeError(
                "内部目标对称化失败: "
                f"error={symmetry_error:.12e}"
            )

        self._publish_base_point(
            self.left_base_pub,
            left_base,
            self.latest_p1.header.stamp,
        )
        self._publish_base_point(
            self.right_base_pub,
            right_base,
            self.latest_p2.header.stamp,
        )

        self.get_logger().info(
            "\n"
            "============ SHORT EDGE MIRROR TARGETS ============\n"
            f"视觉测量 L   = {np.round(measured_left_base,6)} m\n"
            f"视觉测量 R   = {np.round(measured_right_base,6)} m\n"
            "XYZ固定补偿   = DISABLED\n"
            "XYZ固定目标   = DISABLED\n"
            f"原始 centerY   = {raw_center_y:+.6f} m\n"
            f"原始 ΔY        = {raw_y_separation:.6f} m\n"
            f"视觉 common X  = {visual_common_x:.6f} m\n"
            f"视觉 common Z  = {visual_common_z:.6f} m\n"
            f"视觉半间距     = {vision_half_span:.6f} m\n"
            f"最终镜像半间距 = {half_span:.6f} m ({half_span_source})\n"
            f"最终镜像 L   = {np.round(left_base,6)} m\n"
            f"最终镜像 R   = {np.round(right_base,6)} m\n"
            f"目标对称误差 = {symmetry_error:.12f} m\n"
            "要求: XL=XR, YL=-YR, ZL=ZR\n"
            "==================================================="
        )

        y_separation = float(
            left_base[1] - right_base[1]
        )

        sample = np.concatenate(
            [left_base, right_base]
        )
        self.pair_samples.append(sample)

        if (
            len(self.pair_samples)
            < self.stability_frames
        ):
            self.get_logger().info(
                "等待双抓取点稳定: "
                f"{len(self.pair_samples)}/"
                f"{self.stability_frames}"
            )
            return

        samples = np.asarray(
            self.pair_samples,
            dtype=np.float64,
        )
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)

        left_mean = mean[:3]
        right_mean = mean[3:]
        left_std = std[:3]
        right_std = std[3:]

        stable = (
            left_std[0] <= self.stability_xy_m
            and left_std[1] <= self.stability_xy_m
            and left_std[2] <= self.stability_z_m
            and right_std[0] <= self.stability_xy_m
            and right_std[1] <= self.stability_xy_m
            and right_std[2] <= self.stability_z_m
        )

        if not stable:
            self.get_logger().warning(
                "双抓取点还不稳定: "
                f"Lstd(mm)="
                f"{np.round(left_std*1000.0,2)}, "
                f"Rstd(mm)="
                f"{np.round(right_std*1000.0,2)}"
            )
            return

        self._lock_plan_and_maybe_execute(
            left_mean,
            right_mean,
            left_std,
            right_std,
        )

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def _lookup_matrix(
        self,
        target_frame: str,
        source_frame: str,
        stamp,
        *,
        allow_latest_fallback: bool,
    ) -> np.ndarray:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time.from_msg(stamp),
                timeout=Duration(
                    seconds=self.tf_timeout_s
                ),
            )
            return transform_stamped_to_matrix(
                tf_msg
            )
        except TransformException:
            if not allow_latest_fallback:
                raise

        tf_msg = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
            timeout=Duration(
                seconds=self.tf_timeout_s
            ),
        )
        return transform_stamped_to_matrix(
            tf_msg
        )

    def _point_to_base(
        self,
        msg: PointStamped,
    ) -> np.ndarray:
        source_frame = str(
            msg.header.frame_id
        ).strip()
        if not source_frame:
            raise RuntimeError(
                "PointStamped frame_id 为空"
            )

        p_source = np.array(
            [
                float(msg.point.x),
                float(msg.point.y),
                float(msg.point.z),
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(p_source)):
            raise RuntimeError(
                "视觉 3D 点包含 NaN/Inf"
            )

        if source_frame == self.camera_head_frame:
            p_camera_head = p_source
        else:
            T_cameraHead_source = (
                self._lookup_matrix(
                    self.camera_head_frame,
                    source_frame,
                    msg.header.stamp,
                    allow_latest_fallback=True,
                )
            )
            p_camera_head = transform_point(
                T_cameraHead_source,
                p_source,
            )

        p_head_pitch = transform_point(
            self.T_headPitch_cameraHead,
            p_camera_head,
        )

        if self.robot_tf_use_latest:
            tf_base_head = (
                self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.head_pitch_frame,
                    Time(),
                    timeout=Duration(
                        seconds=self.tf_timeout_s
                    ),
                )
            )
            T_base_head = (
                transform_stamped_to_matrix(
                    tf_base_head
                )
            )
        else:
            T_base_head = self._lookup_matrix(
                self.base_frame,
                self.head_pitch_frame,
                msg.header.stamp,
                allow_latest_fallback=False,
            )

        p_base = transform_point(
            T_base_head,
            p_head_pitch,
        )

        return p_base + self.offset_base

    def _publish_base_point(
        self,
        pub,
        point: np.ndarray,
        stamp,
    ) -> None:
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = float(point[2])
        pub.publish(msg)

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------
    @staticmethod
    def _base_to_isaac(
        p_base: np.ndarray,
    ) -> np.ndarray:
        """
        抓取核心约定:
          ROS X = Isaac Y
          ROS Y = -Isaac X
          ROS Z = Isaac Z

        所以:
          Isaac X = -ROS Y
          Isaac Y =  ROS X
          Isaac Z =  ROS Z
        """
        return np.array(
            [
                -float(p_base[1]),
                +float(p_base[0]),
                +float(p_base[2]),
            ],
            dtype=np.float64,
        )

    def _refresh_joint_states(self) -> None:
        for _ in range(5):
            rclpy.spin_once(
                self.grasp_controller,
                timeout_sec=0.03,
            )

    def _plan(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> Tuple[dict, np.ndarray, np.ndarray]:
        self._refresh_joint_states()

        left_isaac = self._base_to_isaac(
            left_base
        )
        right_isaac = self._base_to_isaac(
            right_base
        )

        plan = self.grasp_controller.run(
            left_isaac=tuple(
                float(v)
                for v in left_isaac
            ),
            right_isaac=tuple(
                float(v)
                for v in right_isaac
            ),
            execute=False,
            close_gripper=True,
            left_template_q=self.left_template_q,
            right_template_q=self.right_template_q,
            max_xy_error_m=self.max_xy_error_m,
            max_z_error_m=self.max_z_error_m,
            max_shared_lift_mismatch_m=(
                self.max_shared_lift_mismatch_m
            ),
            max_orientation_error_rad=(
                self.max_orientation_error_rad
            ),
            max_forearm_tilt_rad=(
                self.max_forearm_tilt_rad
            ),
            max_gripper_vertical_error_rad=(
                self.max_gripper_vertical_error_rad
            ),
            max_template_rmse_rad=(
                self.max_template_rmse_rad
            ),
            lateral_weight=self.lateral_weight,
            max_lateral_error_m=(
                self.max_lateral_error_m
            ),
            max_lateral_joint_correction_rad=(
                self.max_lateral_joint_correction_rad
            ),
            shoulder_clearance=(
                self.shoulder_clearance
            ),
        )

        if plan is None:
            raise RuntimeError(
                "双臂规划未返回有效 solution"
            )

        return (
            plan,
            left_isaac,
            right_isaac,
        )

    def _lock_plan_and_maybe_execute(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
        left_std: np.ndarray,
        right_std: np.ndarray,
    ) -> None:
        if (
            self.target_locked
            or self.execution_started
        ):
            return

        try:
            (
                plan,
                left_isaac,
                right_isaac,
            ) = self._plan(
                left_base,
                right_base,
            )
        except Exception as exc:
            self.get_logger().error(
                "❌ 第一次双臂规划失败，"
                "不会锁定/不会运动: "
                f"{type(exc).__name__}: {exc}"
            )
            self.pair_samples.clear()
            return

        self.target_locked = True
        self.locked_left_base = (
            np.asarray(
                left_base,
                dtype=np.float64,
            ).copy()
        )
        self.locked_right_base = (
            np.asarray(
                right_base,
                dtype=np.float64,
            ).copy()
        )

        self._log_plan(
            "✅ 双视觉抓取点已锁定 / 第一次规划通过",
            plan,
            left_base,
            right_base,
            left_std,
            right_std,
            left_isaac,
            right_isaac,
        )

        if not self.enable_motion:
            self.get_logger().warning(
                "\n🛑 enable_motion=false："
                "本次只完成视觉→标定→双臂IK检查，"
                "机器人不会运动。"
            )
            return

        if self.require_execute_confirmation:
            expected = self.execute_confirmation
            self.get_logger().warning(
                "\n⚠️ 即将执行真实双臂抓箱。\n"
                "确认两个短边中点正确，且镜像化后的左右目标仍落在箱体两侧，"
                "箱体没有移动，双臂路径无障碍，急停可触达。\n"
                "最终左右夹爪都会闭合。\n"
                f"若确认，请精确输入: {expected}"
            )

            try:
                answer = input(
                    f"\nType {expected} to execute > "
                ).strip()
            except (
                EOFError,
                KeyboardInterrupt,
            ):
                answer = ""

            if answer != expected:
                self.get_logger().warning(
                    "❎ 未确认，取消执行；"
                    "机器人没有运动。"
                )
                return
        else:
            self.get_logger().warning(
                "\n⚠️ AUTO EXECUTE：enable_motion=true，"
                "第一次镜像IK已经通过。"
                "现在将使用冻结目标 + 最新 joint_states "
                "进行第二次最终规划；通过后自动执行一次。"
            )

        try:
            (
                final_plan,
                left_isaac,
                right_isaac,
            ) = self._plan(
                self.locked_left_base,
                self.locked_right_base,
            )
        except Exception as exc:
            self.get_logger().error(
                "❌ 第二次最终规划失败，取消运动: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        self._log_plan(
            "✅ 第二次最终规划通过，即将执行",
            final_plan,
            self.locked_left_base,
            self.locked_right_base,
            left_std,
            right_std,
            left_isaac,
            right_isaac,
        )

        self.execution_started = True
        self.grasp_controller.arm_motion()

        try:
            self.get_logger().warning(
                "🚨 REAL DUAL-ARM BOX GRASP START"
            )

            self.grasp_controller.execute(
                final_plan["left"],
                final_plan["right"],
                final_plan["body"],
                close_gripper=True,
                shoulder_clearance=(
                    self.shoulder_clearance
                ),
            )

            self.execution_finished = True
            self.get_logger().info(
                "✅ 双臂抓箱完成；"
                "左右夹爪已执行闭合。"
            )

        except Exception as exc:
            self.get_logger().error(
                "❌ 实机抓箱失败: "
                f"{type(exc).__name__}: {exc}"
            )
            try:
                self.grasp_controller.cancel_active_goals()
            except Exception:
                pass
        finally:
            self.grasp_controller.disarm_motion()
            self.get_logger().warning(
                "🔒 双臂运动接口已重新 DISARMED"
            )

    def _log_plan(
        self,
        title: str,
        plan: dict,
        left_base: np.ndarray,
        right_base: np.ndarray,
        left_std: np.ndarray,
        right_std: np.ndarray,
        left_isaac: np.ndarray,
        right_isaac: np.ndarray,
    ) -> None:
        metrics = plan["metrics"]
        xy_stage = plan["xy_stage"]

        self.get_logger().info(
            "\n"
            + title
            + "\n"
            f"left base      = "
            f"{np.round(left_base, 6)} m\n"
            f"right base     = "
            f"{np.round(right_base, 6)} m\n"
            f"left std       = "
            f"{np.round(left_std*1000.0, 2)} mm\n"
            f"right std      = "
            f"{np.round(right_std*1000.0, 2)} mm\n"
            f"left Isaac     = "
            f"{np.round(left_isaac, 6)} m\n"
            f"right Isaac    = "
            f"{np.round(right_isaac, 6)} m\n"
            f"left q(deg)    = "
            f"{np.round(np.degrees(plan['left']), 2)}\n"
            f"right q(deg)   = "
            f"{np.round(np.degrees(plan['right']), 2)}\n"
            f"left XY err    = "
            f"{metrics['left_xy']['xy_error_m']*100.0:.3f} cm\n"
            f"right XY err   = "
            f"{metrics['right_xy']['xy_error_m']*100.0:.3f} cm\n"
            f"left Z err     = "
            f"{metrics['left_final']['z_error_m']*100.0:.3f} cm\n"
            f"right Z err    = "
            f"{metrics['right_final']['z_error_m']*100.0:.3f} cm\n"
            f"left lateral   = "
            f"{metrics['left_xy']['lateral_error_m']*100.0:.3f} cm\n"
            f"right lateral  = "
            f"{metrics['right_xy']['lateral_error_m']*100.0:.3f} cm\n"
            f"shared lift    = "
            f"{plan['body']['base_elevator_joint']:+.6f} m\n"
            f"waist          = "
            f"{math.degrees(plan['body']['elevator_pitch_joint']):+.3f} deg\n"
            f"lift mismatch  = "
            f"{xy_stage['shared_lift_mismatch']*100.0:.3f} cm"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[
        BoxPoseGraspBridge
    ] = None

    try:
        node = BoxPoseGraspBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(
                "节点异常退出: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            print(
                "节点初始化失败: "
                f"{type(exc).__name__}: {exc}"
            )
        raise
    finally:
        if node is not None:
            controller = getattr(
                node,
                "grasp_controller",
                None,
            )
            if controller is not None:
                try:
                    controller.cancel_active_goals()
                except Exception:
                    pass
                try:
                    controller.destroy_node()
                except Exception:
                    pass
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
