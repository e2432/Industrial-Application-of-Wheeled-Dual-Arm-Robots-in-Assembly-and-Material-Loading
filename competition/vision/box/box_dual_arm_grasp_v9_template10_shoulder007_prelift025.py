#!/usr/bin/env python3
"""
robot_a 双臂固定点抓取 —— 精确镜像对称真实机器人 Action 版本
================================================

保留原脚本核心逻辑：
- 本地 FK
- 雅可比伪逆 IK
- dual_ik_fast()
- Isaac Sim 坐标输入与 ROS 坐标转换
- XY 平面模板约束 IK
- 独立 lift Z 定位
- 左右夹爪均使用“最大张开两指中点稍向下”的 grasp TCP
- 6 阶段避障抓取流程
- 强制 q_left = -q_right 精确镜像；waist 固定 0rad

仅替换通信执行层：
- 不再发布 /robot_a/joint_targets
- 改用真实机器人 ros2_control Action

使用的控制器：
- /lift_controller/follow_joint_trajectory
- /waist_controller/follow_joint_trajectory
- /left_arm_root_controller/follow_joint_trajectory
- /right_arm_root_controller/follow_joint_trajectory
- /left_arm_controller/follow_joint_trajectory
- /right_arm_controller/follow_joint_trajectory
- /left_gripper_controller/gripper_cmd
- /right_gripper_controller/gripper_cmd

默认只做 XY 模板约束 IK、独立 Z/lift 计算和安全检查。
必须显式添加 --execute 才会让真实机器人运动。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# ============================================================================
# FK 参数 —— 保留原代码
# ============================================================================

LEFT_ARM_PARAMS = [
    (0.0, -0.363, 0.2428, 0.0, 0.0, -np.pi / 2, 0, 0, 1),
    (0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0, 0, 0, -1),
    (-0.344, 0.0, 0.0, -np.pi / 2, 0.0, -np.pi / 2, 0, 0, -1),
    (0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0, 0, 1),
    (0.0, 0.233, 0.0, -np.pi / 2, 0.0, 0.0, 0, 0, 1),
    (0.0, 0.0, 0.0, -np.pi / 2, 0.0, -np.pi / 2, 0, 0, 1),
    (0.0, -0.211, 0.002, -np.pi / 2, 0.0, np.pi, 0, 0, 1),
]

RIGHT_ARM_PARAMS = [
    (0.0, -0.363, -0.2428, np.pi, 0.0, -np.pi / 2, 0, 0, 1),
    (0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0, 0, 0, 1),
    (-0.344, 0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0, 0, 1),
    (0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0, 0, 1),
    (0.0, -0.233, 0.0, np.pi / 2, 0.0, 0.0, 0, 0, 1),
    (0.0, 0.0, 0.0, -np.pi / 2, 0.0, -np.pi / 2, 0, 0, 1),
    (0.0, -0.211, 0.002, -np.pi / 2, 0.0, np.pi, 0, 0, 1),
]


# ============================================================================
# 真实控制器关节分组
# ============================================================================

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


# ============================================================================
# 标准抓取模板
# 图中关节顺序：
# [pitch_arm_*1,
#  arm_*1_*2, arm_*2_*3, arm_*3_*4,
#  arm_*4_*5, arm_*5_*6, arm_*6_*7]
# ============================================================================

# 抓取标准模板（2026-08-11 更新）
#
# q 顺序:
#   q[0] pitch_arm_*1_joint
#   q[1] arm_*1_arm_*2_joint
#   q[2] arm_*2_arm_*3_joint
#   q[3] arm_*3_arm_*4_joint
#   q[4] arm_*4_arm_*5_joint
#   q[5] arm_*5_arm_*6_joint
#   q[6] arm_*6_arm_*7_joint
#
# 左臂:
#   [-20, 0, +10, -70, -93, 0, 0] deg
# 右臂严格镜像:
#   [+20, 0, -10, +70, +93, 0, 0] deg
#
# 注意：本版本不加入 q2 定向扫描/fallback；
#       其余 IK、精确 q_left=-q_right 镜像、waist=0、
#       独立 lift/Z、6 阶段实机执行逻辑均保持原版。
DEFAULT_LEFT_TEMPLATE_DEG = np.array(
    [-20.0, 0.0, 10.0, -70.0, -93.0, 0.0, 0.0],
    dtype=np.float64,
)
DEFAULT_RIGHT_TEMPLATE_DEG = np.array(
    [20.0, 0.0, -10.0, 70.0, 93.0, 0.0, 0.0],
    dtype=np.float64,
)

# 避障阶段的肩部抬举量。
# 左右肩关节的合法方向相反。
# 初始肩部避障抬举量设为 0.07 rad。
DEFAULT_SHOULDER_CLEARANCE = 0.07


# ============================================================================
# 双臂夹爪抓取 TCP
# ============================================================================
#
# URDF 中 hand_*_link 与 arm_*7_link 是零偏移固定连接。
# 两侧夹爪各有两个 prismatic finger joint，最大开度均为 0.025m。
#
# 最大张开时，两个 finger joint frame 会沿 hand 局部 X 反向移动，
# 所以“两夹爪臂之间的中点”与两个 joint origin 的平均值相同。
#
# 左夹爪最大张开两臂中点（hand_left_link）:
#   (-0.0995000, +0.0328455, +0.1245700) m
#
# 右夹爪最大张开两臂中点（hand_right_link）:
#   (+0.0977285, +0.0369690, +0.1218600) m
#
# 本抓箱模板中两只夹爪的 local X 轴近似竖直，但方向相反：
#   左 hand +X -> 世界向上，因此“稍微靠下” = local -X
#   右 hand +X -> 世界向下，因此“稍微靠下” = local +X
#
# 默认下移 15mm。实机若抓得偏高/偏低，只改这个常量即可。
GRIPPER_MAX_OPEN_M = 0.025
GRASP_TCP_DOWN_OFFSET_M = 0.015

LEFT_GRIPPER_OPEN_CENTER_HAND = np.array(
    [-0.0995000, +0.0328455, +0.1245700],
    dtype=np.float64,
)
RIGHT_GRIPPER_OPEN_CENTER_HAND = np.array(
    [+0.0977285, +0.0369690, +0.1218600],
    dtype=np.float64,
)

LEFT_GRASP_TCP_OFFSET_HAND = (
    LEFT_GRIPPER_OPEN_CENTER_HAND
    + np.array(
        [-GRASP_TCP_DOWN_OFFSET_M, 0.0, 0.0],
        dtype=np.float64,
    )
)
RIGHT_GRASP_TCP_OFFSET_HAND = (
    RIGHT_GRIPPER_OPEN_CENTER_HAND
    + np.array(
        [+GRASP_TCP_DOWN_OFFSET_M, 0.0, 0.0],
        dtype=np.float64,
    )
)


# ============================================================================
# 工具函数 —— 保留原代码
# ============================================================================

def rpy_to_mat(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0],
        [-sp, cp * sr, cp * cr, 0],
        [0, 0, 0, 1],
    ])


def transl_mat(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ])


def rot_z_mat(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


# ============================================================================
# 快速 FK —— 保留原代码
# ============================================================================

def _joint_transform(ox, oy, oz, rr, rp, ry, ax, ay, az, angle):
    del ax, ay
    fixed = transl_mat(ox, oy, oz) @ rpy_to_mat(rr, rp, ry)
    sign = -1 if az < 0 else 1
    return fixed @ rot_z_mat(sign * angle)


def arm_fk(
    joint_angles: np.ndarray,
    lift: float,
    waist: float,
    params: List[Tuple],
) -> np.ndarray:
    transform = transl_mat(0.146, 0, 0.8658) @ transl_mat(0, 0, lift)
    transform = (
        transform
        @ transl_mat(0, 0, 0)
        @ rpy_to_mat(-np.pi / 2, 0, 0)
        @ rot_z_mat(waist)
    )

    for index, params_row in enumerate(params):
        transform = transform @ _joint_transform(
            *params_row,
            joint_angles[index],
        )
    return transform


def _arm_side_from_params(params: List[Tuple]) -> str:
    """根据左右臂第一段固定偏移判断当前是哪一侧。"""
    if not params:
        raise ValueError("FK params 为空")

    # LEFT 第一段 z=+0.2428；RIGHT 第一段 z=-0.2428。
    return "left" if float(params[0][2]) > 0.0 else "right"


def grasp_tcp_offset_hand(params: List[Tuple]) -> np.ndarray:
    """返回当前手臂在 hand frame 下的抓取 TCP 偏移。"""
    side = _arm_side_from_params(params)
    if side == "left":
        return LEFT_GRASP_TCP_OFFSET_HAND
    return RIGHT_GRASP_TCP_OFFSET_HAND


def grasp_tcp_fk(
    joint_angles: np.ndarray,
    lift: float,
    waist: float,
    params: List[Tuple],
) -> np.ndarray:
    """
    base_link -> 实际抓取 TCP。

    TCP = 最大张开两指中点 + 15mm 世界向下等效偏移。
    左右手使用镜像的 local X 偏移，从而都指向世界向下。
    """
    hand_transform = arm_fk(
        joint_angles,
        lift,
        waist,
        params,
    )
    offset = grasp_tcp_offset_hand(params)

    return (
        hand_transform
        @ transl_mat(
            float(offset[0]),
            float(offset[1]),
            float(offset[2]),
        )
    )


# ============================================================================
# 模板约束 IK
# 保留数值雅可比 + 阻尼伪逆核心思路，同时增加：
# - 目标位置
# - 标准模板末端方向
# - 前臂水平
# - 关节接近标准模板
# ============================================================================

def arm_fk_frames(
    joint_angles: np.ndarray,
    lift: float,
    waist: float,
    params: List[Tuple],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """返回末端变换及每个关节之后的变换。"""
    transform = transl_mat(0.146, 0, 0.8658) @ transl_mat(0, 0, lift)
    transform = (
        transform
        @ transl_mat(0, 0, 0)
        @ rpy_to_mat(-np.pi / 2, 0, 0)
        @ rot_z_mat(waist)
    )

    frames: List[np.ndarray] = []
    for index, params_row in enumerate(params):
        transform = transform @ _joint_transform(
            *params_row,
            joint_angles[index],
        )
        frames.append(transform.copy())

    return transform, frames


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """SO(3) 旋转矩阵的旋转向量，单位 rad。"""
    cosine = float(
        np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    )
    angle = math.acos(cosine)

    if angle < 1e-8:
        return 0.5 * np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])

    if abs(math.pi - angle) < 1e-4:
        values, vectors = np.linalg.eig(rotation)
        index = int(np.argmin(np.abs(values - 1.0)))
        axis = np.real(vectors[:, index])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return np.zeros(3)
        return angle * axis / norm

    scale = angle / (2.0 * math.sin(angle))
    return scale * np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])


def arm_joint_limit_arrays(
    limits: Mapping[str, Tuple[float, float, float]],
    side: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if side == "left":
        names = LEFT_ROOT_JOINTS + LEFT_ARM_JOINTS
    elif side == "right":
        names = RIGHT_ROOT_JOINTS + RIGHT_ARM_JOINTS
    else:
        raise ValueError(f"未知手臂: {side}")

    lower = np.array(
        [limits[name][0] for name in names],
        dtype=np.float64,
    )
    upper = np.array(
        [limits[name][1] for name in names],
        dtype=np.float64,
    )
    return lower, upper


def grasp_pose_metrics(
    q: np.ndarray,
    target: Sequence[float],
    template_q: np.ndarray,
    lift: float,
    waist: float,
    params: List[Tuple],
) -> Dict[str, float]:
    """
    计算抓取位姿指标。

    XY 误差与 Z 误差分开返回，避免重新把 Z 混入机械臂 IK。
    """
    transform, frames = arm_fk_frames(
        q,
        lift,
        waist,
        params,
    )
    template_transform, _ = arm_fk_frames(
        template_q,
        lift,
        waist,
        params,
    )

    # 位置误差全部使用真实 grasp TCP；姿态约束仍使用 hand frame。
    tcp_offset = grasp_tcp_offset_hand(params)
    tcp_transform = (
        transform
        @ transl_mat(
            float(tcp_offset[0]),
            float(tcp_offset[1]),
            float(tcp_offset[2]),
        )
    )

    actual_xyz = tcp_transform[:3, 3].astype(float)
    target_xyz = np.asarray(target, dtype=float)

    forward_error = float(
        abs(actual_xyz[0] - target_xyz[0])
    )
    lateral_error = float(
        abs(actual_xyz[1] - target_xyz[1])
    )
    xy_error = float(
        np.linalg.norm(actual_xyz[:2] - target_xyz[:2])
    )
    z_error = float(abs(actual_xyz[2] - target_xyz[2]))

    orientation_error = float(
        np.linalg.norm(
            rotation_vector(
                transform[:3, :3]
                @ template_transform[:3, :3].T
            )
        )
    )

    # frame[3] 到末端代表前臂/腕部整体方向。
    forearm = frames[6][:3, 3] - frames[3][:3, 3]
    forearm_norm = float(np.linalg.norm(forearm))
    if forearm_norm < 1e-9:
        forearm_tilt = math.pi / 2
    else:
        forearm_tilt = math.asin(
            float(
                np.clip(
                    abs(forearm[2]) / forearm_norm,
                    0.0,
                    1.0,
                )
            )
        )

    # 标准模板中夹爪局部 X 轴为竖直轴。
    gripper_axis = transform[:3, 0]
    gripper_vertical_error = math.acos(
        float(
            np.clip(
                abs(
                    np.dot(
                        gripper_axis,
                        np.array([0.0, 0.0, 1.0]),
                    )
                ),
                0.0,
                1.0,
            )
        )
    )

    template_rmse = float(
        np.sqrt(np.mean((q - template_q) ** 2))
    )

    return {
        "x_m": float(actual_xyz[0]),
        "y_m": float(actual_xyz[1]),
        "z_m": float(actual_xyz[2]),
        "xy_error_m": xy_error,
        # 内部 ROS X 对应 Isaac Y（前后）。
        "forward_error_m": forward_error,
        # 内部 ROS Y 对应 -Isaac X（箱体左右夹紧方向）。
        "lateral_error_m": lateral_error,
        "z_error_m": z_error,
        "orientation_error_rad": orientation_error,
        "forearm_tilt_rad": forearm_tilt,
        "gripper_vertical_error_rad": gripper_vertical_error,
        "template_rmse_rad": template_rmse,
    }


def xy_template_task_residual(
    q: np.ndarray,
    target_xy: np.ndarray,
    template_q: np.ndarray,
    lift: float,
    waist: float,
    params: List[Tuple],
    orientation_weight: float,
    forearm_weight: float,
    template_weight: float,
    lateral_weight: float,
) -> np.ndarray:
    """
    XY 平面 IK 残差。

    注意：这里故意不包含 Z 残差。Z 只在 IK 完成后通过 lift 调整。
    """
    transform, frames = arm_fk_frames(
        q,
        lift,
        waist,
        params,
    )
    template_transform, _ = arm_fk_frames(
        template_q,
        lift,
        waist,
        params,
    )

    # XY IK 也必须以夹爪中心 TCP 为末端点。
    tcp_offset = grasp_tcp_offset_hand(params)
    tcp_transform = (
        transform
        @ transl_mat(
            float(tcp_offset[0]),
            float(tcp_offset[1]),
            float(tcp_offset[2]),
        )
    )

    raw_xy_error = tcp_transform[:2, 3] - target_xy

    # 内部第二维 ROS Y 对应 Isaac X，也就是箱体夹紧方向。
    xy_error = np.array([
        raw_xy_error[0],
        float(lateral_weight) * raw_xy_error[1],
    ])

    orientation_error = rotation_vector(
        transform[:3, :3]
        @ template_transform[:3, :3].T
    )

    forearm = frames[6][:3, 3] - frames[3][:3, 3]
    forearm_norm = float(np.linalg.norm(forearm))
    forearm_vertical = (
        1.0
        if forearm_norm < 1e-9
        else float(forearm[2] / forearm_norm)
    )

    template_error = q - template_q

    return np.concatenate([
        xy_error,
        orientation_weight * orientation_error,
        np.array([forearm_weight * forearm_vertical]),
        template_weight * template_error,
    ])


def arm_ik_template_xy(
    target: Tuple[float, float, float],
    reference_lift: float,
    waist: float,
    params: List[Tuple],
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    template_q: np.ndarray,
    current_q: Optional[np.ndarray],
    max_xy_error_m: float,
    max_orientation_error_rad: float,
    max_forearm_tilt_rad: float,
    max_gripper_vertical_error_rad: float,
    max_template_rmse_rad: float,
    lateral_weight: float,
    max_iter: int = 300,
) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
    """
    数值雅可比 + 阻尼最小二乘 XY 模板约束 IK。

    机械臂关节只负责：
    1. X/Y 平面定位；
    2. 前臂水平；
    3. 夹爪竖直；
    4. 接近标准模板。

    Z 完全不参与此函数的优化。
    """
    target_xyz = np.asarray(target, dtype=np.float64)
    target_xy = target_xyz[:2]

    template_q = np.clip(
        np.asarray(template_q, dtype=np.float64),
        lower_limits,
        upper_limits,
    )

    seeds = [template_q.copy()]
    if current_q is not None:
        current_q = np.clip(
            np.asarray(current_q, dtype=np.float64),
            lower_limits,
            upper_limits,
        )
        seeds.extend([
            np.clip(
                0.75 * template_q + 0.25 * current_q,
                lower_limits,
                upper_limits,
            ),
            current_q,
        ])

    orientation_weight = 0.20
    forearm_weight = 0.50
    template_weight = 0.03
    finite_delta = 1e-5
    damping = 0.03
    maximum_step = 0.15

    best_q: Optional[np.ndarray] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_score = float("inf")

    for seed in seeds:
        q = seed.copy()

        for _ in range(max_iter):
            residual = xy_template_task_residual(
                q,
                target_xy,
                template_q,
                reference_lift,
                waist,
                params,
                orientation_weight,
                forearm_weight,
                template_weight,
                lateral_weight,
            )

            metrics = grasp_pose_metrics(
                q,
                target_xyz,
                template_q,
                reference_lift,
                waist,
                params,
            )

            score = (
                4.0 * metrics["forward_error_m"]
                + 10.0 * metrics["lateral_error_m"]
                + 0.20 * metrics["orientation_error_rad"]
                + 0.25 * metrics["forearm_tilt_rad"]
                + 0.15 * metrics["gripper_vertical_error_rad"]
                + 0.04 * metrics["template_rmse_rad"]
            )

            if score < best_score:
                best_score = score
                best_q = q.copy()
                best_metrics = dict(metrics)

            success = (
                metrics["xy_error_m"] <= max_xy_error_m
                and metrics["orientation_error_rad"]
                <= max_orientation_error_rad
                and metrics["forearm_tilt_rad"]
                <= max_forearm_tilt_rad
                and metrics["gripper_vertical_error_rad"]
                <= max_gripper_vertical_error_rad
                and metrics["template_rmse_rad"]
                <= max_template_rmse_rad
            )
            if success:
                return q, metrics

            jacobian = np.zeros((len(residual), 7))

            for joint_index in range(7):
                perturbed_q = q.copy()
                perturbed_q[joint_index] = min(
                    upper_limits[joint_index],
                    perturbed_q[joint_index] + finite_delta,
                )

                if perturbed_q[joint_index] == q[joint_index]:
                    perturbed_q[joint_index] = max(
                        lower_limits[joint_index],
                        q[joint_index] - finite_delta,
                    )

                actual_delta = (
                    perturbed_q[joint_index] - q[joint_index]
                )
                if abs(actual_delta) < 1e-12:
                    continue

                perturbed_residual = xy_template_task_residual(
                    perturbed_q,
                    target_xy,
                    template_q,
                    reference_lift,
                    waist,
                    params,
                    orientation_weight,
                    forearm_weight,
                    template_weight,
                    lateral_weight,
                )
                jacobian[:, joint_index] = (
                    perturbed_residual - residual
                ) / actual_delta

            try:
                normal = (
                    jacobian.T @ jacobian
                    + (damping ** 2) * np.eye(7)
                )
                dq = -np.linalg.solve(
                    normal,
                    jacobian.T @ residual,
                )
            except np.linalg.LinAlgError:
                break

            largest = float(np.max(np.abs(dq)))
            if largest > maximum_step:
                dq *= maximum_step / largest

            q_next = np.clip(
                q + dq,
                lower_limits,
                upper_limits,
            )

            if float(np.linalg.norm(q_next - q)) < 1e-7:
                break

            q = q_next

    if best_q is not None and best_metrics is not None:
        print(
            "     最接近 XY 模板解仍未通过阈值: "
            f"前后={best_metrics['forward_error_m'] * 100:.2f}cm, "
            f"夹紧X={best_metrics['lateral_error_m'] * 100:.2f}cm, "
            f"计算Z={best_metrics['z_m']:.3f}m, "
            f"前臂倾角="
            f"{math.degrees(best_metrics['forearm_tilt_rad']):.2f}°, "
            f"夹爪竖直误差="
            f"{math.degrees(best_metrics['gripper_vertical_error_rad']):.2f}°, "
            f"模板RMSE="
            f"{math.degrees(best_metrics['template_rmse_rad']):.2f}°"
        )

    return None



def refine_lateral_with_arm2_arm3(
    q: np.ndarray,
    target: Sequence[float],
    template_q: np.ndarray,
    reference_lift: float,
    waist: float,
    params: List[Tuple],
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    side: str,
    max_lateral_error_m: float,
    max_forward_error_m: float,
    max_orientation_error_rad: float,
    max_forearm_tilt_rad: float,
    max_gripper_vertical_error_rad: float,
    max_template_rmse_rad: float,
    max_joint_correction_rad: float,
) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
    """
    使用 arm_*2_arm_*3_joint（q[2]）精修箱体夹紧方向。

    Isaac X 在内部对应 ROS Y。右臂向内夹紧时通常表现为
    arm_right2_arm_right3_joint 减小，左臂则镜像增大。
    实际方向由 FK 搜索自动选择。
    """
    if side not in ("left", "right"):
        raise ValueError(f"未知手臂侧: {side}")

    joint_index = 2
    joint_name = (
        "arm_left2_arm_left3_joint"
        if side == "left"
        else "arm_right2_arm_right3_joint"
    )

    target_array = np.asarray(target, dtype=float)
    base_q = np.asarray(q, dtype=float).copy()
    before = grasp_pose_metrics(
        base_q,
        target_array,
        template_q,
        reference_lift,
        waist,
        params,
    )

    if before["lateral_error_m"] <= max_lateral_error_m:
        print(
            f"     {joint_name} 横向精修无需调整: "
            f"夹紧X误差={before['lateral_error_m'] * 100:.2f}cm"
        )
        return base_q, before

    original_angle = float(base_q[joint_index])
    lower = max(
        float(lower_limits[joint_index]),
        original_angle - abs(max_joint_correction_rad),
    )
    upper = min(
        float(upper_limits[joint_index]),
        original_angle + abs(max_joint_correction_rad),
    )

    if upper <= lower:
        return None

    best_q = None
    best_metrics = None
    best_score = float("inf")
    center = original_angle
    half_width = 0.5 * (upper - lower)

    # 三层由粗到细搜索。q[2] 是唯一主动精修变量。
    for _ in range(3):
        local_lower = max(lower, center - half_width)
        local_upper = min(upper, center + half_width)

        for angle in np.linspace(local_lower, local_upper, 51):
            candidate = base_q.copy()
            candidate[joint_index] = float(angle)

            metrics = grasp_pose_metrics(
                candidate,
                target_array,
                template_q,
                reference_lift,
                waist,
                params,
            )

            valid = (
                metrics["forward_error_m"]
                <= max_forward_error_m
                and metrics["orientation_error_rad"]
                <= max_orientation_error_rad
                and metrics["forearm_tilt_rad"]
                <= max_forearm_tilt_rad
                and metrics["gripper_vertical_error_rad"]
                <= max_gripper_vertical_error_rad
                and metrics["template_rmse_rad"]
                <= max_template_rmse_rad
            )
            if not valid:
                continue

            score = (
                100.0 * metrics["lateral_error_m"]
                + 8.0 * metrics["forward_error_m"]
                + 0.15 * metrics["orientation_error_rad"]
                + 0.15 * metrics["forearm_tilt_rad"]
                + 0.04 * metrics["template_rmse_rad"]
                + 0.002 * abs(float(angle) - original_angle)
            )

            if score < best_score:
                best_score = score
                best_q = candidate
                best_metrics = metrics

        if best_q is None:
            return None

        center = float(best_q[joint_index])
        half_width *= 0.18

    assert best_metrics is not None

    if best_metrics["lateral_error_m"] > max_lateral_error_m:
        print(
            f"     {joint_name} 精修后仍未达到横向阈值: "
            f"{best_metrics['lateral_error_m'] * 100:.2f}cm"
        )
        return None

    delta = float(best_q[joint_index] - original_angle)
    if abs(delta) < 1e-9:
        direction = "保持"
    else:
        direction = "增大" if delta > 0.0 else "减小"

    print(
        f"     {joint_name} 横向精修: "
        f"{original_angle:+.5f} → "
        f"{best_q[joint_index]:+.5f}rad "
        f"({direction} {abs(delta):.5f}rad), "
        f"夹紧X误差 "
        f"{before['lateral_error_m'] * 100:.2f}cm "
        f"→ {best_metrics['lateral_error_m'] * 100:.2f}cm"
    )

    return best_q, best_metrics


def dual_ik_xy_then_lift(
    left_target,
    right_target,
    limits: Mapping[str, Tuple[float, float, float]],
    left_template_q: np.ndarray,
    right_template_q: np.ndarray,
    current_left_q: Optional[np.ndarray],
    current_right_q: Optional[np.ndarray],
    current_lift: float,
    current_waist: float,
    max_xy_error_m: float,
    max_z_error_m: float,
    max_shared_lift_mismatch_m: float,
    max_orientation_error_rad: float,
    max_forearm_tilt_rad: float,
    max_gripper_vertical_error_rad: float,
    max_template_rmse_rad: float,
    lateral_weight: float,
    max_lateral_error_m: float,
    max_lateral_joint_correction_rad: float,
) -> Optional[dict]:
    """
    两阶段定位：
    A. 固定当前 lift，只求左右夹爪 XY 和标准模板姿态；
    B. 用 FK 得到此时夹爪 Z，再计算共享 lift 目标。

    因为 base_elevator_joint 是纯 Z 平移：
        lift_target = current_lift + target_z - solved_gripper_z
    """
    left_lower, left_upper = arm_joint_limit_arrays(
        limits,
        "left",
    )
    right_lower, right_upper = arm_joint_limit_arrays(
        limits,
        "right",
    )

    left_target = np.asarray(left_target, dtype=float)
    right_target = np.asarray(right_target, dtype=float)

    symmetric = (
        abs(left_target[0] - right_target[0]) < 0.01
        and abs(left_target[1] + right_target[1]) < 0.01
    )
    if symmetric:
        print("  🔁 左右 XY 对称 → 使用镜像标准模板")

    # lift 固定为当前实机值；只搜索 waist。
    # 编码器零点附近可能出现极小负值，先裁剪到 URDF 合法范围。
    waist_lower, waist_upper, _ = limits[
        "elevator_pitch_joint"
    ]
    raw_current_waist = float(current_waist)
    current_waist = float(
        np.clip(
            raw_current_waist,
            waist_lower,
            waist_upper,
        )
    )
    if abs(current_waist - raw_current_waist) > 1e-9:
        print(
            "  ℹ️ waist 当前反馈已按 URDF 限位修正: "
            f"{raw_current_waist:+.6f} "
            f"→ {current_waist:+.6f}rad"
        )

    waist_search_upper = min(0.9, waist_upper)
    waist_grid = np.linspace(
        waist_lower,
        waist_search_upper,
        8,
    )
    waist_values = sorted(
        {
            current_waist,
            *[float(value) for value in waist_grid],
        },
        key=lambda value: abs(value - current_waist),
    )

    # 优先测试标准模板在 XY 上更接近目标的 waist。
    waist_candidates = []
    for waist in waist_values:
        left_reference = grasp_tcp_fk(
            left_template_q,
            current_lift,
            waist,
            LEFT_ARM_PARAMS,
        )[:2, 3]
        right_reference = grasp_tcp_fk(
            right_template_q,
            current_lift,
            waist,
            RIGHT_ARM_PARAMS,
        )[:2, 3]

        reference_xy_distance = (
            float(
                np.linalg.norm(
                    left_reference - left_target[:2]
                )
            )
            + float(
                np.linalg.norm(
                    right_reference - right_target[:2]
                )
            )
            + 0.02 * abs(waist - current_waist)
        )
        waist_candidates.append(
            (reference_xy_distance, waist)
        )

    waist_candidates.sort(key=lambda item: item[0])

    lift_lower, lift_upper, _ = limits["base_elevator_joint"]
    started = time.time()
    best_failure = None

    for _, waist in waist_candidates:
        left_result = arm_ik_template_xy(
            tuple(left_target),
            current_lift,
            waist,
            LEFT_ARM_PARAMS,
            left_lower,
            left_upper,
            left_template_q,
            current_left_q,
            max_xy_error_m,
            max_orientation_error_rad,
            max_forearm_tilt_rad,
            max_gripper_vertical_error_rad,
            max_template_rmse_rad,
            lateral_weight,
        )
        if left_result is None:
            continue

        left_q, left_xy_metrics = left_result

        right_seed = (
            -left_q if symmetric else current_right_q
        )
        right_result = arm_ik_template_xy(
            tuple(right_target),
            current_lift,
            waist,
            RIGHT_ARM_PARAMS,
            right_lower,
            right_upper,
            right_template_q,
            right_seed,
            max_xy_error_m,
            max_orientation_error_rad,
            max_forearm_tilt_rad,
            max_gripper_vertical_error_rad,
            max_template_rmse_rad,
            lateral_weight,
        )
        if right_result is None:
            continue

        right_q, right_xy_metrics = right_result

        # 主 IK 后，用 arm_*2_arm_*3_joint 专门精修
        # 箱体横向（Isaac X / ROS Y）位置。
        left_refined = refine_lateral_with_arm2_arm3(
            left_q,
            left_target,
            left_template_q,
            current_lift,
            waist,
            LEFT_ARM_PARAMS,
            left_lower,
            left_upper,
            "left",
            max_lateral_error_m,
            max_xy_error_m,
            max_orientation_error_rad,
            max_forearm_tilt_rad,
            max_gripper_vertical_error_rad,
            max_template_rmse_rad,
            max_lateral_joint_correction_rad,
        )
        if left_refined is None:
            best_failure = "左臂横向夹紧关节精修失败"
            continue
        left_q, left_xy_metrics = left_refined

        right_refined = refine_lateral_with_arm2_arm3(
            right_q,
            right_target,
            right_template_q,
            current_lift,
            waist,
            RIGHT_ARM_PARAMS,
            right_lower,
            right_upper,
            "right",
            max_lateral_error_m,
            max_xy_error_m,
            max_orientation_error_rad,
            max_forearm_tilt_rad,
            max_gripper_vertical_error_rad,
            max_template_rmse_rad,
            max_lateral_joint_correction_rad,
        )
        if right_refined is None:
            best_failure = "右臂横向夹紧关节精修失败"
            continue
        right_q, right_xy_metrics = right_refined

        left_z_at_xy = float(left_xy_metrics["z_m"])
        right_z_at_xy = float(right_xy_metrics["z_m"])

        left_required_lift = (
            current_lift
            + float(left_target[2])
            - left_z_at_xy
        )
        right_required_lift = (
            current_lift
            + float(right_target[2])
            - right_z_at_xy
        )

        lift_mismatch = abs(
            left_required_lift - right_required_lift
        )
        shared_lift = 0.5 * (
            left_required_lift + right_required_lift
        )

        if lift_mismatch > max_shared_lift_mismatch_m:
            best_failure = (
                "左右臂所需 lift 不一致: "
                f"left={left_required_lift:+.4f}m, "
                f"right={right_required_lift:+.4f}m, "
                f"差值={lift_mismatch * 100:.2f}cm"
            )
            continue

        if shared_lift < lift_lower or shared_lift > lift_upper:
            best_failure = (
                f"计算 lift={shared_lift:+.4f}m 越界，"
                f"允许范围=[{lift_lower:+.4f}, {lift_upper:+.4f}]m"
            )
            continue

        left_final_metrics = grasp_pose_metrics(
            left_q,
            left_target,
            left_template_q,
            shared_lift,
            waist,
            LEFT_ARM_PARAMS,
        )
        right_final_metrics = grasp_pose_metrics(
            right_q,
            right_target,
            right_template_q,
            shared_lift,
            waist,
            RIGHT_ARM_PARAMS,
        )

        if (
            left_final_metrics["z_error_m"] > max_z_error_m
            or right_final_metrics["z_error_m"] > max_z_error_m
        ):
            best_failure = (
                "共享 lift 后 Z 误差超限: "
                f"left={left_final_metrics['z_error_m'] * 100:.2f}cm, "
                f"right={right_final_metrics['z_error_m'] * 100:.2f}cm"
            )
            continue

        elapsed = time.time() - started
        print(
            f"  ✅ XY 模板 IK + 独立 lift: "
            f"{elapsed * 1000:.0f}ms  "
            f"waist={math.degrees(waist):.1f}°"
        )
        print(
            "     XY 解时 grasp TCP Z: "
            f"左={left_z_at_xy:.4f}m, "
            f"右={right_z_at_xy:.4f}m"
        )
        print(
            "     由 Z 独立计算 lift: "
            f"左需求={left_required_lift:+.4f}m, "
            f"右需求={right_required_lift:+.4f}m, "
            f"共享目标={shared_lift:+.4f}m"
        )

        if symmetric:
            symmetric_error = float(
                np.max(np.abs(left_q + right_q))
            )
            print(
                f"     镜面对称偏差: "
                f"{symmetric_error:.4f}rad"
            )

        return {
            "left": left_q,
            "right": right_q,
            "body": {
                "base_elevator_joint": shared_lift,
                "elevator_pitch_joint": waist,
            },
            "xy_stage": {
                "reference_lift": float(current_lift),
                "left_gripper_z": left_z_at_xy,
                "right_gripper_z": right_z_at_xy,
                "left_required_lift": left_required_lift,
                "right_required_lift": right_required_lift,
                "shared_lift_mismatch": lift_mismatch,
            },
            "metrics": {
                "left_xy": left_xy_metrics,
                "right_xy": right_xy_metrics,
                "left_final": left_final_metrics,
                "right_final": right_final_metrics,
            },
        }

    if best_failure:
        print(f"     最接近结果失败原因: {best_failure}")

    print(
        f"  ❌ XY 模板 IK 或独立 lift 无解"
        f"（耗时 {time.time() - started:.1f}s）"
    )
    return None


# ============================================================================
# URDF 限位
# ============================================================================

def load_urdf_limits(
    urdf_path: str,
) -> Dict[str, Tuple[float, float, float]]:
    path = Path(urdf_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 URDF: {path}")

    root = ET.parse(path).getroot()
    limits: Dict[str, Tuple[float, float, float]] = {}

    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in ALL_MOTION_JOINTS:
            continue

        limit = joint.find("limit")
        if limit is None:
            continue

        lower = limit.get("lower")
        upper = limit.get("upper")
        velocity = limit.get("velocity")

        if lower is None or upper is None or velocity is None:
            continue

        limits[name] = (
            float(lower),
            float(upper),
            float(velocity),
        )

    missing = [
        name
        for name in ALL_MOTION_JOINTS
        if name not in limits
    ]
    if missing:
        raise RuntimeError(
            "URDF 缺少以下运动关节限位:\n  "
            + "\n  ".join(missing)
        )

    return limits


def validate_solution(
    solution: dict,
    limits: Mapping[str, Tuple[float, float, float]],
) -> None:
    values: Dict[str, float] = {
        "base_elevator_joint":
            float(solution["body"]["base_elevator_joint"]),
        "elevator_pitch_joint":
            float(solution["body"]["elevator_pitch_joint"]),
        "pitch_arm_left1_joint":
            float(solution["left"][0]),
        "pitch_arm_right1_joint":
            float(solution["right"][0]),
    }

    for name, value in zip(LEFT_ARM_JOINTS, solution["left"][1:]):
        values[name] = float(value)
    for name, value in zip(RIGHT_ARM_JOINTS, solution["right"][1:]):
        values[name] = float(value)

    problems = []
    for name, value in values.items():
        lower, upper, _ = limits[name]
        if value < lower or value > upper:
            problems.append(
                f"{name}: value={value:+.6f}, "
                f"limit=[{lower:+.6f}, {upper:+.6f}]"
            )

    if problems:
        raise RuntimeError(
            "IK 解超出实机关节限位，拒绝执行:\n  "
            + "\n  ".join(problems)
        )


def verify_grasp_solution(
    solution: dict,
    left_target: Tuple[float, float, float],
    right_target: Tuple[float, float, float],
    left_template_q: np.ndarray,
    right_template_q: np.ndarray,
    max_xy_error_m: float,
    max_z_error_m: float,
    max_shared_lift_mismatch_m: float,
    max_orientation_error_rad: float,
    max_forearm_tilt_rad: float,
    max_gripper_vertical_error_rad: float,
    max_template_rmse_rad: float,
    max_lateral_error_m: float,
) -> None:
    body = solution["body"]
    xy_stage = solution["xy_stage"]

    results = {
        "左": grasp_pose_metrics(
            solution["left"],
            left_target,
            left_template_q,
            body["base_elevator_joint"],
            body["elevator_pitch_joint"],
            LEFT_ARM_PARAMS,
        ),
        "右": grasp_pose_metrics(
            solution["right"],
            right_target,
            right_template_q,
            body["base_elevator_joint"],
            body["elevator_pitch_joint"],
            RIGHT_ARM_PARAMS,
        ),
    }

    print(
        "  XY 求解阶段夹爪 Z: "
        f"左={xy_stage['left_gripper_z']:.4f}m, "
        f"右={xy_stage['right_gripper_z']:.4f}m"
    )
    print(
        "  独立 Z 定位: "
        f"reference_lift={xy_stage['reference_lift']:+.4f}m, "
        f"final_lift={body['base_elevator_joint']:+.4f}m, "
        f"左右需求差="
        f"{xy_stage['shared_lift_mismatch'] * 100:.2f}cm"
    )

    problems = []
    for label, metrics in results.items():
        xy_cm = metrics["xy_error_m"] * 100.0
        forward_cm = metrics["forward_error_m"] * 100.0
        lateral_cm = metrics["lateral_error_m"] * 100.0
        z_cm = metrics["z_error_m"] * 100.0
        orientation_deg = math.degrees(
            metrics["orientation_error_rad"]
        )
        forearm_deg = math.degrees(
            metrics["forearm_tilt_rad"]
        )
        vertical_deg = math.degrees(
            metrics["gripper_vertical_error_rad"]
        )
        template_deg = math.degrees(
            metrics["template_rmse_rad"]
        )

        print(
            f"  {label}臂最终: "
            f"XY误差={xy_cm:.2f}cm, "
            f"前后误差={forward_cm:.2f}cm, "
            f"夹紧X误差={lateral_cm:.2f}cm, "
            f"Z误差={z_cm:.2f}cm, "
            f"模板方向误差={orientation_deg:.2f}°, "
            f"前臂倾角={forearm_deg:.2f}°, "
            f"夹爪竖直误差={vertical_deg:.2f}°, "
            f"模板RMSE={template_deg:.2f}°"
        )

        if metrics["xy_error_m"] > max_xy_error_m:
            problems.append(f"{label}臂 XY 误差超限")
        if metrics["lateral_error_m"] > max_lateral_error_m:
            problems.append(
                f"{label}臂箱体夹紧方向（Isaac X）误差超限"
            )
        if metrics["z_error_m"] > max_z_error_m:
            problems.append(f"{label}臂 Z 误差超限")
        if (
            metrics["orientation_error_rad"]
            > max_orientation_error_rad
        ):
            problems.append(f"{label}臂模板方向误差超限")
        if metrics["forearm_tilt_rad"] > max_forearm_tilt_rad:
            problems.append(f"{label}臂前臂不够水平")
        if (
            metrics["gripper_vertical_error_rad"]
            > max_gripper_vertical_error_rad
        ):
            problems.append(f"{label}臂夹爪不够竖直")
        if metrics["template_rmse_rad"] > max_template_rmse_rad:
            problems.append(f"{label}臂关节偏离模板过大")

    if (
        xy_stage["shared_lift_mismatch"]
        > max_shared_lift_mismatch_m
    ):
        problems.append("左右臂所需 lift 差异过大")

    if problems:
        raise RuntimeError(
            "XY/Z 独立抓取位姿未通过检查，拒绝执行:\n  "
            + "\n  ".join(problems)
        )


# ============================================================================
# ROS Action 实机控制器
# ============================================================================

def to_duration(value: float) -> Tuple[int, int]:
    seconds = int(value)
    nanoseconds = int(round((value - seconds) * 1e9))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds = 0
    return seconds, nanoseconds


class DualArmGraspRealController(Node):
    def __init__(
        self,
        limits: Mapping[str, Tuple[float, float, float]],
        velocity_scale: float,
        gripper_effort: float,
    ) -> None:
        super().__init__("dual_arm_grasp_real")

        if not 0.0 < velocity_scale <= 1.0:
            raise ValueError("velocity_scale 必须位于 (0, 1]")

        self._limits = dict(limits)
        self._velocity_scale = float(velocity_scale)
        self._gripper_effort = float(gripper_effort)
        self._positions: Dict[str, float] = {}
        self._active_goal_handles = []

        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_state,
            20,
        )

        self._lift = ActionClient(
            self,
            FollowJointTrajectory,
            "/lift_controller/follow_joint_trajectory",
        )
        self._waist = ActionClient(
            self,
            FollowJointTrajectory,
            "/waist_controller/follow_joint_trajectory",
        )
        self._left_root = ActionClient(
            self,
            FollowJointTrajectory,
            "/left_arm_root_controller/follow_joint_trajectory",
        )
        self._right_root = ActionClient(
            self,
            FollowJointTrajectory,
            "/right_arm_root_controller/follow_joint_trajectory",
        )
        self._left_arm = ActionClient(
            self,
            FollowJointTrajectory,
            "/left_arm_controller/follow_joint_trajectory",
        )
        self._right_arm = ActionClient(
            self,
            FollowJointTrajectory,
            "/right_arm_controller/follow_joint_trajectory",
        )
        self._left_gripper = ActionClient(
            self,
            GripperCommand,
            "/left_gripper_controller/gripper_cmd",
        )
        self._right_gripper = ActionClient(
            self,
            GripperCommand,
            "/right_gripper_controller/gripper_cmd",
        )

        self._controller_groups = {
            "lift": (
                self._lift,
                LIFT_JOINTS,
            ),
            "waist": (
                self._waist,
                WAIST_JOINTS,
            ),
            "left_root": (
                self._left_root,
                LEFT_ROOT_JOINTS,
            ),
            "right_root": (
                self._right_root,
                RIGHT_ROOT_JOINTS,
            ),
            "left_arm": (
                self._left_arm,
                LEFT_ARM_JOINTS,
            ),
            "right_arm": (
                self._right_arm,
                RIGHT_ARM_JOINTS,
            ),
        }

    def _on_state(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            if name in ALL_MOTION_JOINTS:
                self._positions[name] = float(position)

    def wait_for_servers(self, timeout: float = 8.0) -> bool:
        servers = [
            ("升降", self._lift),
            ("腰部", self._waist),
            ("左臂根部", self._left_root),
            ("右臂根部", self._right_root),
            ("左臂", self._left_arm),
            ("右臂", self._right_arm),
            ("左夹爪", self._left_gripper),
            ("右夹爪", self._right_gripper),
        ]

        all_ok = True
        for label, client in servers:
            if client.wait_for_server(timeout_sec=timeout):
                self.get_logger().info(f"✅ {label}控制器就绪")
            else:
                self.get_logger().error(f"❌ {label}控制器未就绪")
                all_ok = False

        return all_ok

    def wait_for_state(self, timeout: float = 8.0) -> bool:
        deadline = time.monotonic() + timeout

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            missing = [
                name
                for name in ALL_MOTION_JOINTS
                if name not in self._positions
            ]
            if not missing:
                self.get_logger().info(
                    f"✅ 已同步 {len(ALL_MOTION_JOINTS)} 个运动关节"
                )
                return True

        missing = [
            name
            for name in ALL_MOTION_JOINTS
            if name not in self._positions
        ]
        self.get_logger().error(
            "❌ /joint_states 缺少关节:\n  "
            + "\n  ".join(missing)
        )
        return False

    def get_pos(self, name: str) -> float:
        if name not in self._positions:
            raise RuntimeError(f"缺少当前关节状态: {name}")
        return self._positions[name]

    def _wait_all(self, futures: Iterable, timeout: float) -> bool:
        futures = list(futures)
        deadline = time.monotonic() + timeout

        while rclpy.ok() and time.monotonic() < deadline:
            if all(future.done() for future in futures):
                return True
            rclpy.spin_once(self, timeout_sec=0.05)

        return all(future.done() for future in futures)

    def _build_goal(
        self,
        joint_names: Sequence[str],
        points: np.ndarray,
        duration: float,
    ) -> FollowJointTrajectory.Goal:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(joint_names)

        steps = points.shape[0]

        for index, row in enumerate(points, start=1):
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in row]

            timestamp = duration * index / steps
            seconds, nanoseconds = to_duration(timestamp)
            point.time_from_start.sec = seconds
            point.time_from_start.nanosec = nanoseconds

            goal.trajectory.points.append(point)

        return goal

    def _minimum_duration(
        self,
        start: Mapping[str, float],
        targets: Mapping[str, float],
        requested: float,
    ) -> float:
        required = 0.0

        for name, target in targets.items():
            _, _, max_velocity = self._limits[name]
            allowed_velocity = max_velocity * self._velocity_scale
            delta = abs(float(target) - float(start[name]))

            if allowed_velocity > 0.0:
                required = max(
                    required,
                    delta / allowed_velocity,
                )

        # 给 smoothstep 峰值速度留余量。
        required *= 1.6
        return max(float(requested), required, 0.2)

    def animate(
        self,
        targets: Mapping[str, float],
        duration: float = 2.0,
        steps: int = 60,
    ) -> bool:
        """
        保留原 animate() 逻辑：
        - 从当前状态开始
        - smoothstep 插值
        - 未指定关节保持当前位置
        - 多个控制器同时执行
        """
        if not targets:
            return True
        if steps < 2:
            raise ValueError("steps 必须至少为 2")

        for name, value in targets.items():
            if name not in self._limits:
                raise RuntimeError(f"没有关节限位信息: {name}")

            lower, upper, _ = self._limits[name]
            value = float(value)

            if value < lower or value > upper:
                raise RuntimeError(
                    f"{name} 目标越界: value={value:+.6f}, "
                    f"limit=[{lower:+.6f}, {upper:+.6f}]"
                )

        affected = {}

        for group_name, (client, joint_names) in self._controller_groups.items():
            if not any(name in targets for name in joint_names):
                continue

            start = {
                name: self.get_pos(name)
                for name in joint_names
            }
            end = {
                name: float(targets.get(name, start[name]))
                for name in joint_names
            }

            affected[group_name] = (
                client,
                joint_names,
                start,
                end,
            )

        if not affected:
            raise RuntimeError(
                "targets 中没有可由实机控制器执行的关节"
            )

        combined_start = {}
        combined_targets = {}

        for _, joint_names, start, end in affected.values():
            combined_start.update(start)
            combined_targets.update(end)

        safe_duration = self._minimum_duration(
            combined_start,
            combined_targets,
            duration,
        )

        goals = {}
        for group_name, (
            client,
            joint_names,
            start,
            end,
        ) in affected.items():
            rows = []

            for step in range(1, steps + 1):
                ratio = step / steps
                weight = ratio * ratio * (3.0 - 2.0 * ratio)
                rows.append([
                    start[name]
                    + (end[name] - start[name]) * weight
                    for name in joint_names
                ])

            points = np.asarray(rows, dtype=float)
            goal = self._build_goal(
                joint_names,
                points,
                safe_duration,
            )
            goals[group_name] = client.send_goal_async(goal)

        self.get_logger().info(
            f"发送平滑轨迹: targets={dict(targets)}, "
            f"duration={safe_duration:.2f}s, steps={steps}"
        )

        if not self._wait_all(goals.values(), timeout=12.0):
            self.get_logger().error("❌ 等待控制器接受目标超时")
            return False

        handles = {}
        for group_name, future in goals.items():
            handle = future.result()

            if handle is None or not handle.accepted:
                self.get_logger().error(
                    f"❌ {group_name}: 轨迹目标被拒绝"
                )
            else:
                handles[group_name] = handle

        if len(handles) != len(goals):
            for handle in handles.values():
                try:
                    handle.cancel_goal_async()
                except Exception:
                    pass
            return False

        self._active_goal_handles = list(handles.values())

        result_futures = {
            group_name: handle.get_result_async()
            for group_name, handle in handles.items()
        }

        if not self._wait_all(
            result_futures.values(),
            timeout=safe_duration + 15.0,
        ):
            self.get_logger().error(
                "❌ 等待轨迹执行完成超时，正在取消"
            )
            self.cancel_active_goals()
            return False

        all_ok = True

        for group_name, future in result_futures.items():
            wrapped = future.result()

            if wrapped is None:
                self.get_logger().error(
                    f"❌ {group_name}: 执行结果为空"
                )
                all_ok = False
                continue

            result = wrapped.result
            success = (
                wrapped.status == GoalStatus.STATUS_SUCCEEDED
                and result.error_code
                == FollowJointTrajectory.Result.SUCCESSFUL
            )

            if success:
                self.get_logger().info(
                    f"✅ {group_name}: 轨迹完成"
                )
            else:
                self.get_logger().error(
                    f"❌ {group_name}: "
                    f"status={wrapped.status}, "
                    f"error_code={result.error_code}, "
                    f"error_string={result.error_string!r}"
                )
                all_ok = False

        self._active_goal_handles = []

        if all_ok:
            # 处理最新状态回调，不直接伪造内部状态。
            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=0.05)

        return all_ok

    def command_grippers(
        self,
        position: float,
    ) -> bool:
        position = float(position)

        if not 0.0 <= position <= 0.025:
            raise RuntimeError(
                f"夹爪位置越界: {position}; "
                "允许范围为 [0.0, 0.025]"
            )

        send_futures = {}

        for label, client in (
            ("左夹爪", self._left_gripper),
            ("右夹爪", self._right_gripper),
        ):
            goal = GripperCommand.Goal()
            goal.command.position = position
            goal.command.max_effort = self._gripper_effort
            send_futures[label] = client.send_goal_async(goal)

        if not self._wait_all(
            send_futures.values(),
            timeout=10.0,
        ):
            self.get_logger().error("❌ 夹爪目标发送超时")
            return False

        handles = {}
        for label, future in send_futures.items():
            handle = future.result()

            if handle is None or not handle.accepted:
                self.get_logger().error(
                    f"❌ {label}: 目标被拒绝"
                )
            else:
                handles[label] = handle

        if len(handles) != 2:
            return False

        result_futures = {
            label: handle.get_result_async()
            for label, handle in handles.items()
        }

        if not self._wait_all(
            result_futures.values(),
            timeout=12.0,
        ):
            self.get_logger().error("❌ 夹爪执行超时")
            return False

        all_ok = True

        for label, future in result_futures.items():
            wrapped = future.result()
            success = (
                wrapped is not None
                and wrapped.status == GoalStatus.STATUS_SUCCEEDED
            )

            if success:
                result = wrapped.result
                self.get_logger().info(
                    f"✅ {label}: position={result.position:.5f}, "
                    f"reached_goal={result.reached_goal}, "
                    f"stalled={result.stalled}"
                )
            else:
                self.get_logger().error(
                    f"❌ {label}: 执行失败"
                )
                all_ok = False

        return all_ok

    def cancel_active_goals(self) -> None:
        for handle in self._active_goal_handles:
            try:
                handle.cancel_goal_async()
            except Exception:
                pass
        self._active_goal_handles = []

    # ========================================================================
    # 6 阶段 XY/Z 独立抓取流程
    # ========================================================================

    def execute(
        self,
        left_q: np.ndarray,
        right_q: np.ndarray,
        body: Dict[str, float],
        close_gripper: bool,
        shoulder_clearance: float,
    ) -> None:
        """
        PRE. 机械臂运动前，lift 先独立升到最高 0.25m；
        1. 夹爪张开；
        2. 肩部在最终 XY 位姿基础上额外抬举 shoulder_clearance；
        3. 其他机械臂关节运动到 XY 模板 IK 结果；
        4. lift 根据 XY 解对应的夹爪 Z 独立下降/调整；
        5. 肩部从避障位逐渐下降到最终 XY 位姿；
        6. 夹爪闭合。
        """
        log = self.get_logger().info

        final_left_shoulder = float(left_q[1])
        final_right_shoulder = float(right_q[1])

        # shoulder_clearance 按相对最终抓取角处理，而非固定绝对角。
        left_clearance = (
            final_left_shoulder + abs(float(shoulder_clearance))
        )
        right_clearance = (
            final_right_shoulder - abs(float(shoulder_clearance))
        )

        for name, value in (
            ("arm_left1_arm_left2_joint", left_clearance),
            ("arm_right1_arm_right2_joint", right_clearance),
        ):
            lower, upper, _ = self._limits[name]
            if value < lower or value > upper:
                raise RuntimeError(
                    f"{name} 避障位越界: "
                    f"value={value:+.4f}, "
                    f"limit=[{lower:+.4f}, {upper:+.4f}]"
                )

        # --------------------------------------------------------------
        # 预备避障阶段：在任何机械臂关节动作之前，先将 lift 升到最高 0.25m。
        # 后续原 6 阶段逻辑保持不变；原 [4/6] 仍会把 lift 调整到最终抓取高度。
        # --------------------------------------------------------------
        prelift_target_m = 0.25
        lift_lower, lift_upper, _ = self._limits["base_elevator_joint"]
        if (
            prelift_target_m < lift_lower
            or prelift_target_m > lift_upper
        ):
            raise RuntimeError(
                "预抬升 lift 目标越界: "
                f"target={prelift_target_m:+.4f}m, "
                f"limit=[{lift_lower:+.4f}, {lift_upper:+.4f}]"
            )

        log(
            "▶ [PRE] 机械臂动作前先将 lift 升至最高避障位: "
            f"{prelift_target_m:+.4f}m"
        )
        if not self.animate({
            "base_elevator_joint": prelift_target_m,
        }, duration=2.0, steps=80):
            raise RuntimeError("预备 lift 抬升到 0.25m 失败")

        log("▶ [1/6] 夹爪张开")
        if not self.command_grippers(0.025):
            raise RuntimeError("夹爪张开失败")
        time.sleep(0.2)

        log(
            "▶ [2/6] 肩部在最终位基础上抬举 "
            f"{abs(float(shoulder_clearance)):.3f}rad: "
            f"左 {final_left_shoulder:+.3f}"
            f" → {left_clearance:+.3f}, "
            f"右 {final_right_shoulder:+.3f}"
            f" → {right_clearance:+.3f}"
        )
        if not self.animate({
            "arm_left1_arm_left2_joint": left_clearance,
            "arm_right1_arm_right2_joint": right_clearance,
        }, duration=1.2, steps=60):
            raise RuntimeError("肩部避障抬举失败")

        log(
            "▶ [3/6] 其他关节运动到 XY 模板 IK 位姿，"
            "肩部保持避障角"
        )
        arm_targets = {
            "elevator_pitch_joint":
                float(body["elevator_pitch_joint"]),
            "pitch_arm_left1_joint": float(left_q[0]),
            "pitch_arm_right1_joint": float(right_q[0]),
        }

        # q[1] 是肩部抬举关节，暂不下落。
        for name, value in zip(
            LEFT_ARM_JOINTS[1:],
            left_q[2:],
        ):
            arm_targets[name] = float(value)

        for name, value in zip(
            RIGHT_ARM_JOINTS[1:],
            right_q[2:],
        ):
            arm_targets[name] = float(value)

        if not self.animate(
            arm_targets,
            duration=3.0,
            steps=100,
        ):
            raise RuntimeError("XY 机械臂位姿运动失败")

        log(
            "▶ [4/6] 根据 XY 解对应的夹爪 Z 独立调整 lift: "
            f"{body['base_elevator_joint']:+.4f}m"
        )
        if not self.animate({
            "base_elevator_joint":
                float(body["base_elevator_joint"]),
        }, duration=2.0, steps=80):
            raise RuntimeError("独立 lift/Z 调整失败")

        log(
            "▶ [5/6] 肩部逐渐下降到最终 XY 抓取位: "
            f"左={final_left_shoulder:+.3f}rad, "
            f"右={final_right_shoulder:+.3f}rad"
        )
        if not self.animate({
            "arm_left1_arm_left2_joint":
                final_left_shoulder,
            "arm_right1_arm_right2_joint":
                final_right_shoulder,
        }, duration=2.0, steps=100):
            raise RuntimeError("肩部下降到最终抓取位失败")

        if close_gripper:
            log("▶ [6/6] 夹爪闭合")
            if not self.command_grippers(0.0):
                raise RuntimeError("夹爪闭合失败")
        else:
            log("▶ [6/6] 首次测试模式：保持夹爪张开")

        log("✅ XY/Z 独立抓取流程完成")

    @staticmethod
    def isaac_to_ros(x, y, z):
        # Isaac X 左负右正；ROS Y 左正右负。
        return (y, -x, z)

    def run(
        self,
        left_isaac,
        right_isaac,
        execute: bool,
        close_gripper: bool,
        left_template_q: np.ndarray,
        right_template_q: np.ndarray,
        max_xy_error_m: float,
        max_z_error_m: float,
        max_shared_lift_mismatch_m: float,
        max_orientation_error_rad: float,
        max_forearm_tilt_rad: float,
        max_gripper_vertical_error_rad: float,
        max_template_rmse_rad: float,
        lateral_weight: float,
        max_lateral_error_m: float,
        max_lateral_joint_correction_rad: float,
        shoulder_clearance: float,
    ) -> None:
        left_ros = self.isaac_to_ros(*left_isaac)
        right_ros = self.isaac_to_ros(*right_isaac)

        current_left_q = np.array(
            [
                self.get_pos("pitch_arm_left1_joint"),
                *[
                    self.get_pos(name)
                    for name in LEFT_ARM_JOINTS
                ],
            ],
            dtype=np.float64,
        )
        current_right_q = np.array(
            [
                self.get_pos("pitch_arm_right1_joint"),
                *[
                    self.get_pos(name)
                    for name in RIGHT_ARM_JOINTS
                ],
            ],
            dtype=np.float64,
        )
        current_lift = self.get_pos("base_elevator_joint")
        current_waist = self.get_pos("elevator_pitch_joint")

        log = self.get_logger().info
        log("=" * 66)
        log("robot_a 双臂抓取 [新模板 q2=±10° / 精确 q_left=-q_right 镜像 + waist=0 + 独立 lift/Z]")
        log(
            f"[Isaac] 左: "
            f"({left_isaac[0]:.3f},"
            f"{left_isaac[1]:.3f},"
            f"{left_isaac[2]:.3f})"
        )
        log(
            f"[Isaac] 右: "
            f"({right_isaac[0]:.3f},"
            f"{right_isaac[1]:.3f},"
            f"{right_isaac[2]:.3f})"
        )
        log(
            f"当前 lift={current_lift:+.4f}m, "
            f"waist={math.degrees(current_waist):.2f}°"
        )
        log(
            "标准模板左(deg): "
            f"{np.round(np.degrees(left_template_q), 2)}"
        )
        log(
            "标准模板右(deg): "
            f"{np.round(np.degrees(right_template_q), 2)}"
        )
        log(
            f"箱体夹紧方向参数: lateral_weight={lateral_weight:.2f}, "
            f"允许误差={max_lateral_error_m * 100:.2f}cm, "
            f"q2最大精修="
            f"{math.degrees(max_lateral_joint_correction_rad):.1f}°"
        )
        log(
            "左 grasp TCP(hand): "
            f"{np.round(LEFT_GRASP_TCP_OFFSET_HAND, 6)}m；"
            "右 grasp TCP(hand): "
            f"{np.round(RIGHT_GRASP_TCP_OFFSET_HAND, 6)}m；"
            f"两侧均向世界下方偏 {GRASP_TCP_DOWN_OFFSET_M*1000.0:.1f}mm"
        )
        log("=" * 66)

        # --------------------------------------------------------------
        # 精确镜像约束
        # --------------------------------------------------------------
        # 最终 TCP 目标必须关于 base_link 的 Y=0 平面对称：
        #   left_ros  = [X, +Y, Z]
        #   right_ros = [X, -Y, Z]
        mirror_target_tol_m = 0.002
        if (
            abs(float(left_ros[0]) - float(right_ros[0]))
            > mirror_target_tol_m
            or abs(float(left_ros[1]) + float(right_ros[1]))
            > mirror_target_tol_m
            or abs(float(left_ros[2]) - float(right_ros[2]))
            > mirror_target_tol_m
        ):
            raise RuntimeError(
                "精确镜像模式要求左右目标关于 base_link Y=0 对称："
                f"left_ros={np.round(left_ros,6)}, "
                f"right_ros={np.round(right_ros,6)}"
            )

        # waist 非零会破坏机器人本体左右镜像关系，因此本版本固定 waist=0。
        waist_velocity = float(
            self._limits["elevator_pitch_joint"][2]
        )
        mirror_limits = dict(self._limits)
        mirror_limits["elevator_pitch_joint"] = (
            0.0,
            0.0,
            waist_velocity,
        )
        mirror_waist = 0.0

        log(">>> 第一步：镜像约束 XY IK（waist=0，随后强制 q_left=-q_right）...")
        raw_solution = dual_ik_xy_then_lift(
            left_ros,
            right_ros,
            mirror_limits,
            left_template_q,
            right_template_q,
            current_left_q,
            current_right_q,
            current_lift,
            mirror_waist,
            max_xy_error_m,
            max_z_error_m,
            max_shared_lift_mismatch_m,
            max_orientation_error_rad,
            max_forearm_tilt_rad,
            max_gripper_vertical_error_rad,
            max_template_rmse_rad,
            lateral_weight,
            max_lateral_error_m,
            max_lateral_joint_correction_rad,
        )

        if raw_solution is None:
            raise RuntimeError(
                "镜像约束 XY 模板 IK 或独立 lift/Z 计算无解"
            )

        # 原 V2 算法对左右臂分别优化，即使目标对称也只把镜像姿态当 seed。
        # 本版本从左右独立解中构造两个“严格镜像候选”，最终只接受：
        #     left_q == -right_q
        # 这样真实发送到控制器的每一个左右关节均严格镜像。
        def _hard_pose_ok(metrics):
            return (
                float(metrics["xy_error_m"]) <= max_xy_error_m
                and float(metrics["lateral_error_m"])
                <= max_lateral_error_m
                and float(metrics["orientation_error_rad"])
                <= max_orientation_error_rad
                and float(metrics["forearm_tilt_rad"])
                <= max_forearm_tilt_rad
                and float(metrics["gripper_vertical_error_rad"])
                <= max_gripper_vertical_error_rad
            )

        def _build_exact_mirror_candidate(
            right_q_source: np.ndarray,
        ):
            right_q = np.asarray(
                right_q_source,
                dtype=np.float64,
            ).copy()
            left_q = -right_q

            # 先用真实 URDF 限位验证严格镜像后的双方关节。
            temp_solution = {
                "left": left_q,
                "right": right_q,
                "body": {
                    "base_elevator_joint": float(current_lift),
                    "elevator_pitch_joint": 0.0,
                },
            }
            try:
                validate_solution(
                    temp_solution,
                    self._limits,
                )
            except Exception:
                return None

            left_xy_metrics = grasp_pose_metrics(
                left_q,
                left_ros,
                left_template_q,
                current_lift,
                0.0,
                LEFT_ARM_PARAMS,
            )
            right_xy_metrics = grasp_pose_metrics(
                right_q,
                right_ros,
                right_template_q,
                current_lift,
                0.0,
                RIGHT_ARM_PARAMS,
            )

            if (
                not _hard_pose_ok(left_xy_metrics)
                or not _hard_pose_ok(right_xy_metrics)
            ):
                return None

            left_z_at_xy = float(
                left_xy_metrics["z_m"]
            )
            right_z_at_xy = float(
                right_xy_metrics["z_m"]
            )

            left_required_lift = (
                current_lift
                + float(left_ros[2])
                - left_z_at_xy
            )
            right_required_lift = (
                current_lift
                + float(right_ros[2])
                - right_z_at_xy
            )

            lift_mismatch = abs(
                left_required_lift
                - right_required_lift
            )
            if (
                lift_mismatch
                > max_shared_lift_mismatch_m
            ):
                return None

            shared_lift = 0.5 * (
                left_required_lift
                + right_required_lift
            )

            lift_lower, lift_upper, _ = (
                self._limits[
                    "base_elevator_joint"
                ]
            )
            if (
                shared_lift < lift_lower
                or shared_lift > lift_upper
            ):
                return None

            left_final_metrics = grasp_pose_metrics(
                left_q,
                left_ros,
                left_template_q,
                shared_lift,
                0.0,
                LEFT_ARM_PARAMS,
            )
            right_final_metrics = grasp_pose_metrics(
                right_q,
                right_ros,
                right_template_q,
                shared_lift,
                0.0,
                RIGHT_ARM_PARAMS,
            )

            if (
                not _hard_pose_ok(
                    left_final_metrics
                )
                or not _hard_pose_ok(
                    right_final_metrics
                )
                or float(
                    left_final_metrics[
                        "z_error_m"
                    ]
                ) > max_z_error_m
                or float(
                    right_final_metrics[
                        "z_error_m"
                    ]
                ) > max_z_error_m
            ):
                return None

            # 分数只用于在“右臂独立解”和“左臂独立解镜像”
            # 两个严格镜像候选之间选择更贴合目标的一组。
            score = (
                float(
                    left_final_metrics[
                        "xy_error_m"
                    ]
                )
                + float(
                    right_final_metrics[
                        "xy_error_m"
                    ]
                )
                + 2.0
                * (
                    float(
                        left_final_metrics[
                            "lateral_error_m"
                        ]
                    )
                    + float(
                        right_final_metrics[
                            "lateral_error_m"
                        ]
                    )
                )
                + 0.2
                * (
                    float(
                        left_final_metrics[
                            "orientation_error_rad"
                        ]
                    )
                    + float(
                        right_final_metrics[
                            "orientation_error_rad"
                        ]
                    )
                )
            )

            return {
                "score": score,
                "left": left_q,
                "right": right_q,
                "body": {
                    "base_elevator_joint":
                        float(shared_lift),
                    "elevator_pitch_joint": 0.0,
                },
                "xy_stage": {
                    "reference_lift":
                        float(current_lift),
                    "left_gripper_z":
                        left_z_at_xy,
                    "right_gripper_z":
                        right_z_at_xy,
                    "left_required_lift":
                        float(left_required_lift),
                    "right_required_lift":
                        float(right_required_lift),
                    "shared_lift_mismatch":
                        float(lift_mismatch),
                },
                "metrics": {
                    "left_xy":
                        left_xy_metrics,
                    "right_xy":
                        right_xy_metrics,
                    "left_final":
                        left_final_metrics,
                    "right_final":
                        right_final_metrics,
                },
            }

        mirror_candidates = []

        # 候选 A：保留右臂独立 IK 解，左臂完全取负。
        candidate = _build_exact_mirror_candidate(
            raw_solution["right"]
        )
        if candidate is not None:
            mirror_candidates.append(candidate)

        # 候选 B：保留左臂独立 IK 解，把它镜像成右臂。
        candidate = _build_exact_mirror_candidate(
            -np.asarray(
                raw_solution["left"],
                dtype=np.float64,
            )
        )
        if candidate is not None:
            mirror_candidates.append(candidate)

        # 若直接镜像独立解还不够精确，再进行一次“单变量组”联合优化：
        # 只优化 right_q，left_q 永远定义为 -right_q。
        #
        # 优化任务只处理左右 TCP 的 XY；Z 后续仍由共享 lift 独立完成。
        # 这样从求解阶段开始就不会破坏关节镜像。
        left_lower, left_upper = arm_joint_limit_arrays(
            self._limits,
            "left",
        )
        right_lower, right_upper = arm_joint_limit_arrays(
            self._limits,
            "right",
        )

        mirror_lower = np.maximum(
            right_lower,
            -left_upper,
        )
        mirror_upper = np.minimum(
            right_upper,
            -left_lower,
        )

        if np.any(
            mirror_lower > mirror_upper
        ):
            raise RuntimeError(
                "左右URDF限位不存在共同镜像可行域"
            )

        def _mirror_xy_optimize(seed_q):
            q = np.clip(
                np.asarray(
                    seed_q,
                    dtype=np.float64,
                ).copy(),
                mirror_lower,
                mirror_upper,
            )

            delta = 1e-5
            damping = 2e-4
            best_q = q.copy()
            best_norm = float("inf")

            def residual(qr):
                ql = -qr
                pr = grasp_tcp_fk(
                    qr,
                    current_lift,
                    0.0,
                    RIGHT_ARM_PARAMS,
                )[:2, 3]
                pl = grasp_tcp_fk(
                    ql,
                    current_lift,
                    0.0,
                    LEFT_ARM_PARAMS,
                )[:2, 3]

                # ROS X 是前后；ROS Y 是左右夹紧方向。
                # 横向误差沿用 lateral_weight 加权。
                return np.array(
                    [
                        float(
                            pr[0]
                            - right_ros[0]
                        ),
                        float(lateral_weight)
                        * float(
                            pr[1]
                            - right_ros[1]
                        ),
                        float(
                            pl[0]
                            - left_ros[0]
                        ),
                        float(lateral_weight)
                        * float(
                            pl[1]
                            - left_ros[1]
                        ),
                    ],
                    dtype=np.float64,
                )

            for _ in range(220):
                r = residual(q)
                r_norm = float(
                    np.linalg.norm(r)
                )

                if r_norm < best_norm:
                    best_norm = r_norm
                    best_q = q.copy()

                # 约 1mm 级联合残差时即可交给后面的完整安全指标判断。
                if r_norm < 0.0015:
                    break

                J = np.zeros(
                    (4, 7),
                    dtype=np.float64,
                )

                for j in range(7):
                    trial = min(
                        float(mirror_upper[j]),
                        float(q[j] + delta),
                    )
                    if trial == q[j]:
                        trial = max(
                            float(mirror_lower[j]),
                            float(q[j] - delta),
                        )

                    actual_delta = (
                        trial - q[j]
                    )
                    if abs(
                        actual_delta
                    ) < 1e-12:
                        continue

                    q_trial = q.copy()
                    q_trial[j] = trial
                    J[:, j] = (
                        residual(q_trial)
                        - r
                    ) / actual_delta

                # DLS: dq = -J^T (J J^T + λI)^-1 r
                A = (
                    J @ J.T
                    + damping
                    * np.eye(
                        4,
                        dtype=np.float64,
                    )
                )
                try:
                    dq = -J.T @ np.linalg.solve(
                        A,
                        r,
                    )
                except np.linalg.LinAlgError:
                    dq = -np.linalg.pinv(J) @ r

                # 轻微拉回右臂标准模板，避免纯位置任务产生奇异姿态。
                template_pull = (
                    right_template_q
                    - q
                )
                dq += 0.015 * template_pull

                max_step = float(
                    np.max(
                        np.abs(dq)
                    )
                )
                if max_step > 0.12:
                    dq *= 0.12 / max_step

                q = np.clip(
                    q + 0.75 * dq,
                    mirror_lower,
                    mirror_upper,
                )

            return best_q

        optimizer_seeds = [
            np.asarray(
                raw_solution["right"],
                dtype=np.float64,
            ),
            -np.asarray(
                raw_solution["left"],
                dtype=np.float64,
            ),
            np.asarray(
                right_template_q,
                dtype=np.float64,
            ),
            0.5
            * (
                np.asarray(
                    raw_solution["right"],
                    dtype=np.float64,
                )
                - np.asarray(
                    raw_solution["left"],
                    dtype=np.float64,
                )
            ),
        ]

        for seed in optimizer_seeds:
            optimized_right_q = (
                _mirror_xy_optimize(seed)
            )
            candidate = (
                _build_exact_mirror_candidate(
                    optimized_right_q
                )
            )
            if candidate is not None:
                mirror_candidates.append(
                    candidate
                )

        if not mirror_candidates:
            raw_mirror_error = float(
                np.max(
                    np.abs(
                        np.asarray(
                            raw_solution["left"],
                            dtype=np.float64,
                        )
                        + np.asarray(
                            raw_solution["right"],
                            dtype=np.float64,
                        )
                    )
                )
            )
            raise RuntimeError(
                "独立 IK 有解，但强制 q_left=-q_right 后无法同时满足"
                "左右 TCP/URDF 约束；拒绝执行。"
                f" 独立解镜像偏差={raw_mirror_error:.6f}rad"
            )

        mirror_candidates.sort(
            key=lambda item: float(
                item["score"]
            )
        )
        solution = mirror_candidates[0]
        solution.pop("score", None)

        mirror_error = float(
            np.max(
                np.abs(
                    np.asarray(
                        solution["left"],
                        dtype=np.float64,
                    )
                    + np.asarray(
                        solution["right"],
                        dtype=np.float64,
                    )
                )
            )
        )
        if mirror_error > 1e-9:
            raise RuntimeError(
                "内部镜像断言失败："
                f"max|q_left+q_right|="
                f"{mirror_error:.12f}rad"
            )

        print(
            "  ✅ 精确镜像约束通过: "
            f"max|q_left+q_right|="
            f"{mirror_error:.12f}rad, "
            "waist=0.000000rad"
        )

        validate_solution(
            solution,
            self._limits,
        )
        verify_grasp_solution(
            solution,
            left_ros,
            right_ros,
            left_template_q,
            right_template_q,
            max_xy_error_m,
            max_z_error_m,
            max_shared_lift_mismatch_m,
            max_orientation_error_rad,
            max_forearm_tilt_rad,
            max_gripper_vertical_error_rad,
            max_template_rmse_rad,
            max_lateral_error_m,
        )

        xy_stage = solution["xy_stage"]

        print("\nXY/Z 独立定位与安全检查通过：")
        print(
            "  left_q(rad) =",
            np.round(solution["left"], 6),
        )
        print(
            "  left_q(deg) =",
            np.round(np.degrees(solution["left"]), 2),
        )
        print(
            "  right_q(rad)=",
            np.round(solution["right"], 6),
        )
        print(
            "  right_q(deg)=",
            np.round(np.degrees(solution["right"]), 2),
        )
        print(
            "  XY 解时 grasp TCP Z =",
            {
                "left": round(
                    xy_stage["left_gripper_z"],
                    6,
                ),
                "right": round(
                    xy_stage["right_gripper_z"],
                    6,
                ),
            },
        )
        print("  body =", solution["body"])

        if not execute:
            print(
                "\n✅ 仅完成 XY IK、TCP Z 计算、"
                "独立 lift 和姿态限位检查，机器人没有运动。\n"
                "确认路径无碰撞后，添加 --execute 实机运行。"
            )
            return solution

        self.get_logger().warning(
            "即将执行真实机器人 XY/Z 独立 6 阶段抓取流程，"
            "请保持实体急停可触达"
        )

        self.execute(
            solution["left"],
            solution["right"],
            solution["body"],
            close_gripper=close_gripper,
            shoulder_clearance=shoulder_clearance,
        )

        return solution


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "robot_a 双臂抓取 —— "
            "XY 模板 IK + q2 横向夹紧精修 + 独立 lift/Z + 实机 Action"
        ),
        epilog=(
            "示例: /usr/bin/python3.10 "
            "dual_arm_fixed_grasp_xy_lift_squeeze_real.py "
            "-l -0.30 0.65 0.95 "
            "-r 0.30 0.65 0.95"
        ),
    )

    parser.add_argument(
        "-l",
        "--left",
        nargs=3,
        type=float,
        default=[-0.30, 0.65, 0.95],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "-r",
        "--right",
        nargs=3,
        type=float,
        default=[0.30, 0.65, 0.95],
        metavar=("X", "Y", "Z"),
    )

    parser.add_argument(
        "--left-template-deg",
        nargs=7,
        type=float,
        default=DEFAULT_LEFT_TEMPLATE_DEG.tolist(),
        metavar=("Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
    )
    parser.add_argument(
        "--right-template-deg",
        nargs=7,
        type=float,
        default=DEFAULT_RIGHT_TEMPLATE_DEG.tolist(),
        metavar=("Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
    )

    parser.add_argument(
        "--urdf",
        default=(
            "/home/niic/project/20260623/20260116/test_robot_A/"
            "src/robot_A/robot_a_description/urdf/robot_a.urdf"
        ),
    )
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--max-xy-error-cm",
        type=float,
        default=1.0,
        help="机械臂 XY 平面最大误差",
    )
    parser.add_argument(
        "--max-z-error-cm",
        type=float,
        default=1.0,
        help="lift 调整后的最大 Z 误差",
    )
    parser.add_argument(
        "--lateral-weight",
        type=float,
        default=4.0,
        help=(
            "箱体左右夹紧方向（Isaac X）的 IK 权重；"
            "越大则 X 输入变化越敏感"
        ),
    )
    parser.add_argument(
        "--max-lateral-error-cm",
        type=float,
        default=0.5,
        help=(
            "使用 arm_*2_arm_*3_joint 精修后的"
            "箱体夹紧方向最大误差"
        ),
    )
    parser.add_argument(
        "--max-lateral-joint-correction-deg",
        type=float,
        default=20.0,
        help=(
            "arm_*2_arm_*3_joint 相对主 IK 解"
            "允许的最大精修角度"
        ),
    )
    parser.add_argument(
        "--max-shared-lift-mismatch-cm",
        type=float,
        default=1.0,
        help="左右臂分别计算的 lift 目标最大允许差值",
    )
    parser.add_argument(
        "--max-orientation-error-deg",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--max-forearm-tilt-deg",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--max-gripper-vertical-error-deg",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--max-template-rmse-deg",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--shoulder-clearance",
        type=float,
        default=DEFAULT_SHOULDER_CLEARANCE,
        help=(
            "肩部相对最终抓取角额外抬举量，"
            "默认 0.07rad"
        ),
    )
    parser.add_argument(
        "--gripper-effort",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际控制机器人；不添加时只进行检查",
    )
    parser.add_argument(
        "--close-gripper",
        action="store_true",
        help="最终阶段闭合夹爪；首次测试不要添加",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.lateral_weight <= 0.0:
        raise SystemExit("❌ --lateral-weight 必须大于 0")
    if args.max_lateral_error_cm <= 0.0:
        raise SystemExit("❌ --max-lateral-error-cm 必须大于 0")
    if args.max_lateral_joint_correction_deg <= 0.0:
        raise SystemExit(
            "❌ --max-lateral-joint-correction-deg 必须大于 0"
        )

    limits = load_urdf_limits(args.urdf)

    rclpy.init(args=sys.argv)
    controller = DualArmGraspRealController(
        limits=limits,
        velocity_scale=args.velocity_scale,
        gripper_effort=args.gripper_effort,
    )

    try:
        if not controller.wait_for_servers():
            raise RuntimeError("部分实机控制器未就绪")

        if not controller.wait_for_state():
            raise RuntimeError("无法获得完整实机关节状态")

        controller.run(
            left_isaac=args.left,
            right_isaac=args.right,
            execute=args.execute,
            close_gripper=args.close_gripper,
            left_template_q=np.radians(
                np.asarray(args.left_template_deg, dtype=float)
            ),
            right_template_q=np.radians(
                np.asarray(args.right_template_deg, dtype=float)
            ),
            max_xy_error_m=args.max_xy_error_cm / 100.0,
            max_z_error_m=args.max_z_error_cm / 100.0,
            max_shared_lift_mismatch_m=(
                args.max_shared_lift_mismatch_cm / 100.0
            ),
            max_orientation_error_rad=math.radians(
                args.max_orientation_error_deg
            ),
            max_forearm_tilt_rad=math.radians(
                args.max_forearm_tilt_deg
            ),
            max_gripper_vertical_error_rad=math.radians(
                args.max_gripper_vertical_error_deg
            ),
            max_template_rmse_rad=math.radians(
                args.max_template_rmse_deg
            ),
            lateral_weight=args.lateral_weight,
            max_lateral_error_m=(
                args.max_lateral_error_cm / 100.0
            ),
            max_lateral_joint_correction_rad=math.radians(
                args.max_lateral_joint_correction_deg
            ),
            shoulder_clearance=args.shoulder_clearance,
        )
        return 0

    except KeyboardInterrupt:
        controller.cancel_active_goals()
        print("\n⏹️ 用户中断，已请求取消活动轨迹")
        return 130

    except Exception as exc:
        if rclpy.ok():
            controller.get_logger().error(f"❌ {exc}")
        else:
            print(f"❌ {exc}", file=sys.stderr)
        return 1

    finally:
        try:
            controller.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
