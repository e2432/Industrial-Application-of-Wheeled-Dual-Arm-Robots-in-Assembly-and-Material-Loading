#!/usr/bin/env python3
"""
Robot A 右臂纯 IK 抓取 —— 真实机器人 Action 版
================================================

双臂镜像抓取逻辑：
  1. 左右夹爪张开
  2. 双肩镜像抬举：右 -0.3 rad，左 +0.3 rad
  3. 左右机械臂同步进入镜像预抓取姿态
  4. 左右机械臂同步进入最终镜像抓取姿态
  5. Lift 独立调整到目标 Z，并验证真实反馈
  6. 只闭合右夹爪；左夹爪始终保持张开

与原 Isaac/JointState 版本相比：
- 不再发布 /robot_a/joint_targets
- 使用真实机器人 ros2_control Action
- 左右机械臂始终按 q_left=-q_right 镜像同步运动
- 右夹爪用于抓取；左夹爪始终保持张开
- lift 使用多点平滑轨迹，并通过 /joint_states 验证真实运动
- 不控制轮子和头部
- 从 /joint_states 读取当前实机关节位置
- 从 URDF 读取真实位置/速度限位
- IK 使用真实右臂关节限位
- lift 使用当前真实 lift 作为基准，不假设起点为 0
- 默认 dry-run；只有 --execute 才会运动
- 正式闭合夹爪需额外添加 --close-gripper

坐标约定沿用上传的原始右臂脚本：
  ROS X = 输入 Y
  ROS Y = 输入 X
  ROS Z = 输入 Z

例如原脚本示例：
  -x -0.15 -y 0.79 -z 0.94

推荐：
  先 dry-run：
    /usr/bin/python3.10 ik_grasp_dual_mirror_real_v4.py \
      -x -0.15 -y 0.79 -z 0.94

  首次实机保持夹爪张开：
    /usr/bin/python3.10 ik_grasp_dual_mirror_real_v4.py \
      -x -0.15 -y 0.79 -z 0.94 \
      --velocity-scale 0.30 \
      --execute

  正式抓取：
    /usr/bin/python3.10 ik_grasp_dual_mirror_real_v4.py \
      -x -0.15 -y 0.79 -z 0.94 \
      --velocity-scale 0.30 \
      --execute \
      --close-gripper
"""

import argparse
import math
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# 右臂定义
# ============================================================================

LIFT_JOINT = "base_elevator_joint"
WAIST_JOINT = "elevator_pitch_joint"

RIGHT_ROOT_JOINT = "pitch_arm_right1_joint"

RIGHT_ARM_JOINTS = [
    "arm_right1_arm_right2_joint",
    "arm_right2_arm_right3_joint",
    "arm_right3_arm_right4_joint",
    "arm_right4_arm_right5_joint",
    "arm_right5_arm_right6_joint",
    "arm_right6_arm_right7_joint",
]

RIGHT_MOTION_JOINTS = [
    RIGHT_ROOT_JOINT,
    *RIGHT_ARM_JOINTS,
]


LEFT_ROOT_JOINT = "pitch_arm_left1_joint"

LEFT_ARM_JOINTS = [
    "arm_left1_arm_left2_joint",
    "arm_left2_arm_left3_joint",
    "arm_left3_arm_left4_joint",
    "arm_left4_arm_left5_joint",
    "arm_left5_arm_left6_joint",
    "arm_left6_arm_left7_joint",
]

LEFT_MOTION_JOINTS = [
    LEFT_ROOT_JOINT,
    *LEFT_ARM_JOINTS,
]


def mirror_right_q_to_left(right_q: Sequence[float]) -> np.ndarray:
    """Robot A 双臂镜像符号约定：q_left = -q_right。"""
    return -np.asarray(right_q, dtype=np.float64)

# 原脚本模板
TEMPLATE_RIGHT_ARM = np.array(
    [0.6807, -0.3, -0.5061, 0.8901, 0.1745, -0.3840, 0.1571],
    dtype=np.float64,
)

SHOULDER_LIFT = -0.3

GRIPPER_OPEN = 0.025
GRIPPER_CLOSE = 0.0

# 与之前 Robot A 实机限位一致，主要用于 --mock；
# 真机模式仍以 URDF 为准。
FALLBACK_RIGHT_LOWER = np.array(
    [-3.105, -3.14, -3.105, -2.007, -3.105, -2.0769, -3.105],
    dtype=np.float64,
)
FALLBACK_RIGHT_UPPER = np.array(
    [3.105, 0.0, 3.105, 2.007, 3.105, 2.0769, 3.105],
    dtype=np.float64,
)


# ============================================================================
# FK / IK —— 保留原程序运动学模型
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
    ], dtype=np.float64)


def transl_mat(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def rot_z_mat(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)


# 原始右臂 FK 参数
R_P = [
    (0.0, -0.363, -0.2428, np.pi, 0.0, -np.pi / 2, 0, 0, 1),
    (0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0, 0, 0, 1),
    (-0.344, 0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0, 0, 1),
    (0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0, 0, 1),
    (0.0, -0.233, 0.0, np.pi / 2, 0.0, 0.0, 0, 0, 1),
    (0.0, 0.0, 0.0, -np.pi / 2, 0.0, -np.pi / 2, 0, 0, 1),
    (0.0, -0.211, 0.002, -np.pi / 2, 0.0, np.pi, 0, 0, 1),
]


def _jt(ox, oy, oz, rr, rp, ry, ax, ay, az, angle):
    sign = -1 if az < 0 else 1
    return (
        transl_mat(ox, oy, oz)
        @ rpy_to_mat(rr, rp, ry)
        @ rot_z_mat(sign * angle)
    )


def arm_fk(q, lift, waist, params):
    """右臂 FK -> 4x4 变换矩阵。"""
    T = (
        transl_mat(0.146, 0, 0.8658)
        @ transl_mat(0, 0, lift)
        @ rpy_to_mat(-np.pi / 2, 0, 0)
        @ rot_z_mat(waist)
    )

    for i, (ox, oy, oz, rr, rp, ry, ax, ay, az) in enumerate(params):
        T = T @ _jt(
            ox, oy, oz,
            rr, rp, ry,
            ax, ay, az,
            q[i],
        )

    return T


def arm_ik(
    target_ros,
    lift,
    waist,
    params,
    seed,
    lower_limits,
    upper_limits,
    weight_orientation=0.0,
    max_iter=300,
    fixed_joints: Optional[Mapping[int, float]] = None,
):
    """
    原数值雅可比 IK 的实机安全版。

    - 仍使用位置 Jacobian + 伪逆；
    - 使用真实关节上下限，而不是统一 [-3.12, 3.12]；
    - 支持固定关节。

    本抓取流程在 XY 定位阶段必须保持肩部 q[1] = -0.3，
    所以该关节在 IK 中固定，保证实际发送的姿态与 IK 计算一致。
    """
    target = np.asarray(target_ros, dtype=np.float64)
    lower_limits = np.asarray(lower_limits, dtype=np.float64)
    upper_limits = np.asarray(upper_limits, dtype=np.float64)

    q = np.clip(
        np.asarray(seed, dtype=np.float64).copy(),
        lower_limits,
        upper_limits,
    )

    fixed_joints = dict(fixed_joints or {})
    for index, value in fixed_joints.items():
        q[index] = float(
            np.clip(value, lower_limits[index], upper_limits[index])
        )

    delta = 1e-5
    best_err = float("inf")
    best_q = q.copy()
    stuck = 0
    rng = np.random.default_rng(42)

    for _ in range(max_iter):
        for index, value in fixed_joints.items():
            q[index] = float(value)

        T = arm_fk(q, lift, waist, params)
        pos = T[:3, 3]
        pos_err = float(np.linalg.norm(target - pos))

        orient_err = (
            float(np.linalg.norm(q - seed)) * weight_orientation
            if weight_orientation > 0
            else 0.0
        )
        total_err = pos_err + orient_err

        if total_err < best_err:
            best_err = total_err
            best_q = q.copy()
            stuck = 0

        if pos_err < 1e-3:
            return q, pos_err

        stuck += 1
        if stuck > 25:
            perturb = rng.normal(0.0, 0.03, size=7)
            for index in fixed_joints:
                perturb[index] = 0.0

            q = np.clip(
                q + perturb,
                lower_limits,
                upper_limits,
            )
            for index, value in fixed_joints.items():
                q[index] = float(value)
            stuck = 0
            continue

        J = np.zeros((3, 7), dtype=np.float64)

        for jx in range(7):
            if jx in fixed_joints:
                continue

            sv = q[jx]
            trial = min(
                upper_limits[jx],
                sv + delta,
            )

            if trial == sv:
                trial = max(
                    lower_limits[jx],
                    sv - delta,
                )

            actual_delta = trial - sv
            if abs(actual_delta) < 1e-12:
                continue

            q[jx] = trial
            J[:, jx] = (
                arm_fk(q, lift, waist, params)[:3, 3] - pos
            ) / actual_delta
            q[jx] = sv

        try:
            dq = np.linalg.pinv(J) @ (target - pos)

            if weight_orientation > 0:
                dq_orient = -weight_orientation * (q - seed)
                for index in fixed_joints:
                    dq_orient[index] = 0.0
                dq = dq + dq_orient * 0.1

        except Exception:
            continue

        for index in fixed_joints:
            dq[index] = 0.0

        dq_m = float(np.max(np.abs(dq)))
        if dq_m > 0.3:
            dq *= 0.3 / dq_m

        q = np.clip(
            q + dq * 0.8,
            lower_limits,
            upper_limits,
        )

        for index, value in fixed_joints.items():
            q[index] = float(value)

    for index, value in fixed_joints.items():
        best_q[index] = float(value)

    Tf = arm_fk(best_q, lift, waist, params)
    return best_q, float(
        np.linalg.norm(target - Tf[:3, 3])
    )


def solve_ik_with_pose(
    target_xy_isa,
    ref_z_isa,
    seed,
    lift,
    waist,
    lower_limits,
    upper_limits,
):
    """
    求解右臂 XY 抓取 IK。

    沿用原脚本坐标映射：
        Isaac Y -> ROS X
        Isaac X -> ROS Y

    IK 直接求“肩部归零后的最终抓取姿态”。

    q[1] = arm_right1_arm_right2_joint 固定为 0 rad。
    SHOULDER_LIFT=-0.3rad 仅作为实机运动中的避障预姿态，
    不再作为最终 IK 条件。
    """
    tx_ros = float(target_xy_isa[1])
    ty_ros = float(target_xy_isa[0])
    tz_ros = float(ref_z_isa)

    target_ros = np.array(
        [tx_ros, ty_ros, tz_ros],
        dtype=np.float64,
    )

    # 最终抓取时肩部必须归零，因此 IK 在 shoulder=0 条件下求解。
    fixed = {1: 0.0}

    q, err = arm_ik(
        target_ros,
        lift,
        waist,
        R_P,
        seed.copy(),
        lower_limits,
        upper_limits,
        weight_orientation=0.0,
        max_iter=300,
        fixed_joints=fixed,
    )

    if err < 0.05:
        q, err = arm_ik(
            target_ros,
            lift,
            waist,
            R_P,
            q,
            lower_limits,
            upper_limits,
            weight_orientation=0.01,
            max_iter=100,
            fixed_joints=fixed,
        )

    return q, err


# ============================================================================
# URDF 限位
# ============================================================================

def load_urdf_limits(
    urdf_path: str,
) -> Dict[str, Tuple[float, float, float]]:
    path = Path(urdf_path)
    if not path.exists():
        raise FileNotFoundError(
            f"URDF 不存在: {path}"
        )

    root = ET.parse(path).getroot()
    limits: Dict[str, Tuple[float, float, float]] = {}

    required = [
        LIFT_JOINT,
        WAIST_JOINT,
        *RIGHT_MOTION_JOINTS,
        *LEFT_MOTION_JOINTS,
    ]

    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in required:
            continue

        limit = joint.find("limit")
        if limit is None:
            continue

        lower = float(limit.attrib.get("lower", "-inf"))
        upper = float(limit.attrib.get("upper", "inf"))
        velocity = float(limit.attrib.get("velocity", "0"))

        limits[name] = (
            lower,
            upper,
            velocity,
        )

    missing = [name for name in required if name not in limits]
    if missing:
        raise RuntimeError(
            "URDF 缺少以下关节 limit:\n  "
            + "\n  ".join(missing)
        )

    for name in required:
        lower, upper, velocity = limits[name]

        if not math.isfinite(lower) or not math.isfinite(upper):
            raise RuntimeError(
                f"{name} 没有有限位置限位"
            )

        if velocity <= 0.0 or not math.isfinite(velocity):
            raise RuntimeError(
                f"{name} 没有有效速度限位"
            )

    return limits


def right_arm_limit_arrays(
    limits: Mapping[str, Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    names = RIGHT_MOTION_JOINTS

    lower = np.array(
        [limits[name][0] for name in names],
        dtype=np.float64,
    )
    upper = np.array(
        [limits[name][1] for name in names],
        dtype=np.float64,
    )

    return lower, upper


def validate_scalar_limit(
    name: str,
    value: float,
    limits: Mapping[str, Tuple[float, float, float]],
    tolerance: float = 1e-6,
):
    lower, upper, _ = limits[name]
    if value < lower - tolerance or value > upper + tolerance:
        raise RuntimeError(
            f"{name} 越界: value={value:+.6f}, "
            f"limit=[{lower:+.6f}, {upper:+.6f}]"
        )


def validate_right_q(
    q: np.ndarray,
    limits: Mapping[str, Tuple[float, float, float]],
):
    problems = []

    for name, value in zip(RIGHT_MOTION_JOINTS, q):
        lower, upper, _ = limits[name]
        if value < lower - 1e-6 or value > upper + 1e-6:
            problems.append(
                f"{name}: value={value:+.6f}, "
                f"limit=[{lower:+.6f}, {upper:+.6f}]"
            )

    if problems:
        raise RuntimeError(
            "IK 解超出实机关节限位，拒绝执行:\n  "
            + "\n  ".join(problems)
        )



def validate_left_q(
    q: np.ndarray,
    limits: Mapping[str, Tuple[float, float, float]],
):
    problems = []

    for name, value in zip(LEFT_MOTION_JOINTS, q):
        lower, upper, _ = limits[name]
        if value < lower - 1e-6 or value > upper + 1e-6:
            problems.append(
                f"{name}: value={value:+.6f}, "
                f"limit=[{lower:+.6f}, {upper:+.6f}]"
            )

    if problems:
        raise RuntimeError(
            "左臂镜像解超出实机关节限位，拒绝执行:\n  "
            + "\n  ".join(problems)
        )


# ============================================================================
# 真实机器人控制器
# ============================================================================

class IKGraspRealController:
    def __init__(
        self,
        limits,
        velocity_scale=0.30,
        gripper_effort=50.0,
    ):
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from control_msgs.action import (
            FollowJointTrajectory,
            GripperCommand,
        )
        from action_msgs.msg import GoalStatus
        from sensor_msgs.msg import JointState

        self._rclpy = rclpy
        self._ActionClient = ActionClient
        self._FollowJointTrajectory = FollowJointTrajectory
        self._GripperCommand = GripperCommand
        self._GoalStatus = GoalStatus

        self._node = Node("right_arm_ik_grasp_real")
        self._limits = dict(limits)
        self._velocity_scale = float(velocity_scale)
        self._gripper_effort = float(gripper_effort)

        self._pos: Dict[str, float] = {}
        self._pos_lock = threading.Lock()
        self._active_goal_handles = []

        qos_state = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self._node.create_subscription(
            JointState,
            "/joint_states",
            self._on_state,
            qos_state,
        )

        self._clients = {
            "lift": ActionClient(
                self._node,
                FollowJointTrajectory,
                "/lift_controller/follow_joint_trajectory",
            ),
            "right_root": ActionClient(
                self._node,
                FollowJointTrajectory,
                "/right_arm_root_controller/follow_joint_trajectory",
            ),
            "right_arm": ActionClient(
                self._node,
                FollowJointTrajectory,
                "/right_arm_controller/follow_joint_trajectory",
            ),
            "right_gripper": ActionClient(
                self._node,
                GripperCommand,
                "/right_gripper_controller/gripper_cmd",
            ),
            "left_root": ActionClient(
                self._node,
                FollowJointTrajectory,
                "/left_arm_root_controller/follow_joint_trajectory",
            ),
            "left_arm": ActionClient(
                self._node,
                FollowJointTrajectory,
                "/left_arm_controller/follow_joint_trajectory",
            ),
            "left_gripper": ActionClient(
                self._node,
                GripperCommand,
                "/left_gripper_controller/gripper_cmd",
            ),
        }

        self._wait_for_controllers()
        self._wait_for_joint_states()

        self._node.get_logger().info(
            "✅ 右臂纯 IK 实机控制器就绪"
        )

    # ----------------------------------------------------------------------
    # ROS helpers
    # ----------------------------------------------------------------------

    def get_logger(self):
        return self._node.get_logger()

    def _on_state(self, msg):
        with self._pos_lock:
            for name, position in zip(msg.name, msg.position):
                self._pos[name] = float(position)

    def _wait_for_controllers(self):
        labels = {
            "lift": "升降控制器",
            "right_root": "右臂根部控制器",
            "right_arm": "右臂控制器",
            "right_gripper": "右夹爪控制器",
            "left_root": "左臂根部控制器",
            "left_arm": "左臂控制器",
            "left_gripper": "左夹爪控制器",
        }

        for key, client in self._clients.items():
            if not client.wait_for_server(timeout_sec=8.0):
                raise RuntimeError(
                    f"{labels[key]} Action 未就绪"
                )

            self.get_logger().info(
                f"✅ {labels[key]}就绪"
            )

    def _wait_for_joint_states(self):
        required = [
            LIFT_JOINT,
            WAIST_JOINT,
            *RIGHT_MOTION_JOINTS,
            *LEFT_MOTION_JOINTS,
        ]

        deadline = time.monotonic() + 8.0

        while time.monotonic() < deadline:
            self._rclpy.spin_once(
                self._node,
                timeout_sec=0.1,
            )

            with self._pos_lock:
                missing = [
                    name
                    for name in required
                    if name not in self._pos
                ]

            if not missing:
                self.get_logger().info(
                    f"✅ 已同步 {len(required)} 个相关运动关节"
                )
                return

        raise RuntimeError(
            "等待 /joint_states 超时，缺少:\n  "
            + "\n  ".join(missing)
        )

    def get_pos(self, name: str) -> float:
        # 先处理一次最新 joint_states。
        self._rclpy.spin_once(
            self._node,
            timeout_sec=0.01,
        )

        with self._pos_lock:
            if name not in self._pos:
                raise RuntimeError(
                    f"/joint_states 中没有 {name}"
                )
            return float(self._pos[name])

    def current_right_q(self) -> np.ndarray:
        return np.array([
            self.get_pos(RIGHT_ROOT_JOINT),
            *[
                self.get_pos(name)
                for name in RIGHT_ARM_JOINTS
            ],
        ], dtype=np.float64)

    def _safe_duration(
        self,
        targets: Mapping[str, float],
        requested_duration: float,
    ) -> float:
        required = 0.0

        for name, target in targets.items():
            if name not in self._limits:
                continue

            current = self.get_pos(name)
            _, _, vmax = self._limits[name]

            scaled_vmax = (
                vmax * self._velocity_scale
            )

            if scaled_vmax <= 0.0:
                continue

            required = max(
                required,
                abs(float(target) - current)
                / scaled_vmax,
            )

        # 额外 30% 裕量。
        return max(
            float(requested_duration),
            required * 1.30,
            0.20,
        )

    def _wait_future(
        self,
        future,
        timeout_sec: float,
    ) -> bool:
        deadline = time.monotonic() + float(timeout_sec)

        while (
            self._rclpy.ok()
            and time.monotonic() < deadline
        ):
            if future.done():
                return True
            self._rclpy.spin_once(
                self._node,
                timeout_sec=0.05,
            )

        return future.done()

    def _make_follow_goal(
        self,
        joint_names: Sequence[str],
        start_positions: Sequence[float],
        target_positions: Sequence[float],
        duration: float,
        steps: int = 60,
    ):
        """
        构造多点 smoothstep 轨迹。

        之前版本只有一个终点。部分实机控制器虽然接受单点 goal，
        但实际执行可能不明显。这里与已经验证过的双臂实机代码一致，
        显式发送完整平滑轨迹。
        """
        from trajectory_msgs.msg import JointTrajectoryPoint
        from rclpy.duration import Duration

        goal = self._FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(joint_names)

        start = np.asarray(
            start_positions,
            dtype=np.float64,
        )
        target = np.asarray(
            target_positions,
            dtype=np.float64,
        )

        for index in range(1, int(steps) + 1):
            ratio = index / float(steps)
            weight = ratio * ratio * (3.0 - 2.0 * ratio)
            row = start + (target - start) * weight

            point = JointTrajectoryPoint()
            point.positions = [
                float(value)
                for value in row
            ]

            timestamp = float(duration) * ratio
            point.time_from_start = Duration(
                seconds=timestamp
            ).to_msg()

            goal.trajectory.points.append(point)

        return goal

    def _verify_targets_reached(
        self,
        targets: Mapping[str, float],
        timeout_sec: float = 2.0,
    ) -> bool:
        """
        Action 完成后再次用 /joint_states 验证真实位置。

        lift 使用 5 mm 容差；
        旋转关节使用 0.03 rad 容差。
        """
        deadline = time.monotonic() + float(timeout_sec)
        last_errors = {}

        while (
            self._rclpy.ok()
            and time.monotonic() < deadline
        ):
            self._rclpy.spin_once(
                self._node,
                timeout_sec=0.05,
            )

            all_ok = True
            last_errors = {}

            for name, target in targets.items():
                actual = self.get_pos(name)
                error = abs(actual - float(target))
                tolerance = (
                    0.005
                    if name == LIFT_JOINT
                    else 0.03
                )

                last_errors[name] = (
                    actual,
                    float(target),
                    error,
                    tolerance,
                )

                if error > tolerance:
                    all_ok = False

            if all_ok:
                return True

        for name, (
            actual,
            target,
            error,
            tolerance,
        ) in last_errors.items():
            if error > tolerance:
                self.get_logger().error(
                    f"❌ {name} 实际未到目标: "
                    f"actual={actual:+.6f}, "
                    f"target={target:+.6f}, "
                    f"error={error:.6f}, "
                    f"tol={tolerance:.6f}"
                )

        return False

    def send_motion(
        self,
        targets: Mapping[str, float],
        requested_duration: float,
        steps: int = 60,
    ) -> bool:
        """
        将目标拆到 Robot A 实机控制器并同步执行。

        - lift
        - 左右 root
        - 左右 6 关节 arm controller

        未指定的同一控制器关节保持当前实机位置。
        Action 成功后必须通过 /joint_states 二次验证。
        """
        targets = {
            name: float(value)
            for name, value in targets.items()
        }

        for name, value in targets.items():
            validate_scalar_limit(
                name,
                value,
                self._limits,
            )

        duration = self._safe_duration(
            targets,
            requested_duration,
        )

        groups = []

        def add_single(
            key: str,
            joint_name: str,
        ):
            if joint_name not in targets:
                return

            current = self.get_pos(joint_name)
            groups.append((
                key,
                [joint_name],
                [current],
                [targets[joint_name]],
            ))

        add_single("lift", LIFT_JOINT)
        add_single("right_root", RIGHT_ROOT_JOINT)
        add_single("left_root", LEFT_ROOT_JOINT)

        if any(
            name in targets
            for name in RIGHT_ARM_JOINTS
        ):
            start_positions = [
                self.get_pos(name)
                for name in RIGHT_ARM_JOINTS
            ]
            target_positions = [
                targets.get(name, start)
                for name, start in zip(
                    RIGHT_ARM_JOINTS,
                    start_positions,
                )
            ]
            groups.append((
                "right_arm",
                RIGHT_ARM_JOINTS,
                start_positions,
                target_positions,
            ))

        if any(
            name in targets
            for name in LEFT_ARM_JOINTS
        ):
            start_positions = [
                self.get_pos(name)
                for name in LEFT_ARM_JOINTS
            ]
            target_positions = [
                targets.get(name, start)
                for name, start in zip(
                    LEFT_ARM_JOINTS,
                    start_positions,
                )
            ]
            groups.append((
                "left_arm",
                LEFT_ARM_JOINTS,
                start_positions,
                target_positions,
            ))

        if not groups:
            return True

        self.get_logger().info(
            "发送平滑实机轨迹: "
            f"targets={targets}, "
            f"duration={duration:.2f}s, "
            f"steps={steps}"
        )

        send_futures = {}

        for (
            key,
            joint_names,
            start_positions,
            target_positions,
        ) in groups:
            goal = self._make_follow_goal(
                joint_names,
                start_positions,
                target_positions,
                duration,
                steps=steps,
            )
            send_futures[key] = (
                self._clients[key].send_goal_async(
                    goal
                )
            )

        for future in send_futures.values():
            if not self._wait_future(
                future,
                timeout_sec=10.0,
            ):
                self.get_logger().error(
                    "❌ 等待控制器接受目标超时"
                )
                return False

        handles = {}

        for key, future in send_futures.items():
            handle = future.result()

            if (
                handle is None
                or not handle.accepted
            ):
                self.get_logger().error(
                    f"❌ {key} Action goal 被拒绝"
                )
                continue

            handles[key] = handle
            self._active_goal_handles.append(
                handle
            )

        if len(handles) != len(send_futures):
            self.cancel_active_goals()
            return False

        result_futures = {
            key: handle.get_result_async()
            for key, handle in handles.items()
        }

        for future in result_futures.values():
            if not self._wait_future(
                future,
                timeout_sec=duration + 15.0,
            ):
                self.get_logger().error(
                    "❌ 等待轨迹执行完成超时"
                )
                self.cancel_active_goals()
                return False

        ok = True

        for key, future in result_futures.items():
            wrapped = future.result()

            if wrapped is None:
                self.get_logger().error(
                    f"❌ {key} Action 无结果"
                )
                ok = False
                continue

            result = wrapped.result
            status_ok = (
                wrapped.status
                == self._GoalStatus.STATUS_SUCCEEDED
            )
            error_ok = (
                not hasattr(result, "error_code")
                or int(result.error_code) == 0
            )

            if status_ok and error_ok:
                self.get_logger().info(
                    f"✅ {key} 轨迹完成"
                )
            else:
                self.get_logger().error(
                    f"❌ {key} 轨迹失败: "
                    f"status={wrapped.status}, "
                    f"error_code="
                    f"{getattr(result, 'error_code', None)}, "
                    f"error_string="
                    f"{getattr(result, 'error_string', '')!r}"
                )
                ok = False

        self._active_goal_handles.clear()

        if not ok:
            return False

        # 必须用真实 joint_states 复核。
        if not self._verify_targets_reached(
            targets,
            timeout_sec=2.5,
        ):
            return False

        if LIFT_JOINT in targets:
            actual_lift = self.get_pos(
                LIFT_JOINT
            )
            self.get_logger().info(
                "✅ lift 实际反馈: "
                f"{actual_lift:+.6f}m "
                f"(目标 {targets[LIFT_JOINT]:+.6f}m)"
            )

        return True

    def command_gripper(
        self,
        side: str,
        position: float,
    ) -> bool:
        if side not in ("left", "right"):
            raise ValueError(
                f"非法夹爪侧: {side}"
            )

        if not 0.0 <= float(position) <= 0.025:
            raise RuntimeError(
                f"夹爪位置越界: {position}; "
                "允许范围 [0.0, 0.025]"
            )

        client_key = (
            "left_gripper"
            if side == "left"
            else "right_gripper"
        )
        label = (
            "左夹爪"
            if side == "left"
            else "右夹爪"
        )

        goal = self._GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(
            self._gripper_effort
        )

        future = self._clients[
            client_key
        ].send_goal_async(goal)

        if not self._wait_future(
            future,
            timeout_sec=10.0,
        ):
            self.get_logger().error(
                f"❌ {label}目标发送超时"
            )
            return False

        goal_handle = future.result()

        if (
            goal_handle is None
            or not goal_handle.accepted
        ):
            self.get_logger().error(
                f"❌ {label} goal 被拒绝"
            )
            return False

        self._active_goal_handles.append(
            goal_handle
        )

        result_future = (
            goal_handle.get_result_async()
        )

        if not self._wait_future(
            result_future,
            timeout_sec=12.0,
        ):
            self.get_logger().error(
                f"❌ {label}执行超时"
            )
            self.cancel_active_goals()
            return False

        wrapped = result_future.result()

        self._active_goal_handles.clear()

        if (
            wrapped is None
            or wrapped.status
            != self._GoalStatus.STATUS_SUCCEEDED
        ):
            self.get_logger().error(
                f"❌ {label} Action 执行失败"
            )
            return False

        self.get_logger().info(
            f"✅ {label}: "
            f"command_position={position:.5f}"
        )

        return True

    def open_both_grippers(self) -> bool:
        # 左夹爪在整个流程中始终保持张开。
        left_ok = self.command_gripper(
            "left",
            GRIPPER_OPEN,
        )
        right_ok = self.command_gripper(
            "right",
            GRIPPER_OPEN,
        )
        return left_ok and right_ok

    def cancel_active_goals(self):
        for goal_handle in list(
            self._active_goal_handles
        ):
            try:
                future = (
                    goal_handle.cancel_goal_async()
                )
                self._rclpy.spin_until_future_complete(
                    self._node,
                    future,
                    timeout_sec=1.0,
                )
            except Exception:
                pass

        self._active_goal_handles.clear()

    # ----------------------------------------------------------------------
    # IK / plan
    # ----------------------------------------------------------------------

    def calculate_plan(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        max_fk_error_m: float,
        max_final_xy_error_m: float,
    ) -> dict:
        current_lift = self.get_pos(
            LIFT_JOINT
        )
        current_waist_raw = self.get_pos(
            WAIST_JOINT
        )

        waist_lower, waist_upper, _ = (
            self._limits[WAIST_JOINT]
        )

        # 零点附近可能有轻微负反馈。
        current_waist = float(
            np.clip(
                current_waist_raw,
                waist_lower,
                waist_upper,
            )
        )

        if abs(
            current_waist - current_waist_raw
        ) > 1e-9:
            print(
                "  ℹ️ waist 当前反馈按 URDF 限位修正: "
                f"{current_waist_raw:+.6f} "
                f"→ {current_waist:+.6f}rad"
            )

        lower, upper = right_arm_limit_arrays(
            self._limits
        )

        # 当前 lift/waist 下，使用模板的 Z 作为 XY IK 参考高度。
        T_template = arm_fk(
            TEMPLATE_RIGHT_ARM,
            current_lift,
            current_waist,
            R_P,
        )
        ref_z = float(
            T_template[2, 3]
        )

        rarm_ik, fk_err = solve_ik_with_pose(
            np.array(
                [target_x, target_y],
                dtype=np.float64,
            ),
            ref_z,
            TEMPLATE_RIGHT_ARM.copy(),
            current_lift,
            current_waist,
            lower,
            upper,
        )

        validate_right_q(
            rarm_ik,
            self._limits,
        )

        validate_scalar_limit(
            RIGHT_ARM_JOINTS[0],
            SHOULDER_LIFT,
            self._limits,
        )
        validate_scalar_limit(
            RIGHT_ARM_JOINTS[0],
            0.0,
            self._limits,
        )

        if fk_err > max_fk_error_m:
            raise RuntimeError(
                "右臂 XY IK 误差过大，拒绝执行: "
                f"{fk_err * 100:.2f}cm > "
                f"{max_fk_error_m * 100:.2f}cm"
            )

        # 最终抓取 IK 已经在 shoulder=0 条件下求出。
        grasp_q = rarm_ik.copy()
        grasp_q[1] = 0.0

        validate_right_q(
            grasp_q,
            self._limits,
        )

        # 左臂始终按完整关节镜像运动。
        left_grasp_q = mirror_right_q_to_left(
            grasp_q
        )
        validate_left_q(
            left_grasp_q,
            self._limits,
        )

        T_grasp = arm_fk(
            grasp_q,
            current_lift,
            current_waist,
            R_P,
        )
        grasp_ros = T_grasp[:3, 3]

        grasp_x_isa = float(
            grasp_ros[1]
        )
        grasp_y_isa = float(
            grasp_ros[0]
        )
        current_grasp_z = float(
            grasp_ros[2]
        )

        final_xy_error = float(
            math.hypot(
                grasp_x_isa - target_x,
                grasp_y_isa - target_y,
            )
        )

        # 阶段 3 是避障预抓取姿态：
        # 除肩部外，其余关节已经到最终 IK 值；
        # shoulder 暂时保持 -0.3rad。
        # 因此该阶段允许末端暂时偏离目标 XY。
        pregrasp_q = grasp_q.copy()
        pregrasp_q[1] = SHOULDER_LIFT

        validate_right_q(
            pregrasp_q,
            self._limits,
        )

        left_pregrasp_q = mirror_right_q_to_left(
            pregrasp_q
        )
        validate_left_q(
            left_pregrasp_q,
            self._limits,
        )

        T_pregrasp = arm_fk(
            pregrasp_q,
            current_lift,
            current_waist,
            R_P,
        )
        pregrasp_ros = T_pregrasp[:3, 3]

        pregrasp_x_isa = float(
            pregrasp_ros[1]
        )
        pregrasp_y_isa = float(
            pregrasp_ros[0]
        )

        pregrasp_xy_offset = float(
            math.hypot(
                pregrasp_x_isa - target_x,
                pregrasp_y_isa - target_y,
            )
        )

        # 沿用原程序逻辑：肩部归零后只用 lift 修正 Z。
        target_lift = (
            current_lift
            + float(target_z)
            - current_grasp_z
        )

        validate_scalar_limit(
            LIFT_JOINT,
            target_lift,
            self._limits,
        )

        # lift 为纯 Z 平移。
        predicted_final_z = (
            current_grasp_z
            + target_lift
            - current_lift
        )

        final_z_error = abs(
            predicted_final_z
            - float(target_z)
        )

        # 最终 shoulder=0 的抓取姿态必须真正到达目标 XY。
        if final_xy_error > max_final_xy_error_m:
            raise RuntimeError(
                "最终抓取 XY 偏差过大，拒绝执行: "
                f"{final_xy_error * 100:.2f}cm > "
                f"{max_final_xy_error_m * 100:.2f}cm"
            )

        return {
            "current_lift": current_lift,
            "current_waist": current_waist,
            "reference_z": ref_z,
            "ik_q": rarm_ik,
            "pregrasp_q": pregrasp_q,
            "grasp_q": grasp_q,
            "left_pregrasp_q": left_pregrasp_q,
            "left_grasp_q": left_grasp_q,
            "ik_fk_error_m": fk_err,
            "pregrasp_xy_offset_m": pregrasp_xy_offset,
            "pregrasp_predicted": (
                pregrasp_x_isa,
                pregrasp_y_isa,
                float(pregrasp_ros[2]),
            ),
            "grasp_xy_error_m": final_xy_error,
            "grasp_predicted_before_lift": (
                grasp_x_isa,
                grasp_y_isa,
                current_grasp_z,
            ),
            "target_lift": target_lift,
            "lift_delta": (
                target_lift - current_lift
            ),
            "predicted_final_z": predicted_final_z,
            "final_z_error_m": final_z_error,
        }

    # ----------------------------------------------------------------------
    # 6 阶段实机抓取
    # ----------------------------------------------------------------------

    def execute_plan(
        self,
        plan: dict,
        close_gripper: bool,
    ):
        """
        双臂对称运动，右夹爪负责抓取，左夹爪始终张开。

        1. 左右夹爪张开
        2. 双肩镜像抬举：右 -0.3，左 +0.3
        3. 左右臂同时进入镜像预抓取姿态
        4. 双肩同时回到最终镜像抓取姿态
        5. lift 真实运动并通过 /joint_states 复核
        6. 只闭合右夹爪，左夹爪保持张开
        """
        log = self.get_logger().info

        pregrasp_q = np.asarray(
            plan["pregrasp_q"],
            dtype=np.float64,
        )
        grasp_q = np.asarray(
            plan["grasp_q"],
            dtype=np.float64,
        )
        left_pregrasp_q = np.asarray(
            plan["left_pregrasp_q"],
            dtype=np.float64,
        )
        left_grasp_q = np.asarray(
            plan["left_grasp_q"],
            dtype=np.float64,
        )

        log(
            "▶ [1/6] 左右夹爪张开；"
            "左夹爪后续始终保持张开"
        )
        if not self.open_both_grippers():
            raise RuntimeError(
                "左右夹爪张开失败"
            )

        log(
            "▶ [2/6] 双肩镜像抬举: "
            f"右={SHOULDER_LIFT:+.3f}rad, "
            f"左={-SHOULDER_LIFT:+.3f}rad"
        )
        if not self.send_motion({
            RIGHT_ARM_JOINTS[0]:
                SHOULDER_LIFT,
            LEFT_ARM_JOINTS[0]:
                -SHOULDER_LIFT,
        }, requested_duration=1.5):
            raise RuntimeError(
                "双肩镜像抬举失败"
            )

        log(
            "▶ [3/6] 左右机械臂同步进入镜像预抓取姿态"
        )

        stage3_targets = {
            RIGHT_ROOT_JOINT:
                float(pregrasp_q[0]),
            LEFT_ROOT_JOINT:
                float(left_pregrasp_q[0]),
        }

        for name, value in zip(
            RIGHT_ARM_JOINTS,
            pregrasp_q[1:],
        ):
            stage3_targets[name] = float(
                value
            )

        for name, value in zip(
            LEFT_ARM_JOINTS,
            left_pregrasp_q[1:],
        ):
            stage3_targets[name] = float(
                value
            )

        if not self.send_motion(
            stage3_targets,
            requested_duration=3.0,
            steps=90,
        ):
            raise RuntimeError(
                "双臂镜像预抓取运动失败"
            )

        log(
            "▶ [4/6] 双臂同步进入最终镜像抓取姿态；"
            "右肩 -0.3→0，左肩 +0.3→0"
        )

        stage4_targets = {
            RIGHT_ROOT_JOINT:
                float(grasp_q[0]),
            LEFT_ROOT_JOINT:
                float(left_grasp_q[0]),
        }

        for name, value in zip(
            RIGHT_ARM_JOINTS,
            grasp_q[1:],
        ):
            stage4_targets[name] = float(
                value
            )

        for name, value in zip(
            LEFT_ARM_JOINTS,
            left_grasp_q[1:],
        ):
            stage4_targets[name] = float(
                value
            )

        if not self.send_motion(
            stage4_targets,
            requested_duration=1.8,
            steps=70,
        ):
            raise RuntimeError(
                "双臂最终镜像抓取姿态运动失败"
            )

        current_lift_feedback = self.get_pos(
            LIFT_JOINT
        )

        log(
            "▶ [5/6] Lift Z 定位: "
            f"实机当前={current_lift_feedback:+.4f}m, "
            f"计划起点={plan['current_lift']:+.4f}m, "
            f"目标={plan['target_lift']:+.4f}m, "
            f"Δ={plan['target_lift'] - current_lift_feedback:+.4f}m"
        )

        if not self.send_motion({
            LIFT_JOINT:
                float(plan["target_lift"]),
        }, requested_duration=2.5, steps=80):
            raise RuntimeError(
                "Lift Z 定位失败："
                "Action 或真实 joint_states 未到目标"
            )

        actual_lift = self.get_pos(
            LIFT_JOINT
        )

        log(
            "     Lift 已验证: "
            f"actual={actual_lift:+.6f}m, "
            f"target={plan['target_lift']:+.6f}m"
        )

        if close_gripper:
            log(
                "▶ [6/6] 只闭合右夹爪；"
                "左夹爪保持张开"
            )

            # 先再次确认左夹爪是张开的。
            if not self.command_gripper(
                "left",
                GRIPPER_OPEN,
            ):
                raise RuntimeError(
                    "左夹爪保持张开命令失败"
                )

            if not self.command_gripper(
                "right",
                GRIPPER_CLOSE,
            ):
                raise RuntimeError(
                    "右夹爪闭合失败"
                )
        else:
            log(
                "▶ [6/6] 测试模式："
                "左右夹爪都保持张开"
            )
            if not self.command_gripper(
                "left",
                GRIPPER_OPEN,
            ):
                raise RuntimeError(
                    "左夹爪保持张开失败"
                )

        log(
            "✅ 双臂镜像运动完成；"
            "左夹爪保持张开"
        )

    def shutdown(self):
        try:
            self.cancel_active_goals()
        except Exception:
            pass

        try:
            self._node.destroy_node()
        except Exception:
            pass


# ============================================================================
# Mock
# ============================================================================

def mock_execute(
    tx: float,
    ty: float,
    tz: float,
):
    print(
        f"\n🎭 Mock: "
        f"target=({tx:.3f}, {ty:.3f}, {tz:.3f})"
    )

    T_template = arm_fk(
        TEMPLATE_RIGHT_ARM,
        0.0,
        0.0,
        R_P,
    )
    ref_z = float(T_template[2, 3])

    q, fk_err = solve_ik_with_pose(
        np.array([tx, ty]),
        ref_z,
        TEMPLATE_RIGHT_ARM.copy(),
        0.0,
        0.0,
        FALLBACK_RIGHT_LOWER,
        FALLBACK_RIGHT_UPPER,
    )

    grasp_q = q.copy()
    grasp_q[1] = 0.0

    T_grasp = arm_fk(
        grasp_q,
        0.0,
        0.0,
        R_P,
    )
    grasp_pos = T_grasp[:3, 3]

    gx = float(grasp_pos[1])
    gy = float(grasp_pos[0])
    gz = float(grasp_pos[2])

    final_xy_err = math.hypot(
        gx - tx,
        gy - ty,
    )

    pregrasp_q = grasp_q.copy()
    pregrasp_q[1] = SHOULDER_LIFT
    T_pregrasp = arm_fk(
        pregrasp_q,
        0.0,
        0.0,
        R_P,
    )
    pre = T_pregrasp[:3, 3]
    px = float(pre[1])
    py = float(pre[0])
    pz = float(pre[2])
    pregrasp_offset = math.hypot(
        px - tx,
        py - ty,
    )

    lift = tz - gz

    print(
        "   最终 IK 解关节:",
        np.round(grasp_q, 4),
    )
    print(
        f"   避障预抓取末端: "
        f"({px:.3f}, {py:.3f}, {pz:.3f})"
    )
    print(
        f"   预抓取相对目标 XY 偏移: "
        f"{pregrasp_offset * 100:.2f}cm"
    )
    print(
        f"   肩部归零后最终末端: "
        f"({gx:.3f}, {gy:.3f}, {gz:.3f})"
    )
    print(
        f"   肩部归零后 XY误差: "
        f"{final_xy_err * 100:.2f}cm"
    )
    print(
        f"   目标 lift: {lift:+.4f}m"
    )
    print(
        f"   IK FK误差: {fk_err * 100:.2f}cm"
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Robot A 右臂纯 IK 抓取 —— "
            "真实机器人 ros2_control Action"
        )
    )

    parser.add_argument(
        "-x",
        "--target-x",
        type=float,
        required=True,
        help=(
            "沿用原脚本坐标：Isaac X / ROS Y，"
            "右侧示例为负值"
        ),
    )
    parser.add_argument(
        "-y",
        "--target-y",
        type=float,
        required=True,
        help="沿用原脚本坐标：Isaac Y / ROS X",
    )
    parser.add_argument(
        "-z",
        "--target-z",
        type=float,
        required=True,
        help="目标地面参考 Z，单位 m",
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
        default=0.30,
        help="URDF 最大速度缩放，默认 0.30",
    )

    parser.add_argument(
        "--gripper-effort",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--max-fk-error-cm",
        type=float,
        default=5.0,
        help=(
            "最终 shoulder=0 条件下的 IK 最大位置误差，"
            "默认 5cm"
        ),
    )

    parser.add_argument(
        "--max-final-xy-error-cm",
        type=float,
        default=5.0,
        help=(
            "最终 shoulder=0 抓取姿态的 XY 最大误差。"
        ),
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="完全离线，只进行 FK/IK 测试",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="允许真实机器人运动；默认仅安全检查",
    )

    parser.add_argument(
        "--close-gripper",
        action="store_true",
        help=(
            "第6阶段闭合右夹爪；"
            "首次路径测试不要添加"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not (
        0.0 < args.velocity_scale <= 1.0
    ):
        raise SystemExit(
            "❌ --velocity-scale 必须位于 (0, 1]"
        )

    if args.max_fk_error_cm <= 0.0:
        raise SystemExit(
            "❌ --max-fk-error-cm 必须大于 0"
        )

    if args.max_final_xy_error_cm <= 0.0:
        raise SystemExit(
            "❌ --max-final-xy-error-cm 必须大于 0"
        )

    if args.mock:
        mock_execute(
            args.target_x,
            args.target_y,
            args.target_z,
        )
        return 0

    import rclpy

    limits = load_urdf_limits(
        args.urdf
    )

    rclpy.init(args=None)
    controller = None

    try:
        controller = IKGraspRealController(
            limits,
            velocity_scale=args.velocity_scale,
            gripper_effort=args.gripper_effort,
        )

        log = controller.get_logger().info

        log("=" * 64)
        log(
            "Robot A 右臂纯 IK 抓取 "
            "[真实机器人 Action]"
        )
        log(
            f"目标: "
            f"({args.target_x:.3f}, "
            f"{args.target_y:.3f}, "
            f"{args.target_z:.3f})"
        )
        log(
            "坐标映射沿用原脚本: "
            "ROS X=输入Y, ROS Y=输入X"
        )
        log("=" * 64)

        log(">>> IK 与实机安全检查...")

        plan = controller.calculate_plan(
            args.target_x,
            args.target_y,
            args.target_z,
            max_fk_error_m=(
                args.max_fk_error_cm / 100.0
            ),
            max_final_xy_error_m=(
                args.max_final_xy_error_cm
                / 100.0
            ),
        )

        print(
            "\nIK 与安全检查通过："
        )
        print(
            "  ik_q(rad) =",
            np.round(
                plan["ik_q"],
                6,
            ),
        )
        print(
            "  ik_q(deg) =",
            np.round(
                np.degrees(
                    plan["ik_q"]
                ),
                2,
            ),
        )
        print(
            "  pregrasp_predicted =",
            tuple(
                round(v, 6)
                for v in plan[
                    "pregrasp_predicted"
                ]
            ),
        )
        print(
            "  pregrasp_XY_offset_cm =",
            round(
                plan[
                    "pregrasp_xy_offset_m"
                ] * 100.0,
                3,
            ),
        )
        print(
            "  final_shoulder_zero_predicted =",
            tuple(
                round(v, 6)
                for v in plan[
                    "grasp_predicted_before_lift"
                ]
            ),
        )
        print(
            "  final_XY_error_cm =",
            round(
                plan[
                    "grasp_xy_error_m"
                ] * 100.0,
                3,
            ),
        )
        print(
            "  left_pregrasp_q(deg) =",
            np.round(
                np.degrees(
                    plan["left_pregrasp_q"]
                ),
                2,
            ),
        )
        print(
            "  left_grasp_q(deg) =",
            np.round(
                np.degrees(
                    plan["left_grasp_q"]
                ),
                2,
            ),
        )
        print(
            "  mirror_error_pregrasp_rad =",
            round(
                float(
                    np.max(
                        np.abs(
                            plan["pregrasp_q"]
                            + plan["left_pregrasp_q"]
                        )
                    )
                ),
                6,
            ),
        )
        print(
            "  mirror_error_grasp_rad =",
            round(
                float(
                    np.max(
                        np.abs(
                            plan["grasp_q"]
                            + plan["left_grasp_q"]
                        )
                    )
                ),
                6,
            ),
        )
        print(
            "  current_lift =",
            round(
                plan["current_lift"],
                6,
            ),
        )
        print(
            "  target_lift =",
            round(
                plan["target_lift"],
                6,
            ),
        )
        print(
            "  lift_delta =",
            round(
                plan["lift_delta"],
                6,
            ),
        )
        print(
            "  predicted_final_z =",
            round(
                plan["predicted_final_z"],
                6,
            ),
        )

        if not args.execute:
            print(
                "\n✅ 仅完成 IK、FK、URDF 限位、"
                "控制器和 lift 检查，机器人没有运动。\n"
                "确认路径安全后添加 --execute。"
            )
            return 0

        controller.get_logger().warning(
            "即将控制真实机器人右臂。"
            "请保持实体急停可立即触达。"
        )

        controller.execute_plan(
            plan,
            close_gripper=args.close_gripper,
        )

        return 0

    except KeyboardInterrupt:
        if controller is not None:
            controller.get_logger().warning(
                "收到 Ctrl+C，正在请求取消活动 Action。"
            )
            controller.cancel_active_goals()
        return 130

    except Exception as exc:
        if controller is not None:
            controller.get_logger().error(
                f"❌ {exc}"
            )
        else:
            print(
                f"❌ {exc}",
                file=sys.stderr,
            )
        return 1

    finally:
        if controller is not None:
            controller.shutdown()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
