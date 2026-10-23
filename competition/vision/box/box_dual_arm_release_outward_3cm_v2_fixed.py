#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot A 双臂释放箱子 V2：保持当前抓取姿态，再左右各向外退
====================================================

用途：
    在抓箱 + 放箱动作完成之后单独运行。

实际执行顺序：
    0. 先在当前真实关节状态上完整规划并检查外退轨迹；
       只有整条轨迹可行才允许进入真实释放。
    1. 左右夹爪先张开到最大位置 0.025m；
    2. 确认夹爪张开动作完成；
    3. 左右 grasp TCP 分别沿当前双夹爪连线的“外侧方向”
       各退 --retreat-cm（默认 3cm）；
    4. 外退过程中保持当前 TCP Z，高度不主动变化；
    5. 前臂保持水平，手部姿态尽量保持当前释放姿态；
    6. 使用 2mm Cartesian waypoint + V4 minimum-jerk 时间律，
       让外退动作更平滑。

外退方向：
    span_xy = left_tcp_xy - right_tcp_xy
    left  向 +normalize(span_xy) 退
    right 向 -normalize(span_xy) 退

因此默认 --retreat-cm 3.0 表示：
    左夹爪向左外侧约 3cm；
    右夹爪向右外侧约 3cm；
    两夹爪最终总间距约增加 6cm。

安全：
    - 默认 dry-run，不加 --execute 绝不运动，也不会张开夹爪。
    - 真正执行前先完整规划，防止“先松箱后发现后退无解”。
    - 本脚本依赖：
        box_dual_arm_grasp_v9_template10_shoulder007_prelift025.py
        box_dual_arm_place_height_constrained_v4_smooth.py
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, Optional

import numpy as np
import rclpy

from box_dual_arm_grasp_v9_template10_shoulder007_prelift025 import (
    DualArmGraspRealController,
    LEFT_ARM_PARAMS,
    RIGHT_ARM_PARAMS,
    arm_joint_limit_arrays,
    grasp_tcp_fk,
    load_urdf_limits,
    rotation_vector,
)

from box_dual_arm_place_height_constrained_v4_smooth import (
    arm_q,
    forearm_tilt,
    hand_R,
    signed_forearm_vertical,
    gripper_vertical_error,
    execute_arm_path,
)


GRIPPER_MAX_OPEN_M = 0.025


def retreat_residual(
    q: np.ndarray,
    *,
    target_xyz: np.ndarray,
    reference_R: np.ndarray,
    reference_q: np.ndarray,
    lift: float,
    waist: float,
    params,
) -> np.ndarray:
    """
    单臂外退 IK 残差。

    硬目标：
      - grasp TCP XYZ 到 waypoint。

    软目标：
      - hand 姿态尽量保持当前释放姿态；
      - 与上一 waypoint 关节姿态连续。

    注意：
      不再强制“前臂绝对水平”。release 从任意实际抓取姿态开始，
      如果当前前臂本来不是水平，旧逻辑会在第一个 2mm waypoint
      就把本来可行的外退误判为 IK 无解。
    """
    q = np.asarray(q, dtype=float)

    tcp = grasp_tcp_fk(
        q,
        lift,
        waist,
        params,
    )[:3, 3]

    R = hand_R(
        q,
        lift,
        waist,
        params,
    )

    return np.concatenate([
        # 外退位置是核心约束，XYZ 都约束。
        16.0 * (
            tcp
            - np.asarray(
                target_xyz,
                dtype=float,
            )
        ),

        # 不再强制前臂绝对水平。
        # 外退时只需要从“当前真实姿态”平滑退出，
        # 因此保持当前 hand 方向即可。

        # 手部方向做连续性约束。
        0.050 * rotation_vector(
            R @ reference_R.T
        ),

        # 相邻 waypoint 变化尽量小。
        0.005 * (
            q
            - np.asarray(
                reference_q,
                dtype=float,
            )
        ),
    ])


def solve_retreat_waypoint_from_seed(
    *,
    seed_q: np.ndarray,
    reference_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_xyz: np.ndarray,
    reference_R: np.ndarray,
    lift: float,
    waist: float,
    params,
    max_tcp_error_m: float,
    max_forearm_tilt_rad: float,
    max_gripper_vertical_rad: float,
    max_orientation_drift_rad: float,
) -> Optional[Dict[str, object]]:
    q = np.clip(
        np.asarray(seed_q, dtype=float).copy(),
        lower,
        upper,
    )

    finite_delta = 1e-5
    damping = 0.020
    maximum_step = 0.10

    best = None
    best_score = float("inf")

    for _ in range(280):
        residual = retreat_residual(
            q,
            target_xyz=target_xyz,
            reference_R=reference_R,
            reference_q=reference_q,
            lift=lift,
            waist=waist,
            params=params,
        )

        tcp = grasp_tcp_fk(
            q,
            lift,
            waist,
            params,
        )[:3, 3]

        R = hand_R(
            q,
            lift,
            waist,
            params,
        )

        tcp_error = float(
            np.linalg.norm(
                tcp - target_xyz
            )
        )

        tilt = forearm_tilt(
            q,
            lift,
            waist,
            params,
        )

        vertical_error = (
            gripper_vertical_error(R)
        )

        orientation_drift = float(
            np.linalg.norm(
                rotation_vector(
                    R @ reference_R.T
                )
            )
        )

        qjump = float(
            np.linalg.norm(
                q - reference_q
            )
        )

        # tilt / gripper_vertical 仅用于日志观察，
        # 不再把它们当成 release 可行性的硬条件。
        score = (
            30.0 * tcp_error
            + 0.10 * orientation_drift
            + 0.02 * qjump
        )

        if score < best_score:
            best_score = score
            best = {
                "q": q.copy(),
                "tcp": tcp.copy(),
                "tcp_error_m":
                    tcp_error,
                "forearm_tilt_rad":
                    float(tilt),
                "gripper_vertical_rad":
                    float(vertical_error),
                "orientation_drift_rad":
                    orientation_drift,
                "qjump_rad":
                    qjump,
            }

        # release 的真正硬条件：
        # 1) TCP 到达外退 waypoint；
        # 2) 手部姿态不要相对当前释放姿态漂得太多。
        #
        # 不再要求前臂在第一个 waypoint 就突然变成绝对水平，
        # 也不要求夹爪达到某个绝对竖直角。
        if (
            tcp_error <= max_tcp_error_m
            and orientation_drift <= max_orientation_drift_rad
        ):
            return best

        J = np.zeros(
            (len(residual), 7),
            dtype=float,
        )

        for j in range(7):
            qp = q.copy()
            qp[j] = min(
                upper[j],
                qp[j] + finite_delta,
            )
            if qp[j] == q[j]:
                qp[j] = max(
                    lower[j],
                    q[j] - finite_delta,
                )

            actual_delta = (
                qp[j] - q[j]
            )
            if abs(actual_delta) < 1e-12:
                continue

            J[:, j] = (
                retreat_residual(
                    qp,
                    target_xyz=target_xyz,
                    reference_R=reference_R,
                    reference_q=reference_q,
                    lift=lift,
                    waist=waist,
                    params=params,
                )
                - residual
            ) / actual_delta

        try:
            dq = -np.linalg.solve(
                J.T @ J
                + (damping ** 2)
                * np.eye(7),
                J.T @ residual,
            )
        except np.linalg.LinAlgError:
            break

        max_dq = float(
            np.max(np.abs(dq))
        )
        if max_dq > maximum_step:
            dq *= maximum_step / max_dq

        q_next = np.clip(
            q + dq,
            lower,
            upper,
        )

        if (
            float(
                np.linalg.norm(
                    q_next - q
                )
            )
            < 1e-8
        ):
            break

        q = q_next

    return None


def solve_retreat_waypoint(
    *,
    reference_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_xyz: np.ndarray,
    reference_R: np.ndarray,
    lift: float,
    waist: float,
    params,
    max_tcp_error_m: float,
    max_forearm_tilt_rad: float,
    max_gripper_vertical_rad: float,
    max_orientation_drift_rad: float,
) -> Optional[Dict[str, object]]:
    """
    首先使用上一 waypoint 作为 seed；
    失败时做少量确定性 multi-start。
    """
    first = solve_retreat_waypoint_from_seed(
        seed_q=reference_q,
        reference_q=reference_q,
        lower=lower,
        upper=upper,
        target_xyz=target_xyz,
        reference_R=reference_R,
        lift=lift,
        waist=waist,
        params=params,
        max_tcp_error_m=max_tcp_error_m,
        max_forearm_tilt_rad=
            max_forearm_tilt_rad,
        max_gripper_vertical_rad=
            max_gripper_vertical_rad,
        max_orientation_drift_rad=
            max_orientation_drift_rad,
    )

    if first is not None:
        return first

    rng = np.random.default_rng(
        20260814
    )

    candidates = []

    for sigma in (
        0.020,
        0.040,
        0.070,
    ):
        for _ in range(2):
            trial_seed = np.clip(
                reference_q
                + rng.normal(
                    0.0,
                    sigma,
                    size=7,
                ),
                lower,
                upper,
            )

            sol = (
                solve_retreat_waypoint_from_seed(
                    seed_q=trial_seed,
                    reference_q=reference_q,
                    lower=lower,
                    upper=upper,
                    target_xyz=target_xyz,
                    reference_R=reference_R,
                    lift=lift,
                    waist=waist,
                    params=params,
                    max_tcp_error_m=
                        max_tcp_error_m,
                    max_forearm_tilt_rad=
                        max_forearm_tilt_rad,
                    max_gripper_vertical_rad=
                        max_gripper_vertical_rad,
                    max_orientation_drift_rad=
                        max_orientation_drift_rad,
                )
            )

            if sol is not None:
                candidates.append(sol)

    if not candidates:
        return None

    candidates.sort(
        key=lambda item:
            float(
                item["qjump_rad"]
            )
    )

    return candidates[0]


def plan_outward_retreat(
    controller: DualArmGraspRealController,
    *,
    retreat_m: float,
    path_step_m: float,
    max_tcp_error_m: float,
    max_forearm_tilt_rad: float,
    max_gripper_vertical_rad: float,
    max_orientation_drift_rad: float,
):
    """
    根据当前真实 TCP 自动判断“外侧”方向，
    左右分别退 retreat_m。
    """
    lift = controller.get_pos(
        "base_elevator_joint"
    )
    waist = controller.get_pos(
        "elevator_pitch_joint"
    )

    left_q0 = arm_q(
        controller,
        "left",
    )
    right_q0 = arm_q(
        controller,
        "right",
    )

    left_tcp0 = grasp_tcp_fk(
        left_q0,
        lift,
        waist,
        LEFT_ARM_PARAMS,
    )[:3, 3].astype(float)

    right_tcp0 = grasp_tcp_fk(
        right_q0,
        lift,
        waist,
        RIGHT_ARM_PARAMS,
    )[:3, 3].astype(float)

    span_xy = (
        left_tcp0[:2]
        - right_tcp0[:2]
    )

    span_xy_norm = float(
        np.linalg.norm(span_xy)
    )

    if span_xy_norm < 0.05:
        raise RuntimeError(
            "左右 TCP 水平分离过小，"
            "无法可靠判断箱体外侧方向: "
            f"{span_xy_norm:.4f}m"
        )

    outward_xy = (
        span_xy / span_xy_norm
    )

    left_outward = np.array(
        [
            outward_xy[0],
            outward_xy[1],
            0.0,
        ],
        dtype=float,
    )

    right_outward = -left_outward

    left_R0 = hand_R(
        left_q0,
        lift,
        waist,
        LEFT_ARM_PARAMS,
    )
    right_R0 = hand_R(
        right_q0,
        lift,
        waist,
        RIGHT_ARM_PARAMS,
    )

    left_lower, left_upper = (
        arm_joint_limit_arrays(
            controller._limits,
            "left",
        )
    )
    right_lower, right_upper = (
        arm_joint_limit_arrays(
            controller._limits,
            "right",
        )
    )

    waypoint_count = max(
        1,
        int(
            math.ceil(
                float(retreat_m)
                / max(
                    float(path_step_m),
                    0.0005,
                )
            )
        ),
    )

    left_path = []
    right_path = []

    left_seed = left_q0.copy()
    right_seed = right_q0.copy()

    print("\n外退轨迹规划：")
    print(
        f"  current left TCP  = "
        f"{np.round(left_tcp0, 5)}"
    )
    print(
        f"  current right TCP = "
        f"{np.round(right_tcp0, 5)}"
    )
    print(
        "  outward XY unit   = "
        f"{np.round(outward_xy, 5)}"
    )
    print(
        f"  each hand retreat = "
        f"{retreat_m*100:.1f}cm"
    )
    print(
        f"  Cartesian step    = "
        f"{path_step_m*1000:.1f}mm"
    )

    for i in range(
        1,
        waypoint_count + 1,
    ):
        ratio = (
            i / waypoint_count
        )

        distance = (
            float(retreat_m)
            * ratio
        )

        left_target = (
            left_tcp0
            + left_outward
            * distance
        )

        right_target = (
            right_tcp0
            + right_outward
            * distance
        )

        left_sol = (
            solve_retreat_waypoint(
                reference_q=left_seed,
                lower=left_lower,
                upper=left_upper,
                target_xyz=left_target,
                reference_R=left_R0,
                lift=lift,
                waist=waist,
                params=LEFT_ARM_PARAMS,
                max_tcp_error_m=
                    max_tcp_error_m,
                max_forearm_tilt_rad=
                    max_forearm_tilt_rad,
                max_gripper_vertical_rad=
                    max_gripper_vertical_rad,
                max_orientation_drift_rad=
                    max_orientation_drift_rad,
            )
        )

        if left_sol is None:
            raise RuntimeError(
                "左臂外退 IK 无解: "
                f"waypoint={i}/"
                f"{waypoint_count}, "
                f"distance="
                f"{distance*100:.2f}cm"
            )

        right_sol = (
            solve_retreat_waypoint(
                reference_q=right_seed,
                lower=right_lower,
                upper=right_upper,
                target_xyz=right_target,
                reference_R=right_R0,
                lift=lift,
                waist=waist,
                params=RIGHT_ARM_PARAMS,
                max_tcp_error_m=
                    max_tcp_error_m,
                max_forearm_tilt_rad=
                    max_forearm_tilt_rad,
                max_gripper_vertical_rad=
                    max_gripper_vertical_rad,
                max_orientation_drift_rad=
                    max_orientation_drift_rad,
            )
        )

        if right_sol is None:
            raise RuntimeError(
                "右臂外退 IK 无解: "
                f"waypoint={i}/"
                f"{waypoint_count}, "
                f"distance="
                f"{distance*100:.2f}cm"
            )

        left_seed = np.asarray(
            left_sol["q"],
            dtype=float,
        )
        right_seed = np.asarray(
            right_sol["q"],
            dtype=float,
        )

        left_path.append(
            left_seed.copy()
        )
        right_path.append(
            right_seed.copy()
        )

        print(
            f"  waypoint "
            f"{i:02d}/"
            f"{waypoint_count:02d}: "
            f"out={distance*100:.2f}cm, "
            f"TCPerr L/R="
            f"{float(left_sol['tcp_error_m'])*1000:.1f}/"
            f"{float(right_sol['tcp_error_m'])*1000:.1f}mm, "
            f"forearm L/R="
            f"{math.degrees(float(left_sol['forearm_tilt_rad'])):.2f}/"
            f"{math.degrees(float(right_sol['forearm_tilt_rad'])):.2f}deg"
        )

    left_path = np.asarray(
        left_path,
        dtype=float,
    )
    right_path = np.asarray(
        right_path,
        dtype=float,
    )

    # 最终 FK 再核对一次。
    final_left_tcp = grasp_tcp_fk(
        left_path[-1],
        lift,
        waist,
        LEFT_ARM_PARAMS,
    )[:3, 3]

    final_right_tcp = grasp_tcp_fk(
        right_path[-1],
        lift,
        waist,
        RIGHT_ARM_PARAMS,
    )[:3, 3]

    left_actual_retreat = float(
        np.dot(
            final_left_tcp
            - left_tcp0,
            left_outward,
        )
    )

    right_actual_retreat = float(
        np.dot(
            final_right_tcp
            - right_tcp0,
            right_outward,
        )
    )

    z_change_left = float(
        final_left_tcp[2]
        - left_tcp0[2]
    )
    z_change_right = float(
        final_right_tcp[2]
        - right_tcp0[2]
    )

    print("\n最终外退检查：")
    print(
        f"  left retreat  = "
        f"{left_actual_retreat*100:.2f}cm"
    )
    print(
        f"  right retreat = "
        f"{right_actual_retreat*100:.2f}cm"
    )
    print(
        f"  TCP Z change L/R = "
        f"{z_change_left*1000:+.1f}/"
        f"{z_change_right*1000:+.1f}mm"
    )

    if (
        abs(
            left_actual_retreat
            - retreat_m
        )
        > max_tcp_error_m
        or abs(
            right_actual_retreat
            - retreat_m
        )
        > max_tcp_error_m
    ):
        raise RuntimeError(
            "最终外退距离误差超限"
        )

    if (
        abs(z_change_left)
        > max_tcp_error_m
        or abs(z_change_right)
        > max_tcp_error_m
    ):
        raise RuntimeError(
            "外退过程中最终 TCP 高度变化超限"
        )

    return {
        "left_path":
            left_path,
        "right_path":
            right_path,
        "left_tcp0":
            left_tcp0,
        "right_tcp0":
            right_tcp0,
        "left_final_tcp":
            final_left_tcp,
        "right_final_tcp":
            final_right_tcp,
        "lift":
            float(lift),
        "waist":
            float(waist),
    }


def run_release(
    controller: DualArmGraspRealController,
    *,
    retreat_m: float,
    path_step_m: float,
    execute: bool,
    velocity_min_dt: float,
    smooth_duration_scale: float,
    max_tcp_error_m: float,
    max_forearm_tilt_rad: float,
    max_gripper_vertical_rad: float,
    max_orientation_drift_rad: float,
):
    log = controller.get_logger().info

    log("=" * 70)
    log(
        "释放箱子：先规划完整外退路径，"
        "然后 张开夹爪 -> 双手向外退"
    )
    log(
        f"计划每只手向外退 "
        f"{retreat_m*100:.1f}cm"
    )
    log("=" * 70)

    # 先规划，不能先松箱。
    plan = plan_outward_retreat(
        controller,
        retreat_m=retreat_m,
        path_step_m=path_step_m,
        max_tcp_error_m=
            max_tcp_error_m,
        max_forearm_tilt_rad=
            max_forearm_tilt_rad,
        max_gripper_vertical_rad=
            max_gripper_vertical_rad,
        max_orientation_drift_rad=
            max_orientation_drift_rad,
    )

    if not execute:
        print(
            "\n✅ DRY-RUN 完成："
            "外退路径可行。"
            "没有张开夹爪，也没有发送机械臂运动。"
        )
        print(
            "实际执行顺序将是："
            "夹爪最大张开 -> "
            f"左右各向外退 "
            f"{retreat_m*100:.1f}cm"
        )
        return

    log(
        "▶ [1/2] 左右夹爪先张开到最大 "
        f"{GRIPPER_MAX_OPEN_M:.3f}m"
    )

    if not controller.command_grippers(
        GRIPPER_MAX_OPEN_M
    ):
        raise RuntimeError(
            "双夹爪最大张开失败；"
            "不会执行后续外退"
        )

    # 处理最新 joint state；
    # 张开夹爪不会改变机械臂关节，但仍做一次状态刷新。
    for _ in range(5):
        rclpy.spin_once(
            controller,
            timeout_sec=0.05,
        )

    log(
        "✅ 双夹爪已完成张开命令"
    )

    log(
        "▶ [2/2] 双手同步向各自外侧退，"
        f"每侧约 {retreat_m*100:.1f}cm"
    )

    if not execute_arm_path(
        controller,
        plan["left_path"],
        plan["right_path"],
        velocity_min_dt,
        smooth_duration_scale=
            smooth_duration_scale,
    ):
        raise RuntimeError(
            "双臂向外释放轨迹执行失败"
        )

    # 最终实机 FK 检查。
    lift = controller.get_pos(
        "base_elevator_joint"
    )
    waist = controller.get_pos(
        "elevator_pitch_joint"
    )

    left_q = arm_q(
        controller,
        "left",
    )
    right_q = arm_q(
        controller,
        "right",
    )

    left_tcp = grasp_tcp_fk(
        left_q,
        lift,
        waist,
        LEFT_ARM_PARAMS,
    )[:3, 3]

    right_tcp = grasp_tcp_fk(
        right_q,
        lift,
        waist,
        RIGHT_ARM_PARAMS,
    )[:3, 3]

    original_span_xy = (
        plan["left_tcp0"][:2]
        - plan["right_tcp0"][:2]
    )
    unit = (
        original_span_xy
        / np.linalg.norm(
            original_span_xy
        )
    )

    left_retreat_actual = float(
        np.dot(
            left_tcp[:2]
            - plan["left_tcp0"][:2],
            unit,
        )
    )
    right_retreat_actual = float(
        np.dot(
            right_tcp[:2]
            - plan["right_tcp0"][:2],
            -unit,
        )
    )

    log(
        "✅ 箱子释放动作完成："
        f"左外退={left_retreat_actual*100:.2f}cm, "
        f"右外退={right_retreat_actual*100:.2f}cm"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Robot A 放箱后释放："
            "先双夹爪最大张开，"
            "再左右 grasp TCP 各向外退"
        )
    )

    parser.add_argument(
        "--retreat-cm",
        type=float,
        default=3.0,
        help=(
            "每只夹爪向各自外侧退开的距离(cm)，"
            "默认 3.0cm"
        ),
    )

    parser.add_argument(
        "--path-step-mm",
        type=float,
        default=2.0,
        help=(
            "外退 Cartesian waypoint 间距(mm)，"
            "默认 2mm"
        ),
    )

    parser.add_argument(
        "--urdf",
        default=(
            "/home/niic/project/20260623/"
            "20260116/test_robot_A/src/"
            "robot_A/robot_a_description/"
            "urdf/robot_a.urdf"
        ),
    )

    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=0.20,
        help=(
            "关节速度比例，"
            "释放外退默认 0.20"
        ),
    )

    parser.add_argument(
        "--gripper-effort",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--min-waypoint-dt",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--smooth-duration-scale",
        type=float,
        default=2.0,
        help=(
            "minimum-jerk 时间倍率，"
            "默认 2.0；越大越慢、更柔和"
        ),
    )

    parser.add_argument(
        "--max-tcp-error-mm",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--max-forearm-tilt-deg",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--max-gripper-vertical-deg",
        type=float,
        default=12.0,
    )

    parser.add_argument(
        "--max-orientation-drift-deg",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "实际执行；默认只规划检查，"
            "不会张开夹爪"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.retreat_cm <= 0.0:
        raise SystemExit(
            "❌ --retreat-cm 必须 > 0"
        )

    if args.path_step_mm <= 0.0:
        raise SystemExit(
            "❌ --path-step-mm 必须 > 0"
        )

    if not (
        0.0
        < args.velocity_scale
        <= 1.0
    ):
        raise SystemExit(
            "❌ --velocity-scale 必须位于 (0,1]"
        )

    if args.smooth_duration_scale <= 0:
        raise SystemExit(
            "❌ --smooth-duration-scale 必须 > 0"
        )

    limits = load_urdf_limits(
        args.urdf
    )

    rclpy.init(
        args=sys.argv
    )

    controller = (
        DualArmGraspRealController(
            limits=limits,
            velocity_scale=
                args.velocity_scale,
            gripper_effort=
                args.gripper_effort,
        )
    )

    try:
        if not controller.wait_for_servers():
            raise RuntimeError(
                "部分实机控制器未就绪"
            )

        if not controller.wait_for_state():
            raise RuntimeError(
                "无法获得完整 joint_states"
            )

        run_release(
            controller,
            retreat_m=
                float(args.retreat_cm)
                / 100.0,
            path_step_m=
                float(args.path_step_mm)
                / 1000.0,
            execute=bool(
                args.execute
            ),
            velocity_min_dt=
                float(
                    args.min_waypoint_dt
                ),
            smooth_duration_scale=
                float(
                    args.smooth_duration_scale
                ),
            max_tcp_error_m=
                float(
                    args.max_tcp_error_mm
                ) / 1000.0,
            max_forearm_tilt_rad=
                math.radians(
                    float(
                        args.max_forearm_tilt_deg
                    )
                ),
            max_gripper_vertical_rad=
                math.radians(
                    float(
                        args.max_gripper_vertical_deg
                    )
                ),
            max_orientation_drift_rad=
                math.radians(
                    float(
                        args.max_orientation_drift_deg
                    )
                ),
        )

        return 0

    except KeyboardInterrupt:
        controller.cancel_active_goals()
        print(
            "\n⏹️ 用户中断，"
            "已请求取消活动轨迹"
        )
        return 130

    except Exception as exc:
        if rclpy.ok():
            controller.get_logger().error(
                f"❌ {type(exc).__name__}: "
                f"{exc}"
            )
        else:
            print(
                f"❌ {type(exc).__name__}: "
                f"{exc}",
                file=sys.stderr,
            )
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
