#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot A 抓箱后的指定高度放置脚本。

必须在 box_dual_arm_grasp_v9_template10_shoulder007_prelift025.py
抓取完成、双夹爪已经夹紧箱子之后单独运行。

--target-height 表示 base_link 下左右 grasp TCP 中点的 Z 高度，单位 m。

策略：
1) 优先只动 lift。能只靠 lift 到目标高度时，机械臂完全不动。
2) 若目标过高、lift 单独到不了：
   lift -> URDF 上限；
   lift 保持最高；
   左右臂独立求解一串竖直 Cartesian waypoint；
   每个 waypoint 保持左右 TCP 原相对向量、前臂水平；允许箱体共同XY小幅平移；
   再同步发送左右臂多点轨迹。
3) 执行采用 minimum-jerk S 曲线时间律，起步/停止更柔和。
4) 搬运过程中夹爪持续闭合。
5) 默认到位后仍不松手；只有显式 --release 才张开夹爪。
"""
from __future__ import annotations
import argparse, math, sys
from typing import Dict, Mapping, Optional, Tuple
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory

from box_dual_arm_grasp_v9_template10_shoulder007_prelift025 import (
    DualArmGraspRealController,
    LEFT_ARM_PARAMS, RIGHT_ARM_PARAMS,
    LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS,
    LEFT_ROOT_JOINTS, RIGHT_ROOT_JOINTS,
    arm_fk_frames, arm_joint_limit_arrays,
    grasp_tcp_fk, load_urdf_limits, rotation_vector, to_duration,
)

def arm_q(c, side):
    names = ((LEFT_ROOT_JOINTS + LEFT_ARM_JOINTS) if side == "left"
             else (RIGHT_ROOT_JOINTS + RIGHT_ARM_JOINTS))
    return np.asarray([c.get_pos(n) for n in names], dtype=float)

def forearm_tilt(q, lift, waist, params):
    _, f = arm_fk_frames(np.asarray(q,float), lift, waist, params)
    v = f[6][:3,3] - f[3][:3,3]
    n = float(np.linalg.norm(v))
    return math.pi/2 if n < 1e-9 else math.asin(
        float(np.clip(abs(v[2])/n, 0.0, 1.0))
    )

def hand_R(q, lift, waist, params):
    T, _ = arm_fk_frames(np.asarray(q,float), lift, waist, params)
    return T[:3,:3].copy()

def mirror_bounds(limits):
    ll, lu = arm_joint_limit_arrays(limits, "left")
    rl, ru = arm_joint_limit_arrays(limits, "right")
    lo = np.maximum(rl, -lu)
    hi = np.minimum(ru, -ll)
    if np.any(lo > hi):
        raise RuntimeError("左右镜像联合关节限位无交集")
    return lo, hi

def dual_state(qr, lift, waist):
    qr = np.asarray(qr,float); ql = -qr
    lt = grasp_tcp_fk(ql,lift,waist,LEFT_ARM_PARAMS)[:3,3]
    rt = grasp_tcp_fk(qr,lift,waist,RIGHT_ARM_PARAMS)[:3,3]
    return dict(
        ql=ql, qr=qr, lt=lt, rt=rt,
        lR=hand_R(ql,lift,waist,LEFT_ARM_PARAMS),
        rR=hand_R(qr,lift,waist,RIGHT_ARM_PARAMS),
        ltilt=forearm_tilt(ql,lift,waist,LEFT_ARM_PARAMS),
        rtilt=forearm_tilt(qr,lift,waist,RIGHT_ARM_PARAMS),
    )

def signed_forearm_vertical(q, lift, waist, params):
    _, f = arm_fk_frames(q,lift,waist,params)
    v = f[6][:3,3] - f[3][:3,3]
    n = float(np.linalg.norm(v))
    return 1.0 if n < 1e-9 else float(v[2]/n)

def gripper_vertical_error(R):
    """hand local X 轴偏离世界竖直方向的角度。"""
    axis = np.asarray(R, dtype=float)[:3, 0]
    return math.acos(float(np.clip(abs(axis[2]), 0.0, 1.0)))


def dual_independent_state(ql, qr, lift, waist):
    """
    左右臂独立关节，但都作用在同一个刚体箱上。
    不再要求 q_left = -q_right。
    """
    ql = np.asarray(ql, dtype=float)
    qr = np.asarray(qr, dtype=float)

    lt = grasp_tcp_fk(
        ql, lift, waist, LEFT_ARM_PARAMS
    )[:3, 3].astype(float)
    rt = grasp_tcp_fk(
        qr, lift, waist, RIGHT_ARM_PARAMS
    )[:3, 3].astype(float)

    lR = hand_R(ql, lift, waist, LEFT_ARM_PARAMS)
    rR = hand_R(qr, lift, waist, RIGHT_ARM_PARAMS)

    return {
        "ql": ql,
        "qr": qr,
        "lt": lt,
        "rt": rt,
        "lR": lR,
        "rR": rR,
        "ltilt": forearm_tilt(
            ql, lift, waist, LEFT_ARM_PARAMS
        ),
        "rtilt": forearm_tilt(
            qr, lift, waist, RIGHT_ARM_PARAMS
        ),
    }


def rigid_box_residual_independent(
    x, lift, waist,
    target_mid_z, span0, center_xy0,
    lR0, rR0, ql_ref, qr_ref,
    center_xy_soft_weight,
):
    """
    14 自由度（左7 + 右7）的箱体刚体约束残差。

    硬几何目标：
      1) 左右 TCP 相对向量 L-R 不变
         -> 夹爪间距不变
         -> 左右TCP高度差不变
         -> 箱体不会被双臂拉伸/压缩/扭斜
      2) TCP 中点 Z 到指定高度
      3) 左右前臂水平

    软目标：
      - 箱体共同 XY 尽量少移动
      - 两只手方向尽量接近抓取后的方向
      - 每一步关节变化尽量小

    注意：不再要求 q_left=-q_right。
    """
    x = np.asarray(x, dtype=float)
    ql = x[:7]
    qr = x[7:]
    st = dual_independent_state(
        ql, qr, lift, waist
    )
    center = 0.5 * (st["lt"] + st["rt"])
    span = st["lt"] - st["rt"]

    return np.concatenate([
        # 刚体箱最重要的相对向量约束
        14.0 * (span - span0),

        # 指定高度
        np.array([
            14.0 * (
                float(center[2])
                - float(target_mid_z)
            )
        ]),

        # 两侧前臂水平
        1.40 * np.array([
            signed_forearm_vertical(
                ql, lift, waist, LEFT_ARM_PARAMS
            ),
            signed_forearm_vertical(
                qr, lift, waist, RIGHT_ARM_PARAMS
            ),
        ]),

        # 箱子共同XY允许变化，但尽量少变
        float(center_xy_soft_weight) * (
            center[:2] - np.asarray(
                center_xy0, dtype=float
            )
        ),

        # 手部完整方向仅软约束
        0.020 * rotation_vector(
            st["lR"] @ lR0.T
        ),
        0.020 * rotation_vector(
            st["rR"] @ rR0.T
        ),

        # 相邻 waypoint 连续性
        0.004 * (
            ql - np.asarray(ql_ref, dtype=float)
        ),
        0.004 * (
            qr - np.asarray(qr_ref, dtype=float)
        ),
    ])


def joint_limit_report(
    ql, qr, left_lower, left_upper,
    right_lower, right_upper,
):
    """返回最接近 URDF 限位的若干关节。"""
    names_left = LEFT_ROOT_JOINTS + LEFT_ARM_JOINTS
    names_right = RIGHT_ROOT_JOINTS + RIGHT_ARM_JOINTS
    rows = []

    for side, names, q, lo, hi in (
        ("L", names_left, ql, left_lower, left_upper),
        ("R", names_right, qr, right_lower, right_upper),
    ):
        for i, name in enumerate(names):
            low_margin = float(q[i] - lo[i])
            high_margin = float(hi[i] - q[i])
            margin = min(low_margin, high_margin)
            rows.append(
                (
                    margin,
                    side,
                    name,
                    float(q[i]),
                    float(lo[i]),
                    float(hi[i]),
                )
            )

    rows.sort(key=lambda r: r[0])
    return rows[:6]


def _solve_waypoint_independent_from_seed(
    x_seed, path_ql_ref, path_qr_ref,
    left_lower, left_upper,
    right_lower, right_upper,
    lift, waist,
    target_mid_z, span0, center_xy0,
    lR0, rR0,
    height_tol, span_tol, tilt_tol,
    gripper_vertical_tol,
    max_common_xy_shift_m,
    center_xy_soft_weight,
):
    lower = np.concatenate(
        [left_lower, right_lower]
    )
    upper = np.concatenate(
        [left_upper, right_upper]
    )

    x = np.clip(
        np.asarray(x_seed, dtype=float).copy(),
        lower,
        upper,
    )

    finite_delta = 1e-5
    damping = 0.020
    maximum_step = 0.10

    best = None
    best_score = float("inf")

    for _ in range(360):
        rr = rigid_box_residual_independent(
            x, lift, waist,
            target_mid_z, span0, center_xy0,
            lR0, rR0,
            path_ql_ref, path_qr_ref,
            center_xy_soft_weight,
        )

        ql = x[:7]
        qr = x[7:]
        st = dual_independent_state(
            ql, qr, lift, waist
        )

        center = 0.5 * (
            st["lt"] + st["rt"]
        )
        span = st["lt"] - st["rt"]

        zerr = abs(
            float(center[2])
            - float(target_mid_z)
        )
        sve = float(
            np.linalg.norm(span - span0)
        )
        sle = abs(
            float(np.linalg.norm(span))
            - float(np.linalg.norm(span0))
        )
        xyshift = float(
            np.linalg.norm(
                center[:2] - center_xy0
            )
        )

        lgve = gripper_vertical_error(
            st["lR"]
        )
        rgve = gripper_vertical_error(
            st["rR"]
        )

        qjump = float(
            np.linalg.norm(
                ql - path_ql_ref
            )
            + np.linalg.norm(
                qr - path_qr_ref
            )
        )

        score = (
            25.0 * zerr
            + 35.0 * sve
            + 4.0 * (
                st["ltilt"] + st["rtilt"]
            )
            + 0.5 * xyshift
            + 0.04 * (lgve + rgve)
            + 0.015 * qjump
        )

        if score < best_score:
            best_score = score
            best = {
                "ql": ql.copy(),
                "qr": qr.copy(),
                "s": st,
                "zerr": zerr,
                "sve": sve,
                "sle": sle,
                "xyshift": xyshift,
                "lgve": lgve,
                "rgve": rgve,
                "qjump": qjump,
            }

        if (
            zerr <= height_tol
            and sve <= span_tol
            and sle <= span_tol
            and st["ltilt"] <= tilt_tol
            and st["rtilt"] <= tilt_tol
            and lgve <= gripper_vertical_tol
            and rgve <= gripper_vertical_tol
            and xyshift <= max_common_xy_shift_m
        ):
            return best

        J = np.zeros(
            (len(rr), 14),
            dtype=float,
        )

        for j in range(14):
            xp = x.copy()
            xp[j] = min(
                upper[j],
                xp[j] + finite_delta,
            )
            if xp[j] == x[j]:
                xp[j] = max(
                    lower[j],
                    x[j] - finite_delta,
                )

            dd = xp[j] - x[j]
            if abs(dd) < 1e-12:
                continue

            J[:, j] = (
                rigid_box_residual_independent(
                    xp, lift, waist,
                    target_mid_z,
                    span0, center_xy0,
                    lR0, rR0,
                    path_ql_ref, path_qr_ref,
                    center_xy_soft_weight,
                ) - rr
            ) / dd

        try:
            dq = -np.linalg.solve(
                J.T @ J
                + (damping ** 2) * np.eye(14),
                J.T @ rr,
            )
        except np.linalg.LinAlgError:
            break

        m = float(
            np.max(np.abs(dq))
        )
        if m > maximum_step:
            dq *= maximum_step / m

        xn = np.clip(
            x + dq,
            lower,
            upper,
        )

        if (
            float(np.linalg.norm(xn - x))
            < 1e-8
        ):
            break

        x = xn

    return None


def solve_waypoint_independent(
    path_ql_ref, path_qr_ref,
    left_lower, left_upper,
    right_lower, right_upper,
    lift, waist,
    target_mid_z, span0, center_xy0,
    lR0, rR0,
    height_tol, span_tol, tilt_tol,
    gripper_vertical_tol,
    max_common_xy_shift_m,
    center_xy_soft_weight,
):
    """
    优先从上一 waypoint 连续求解。
    失败后才对左右臂分别做小扰动 multi-start。
    """
    x0 = np.concatenate([
        np.asarray(path_ql_ref, dtype=float),
        np.asarray(path_qr_ref, dtype=float),
    ])

    sol = _solve_waypoint_independent_from_seed(
        x0,
        path_ql_ref,
        path_qr_ref,
        left_lower, left_upper,
        right_lower, right_upper,
        lift, waist,
        target_mid_z,
        span0, center_xy0,
        lR0, rR0,
        height_tol, span_tol, tilt_tol,
        gripper_vertical_tol,
        max_common_xy_shift_m,
        center_xy_soft_weight,
    )
    if sol is not None:
        return sol

    rng = np.random.default_rng(20260814)
    candidates = []

    for sigma in (0.025, 0.050, 0.090):
        for _ in range(3):
            trial = x0.copy()
            trial[:7] += rng.normal(
                0.0, sigma, size=7
            )
            trial[7:] += rng.normal(
                0.0, sigma, size=7
            )

            sol = _solve_waypoint_independent_from_seed(
                trial,
                path_ql_ref,
                path_qr_ref,
                left_lower, left_upper,
                right_lower, right_upper,
                lift, waist,
                target_mid_z,
                span0, center_xy0,
                lR0, rR0,
                height_tol, span_tol, tilt_tol,
                gripper_vertical_tol,
                max_common_xy_shift_m,
                center_xy_soft_weight,
            )
            if sol is not None:
                candidates.append(sol)

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: float(
            item["qjump"]
        )
    )
    return candidates[0]


def plan_vertical_path(
    ql0, qr0,
    lift, target_mid_z,
    limits, step_m,
    height_tol, span_tol,
    tilt_tol, gripper_vertical_tol,
    max_common_xy_shift_m,
    center_xy_soft_weight,
):
    """
    V3：双臂独立关节 + 同一个刚体箱约束。

    不再强制 q_left=-q_right。
    真正硬约束的是箱体：
      - 左右 TCP 相对向量固定
      - 夹爪间距固定
      - 左右 TCP 高度差固定
      - 中点 Z 按路径上升
      - 两侧前臂水平

    这样左右臂可以采用不同的关节角来共同托住同一个箱子。
    """
    left_lower, left_upper = (
        arm_joint_limit_arrays(
            limits, "left"
        )
    )
    right_lower, right_upper = (
        arm_joint_limit_arrays(
            limits, "right"
        )
    )

    ql = np.clip(
        np.asarray(ql0, dtype=float),
        left_lower,
        left_upper,
    )
    qr = np.clip(
        np.asarray(qr0, dtype=float),
        right_lower,
        right_upper,
    )

    s0 = dual_independent_state(
        ql, qr, lift, 0.0
    )

    span0 = (
        s0["lt"] - s0["rt"]
    )
    center0 = 0.5 * (
        s0["lt"] + s0["rt"]
    )
    center_xy0 = center0[:2].copy()

    start_z = float(center0[2])
    target_z = float(target_mid_z)

    direction = (
        1.0
        if target_z >= start_z
        else -1.0
    )

    nominal_step = max(
        float(step_m),
        0.001,
    )
    current_step = nominal_step
    min_step = 0.001

    left_path = []
    right_path = []

    ql_seed = ql.copy()
    qr_seed = qr.copy()
    current_z = start_z
    waypoint_index = 0

    while (
        direction
        * (target_z - current_z)
        > 1e-6
    ):
        remaining = abs(
            target_z - current_z
        )
        step = min(
            current_step,
            remaining,
        )
        next_z = (
            current_z
            + direction * step
        )

        sol = solve_waypoint_independent(
            ql_seed, qr_seed,
            left_lower, left_upper,
            right_lower, right_upper,
            lift, 0.0,
            next_z,
            span0,
            center_xy0,
            s0["lR"],
            s0["rR"],
            height_tol,
            span_tol,
            tilt_tol,
            gripper_vertical_tol,
            max_common_xy_shift_m,
            center_xy_soft_weight,
        )

        if sol is None:
            if current_step > min_step + 1e-9:
                current_step = max(
                    min_step,
                    0.5 * current_step,
                )
                print(
                    f"  ↪ Z={next_z:.4f}m "
                    "独立双臂连续解失败，"
                    f"步长缩小到 "
                    f"{current_step*1000:.1f}mm"
                )
                continue

            # 最小步长仍失败，打印当前路径末端的关节限位余量。
            report = joint_limit_report(
                ql_seed,
                qr_seed,
                left_lower,
                left_upper,
                right_lower,
                right_upper,
            )
            print(
                "\n最接近关节限位的关节："
            )
            for (
                margin,
                side,
                name,
                value,
                lower,
                upper,
            ) in report:
                print(
                    f"  {side} {name}: "
                    f"q={value:+.4f}, "
                    f"limit=[{lower:+.4f},"
                    f"{upper:+.4f}], "
                    f"margin={margin:.4f}rad"
                )

            raise RuntimeError(
                "双臂独立刚体约束 IK 在 1mm "
                "最小步长仍无解；"
                f"已规划到 Z={current_z:.4f}m，"
                f"下一目标 Z={next_z:.4f}m。"
            )

        ql_seed = np.asarray(
            sol["ql"], dtype=float
        )
        qr_seed = np.asarray(
            sol["qr"], dtype=float
        )

        left_path.append(
            ql_seed.copy()
        )
        right_path.append(
            qr_seed.copy()
        )

        current_z = next_z
        waypoint_index += 1

        current_step = min(
            nominal_step,
            max(
                current_step,
                min_step,
            ) * 1.35,
        )

        st = sol["s"]
        mirror_error = float(
            np.max(
                np.abs(
                    ql_seed + qr_seed
                )
            )
        )

        print(
            f"waypoint {waypoint_index:03d}: "
            f"Z={current_z:.4f}m, "
            f"heightErr="
            f"{sol['zerr']*1000:.1f}mm, "
            f"spanErr="
            f"{sol['sve']*1000:.1f}mm, "
            f"commonXYshift="
            f"{sol['xyshift']*100:.1f}cm, "
            f"forearm="
            f"{math.degrees(st['ltilt']):.2f}/"
            f"{math.degrees(st['rtilt']):.2f}deg, "
            f"gripperVertical="
            f"{math.degrees(sol['lgve']):.1f}/"
            f"{math.degrees(sol['rgve']):.1f}deg, "
            f"mirrorErr(now optional)="
            f"{mirror_error:.3f}rad"
        )

    return (
        np.asarray(left_path, dtype=float),
        np.asarray(right_path, dtype=float),
        span0,
    )

def _minimum_jerk_s(u):
    """
    五次 minimum-jerk 时间律：
        s(0)=0, s(1)=1
        起点/终点速度=0
        起点/终点加速度=0
    """
    u = float(np.clip(u, 0.0, 1.0))
    return (
        10.0 * u**3
        - 15.0 * u**4
        + 6.0 * u**5
    )


def _inverse_minimum_jerk(s_target):
    """
    已知几何路径进度 s，反求 minimum-jerk 时间进度 u。
    用确定性的二分法，避免额外依赖。
    """
    target = float(np.clip(s_target, 0.0, 1.0))
    if target <= 0.0:
        return 0.0
    if target >= 1.0:
        return 1.0

    lo = 0.0
    hi = 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _minimum_jerk_s(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _build_goal_with_times(
    c,
    joint_names,
    points,
    times,
):
    """
    和原控制器 _build_goal() 相同的数据结构，
    但允许每个 waypoint 使用非均匀时间戳。
    """
    points = np.asarray(points, dtype=float)
    times = np.asarray(times, dtype=float)

    if points.shape[0] != len(times):
        raise RuntimeError("points/times 数量不一致")

    goal = FollowJointTrajectory.Goal()
    # 复用控制器原方法生成一个空壳，再替换点，
    # 避免重复依赖 trajectory_msgs 的 import。
    dummy = c._build_goal(
        joint_names,
        points[:1],
        max(float(times[-1]), 0.2),
    )
    goal.trajectory = dummy.trajectory
    goal.trajectory.joint_names = list(joint_names)
    goal.trajectory.points = []

    from trajectory_msgs.msg import JointTrajectoryPoint

    for row, timestamp in zip(points, times):
        p = JointTrajectoryPoint()
        p.positions = [
            float(v) for v in row
        ]
        sec, nsec = to_duration(
            float(timestamp)
        )
        p.time_from_start.sec = sec
        p.time_from_start.nanosec = nsec
        goal.trajectory.points.append(p)

    return goal


def execute_arm_path(
    c,
    lp,
    rp,
    min_dt,
    smooth_duration_scale=1.80,
):
    """
    V4 平滑执行：

    1) 几何 waypoint 完全使用 V3 规划结果，不修改任何关节目标；
       因此不会为了“平滑”破坏 TCP 相对向量/前臂水平约束。

    2) 只修改 waypoint 的时间戳：
       使用 minimum-jerk 五次 S 曲线，
       起步和停止的速度/加速度都趋近 0。

    3) 根据每个真实关节的 URDF 最大速度反推最小总时长，
       防止 S 曲线中段速度过快。

    smooth_duration_scale 默认 1.8：
       相比原等时间执行更柔和，且不会为了 S 曲线中段加速而明显提高峰值速度。
    """
    lp = np.asarray(lp, float)
    rp = np.asarray(rp, float)

    if (
        lp.shape != rp.shape
        or lp.ndim != 2
        or lp.shape[1] != 7
    ):
        raise RuntimeError(
            f"非法路径 shape L={lp.shape}, R={rp.shape}"
        )
    if len(lp) == 0:
        return True

    lnames = LEFT_ROOT_JOINTS + LEFT_ARM_JOINTS
    rnames = RIGHT_ROOT_JOINTS + RIGHT_ARM_JOINTS
    lstart = arm_q(c, "left")
    rstart = arm_q(c, "right")

    n = len(lp)

    # 几何 waypoint 是沿路径顺序排列的。
    # 对第 i 个 waypoint 的几何进度 s=i/N，
    # 反求 minimum-jerk 对应的时间进度 u。
    u = np.asarray(
        [
            _inverse_minimum_jerk(
                (i + 1) / n
            )
            for i in range(n)
        ],
        dtype=float,
    )

    # 原版本的基准总时间约为 min_dt * waypoint_count。
    # V4 默认乘 1.8，使 minimum-jerk 中间段不会显著快于旧版。
    requested_total = max(
        float(min_dt)
        * n
        * float(smooth_duration_scale),
        1.0,
    )

    # 用 URDF 速度限位校正总时长。
    # 对每一小段：
    #   segment_time = total_time * delta_u
    # 必须 >= 1.8 * |delta_q| / allowed_velocity
    required_total = 0.0

    for names, start, path in (
        (lnames, lstart, lp),
        (rnames, rstart, rp),
    ):
        prev_q = np.asarray(
            start, dtype=float
        )
        prev_u = 0.0

        for row, current_u in zip(
            path, u
        ):
            du = float(
                current_u - prev_u
            )
            if du <= 1e-9:
                raise RuntimeError(
                    "minimum-jerk 时间序列退化"
                )

            for j, name in enumerate(names):
                vmax = (
                    float(c._limits[name][2])
                    * c._velocity_scale
                )
                if vmax <= 0.0:
                    continue

                segment_required = (
                    1.8
                    * abs(
                        float(row[j])
                        - float(prev_q[j])
                    )
                    / vmax
                )
                required_total = max(
                    required_total,
                    segment_required / du,
                )

            prev_q = row
            prev_u = float(current_u)

    total = max(
        requested_total,
        required_total,
        1.0,
    )
    times = total * u

    # 额外检查时间必须严格递增。
    if (
        np.any(np.diff(times) <= 0.0)
        or times[0] <= 0.0
    ):
        raise RuntimeError(
            "生成的 S 曲线时间戳非法"
        )

    groups = {
        "left_root": (
            c._left_root,
            LEFT_ROOT_JOINTS,
            lp[:, :1],
        ),
        "right_root": (
            c._right_root,
            RIGHT_ROOT_JOINTS,
            rp[:, :1],
        ),
        "left_arm": (
            c._left_arm,
            LEFT_ARM_JOINTS,
            lp[:, 1:],
        ),
        "right_arm": (
            c._right_arm,
            RIGHT_ARM_JOINTS,
            rp[:, 1:],
        ),
    }

    sends = {}
    for label, (
        client,
        names,
        pts,
    ) in groups.items():
        goal = _build_goal_with_times(
            c,
            names,
            pts,
            times,
        )
        sends[label] = (
            client.send_goal_async(goal)
        )

    c.get_logger().warning(
        "发送双臂受约束 minimum-jerk 多点轨迹: "
        f"waypoints={n}, "
        f"duration={total:.2f}s, "
        f"first_dt={times[0]:.3f}s, "
        f"middle_dt≈"
        f"{np.median(np.diff(times)):.3f}s, "
        f"last_dt="
        f"{times[-1]-times[-2]:.3f}s"
    )

    if not c._wait_all(
        sends.values(),
        12.0,
    ):
        return False

    handles = {}
    for label, future in sends.items():
        handle = future.result()
        if (
            handle is None
            or not handle.accepted
        ):
            c.get_logger().error(
                f"{label}: 轨迹被拒绝"
            )
        else:
            handles[label] = handle

    if len(handles) != len(groups):
        for handle in handles.values():
            try:
                handle.cancel_goal_async()
            except Exception:
                pass
        return False

    c._active_goal_handles = list(
        handles.values()
    )

    results = {
        key: handle.get_result_async()
        for key, handle in handles.items()
    }

    if not c._wait_all(
        results.values(),
        total + 20.0,
    ):
        c.cancel_active_goals()
        return False

    ok = True
    for label, future in results.items():
        wrapped = future.result()
        good = (
            wrapped is not None
            and wrapped.status
            == GoalStatus.STATUS_SUCCEEDED
            and wrapped.result.error_code
            == FollowJointTrajectory.Result.SUCCESSFUL
        )
        c.get_logger().info(
            ("✅ " if good else "❌ ")
            + f"{label}: 平滑轨迹完成"
        )
        ok = ok and good

    c._active_goal_handles = []

    if ok:
        for _ in range(6):
            rclpy.spin_once(
                c,
                timeout_sec=0.05,
            )

    return ok

def verify(c, span0, target_z, ztol, spantol, tilttol):
    lift=c.get_pos("base_elevator_joint")
    waist=c.get_pos("elevator_pitch_joint")
    lq=arm_q(c,"left"); rq=arm_q(c,"right")
    lt=grasp_tcp_fk(lq,lift,waist,LEFT_ARM_PARAMS)[:3,3]
    rt=grasp_tcp_fk(rq,lift,waist,RIGHT_ARM_PARAMS)[:3,3]
    span=lt-rt
    sve=float(np.linalg.norm(span-span0))
    sle=abs(float(np.linalg.norm(span))-float(np.linalg.norm(span0)))
    midz=0.5*(float(lt[2])+float(rt[2]))
    ltilt=forearm_tilt(lq,lift,waist,LEFT_ARM_PARAMS)
    rtilt=forearm_tilt(rq,lift,waist,RIGHT_ARM_PARAMS)
    if sve>spantol or sle>spantol:
        raise RuntimeError(
            f"夹爪间距/相对向量变化超限: vec={sve*1000:.1f}mm, "
            f"len={sle*1000:.1f}mm"
        )
    if ltilt>tilttol or rtilt>tilttol:
        raise RuntimeError(
            f"前臂不水平: L={math.degrees(ltilt):.2f}°, "
            f"R={math.degrees(rtilt):.2f}°"
        )
    if target_z is not None and abs(midz-target_z)>ztol:
        raise RuntimeError(
            f"目标高度误差过大: actual={midz:.4f}, target={target_z:.4f}"
        )
    return dict(midz=midz,span=float(np.linalg.norm(span)),
                ltilt=ltilt,rtilt=rtilt,lift=lift)

def run(c, target_z, execute, release, step_m, tcp_tol, span_tol,
        tilt_tol, orient_tol, ztol, mirror_tol, initial_z_mismatch_tol,
        min_dt, max_common_xy_shift_m, center_xy_soft_weight,
        smooth_duration_scale):
    lift=c.get_pos("base_elevator_joint")
    waist=c.get_pos("elevator_pitch_joint")
    lq=arm_q(c,"left"); rq=arm_q(c,"right")
    lt=grasp_tcp_fk(lq,lift,waist,LEFT_ARM_PARAMS)[:3,3]
    rt=grasp_tcp_fk(rq,lift,waist,RIGHT_ARM_PARAMS)[:3,3]
    span0=lt-rt
    span_len=float(np.linalg.norm(span0))
    midz=0.5*(float(lt[2])+float(rt[2]))
    zmis=abs(float(lt[2])-float(rt[2]))
    merr=float(np.max(np.abs(lq+rq)))
    ltilt=forearm_tilt(lq,lift,waist,LEFT_ARM_PARAMS)
    rtilt=forearm_tilt(rq,lift,waist,RIGHT_ARM_PARAMS)

    log=c.get_logger().info
    log("="*70)
    log(f"当前 lift={lift:+.4f}m, waist={waist:+.4f}rad")
    log(f"当前 TCP L={np.round(lt,5)}, R={np.round(rt,5)}")
    log(f"TCP中点Z={midz:.4f}m -> 目标Z={target_z:.4f}m")
    log(f"夹爪间距={span_len:.4f}m, 左右Z差={zmis*1000:.1f}mm")
    log(f"前臂倾角 L/R={math.degrees(ltilt):.2f}/{math.degrees(rtilt):.2f}deg")
    log(f"镜像误差 max|qL+qR|={merr:.5f}rad")
    log("="*70)

    if zmis>initial_z_mismatch_tol:
        raise RuntimeError("初始左右TCP高度差过大，箱体可能已不水平")
    if ltilt>tilt_tol or rtilt>tilt_tol:
        raise RuntimeError("初始前臂不满足水平阈值，拒绝搬箱")

    lmin,lmax,_=c._limits["base_elevator_joint"]
    lift_only=lift+(float(target_z)-midz)

    # A: 只调 lift
    if lmin<=lift_only<=lmax:
        log("✅ 可只调 lift；双臂所有关节保持不动")
        log(f"lift {lift:+.4f} -> {lift_only:+.4f}m")
        if not execute:
            print("[DRY-RUN] lift-only 可达；不会改变夹爪距离/前臂姿态。")
            return
        if not c.command_grippers(0.0):
            raise RuntimeError("无法保持双夹爪闭合")
        if not c.animate({"base_elevator_joint":float(lift_only)},
                         duration=2.0,steps=100):
            raise RuntimeError("lift-only 执行失败")
        v=verify(c,span0,target_z,ztol,span_tol,tilt_tol)
        log(f"✅ 到位 midZ={v['midz']:.4f}m, span={v['span']:.4f}m")
        if release:
            log("▶ --release：张开双夹爪")
            if not c.command_grippers(0.025):
                raise RuntimeError("释放箱体失败")
        return

    if lift_only<lmin:
        raise RuntimeError(
            f"目标过低：需要 lift={lift_only:+.4f}m < 下限 {lmin:+.4f}m；"
            "本版本按需求只对“目标过高”做机械臂补偿。"
        )
    if abs(waist)>0.02:
        raise RuntimeError(f"高位双臂补偿要求 waist≈0，当前={waist:+.4f}rad")

    log("⚠️ 目标过高：lift-only 不可达")
    if execute:
        if not c.command_grippers(0.0):
            raise RuntimeError("无法保持双夹爪闭合")
        log(f"▶ [1/3] lift 先升至最高 {lmax:+.4f}m")
        if not c.animate({"base_elevator_joint":float(lmax)},
                         duration=2.0,steps=100):
            raise RuntimeError("lift 升最高失败")
        for _ in range(5): rclpy.spin_once(c,timeout_sec=0.05)
        planning_lift=c.get_pos("base_elevator_joint")
        if abs(planning_lift-lmax)>0.005:
            raise RuntimeError("lift 最高位反馈误差过大")
        rq=arm_q(c,"right")
        lq=arm_q(c,"left")
    else:
        planning_lift=float(lmax)
        print(f"[DRY-RUN] 虚拟 lift={planning_lift:.4f}m")

    # V3 不再把真实抓取姿态强制投影回精确镜像姿态。
    # 直接以 lift 最高位的真实左右关节状态作为刚体路径起点，
    # 因此不会在进入机械臂补高阶段前发生“镜像校正跳变”。
    log("▶ [2/3] 计算夹爪间距固定 + 前臂水平的双臂刚体路径")
    lp,rp,path_span0=plan_vertical_path(
        lq, rq,
        planning_lift,
        target_z,
        c._limits,
        step_m,
        tcp_tol,
        span_tol,
        tilt_tol,
        orient_tol,
        max_common_xy_shift_m,
        center_xy_soft_weight,
    )
    print(f"✅ 受约束路径规划完成：{len(rp)} 个 waypoint")

    if not execute:
        print("✅ DRY-RUN 完成；未发送实机命令。加 --execute 后执行。")
        return

    log("▶ [3/3] lift 保持最高，同步执行双臂多点轨迹")
    if not execute_arm_path(
        c, lp, rp, min_dt,
        smooth_duration_scale=smooth_duration_scale,
    ):
        raise RuntimeError("受约束双臂轨迹执行失败")

    v=verify(c,path_span0,target_z,ztol,span_tol,tilt_tol)
    # 额外确认最终“距离长度”相对抓箱完成时没有改变。
    if abs(v["span"]-span_len)>span_tol:
        raise RuntimeError(
            f"最终夹爪距离相对抓取结束变化 "
            f"{abs(v['span']-span_len)*1000:.1f}mm"
        )
    log(
        f"✅ 放置高度到位 midZ={v['midz']:.4f}m, "
        f"span={v['span']:.4f}m, "
        f"forearm={math.degrees(v['ltilt']):.2f}/"
        f"{math.degrees(v['rtilt']):.2f}deg"
    )
    if release:
        log("▶ --release：确认箱体已由放置面承重后张开双夹爪")
        if not c.command_grippers(0.025):
            raise RuntimeError("释放箱体失败")


def parse_args():
    ap=argparse.ArgumentParser(
        description="抓箱后指定高度放置：优先 lift-only；高位时 lift=max + 双臂刚性约束路径"
    )
    ap.add_argument("-z","--target-height",type=float,required=True,
                    help="base_link 下左右 grasp TCP 中点目标 Z，高度单位 m")
    ap.add_argument("--urdf",default=(
        "/home/niic/project/20260623/20260116/test_robot_A/"
        "src/robot_A/robot_a_description/urdf/robot_a.urdf"))
    ap.add_argument("--velocity-scale",type=float,default=0.30)
    ap.add_argument("--gripper-effort",type=float,default=50.0)
    ap.add_argument(
        "--path-step-mm",
        type=float,
        default=2.0,
        help=(
            "Cartesian 约束 waypoint 最大步长(mm)。"
            "越小越能降低相邻点关节插值造成的夹爪几何漂移；"
            "它不是速度参数。V4 默认 2mm"
        ),
    )
    ap.add_argument("--max-tcp-error-mm",type=float,default=4.0)
    ap.add_argument("--max-span-error-mm",type=float,default=3.0)
    ap.add_argument("--max-forearm-tilt-deg",type=float,default=3.0)
    ap.add_argument(
        "--max-orientation-drift-deg",
        type=float,
        default=10.0,
        help=(
            "兼容旧参数名：V2 中表示 hand local X 偏离世界竖直的"
            "最大角度，而不再锁死完整手部姿态"
        ),
    )
    ap.add_argument(
        "--max-common-xy-shift-cm",
        type=float,
        default=12.0,
        help=(
            "机械臂补高时允许箱子整体在水平面共同移动的最大距离。"
            "夹爪相对向量/间距仍保持不变"
        ),
    )
    ap.add_argument(
        "--center-xy-soft-weight",
        type=float,
        default=0.20,
        help=(
            "箱体共同XY位置软约束权重；越大越倾向保持原XY，"
            "但过大可能再次造成高位无解"
        ),
    )
    ap.add_argument("--max-height-error-mm",type=float,default=5.0)
    ap.add_argument("--max-initial-mirror-error-deg",type=float,default=2.0)
    ap.add_argument("--max-initial-height-mismatch-mm",type=float,default=5.0)
    ap.add_argument("--min-waypoint-dt",type=float,default=0.12)
    ap.add_argument(
        "--smooth-duration-scale",
        type=float,
        default=1.80,
        help=(
            "minimum-jerk 总时长倍率。"
            "越大越慢、更柔和；默认 1.8。"
            "建议范围 1.5~2.5"
        ),
    )
    ap.add_argument("--execute",action="store_true",
                    help="实际执行；默认只规划/检查")
    ap.add_argument("--release",action="store_true",
                    help="到位后张开双夹爪；默认保持夹紧")
    return ap.parse_args()

def main():
    a=parse_args()
    if not np.isfinite(a.target_height):
        raise SystemExit("target-height 必须是有限数值")
    if a.path_step_mm<=0 or not (0<a.velocity_scale<=1):
        raise SystemExit("参数非法")
    if a.smooth_duration_scale <= 0.0:
        raise SystemExit("--smooth-duration-scale 必须 > 0")
    limits=load_urdf_limits(a.urdf)
    rclpy.init(args=sys.argv)
    c=DualArmGraspRealController(
        limits=limits,
        velocity_scale=a.velocity_scale,
        gripper_effort=a.gripper_effort,
    )
    try:
        if not c.wait_for_servers():
            raise RuntimeError("部分控制器未就绪")
        if not c.wait_for_state():
            raise RuntimeError("无法获得完整 joint_states")
        run(
            c,
            target_z=float(a.target_height),
            execute=bool(a.execute),
            release=bool(a.release),
            step_m=a.path_step_mm/1000.0,
            tcp_tol=a.max_tcp_error_mm/1000.0,
            span_tol=a.max_span_error_mm/1000.0,
            tilt_tol=math.radians(a.max_forearm_tilt_deg),
            orient_tol=math.radians(a.max_orientation_drift_deg),
            ztol=a.max_height_error_mm/1000.0,
            mirror_tol=math.radians(a.max_initial_mirror_error_deg),
            initial_z_mismatch_tol=a.max_initial_height_mismatch_mm/1000.0,
            min_dt=float(a.min_waypoint_dt),
            max_common_xy_shift_m=
                float(a.max_common_xy_shift_cm) / 100.0,
            center_xy_soft_weight=
                float(a.center_xy_soft_weight),
            smooth_duration_scale=
                float(a.smooth_duration_scale),
        )
        return 0
    except KeyboardInterrupt:
        c.cancel_active_goals()
        print("\n⏹️ 用户中断，已请求取消活动轨迹")
        return 130
    except Exception as e:
        if rclpy.ok():
            c.get_logger().error(f"❌ {type(e).__name__}: {e}")
        else:
            print(f"❌ {type(e).__name__}: {e}",file=sys.stderr)
        return 1
    finally:
        try: c.destroy_node()
        except Exception: pass
        if rclpy.ok(): rclpy.shutdown()

if __name__=="__main__":
    raise SystemExit(main())
