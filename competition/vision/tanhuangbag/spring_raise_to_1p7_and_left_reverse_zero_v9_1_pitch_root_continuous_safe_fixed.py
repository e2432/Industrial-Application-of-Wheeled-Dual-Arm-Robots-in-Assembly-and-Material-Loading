#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot A 弹簧抓取后：主动抬升 pitch_arm_right1_joint 到约1.7m + 连续性安全约束 + 左臂逆序归零

执行顺序：
1) 不向右夹爪发送任何新命令，保持原抓取闭合状态；
2) Lift 先独立升到 URDF 上限；
3) Lift 保持最高，主动增加 pitch_arm_right1_joint：
   - TCP 不要求垂直上升，X/Y允许自然变化；
   - hand 完整姿态保持抓取时姿态；
   - 前臂不要求完全水平，只要求倾角不超过默认12deg；
   - 其余6个右臂关节由IK配合补偿；
4) 右臂保持高位，左臂按原抓取路径逆序归零：
   A. 撤销左 root 额外 -0.10rad；
   B. 左肩 arm_left1_arm_left2_joint 从0恢复到 +0.7rad；
   C. 肩保持+0.7rad，其余左臂关节逐渐回0；
   D. 最后肩 +0.7rad -> 0。
5) 最终复核。

默认 DRY-RUN；只有 --execute 才真实运动。
本脚本不会张开右夹爪。
"""

from __future__ import annotations
import argparse
import math
import sys
from typing import Dict, Optional

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.duration import Duration

from ik_grasp_dual_mirror_real_v9_grasp_verify_retry import (
    IKGraspRealController,
    load_urdf_limits,
    LIFT_JOINT,
    WAIST_JOINT,
    RIGHT_ROOT_JOINT,
    RIGHT_ARM_JOINTS,
    RIGHT_MOTION_JOINTS,
    LEFT_ROOT_JOINT,
    LEFT_ARM_JOINTS,
    LEFT_MOTION_JOINTS,
    LEFT_ROOT_EXTRA_LIFT_RAD,
    SHOULDER_LIFT,
    R_P,
    GRASP_TCP_OFFSET_HAND,
    transl_mat,
    rpy_to_mat,
    rot_z_mat,
)

DEFAULT_URDF = (
    "/home/niic/project/20260623/20260116/test_robot_A/"
    "src/robot_A/robot_a_description/urdf/robot_a.urdf"
)

def rotation_vector(R):
    R = np.asarray(R, dtype=float)
    c = float(np.clip((np.trace(R)-1.0)*0.5, -1.0, 1.0))
    a = math.acos(c)
    if a < 1e-8:
        return 0.5*np.array([
            R[2,1]-R[1,2],
            R[0,2]-R[2,0],
            R[1,0]-R[0,1],
        ])
    s = math.sin(a)
    if abs(s) < 1e-8:
        return np.zeros(3)
    k = a/(2.0*s)
    return k*np.array([
        R[2,1]-R[1,2],
        R[0,2]-R[2,0],
        R[1,0]-R[0,1],
    ])

def _joint_transform(row, angle):
    ox,oy,oz,rr,rp,ry,ax,ay,az = row
    sign = -1 if az < 0 else 1
    return (
        transl_mat(ox,oy,oz)
        @ rpy_to_mat(rr,rp,ry)
        @ rot_z_mat(sign*float(angle))
    )

def right_fk_frames(q, lift, waist):
    q = np.asarray(q, dtype=float)
    T = (
        transl_mat(0.146,0.0,0.8658)
        @ transl_mat(0.0,0.0,float(lift))
        @ rpy_to_mat(-np.pi/2,0.0,0.0)
        @ rot_z_mat(float(waist))
    )
    frames = []
    for i,row in enumerate(R_P):
        T = T @ _joint_transform(row, q[i])
        frames.append(T.copy())
    return T, frames

def right_state(q, lift, waist):
    hand_T, frames = right_fk_frames(q, lift, waist)
    tcp_T = hand_T @ transl_mat(
        float(GRASP_TCP_OFFSET_HAND[0]),
        float(GRASP_TCP_OFFSET_HAND[1]),
        float(GRASP_TCP_OFFSET_HAND[2]),
    )
    forearm = frames[6][:3,3] - frames[3][:3,3]
    n = float(np.linalg.norm(forearm))
    if n < 1e-9:
        fv, tilt = 1.0, math.pi/2
    else:
        fv = float(forearm[2]/n)
        tilt = math.asin(float(np.clip(abs(fv),0.0,1.0)))
    return {
        "tcp": tcp_T[:3,3].copy(),
        "hand_R": hand_T[:3,:3].copy(),
        "forearm_vertical": fv,
        "forearm_tilt_rad": tilt,
    }

def current_q(c, names):
    return np.asarray([c.get_pos(n) for n in names], dtype=float)

def right_limits(limits):
    lo = np.asarray([limits[n][0] for n in RIGHT_MOTION_JOINTS], dtype=float)
    hi = np.asarray([limits[n][1] for n in RIGHT_MOTION_JOINTS], dtype=float)
    return lo, hi

def exact_pose_residual(
    q,
    lift,
    waist,
    target_xyz,
    grasp_R,
    qref,
):
    """
    V5 精确候选位姿残差：
      - TCP XYZ 都针对一个明确候选点；
      - hand 完整姿态保持抓取时姿态；
      - 前臂完全不参与约束；
      - qref 只做很弱的连续性正则。
    """
    st = right_state(q, lift, waist)
    tcp = np.asarray(st["tcp"], dtype=float)
    pos = tcp - np.asarray(target_xyz, dtype=float)
    ori = rotation_vector(st["hand_R"] @ grasp_R.T)

    return np.concatenate([
        18.0 * pos,
        3.5 * ori,
        0.003 * (q - qref),
    ])


def solve_exact_pose_seed(
    seed,
    qref,
    lo,
    hi,
    lift,
    waist,
    target_xyz,
    grasp_R,
    tcp_tol,
    ori_tol,
):
    q = np.clip(np.asarray(seed, dtype=float).copy(), lo, hi)

    delta = 1e-5
    damping = 0.022
    max_step = 0.10

    best_valid = None
    best_valid_score = float("inf")

    for _ in range(520):
        r = exact_pose_residual(
            q,
            lift,
            waist,
            target_xyz,
            grasp_R,
            qref,
        )

        st = right_state(q, lift, waist)
        tcp = np.asarray(st["tcp"], dtype=float)

        pe = float(
            np.linalg.norm(
                tcp - np.asarray(target_xyz, dtype=float)
            )
        )
        oe = float(
            np.linalg.norm(
                rotation_vector(
                    st["hand_R"] @ grasp_R.T
                )
            )
        )
        ft = float(st["forearm_tilt_rad"])

        # 只有真正满足硬阈值的解才进入 valid 集合。
        if pe <= tcp_tol and oe <= ori_tol:
            score = (
                float(np.linalg.norm(q - qref))
                + 4.0 * pe
                + 1.5 * oe
            )

            if score < best_valid_score:
                best_valid_score = score
                best_valid = {
                    "q": q.copy(),
                    "state": st,
                    "tcp_error_m": pe,
                    "z_error_m": abs(
                        float(tcp[2])
                        - float(target_xyz[2])
                    ),
                    "orientation_error_rad": oe,
                    "forearm_tilt_rad": ft,
                }

            # 当前迭代就是满足条件的解，直接返回当前解。
            # 修复 V4 “返回历史 best 导致日志误差超阈值”的问题。
            return best_valid

        J = np.zeros((len(r), 7), dtype=float)

        for j in range(7):
            qp = q.copy()
            qp[j] = min(hi[j], qp[j] + delta)

            if qp[j] == q[j]:
                qp[j] = max(lo[j], q[j] - delta)

            dd = qp[j] - q[j]
            if abs(dd) < 1e-12:
                continue

            J[:, j] = (
                exact_pose_residual(
                    qp,
                    lift,
                    waist,
                    target_xyz,
                    grasp_R,
                    qref,
                )
                - r
            ) / dd

        try:
            dq = -np.linalg.solve(
                J.T @ J
                + (damping ** 2) * np.eye(7),
                J.T @ r,
            )
        except np.linalg.LinAlgError:
            break

        m = float(np.max(np.abs(dq)))
        if m > max_step:
            dq *= max_step / m

        qn = np.clip(q + dq, lo, hi)

        if float(np.linalg.norm(qn - q)) < 1e-10:
            break

        q = qn

    return best_valid


def solve_exact_pose(
    qref,
    lo,
    hi,
    lift,
    waist,
    target_xyz,
    grasp_R,
    tcp_tol,
    ori_tol,
    *,
    heavy_multistart=False,
):
    """
    优先从上一 waypoint 解继续。
    rescue 模式会扩大 multistart，帮助跳出局部 IK 分支。
    """
    sol = solve_exact_pose_seed(
        qref,
        qref,
        lo,
        hi,
        lift,
        waist,
        target_xyz,
        grasp_R,
        tcp_tol,
        ori_tol,
    )
    if sol is not None:
        return sol

    rng = np.random.default_rng(20260815)

    if heavy_multistart:
        sigmas = (0.025, 0.05, 0.09, 0.14, 0.20)
        count = 5
    else:
        sigmas = (0.025, 0.05, 0.09)
        count = 3

    candidates = []

    for sigma in sigmas:
        for _ in range(count):
            seed = np.clip(
                qref + rng.normal(
                    0.0,
                    sigma,
                    size=7,
                ),
                lo,
                hi,
            )

            sol = solve_exact_pose_seed(
                seed,
                qref,
                lo,
                hi,
                lift,
                waist,
                target_xyz,
                grasp_R,
                tcp_tol,
                ori_tol,
            )

            if sol is not None:
                candidates.append(sol)

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: float(
            np.linalg.norm(x["q"] - qref)
        )
    )

    return candidates[0]


def root_pitch_step_residual(
    x6,
    *,
    root_target,
    q_reference,
    reference_xy,
    lift,
    waist,
    target_z,
    grasp_R,
):
    """
    V7:
      pitch_arm_right1_joint 由外层明确指定为 root_target；
      只求剩余 6 个右臂关节。

    约束：
      - TCP Z -> 当前阶段目标高度；
      - hand 完整姿态保持抓取时姿态；
      - 前臂 signed vertical -> 当前阶段目标（最终为0，即水平）；
      - X/Y 完全不约束；
      - 相邻 waypoint 连续性弱正则。
    """
    q = np.concatenate([
        np.array([float(root_target)]),
        np.asarray(x6, dtype=float),
    ])

    st = right_state(q, lift, waist)

    z_error = (
        float(st["tcp"][2])
        - float(target_z)
    )

    ori_error = rotation_vector(
        np.asarray(st["hand_R"], dtype=float)
        @ np.asarray(grasp_R, dtype=float).T
    )

    qref = np.asarray(q_reference, dtype=float)
    reference_xy = np.asarray(reference_xy, dtype=float)
    xy_error = (
        np.asarray(st["tcp"], dtype=float)[:2]
        - reference_xy
    )

    return np.concatenate([
        # Z 是强约束；X/Y 不固定在全局位置，
        # 但弱惩罚“相邻 waypoint 突然横跳”。
        np.array([18.0 * z_error]),
        1.2 * xy_error,

        # 右夹爪保持抓取时 hand 姿态。
        3.8 * ori_error,

        # V8/V9：前臂只做非常弱的“更水平”偏好。
        # 不再要求它收敛到0°，真正的硬条件只在外部检查
        # forearm_tilt <= forearm_max。
        np.array([
            0.25 * float(st["forearm_vertical"])
        ]),

        # 只对 6 个内部关节做弱连续性正则。
        0.003 * (
            q[1:]
            - qref[1:]
        ),
    ])


def solve_root_pitch_step_seed(
    *,
    seed_q,
    q_reference,
    reference_xy,
    root_target,
    lower,
    upper,
    lift,
    waist,
    target_z,
    grasp_R,
    tcp_z_tol_m,
    orientation_tol_rad,
    forearm_max_rad,
    max_xy_step_m,
    max_internal_joint_step_rad,
):
    """
    root_target 固定，只优化 q[1:7]。
    """
    seed_q = np.asarray(seed_q, dtype=float)
    qref = np.asarray(q_reference, dtype=float)

    x = np.clip(
        seed_q[1:].copy(),
        lower[1:],
        upper[1:],
    )

    delta = 1e-5
    damping = 0.020
    max_step = 0.10

    for _ in range(520):
        residual = root_pitch_step_residual(
            x,
            root_target=root_target,
            q_reference=qref,
            reference_xy=reference_xy,
            lift=lift,
            waist=waist,
            target_z=target_z,
            grasp_R=grasp_R,
        )

        q = np.concatenate([
            np.array([float(root_target)]),
            x,
        ])

        st = right_state(
            q,
            lift,
            waist,
        )

        z_error = abs(
            float(st["tcp"][2])
            - float(target_z)
        )

        orientation_error = float(
            np.linalg.norm(
                rotation_vector(
                    np.asarray(
                        st["hand_R"],
                        dtype=float,
                    )
                    @ np.asarray(
                        grasp_R,
                        dtype=float,
                    ).T
                )
            )
        )

        forearm_tilt = float(
            st["forearm_tilt_rad"]
        )

        tcp_xy = np.asarray(
            st["tcp"],
            dtype=float,
        )[:2]

        xy_step = float(
            np.linalg.norm(
                tcp_xy
                - np.asarray(reference_xy, dtype=float)
            )
        )

        internal_joint_step = float(
            np.max(
                np.abs(
                    q[1:]
                    - qref[1:]
                )
            )
        )

        if (
            z_error <= float(tcp_z_tol_m)
            and orientation_error
            <= float(orientation_tol_rad)
            and forearm_tilt
            <= float(forearm_max_rad)
            and xy_step
            <= float(max_xy_step_m)
            and internal_joint_step
            <= float(max_internal_joint_step_rad)
        ):
            return {
                "q": q.copy(),
                "state": st,
                "z_error_m":
                    z_error,
                "orientation_error_rad":
                    orientation_error,
                "forearm_tilt_rad":
                    forearm_tilt,
                "xy_step_m":
                    xy_step,
                "internal_joint_step_rad":
                    internal_joint_step,
            }

        J = np.zeros(
            (len(residual), 6),
            dtype=float,
        )

        for j in range(6):
            xp = x.copy()

            xp[j] = min(
                upper[j + 1],
                xp[j] + delta,
            )

            if xp[j] == x[j]:
                xp[j] = max(
                    lower[j + 1],
                    x[j] - delta,
                )

            dd = xp[j] - x[j]

            if abs(dd) < 1e-12:
                continue

            J[:, j] = (
                root_pitch_step_residual(
                    xp,
                    root_target=
                        root_target,
                    q_reference=qref,
                    reference_xy=reference_xy,
                    lift=lift,
                    waist=waist,
                    target_z=target_z,
                    grasp_R=grasp_R,
                )
                - residual
            ) / dd

        try:
            dx = -np.linalg.solve(
                J.T @ J
                + (
                    damping ** 2
                ) * np.eye(6),
                J.T @ residual,
            )
        except np.linalg.LinAlgError:
            return None

        m = float(
            np.max(
                np.abs(dx)
            )
        )

        if m > max_step:
            dx *= max_step / m

        xn = np.clip(
            x + dx,
            lower[1:],
            upper[1:],
        )

        if (
            float(
                np.linalg.norm(
                    xn - x
                )
            )
            < 1e-10
        ):
            return None

        x = xn

    return None


def solve_root_pitch_step(
    *,
    q_reference,
    reference_xy,
    root_target,
    lower,
    upper,
    lift,
    waist,
    target_z,
    grasp_R,
    tcp_z_tol_m,
    orientation_tol_rad,
    forearm_max_rad,
    max_xy_step_m,
    max_internal_joint_step_rad,
):
    qref = np.asarray(
        q_reference,
        dtype=float,
    )

    # 先从上一 waypoint 连续求。
    sol = solve_root_pitch_step_seed(
        seed_q=qref,
        q_reference=qref,
        reference_xy=reference_xy,
        root_target=root_target,
        lower=lower,
        upper=upper,
        lift=lift,
        waist=waist,
        target_z=target_z,
        grasp_R=grasp_R,
        tcp_z_tol_m=tcp_z_tol_m,
        orientation_tol_rad=
            orientation_tol_rad,
        forearm_max_rad=
            forearm_max_rad,
        max_xy_step_m=
            max_xy_step_m,
        max_internal_joint_step_rad=
            max_internal_joint_step_rad,
    )

    if sol is not None:
        return sol

    # 小范围 multistart，仍围绕上一姿态，避免跳到很远的支路。
    rng = np.random.default_rng(
        20260815
    )

    candidates = []

    for sigma in (
        0.020,
        0.045,
        0.080,
        0.120,
    ):
        for _ in range(4):
            seed = qref.copy()
            seed[0] = float(
                root_target
            )
            seed[1:] = np.clip(
                qref[1:]
                + rng.normal(
                    0.0,
                    sigma,
                    size=6,
                ),
                lower[1:],
                upper[1:],
            )

            candidate = (
                solve_root_pitch_step_seed(
                    seed_q=seed,
                    q_reference=qref,
                    reference_xy=reference_xy,
                    root_target=
                        root_target,
                    lower=lower,
                    upper=upper,
                    lift=lift,
                    waist=waist,
                    target_z=target_z,
                    grasp_R=grasp_R,
                    tcp_z_tol_m=
                        tcp_z_tol_m,
                    orientation_tol_rad=
                        orientation_tol_rad,
                    forearm_max_rad=
                        forearm_max_rad,
                    max_xy_step_m=
                        max_xy_step_m,
                    max_internal_joint_step_rad=
                        max_internal_joint_step_rad,
                )
            )

            if candidate is not None:
                candidates.append(
                    candidate
                )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            float(item.get("xy_step_m", 1e9)),
            float(item.get("internal_joint_step_rad", 1e9)),
            float(
                np.linalg.norm(
                    np.asarray(
                        item["q"],
                        dtype=float,
                    )[1:]
                    - qref[1:]
                )
            ),
        )
    )

    return candidates[0]


def right_joint_margin_report(
    q,
    lo,
    hi,
):
    q = np.asarray(q, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    margins = np.minimum(
        q - lo,
        hi - q,
    )

    order = np.argsort(margins)

    rows = []

    for idx in order[:4]:
        rows.append(
            (
                RIGHT_MOTION_JOINTS[
                    int(idx)
                ],
                float(q[int(idx)]),
                float(
                    margins[int(idx)]
                ),
            )
        )

    return rows


def plan_right_raise(
    q_start,
    lift_max,
    waist,
    target_z,
    lo,
    hi,
    step_m,       # 兼容旧CLI；V7不用于路径方向
    tcp_tol,
    ori_tol,
    forearm_tol,
    max_inward_shift_m,   # 兼容V6，不使用
    max_lateral_shift_m,  # 兼容V6，不使用
    xy_soft_weight,       # 兼容V6，不使用
    rescue_inward_step_m, # 兼容V6，不使用
    rescue_lateral_step_m,# 兼容V6，不使用
    rescue_cart_step_m,   # 兼容V6，不使用
    root_step_rad,
    root_limit_margin_rad,
    level_over_root_rad,
    max_xy_step_m,
    max_internal_joint_step_rad,
):
    """
    V7: pitch_arm_right1_joint 主动抬升模式。

    不要求 TCP 垂直上升，也不约束 X/Y。

    路径参数由 RIGHT_ROOT_JOINT = pitch_arm_right1_joint 驱动：
        root_start -> root_upper - margin

    每个 root waypoint：
        - root 精确单调增加；
        - 剩余6关节做补偿；
        - TCP Z 按 root 进度逐步升向 target_z；
        - hand 姿态保持；
        - 前臂不要求完全水平，只要求倾角不超过用户上限。
    """
    q_start = np.asarray(
        q_start,
        dtype=float,
    )

    st0 = right_state(
        q_start,
        lift_max,
        waist,
    )

    root_start = float(
        q_start[0]
    )

    root_upper_safe = float(
        hi[0]
        - float(
            root_limit_margin_rad
        )
    )

    available_root = (
        root_upper_safe
        - root_start
    )

    if available_root <= math.radians(1.0):
        raise RuntimeError(
            f"{RIGHT_ROOT_JOINT} 正方向可用行程过小: "
            f"current="
            f"{math.degrees(root_start):.2f}deg, "
            f"safe_upper="
            f"{math.degrees(root_upper_safe):.2f}deg"
        )

    start_z = float(
        st0["tcp"][2]
    )

    grasp_R = np.asarray(
        st0["hand_R"],
        dtype=float,
    )

    root_step = max(
        math.radians(0.25),
        float(root_step_rad),
    )

    min_root_step = math.radians(
        0.25
    )

    root_now = root_start
    qref = q_start.copy()
    previous_tcp = np.asarray(
        st0["tcp"],
        dtype=float,
    ).copy()
    path = []

    print(
        "\nV9.1 root-pitch 规划参数："
    )
    print(
        f"  {RIGHT_ROOT_JOINT}: "
        f"{math.degrees(root_start):.2f}deg "
        f"-> safe upper "
        f"{math.degrees(root_upper_safe):.2f}deg"
    )
    print(
        f"  available root lift = "
        f"{math.degrees(available_root):.2f}deg"
    )
    print(
        f"  initial TCP Z       = "
        f"{start_z:.4f}m"
    )
    print(
        f"  target TCP Z        = "
        f"{float(target_z):.4f}m"
    )
    print(
        f"  initial forearm     = "
        f"{math.degrees(float(st0['forearm_tilt_rad'])):.2f}deg"
    )
    print(
        f"  forearm max allowed = "
        f"{math.degrees(float(forearm_tol)):.1f}deg"
    )
    print(
        f"  max waypoint XY step= "
        f"{float(max_xy_step_m)*100:.1f}cm"
    )
    print(
        f"  max internal q step = "
        f"{math.degrees(float(max_internal_joint_step_rad)):.1f}deg"
    )

    while (
        root_upper_safe
        - root_now
        > 1e-8
    ):
        next_root = min(
            root_now + root_step,
            root_upper_safe,
        )

        root_progress = (
            next_root - root_start
        ) / available_root

        root_progress = float(
            np.clip(
                root_progress,
                0.0,
                1.0,
            )
        )

        # 不要求垂直路径，只用 root 进度平滑增加 Z 目标。
        desired_z = (
            start_z
            + root_progress
            * (
                float(target_z)
                - start_z
            )
        )


        sol = solve_root_pitch_step(
            q_reference=qref,
            reference_xy=previous_tcp[:2],
            root_target=next_root,
            lower=lo,
            upper=hi,
            lift=lift_max,
            waist=waist,
            target_z=desired_z,
            grasp_R=grasp_R,
            tcp_z_tol_m=
                tcp_tol,
            orientation_tol_rad=
                ori_tol,
            forearm_max_rad=
                forearm_tol,
            max_xy_step_m=
                max_xy_step_m,
            max_internal_joint_step_rad=
                max_internal_joint_step_rad,
        )

        if sol is None:
            if (
                root_step
                > min_root_step
                + 1e-9
            ):
                root_step = max(
                    min_root_step,
                    0.5
                    * root_step,
                )

                print(
                    f"  ↪ root="
                    f"{math.degrees(next_root):.2f}deg "
                    "当前约束/连续性无解，"
                    f"root步长缩小到 "
                    f"{math.degrees(root_step):.2f}deg"
                )
                continue

            rows = right_joint_margin_report(
                qref,
                lo,
                hi,
            )

            margin_text = "; ".join(
                f"{name}: "
                f"q={math.degrees(qv):+.1f}deg, "
                f"margin="
                f"{math.degrees(margin):.1f}deg"
                for (
                    name,
                    qv,
                    margin,
                ) in rows
            )

            raise RuntimeError(
                "V7 pitch-root 抬升无法继续。"
                f" root="
                f"{math.degrees(root_now):.2f}deg, "
                f"TCP Z="
                f"{float(right_state(qref, lift_max, waist)['tcp'][2]):.4f}m. "
                f"最接近关节限位: "
                f"{margin_text}"
            )

        qref = np.asarray(
            sol["q"],
            dtype=float,
        ).copy()

        st = sol["state"]
        tcp = np.asarray(
            st["tcp"],
            dtype=float,
        )

        path.append(
            qref.copy()
        )

        previous_tcp = tcp.copy()

        root_now = float(
            next_root
        )

        print(
            f"right waypoint "
            f"{len(path):03d} [PITCH-UP]: "
            f"root="
            f"{math.degrees(root_now):.2f}deg, "
            f"TCP=("
            f"{tcp[0]:+.4f},"
            f"{tcp[1]:+.4f},"
            f"{tcp[2]:+.4f}), "
            f"zErr="
            f"{sol['z_error_m']*1000:.2f}mm, "
            f"handErr="
            f"{math.degrees(sol['orientation_error_rad']):.2f}deg, "
            f"forearm="
            f"{math.degrees(sol['forearm_tilt_rad']):.2f}deg, "
            f"XYstep="
            f"{sol['xy_step_m']*100:.2f}cm, "
            f"qStep="
            f"{math.degrees(sol['internal_joint_step_rad']):.2f}deg"
        )

        # 一旦达到目标高度，且前臂倾角仍在允许范围内，就停止继续抬 root。
        if (
            float(tcp[2])
            >= float(target_z)
            - max(
                0.003,
                float(tcp_tol),
            )
            and float(
                st["forearm_tilt_rad"]
            )
            <= float(forearm_tol)
        ):
            print(
                "✅ 已提前达到目标高度，"
                "且前臂倾角仍在允许范围内；"
                "无需继续增加 pitch root。"
            )
            break

        # 成功后逐渐恢复到用户给定的 nominal root step。
        root_step = min(
            float(root_step_rad),
            max(
                min_root_step,
                root_step * 1.30,
            ),
        )

    final_state = right_state(
        qref,
        lift_max,
        waist,
    )

    final_z = float(
        final_state["tcp"][2]
    )

    final_forearm = float(
        final_state[
            "forearm_tilt_rad"
        ]
    )

    if (
        final_z
        < float(target_z)
        - max(
            0.005,
            2.0 * float(tcp_tol),
        )
    ):
        raise RuntimeError(
            "pitch_arm_right1_joint 已到安全上限附近，"
            "但仍未达到目标高度: "
            f"TCP Z={final_z:.4f}m < "
            f"target={float(target_z):.4f}m"
        )

    if (
        final_forearm
        > float(forearm_tol)
    ):
        raise RuntimeError(
            "最终高度虽然达到，但右前臂倾角超过允许上限: "
            f"{math.degrees(final_forearm):.2f}deg > "
            f"{math.degrees(float(forearm_tol)):.2f}deg"
        )

    return (
        np.asarray(
            path,
            dtype=float,
        ),
        st0,
        grasp_R,
    )


# ============================================================================
# 右臂多 waypoint minimum-jerk 平滑执行
# ============================================================================

def minimum_jerk_s(u):
    u = float(np.clip(u,0.0,1.0))
    return 10*u**3 - 15*u**4 + 6*u**5

def inverse_minimum_jerk(s_target):
    target = float(np.clip(s_target,0.0,1.0))
    if target <= 0.0:
        return 0.0
    if target >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if minimum_jerk_s(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

def make_goal(joint_names, points, times):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(joint_names)
    for row,t in zip(points,times):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in row]
        pt.time_from_start = Duration(seconds=float(t)).to_msg()
        goal.trajectory.points.append(pt)
    return goal

def send_right_path(c, path, min_dt, smooth_scale):
    path = np.asarray(path,float)
    if len(path) == 0:
        return True
    if path.ndim != 2 or path.shape[1] != 7:
        raise RuntimeError(f"右臂path shape非法: {path.shape}")

    q0 = current_q(c, RIGHT_MOTION_JOINTS)
    n = len(path)
    u = np.asarray([
        inverse_minimum_jerk((i+1)/n)
        for i in range(n)
    ], dtype=float)

    requested_total = max(1.0, float(min_dt)*n*float(smooth_scale))
    required_total = 0.0
    prev_q = q0.copy()
    prev_u = 0.0

    for q,cu in zip(path,u):
        du = float(cu-prev_u)
        if du <= 1e-9:
            raise RuntimeError("右臂时间参数化退化")
        for j,name in enumerate(RIGHT_MOTION_JOINTS):
            vmax = float(c._limits[name][2])*float(c._velocity_scale)
            if vmax <= 0.0:
                continue
            seg = 1.8*abs(float(q[j])-float(prev_q[j]))/vmax
            required_total = max(required_total, seg/du)
        prev_q = q
        prev_u = float(cu)

    total = max(requested_total, required_total)
    times = total*u

    root_goal = make_goal(
        [RIGHT_ROOT_JOINT],
        path[:,:1],
        times,
    )
    arm_goal = make_goal(
        RIGHT_ARM_JOINTS,
        path[:,1:],
        times,
    )

    c.get_logger().warning(
        "▶ 发送右臂高位 minimum-jerk 约束轨迹: "
        f"waypoints={n}, duration={total:.2f}s"
    )

    f_root = c._clients["right_root"].send_goal_async(root_goal)
    f_arm = c._clients["right_arm"].send_goal_async(arm_goal)

    for f in (f_root,f_arm):
        if not c._wait_future(f, timeout_sec=10.0):
            c.cancel_active_goals()
            return False

    h_root = f_root.result()
    h_arm = f_arm.result()
    if (
        h_root is None or h_arm is None
        or not h_root.accepted
        or not h_arm.accepted
    ):
        c.cancel_active_goals()
        return False

    c._active_goal_handles.extend([h_root,h_arm])

    r_root = h_root.get_result_async()
    r_arm = h_arm.get_result_async()

    for f in (r_root,r_arm):
        if not c._wait_future(f, timeout_sec=total+20.0):
            c.cancel_active_goals()
            return False

    ok = True
    for label,f in (("right_root",r_root),("right_arm",r_arm)):
        wrapped = f.result()
        good = (
            wrapped is not None
            and wrapped.status == c._GoalStatus.STATUS_SUCCEEDED
        )
        c.get_logger().info(
            ("✅ " if good else "❌ ") + f"{label}: 高位轨迹完成"
        )
        ok = ok and good

    c._active_goal_handles.clear()

    if ok:
        targets = {
            name: float(v)
            for name,v in zip(RIGHT_MOTION_JOINTS,path[-1])
        }
        ok = c._verify_targets_reached(targets, timeout_sec=3.0)

    return bool(ok)


# ============================================================================
# 左臂抓取后姿态检查 + 原路径逆序归零
# ============================================================================

def verify_left_post_grasp(left_q, original_right_q, tol_rad):
    left_q = np.asarray(left_q,float)
    right_q = np.asarray(original_right_q,float)

    expected_left_grasp = -right_q
    expected_post = expected_left_grasp.copy()
    expected_post[0] -= float(LEFT_ROOT_EXTRA_LIFT_RAD)

    err = np.abs(left_q-expected_post)
    print("\n左臂抓取后姿态检查：")
    print("  actual left(deg)  =", np.round(np.degrees(left_q),2))
    print("  expected(deg)     =", np.round(np.degrees(expected_post),2))
    print("  error(deg)        =", np.round(np.degrees(err),2))

    if float(np.max(err)) > float(tol_rad):
        raise RuntimeError(
            "当前左臂不符合本抓取程序的最终姿态，"
            "不允许假定可沿原路径逆序归零。"
            f" max_error={math.degrees(float(np.max(err))):.2f}deg"
        )
    return expected_left_grasp

def execute_left_reverse_zero(c, original_right_q):
    """
    原抓取：
      stage3 shoulder clearance
      stage4 pregrasp
      stage5 final grasp
      stage6 left root extra -0.10
    这里严格逆序：
      stage6^-1 -> stage5^-1 -> stage4^-1 -> stage3^-1
    """
    left_grasp = -np.asarray(original_right_q,float)
    left_shoulder_clearance = float(-SHOULDER_LIFT)  # +0.7

    c.get_logger().warning(
        "▶ 左臂逆序 A/4：撤销 left root 额外 -0.10rad"
    )
    if not c.send_motion(
        {LEFT_ROOT_JOINT: float(left_grasp[0])},
        requested_duration=1.2,
        steps=50,
    ):
        return False

    c.get_logger().warning(
        "▶ 左臂逆序 B/4：反向 final grasp -> pregrasp；"
        f"{LEFT_ARM_JOINTS[0]} -> {left_shoulder_clearance:+.3f}rad"
    )
    if not c.send_motion(
        {LEFT_ARM_JOINTS[0]: left_shoulder_clearance},
        requested_duration=1.8,
        steps=70,
    ):
        return False

    c.get_logger().warning(
        "▶ 左臂逆序 C/4：肩保持 +0.7rad，"
        "其余 left root/arm 沿原通道逐渐归零"
    )
    stage_c = {LEFT_ROOT_JOINT: 0.0}
    for name in LEFT_ARM_JOINTS:
        stage_c[name] = 0.0
    stage_c[LEFT_ARM_JOINTS[0]] = left_shoulder_clearance

    if not c.send_motion(
        stage_c,
        requested_duration=3.0,
        steps=90,
    ):
        return False

    c.get_logger().warning(
        "▶ 左臂逆序 D/4：其余关节已归零，"
        "最后 shoulder +0.7rad -> 0"
    )
    if not c.send_motion(
        {LEFT_ARM_JOINTS[0]: 0.0},
        requested_duration=1.5,
        steps=60,
    ):
        return False

    qf = current_q(c, LEFT_MOTION_JOINTS)
    if float(np.max(np.abs(qf))) > 0.04:
        c.get_logger().error(
            "❌ 左臂最终未充分归零: "
            f"{np.round(np.degrees(qf),2)} deg"
        )
        return False

    c.get_logger().info(
        "✅ 左臂逆序归零完成: "
        f"{np.round(np.degrees(qf),2)} deg"
    )
    return True


# ============================================================================
# 总流程
# ============================================================================

def plan_and_run(
    c,
    limits,
    *,
    target_z,
    path_step_m,
    tcp_tol_m,
    ori_tol_rad,
    forearm_tol_rad,
    max_inward_shift_m,
    max_lateral_shift_m,
    xy_soft_weight,
    rescue_inward_step_m,
    rescue_lateral_step_m,
    rescue_cart_step_m,
    root_step_rad,
    root_limit_margin_rad,
    level_over_root_rad,
    max_xy_step_m,
    max_internal_joint_step_rad,
    left_tol_rad,
    min_dt,
    smooth_scale,
    execute,
):
    log = c.get_logger().info

    lift_now = c.get_pos(LIFT_JOINT)
    waist_now = c.get_pos(WAIST_JOINT)
    right_q0 = current_q(c, RIGHT_MOTION_JOINTS)
    left_q0 = current_q(c, LEFT_MOTION_JOINTS)

    lift_lo, lift_hi, _ = limits[LIFT_JOINT]
    lift_max = float(lift_hi)

    st_now = right_state(right_q0, lift_now, waist_now)
    st_liftmax = right_state(right_q0, lift_max, waist_now)

    print("\n"+"="*76)
    print("当前抓取后状态")
    print(f"  lift              = {lift_now:+.4f}m")
    print(f"  lift URDF         = [{lift_lo:+.4f}, {lift_hi:+.4f}]m")
    print(f"  waist             = {waist_now:+.4f}rad")
    print("  right TCP         =", np.round(st_now["tcp"],5))
    print(
        f"  right forearm     = "
        f"{math.degrees(st_now['forearm_tilt_rad']):.2f}deg"
    )
    print("  right q(deg)      =", np.round(np.degrees(right_q0),2))
    print("  left q(deg)       =", np.round(np.degrees(left_q0),2))
    print("="*76)

    # 逆序路径前提：当前确实是我们抓取代码执行后的左臂姿态。
    verify_left_post_grasp(left_q0, right_q0, left_tol_rad)

    print(
        f"\n[DRY PLAN 1] Lift先到最高: "
        f"{lift_now:+.4f} -> {lift_max:+.4f}m"
    )
    print("  Lift最高后虚拟 right TCP =", np.round(st_liftmax["tcp"],5))

    lo,hi = right_limits(limits)

    print(
        f"\n[DRY PLAN 2] 主动抬升 {RIGHT_ROOT_JOINT}；"
        "TCP不要求垂直、X/Y自由；"
        "右hand保持抓取姿态，前臂只限制最大倾角；"
        f"目标 TCP Z={target_z:.4f}m"
    )
    path, _, grasp_R = plan_right_raise(
        right_q0,
        lift_max,
        waist_now,
        float(target_z),
        lo,
        hi,
        float(path_step_m),
        float(tcp_tol_m),
        float(ori_tol_rad),
        float(forearm_tol_rad),
        float(max_inward_shift_m),
        float(max_lateral_shift_m),
        float(xy_soft_weight),
        float(rescue_inward_step_m),
        float(rescue_lateral_step_m),
        float(rescue_cart_step_m),
        float(root_step_rad),
        float(root_limit_margin_rad),
        float(level_over_root_rad),
        float(max_xy_step_m),
        float(max_internal_joint_step_rad),
    )

    planned_final = (
        right_state(path[-1],lift_max,waist_now)
        if len(path)>0
        else st_liftmax
    )
    planned_ori = float(np.linalg.norm(
        rotation_vector(planned_final["hand_R"] @ grasp_R.T)
    ))

    # V9 轨迹连续性二次审计：即使求解器某处有逻辑遗漏，
    # dry-run 也不会把含大跳变的路径判为可执行。
    audit_prev_q = right_q0.copy()
    audit_prev_tcp = np.asarray(
        st_liftmax["tcp"],
        dtype=float,
    ).copy()

    max_audit_xy = 0.0
    max_audit_q = 0.0
    worst_index = 0

    for idx, q_cmd in enumerate(path, start=1):
        ast = right_state(
            q_cmd,
            lift_max,
            waist_now,
        )
        atcp = np.asarray(
            ast["tcp"],
            dtype=float,
        )

        xy_jump = float(
            np.linalg.norm(
                atcp[:2] - audit_prev_tcp[:2]
            )
        )
        q_jump = float(
            np.max(
                np.abs(
                    np.asarray(q_cmd)[1:]
                    - audit_prev_q[1:]
                )
            )
        )

        if (
            xy_jump > max_audit_xy
            or q_jump > max_audit_q
        ):
            worst_index = idx

        max_audit_xy = max(
            max_audit_xy,
            xy_jump,
        )
        max_audit_q = max(
            max_audit_q,
            q_jump,
        )

        audit_prev_q = np.asarray(
            q_cmd,
            dtype=float,
        )
        audit_prev_tcp = atcp

    print("\nV9.1 连续性审计：")
    print(
        f"  max adjacent XY step = "
        f"{max_audit_xy*100:.2f}cm "
        f"(limit {max_xy_step_m*100:.2f}cm)"
    )
    print(
        f"  max internal q step  = "
        f"{math.degrees(max_audit_q):.2f}deg "
        f"(limit "
        f"{math.degrees(max_internal_joint_step_rad):.2f}deg)"
    )

    if (
        max_audit_xy > max_xy_step_m + 1e-6
        or max_audit_q
        > max_internal_joint_step_rad + 1e-6
    ):
        raise RuntimeError(
            "V9.1 最终连续性审计失败，禁止执行。"
            f" worst waypoint≈{worst_index}"
        )

    print("\n右臂最终规划检查：")
    print("  planned TCP       =", np.round(planned_final["tcp"],5))
    print(
        f"  Z error           = "
        f"{abs(float(planned_final['tcp'][2])-target_z)*1000:.2f}mm"
    )
    print(
        f"  orientation drift = "
        f"{math.degrees(planned_ori):.2f}deg"
    )
    final_dx = float(
        planned_final["tcp"][0]
        - st_liftmax["tcp"][0]
    )
    final_dy = float(
        planned_final["tcp"][1]
        - st_liftmax["tcp"][1]
    )
    print(
        f"  inward shift      = "
        f"{max(0.0, -final_dx)*100:.2f}cm"
    )
    print(
        f"  lateral shift     = "
        f"{final_dy*100:+.2f}cm"
    )
    print(
        f"  forearm tilt(log) = "
        f"{math.degrees(planned_final['forearm_tilt_rad']):.2f}deg"
    )
    print(f"  waypoints         = {len(path)}")

    print("\n[DRY PLAN 3] 左臂逆序归零：")
    print("  A) 撤销 left root extra -0.10rad")
    print("  B) left shoulder -> +0.7rad")
    print("  C) shoulder保持+0.7rad，其余左臂 -> 0")
    print("  D) left shoulder +0.7rad -> 0")

    if not execute:
        print(
            "\n✅ DRY-RUN完成；未发送任何运动命令，"
            "也没有向右夹爪发送任何命令。"
        )
        return

    # -------------------- REAL [1/3] Lift --------------------
    log(
        "▶ [1/3] Lift先独立升到URDF最高；"
        "左右手臂保持当前姿态"
    )
    if not c.send_motion(
        {LIFT_JOINT: lift_max},
        requested_duration=3.0,
        steps=100,
    ):
        raise RuntimeError("Lift升到最高位失败")

    lift_actual = c.get_pos(LIFT_JOINT)
    if abs(lift_actual-lift_max) > 0.005:
        raise RuntimeError(
            f"Lift反馈未到最高: actual={lift_actual:+.6f}, "
            f"target={lift_max:+.6f}"
        )

    # -------------------- REAL [2/3] Right raise --------------------
    log(
        "▶ [2/3] Lift保持最高；右臂平滑抬高弹簧；"
        "右夹爪不发送新命令"
    )
    if not send_right_path(c,path,min_dt,smooth_scale):
        raise RuntimeError("右臂1.7m约束轨迹执行失败")

    qr = current_q(c, RIGHT_MOTION_JOINTS)
    strr = right_state(qr,lift_actual,waist_now)
    zerr = abs(float(strr["tcp"][2])-float(target_z))
    oerr = float(np.linalg.norm(rotation_vector(strr["hand_R"]@grasp_R.T)))
    ftilt = float(strr["forearm_tilt_rad"])

    if zerr > max(0.005,2.0*tcp_tol_m):
        raise RuntimeError(
            f"右TCP实际高度误差过大: "
            f"Z={strr['tcp'][2]:.4f}, target={target_z:.4f}, "
            f"err={zerr*1000:.1f}mm"
        )
    if oerr > ori_tol_rad + math.radians(1.0):
        raise RuntimeError(
            f"右夹爪姿态漂移超限: {math.degrees(oerr):.2f}deg"
        )

    log(
        "✅ 右臂高位验证通过: "
        f"TCP Z={strr['tcp'][2]:.4f}m, "
        f"hand drift={math.degrees(oerr):.2f}deg, "
        f"forearm(log only)={math.degrees(ftilt):.2f}deg"
    )

    # -------------------- REAL [3/3] Left reverse --------------------
    log(
        "▶ [3/3] 右臂保持高位抓取弹簧；"
        "左臂按原抓取路径逆序归零"
    )
    if not execute_left_reverse_zero(c,right_q0):
        raise RuntimeError("左臂逆序归零失败")

    # 最终复核右臂仍未漂动。
    qrf = current_q(c, RIGHT_MOTION_JOINTS)
    lift_final = c.get_pos(LIFT_JOINT)
    waist_final = c.get_pos(WAIST_JOINT)
    stf = right_state(qrf,lift_final,waist_final)
    of = float(np.linalg.norm(rotation_vector(stf["hand_R"]@grasp_R.T)))

    print("\n"+"="*76)
    print("✅ 最终状态")
    print(f"  Lift               = {lift_final:+.4f}m")
    print(f"  Right TCP Z        = {stf['tcp'][2]:.4f}m")
    print(f"  Right hand drift   = {math.degrees(of):.2f}deg")
    print(
        f"  Right forearm(log) = "
        f"{math.degrees(stf['forearm_tilt_rad']):.2f}deg"
    )
    print("  Left arm           = ZERO")
    print("  Right gripper      = 保持原抓取控制状态（本脚本未发送张开命令）")
    print("="*76)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "弹簧抓取后：Lift最高 -> "
            "右TCP保持抓取姿态抬到约1.7m -> "
            "左臂沿原路径逆序归零"
        )
    )
    ap.add_argument(
        "-z","--target-z",
        type=float, default=1.70,
        help="右grasp TCP在base_link下目标Z，默认1.70m"
    )
    ap.add_argument("--urdf",default=DEFAULT_URDF)
    ap.add_argument(
        "--velocity-scale",type=float,default=0.20,
        help="关节速度缩放，默认0.20"
    )
    ap.add_argument(
        "--gripper-effort",type=float,default=50.0,
        help="仅用于构造已有控制器；本脚本不会发送夹爪新命令"
    )
    ap.add_argument(
        "--path-step-mm",type=float,default=2.0,
        help="右TCP竖直路径步长，默认2mm"
    )
    ap.add_argument(
        "--tcp-tol-mm",type=float,default=1.5,
        help="右TCP waypoint位置容差，默认1.5mm"
    )
    ap.add_argument(
        "--orientation-tol-deg",type=float,default=2.0,
        help="右hand相对抓取姿态允许漂移，默认2deg"
    )
    ap.add_argument(
        "--max-inward-shift-cm",
        type=float,
        default=8.0,
        help=(
            "右TCP为了升高允许向机器人本体方向(base_link X减小)"
            "回收的最大距离，默认8cm"
        ),
    )
    ap.add_argument(
        "--max-lateral-shift-cm",
        type=float,
        default=2.0,
        help=(
            "右TCP在base_link Y方向允许的最大漂移，默认2cm"
        ),
    )
    ap.add_argument(
        "--xy-soft-weight",
        type=float,
        default=0.8,
        help=(
            "XY软锚点权重，越小越容易向内收；默认0.8"
        ),
    )
    ap.add_argument(
        "--rescue-inward-step-cm",
        type=float,
        default=0.5,
        help=(
            "当前XY支路失败时，每次连续向内回收的总距离，默认0.5cm"
        ),
    )
    ap.add_argument(
        "--rescue-lateral-step-cm",
        type=float,
        default=0.5,
        help=(
            "兼容V5参数；V6默认不主动改变Y"
        ),
    )
    ap.add_argument(
        "--rescue-cart-step-mm",
        type=float,
        default=2.0,
        help=(
            "兼容V6参数；V7不使用"
        ),
    )
    ap.add_argument(
        "--root-step-deg",
        type=float,
        default=1.0,
        help=(
            "pitch_arm_right1_joint 每个规划步主动抬升角度，"
            "默认1deg；失败会自动减小到0.25deg"
        ),
    )
    ap.add_argument(
        "--max-waypoint-xy-step-cm",
        type=float,
        default=3.0,
        help=(
            "相邻右TCP waypoint允许的最大水平跳变，默认3cm。"
            "用于拒绝IK换支路造成的突然横移"
        ),
    )
    ap.add_argument(
        "--max-internal-joint-step-deg",
        type=float,
        default=8.0,
        help=(
            "相邻waypoint其余6个右臂关节的最大单关节变化，"
            "默认8deg"
        ),
    )
    ap.add_argument(
        "--root-limit-margin-deg",
        type=float,
        default=3.0,
        help=(
            "不把 pitch_arm_right1_joint 用到URDF硬上限，"
            "默认保留3deg安全余量"
        ),
    )
    ap.add_argument(
        "--level-over-root-deg",
        type=float,
        default=6.0,
        help=(
            "兼容V7参数；V8不再强制前臂逐渐归零，因此该参数不使用"
        ),
    )
    ap.add_argument(
        "--forearm-tol-deg",
        type=float,
        default=12.0,
        help=(
            "V8 前臂最大允许倾角，默认12deg。"
            "不要求完全水平，只要不超过该角度即可"
        )
    )
    ap.add_argument(
        "--left-post-grasp-tol-deg",type=float,default=5.0,
        help="当前左臂与预期抓取后姿态最大允许误差，默认5deg"
    )
    ap.add_argument("--min-waypoint-dt",type=float,default=0.10)
    ap.add_argument(
        "--smooth-duration-scale",type=float,default=1.8,
        help="右臂minimum-jerk时间倍率，越大越慢，默认1.8"
    )
    ap.add_argument(
        "--execute",action="store_true",
        help="真实执行；默认仅dry-run"
    )
    return ap.parse_args()


def main():
    a = parse_args()
    if a.target_z <= 0 or a.path_step_mm <= 0 or a.tcp_tol_mm <= 0:
        raise SystemExit("Z/步长/位置容差必须 > 0")
    if (
        a.max_inward_shift_cm <= 0
        or a.max_lateral_shift_cm <= 0
        or a.xy_soft_weight <= 0
        or a.rescue_inward_step_cm <= 0
        or a.rescue_lateral_step_cm <= 0
        or a.rescue_cart_step_mm <= 0
        or a.root_step_deg <= 0
        or a.max_waypoint_xy_step_cm <= 0
        or a.max_internal_joint_step_deg <= 0
        or a.root_limit_margin_deg < 0
        or a.level_over_root_deg < 0
        or a.forearm_tol_deg <= 0
    ):
        raise SystemExit(
            "XY允许位移和--xy-soft-weight必须 > 0"
        )
    if not (0.0 < a.velocity_scale <= 1.0):
        raise SystemExit("--velocity-scale必须位于(0,1]")
    if (
        a.orientation_tol_deg <= 0
        or a.left_post_grasp_tol_deg <= 0
        or a.smooth_duration_scale <= 0
    ):
        raise SystemExit("角度容差和smooth scale必须 > 0")

    print(
        f"[CLI] target_z={float(a.target_z):.4f}m, "
        f"driver={RIGHT_ROOT_JOINT}, "
        f"root_step={float(a.root_step_deg):.2f}deg, "
        f"forearm_max={float(a.forearm_tol_deg):.2f}deg, "
        f"XYstep_max={float(a.max_waypoint_xy_step_cm):.1f}cm, "
        f"qStep_max={float(a.max_internal_joint_step_deg):.1f}deg"
    )

    limits = load_urdf_limits(a.urdf)
    rclpy.init(args=sys.argv)
    c = None
    try:
        c = IKGraspRealController(
            limits,
            velocity_scale=a.velocity_scale,
            gripper_effort=a.gripper_effort,
        )
        plan_and_run(
            c,limits,
            target_z=float(a.target_z),
            path_step_m=float(a.path_step_mm)/1000.0,
            tcp_tol_m=float(a.tcp_tol_mm)/1000.0,
            ori_tol_rad=math.radians(float(a.orientation_tol_deg)),
            forearm_tol_rad=math.radians(float(a.forearm_tol_deg)),
            max_inward_shift_m=float(a.max_inward_shift_cm)/100.0,
            max_lateral_shift_m=float(a.max_lateral_shift_cm)/100.0,
            xy_soft_weight=float(a.xy_soft_weight),
            rescue_inward_step_m=float(a.rescue_inward_step_cm)/100.0,
            rescue_lateral_step_m=float(a.rescue_lateral_step_cm)/100.0,
            rescue_cart_step_m=float(a.rescue_cart_step_mm)/1000.0,
            root_step_rad=math.radians(float(a.root_step_deg)),
            root_limit_margin_rad=math.radians(
                float(a.root_limit_margin_deg)
            ),
            level_over_root_rad=math.radians(
                float(a.level_over_root_deg)
            ),
            max_xy_step_m=float(
                a.max_waypoint_xy_step_cm
            )/100.0,
            max_internal_joint_step_rad=math.radians(
                float(a.max_internal_joint_step_deg)
            ),
            left_tol_rad=math.radians(float(a.left_post_grasp_tol_deg)),
            min_dt=float(a.min_waypoint_dt),
            smooth_scale=float(a.smooth_duration_scale),
            execute=bool(a.execute),
        )
        return 0
    except KeyboardInterrupt:
        if c is not None:
            try:
                c.cancel_active_goals()
            except Exception:
                pass
        print("\n⏹️ 用户中断，已请求取消活动轨迹。")
        return 130
    except Exception as exc:
        if c is not None:
            c.get_logger().error(f"❌ {type(exc).__name__}: {exc}")
        else:
            print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if c is not None:
            try:
                c.shutdown()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
