#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot A 双臂抓箱：大倾角/大偏航抓取增强 V2

目标
====
保持已经实机验证过的抓取动作顺序不变：

  1. 双臂预抬升
  2. 双夹爪张开
  3. 左右 TCP 沿真实抓取点连线向箱体外侧扩开
  4. 保持外扩姿态下降
  5. 低位多 waypoint 逐步向内收拢
  6. 双夹爪闭合并验证 position / reached_goal / stalled
  7. 抓后 Lift 抬升
  8. 首次闭合失败时保持当前夹持状态，先抬升再闭合一次

V2 只增强“大歪抓取点”的规划能力，不修改视觉节点和手眼标定。

为什么需要 V2
============
旧 large_skew 版本只在箱体 XY 横向轴 rotation > 35deg 时增加额外 IK seed。
实际日志中困难箱体 rotation=30.57deg，因此额外 seed 没有真正触发。

同时，最接近 IK 解表现为：
  - 左臂 TCP 已几乎到点；
  - 右臂 XY 误差约 1.5~1.75cm；
  - 左臂为了够到目标会比标准模板偏离更多；
  - 第二阶段恢复标准模板权重后，位置会被模板约束“拉回来”。

V2 因此采用：
  A. large-skew 触发角降到默认 24deg；
  B. 普通箱子完全沿用原严格 IK；
  C. 仅 large-skew 模式下：
       - reach/refine 两阶段都更偏向 TCP 位置；
       - 模板只是弱正则，不再压过位置；
       - XY 总误差允许到默认 1.9cm；
       - 箱体左右夹紧方向 lateral 仍严格限制到默认 0.8cm；
       - forearm 最大默认放宽到 5deg；
       - gripper vertical 仍保持原 5deg；
       - template RMSE 默认只放宽到 30deg，而不是无限放宽；
  D. 大角度时增加确定性多 seed；
  E. 原 URDF 限位、共享 Lift、waypoint 跳变、夹爪验证、
     common-height、宽位下降、post-lift 全部保留。

注意
====
本系统没有完整 3D collision scene。第一次测试必须 enable_motion:=false。
确认整条路径连续且没有扫到机器人本体/环境后，再低速实机。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import rclpy

from bridge_xyz_offset_box_wide_descend_postlift import (
    WideDescendPostLiftBridge,
)


class LargeSkewV2Bridge(WideDescendPostLiftBridge):
    """原抓取流程 + 大歪目标自适应 IK。"""

    def __init__(self) -> None:
        super().__init__()

        # --------------------------------------------------------------
        # 几何范围：继续允许更大的平面内偏航。
        # --------------------------------------------------------------
        self.declare_parameter(
            "large_skew_max_side_axis_rotation_deg",
            68.0,
        )
        self.declare_parameter(
            "large_skew_max_target_x_difference_m",
            0.60,
        )
        self.declare_parameter(
            "large_skew_centerline_margin_m",
            0.005,
        )

        # 不默认允许左右机械臂跨过中心线交换目标。
        self.declare_parameter(
            "large_skew_require_targets_straddle_centerline",
            True,
        )

        # --------------------------------------------------------------
        # 关键：large-skew 模式触发阈值。
        # 当前实机困难样本 rotation=30.57deg，因此默认 24deg。
        # --------------------------------------------------------------
        self.declare_parameter(
            "large_skew_trigger_rotation_deg",
            24.0,
        )
        self.declare_parameter(
            "large_skew_trigger_dx_m",
            0.140,
        )

        # --------------------------------------------------------------
        # Large-skew 最终抓取允许值。
        #
        # XY 可以主要在“前后”方向有一定捕获误差；
        # 箱体左右夹紧方向 lateral 仍保持明显更严格。
        # --------------------------------------------------------------
        self.declare_parameter(
            "large_skew_max_xy_error_cm",
            1.90,
        )
        self.declare_parameter(
            "large_skew_max_lateral_error_cm",
            0.80,
        )
        self.declare_parameter(
            "large_skew_max_orientation_error_deg",
            6.0,
        )
        self.declare_parameter(
            "large_skew_max_forearm_tilt_deg",
            5.0,
        )
        self.declare_parameter(
            "large_skew_max_gripper_vertical_error_deg",
            5.0,
        )
        self.declare_parameter(
            "large_skew_max_template_rmse_deg",
            30.0,
        )

        # --------------------------------------------------------------
        # 偏置捕获与 IK 搜索范围。
        # --------------------------------------------------------------
        self.declare_parameter(
            "large_skew_adaptive_max_outset_m",
            0.030,
        )
        self.declare_parameter(
            "large_skew_adaptive_cross_side_max_m",
            0.016,
        )
        self.declare_parameter(
            "large_skew_seed_limit",
            12,
        )
        self.declare_parameter(
            "large_skew_reach_iterations",
            180,
        )
        self.declare_parameter(
            "large_skew_refine_iterations",
            360,
        )
        self.declare_parameter(
            "large_skew_line_search_steps",
            8,
        )

        # --------------------------------------------------------------
        # 读取参数。
        # --------------------------------------------------------------
        max_rotation_deg = float(
            self.get_parameter(
                "large_skew_max_side_axis_rotation_deg"
            ).value
        )
        max_dx = float(
            self.get_parameter(
                "large_skew_max_target_x_difference_m"
            ).value
        )
        centerline_margin = float(
            self.get_parameter(
                "large_skew_centerline_margin_m"
            ).value
        )
        require_straddle = bool(
            self.get_parameter(
                "large_skew_require_targets_straddle_centerline"
            ).value
        )
        self.large_skew_trigger_rotation_rad = math.radians(
            float(
                self.get_parameter(
                    "large_skew_trigger_rotation_deg"
                ).value
            )
        )
        self.large_skew_trigger_dx_m = float(
            self.get_parameter(
                "large_skew_trigger_dx_m"
            ).value
        )

        self.large_skew_xy_error_m = (
            float(
                self.get_parameter(
                    "large_skew_max_xy_error_cm"
                ).value
            )
            / 100.0
        )
        self.large_skew_lateral_error_m = (
            float(
                self.get_parameter(
                    "large_skew_max_lateral_error_cm"
                ).value
            )
            / 100.0
        )
        self.large_skew_orientation_rad = math.radians(
            float(
                self.get_parameter(
                    "large_skew_max_orientation_error_deg"
                ).value
            )
        )
        self.large_skew_forearm_rad = math.radians(
            float(
                self.get_parameter(
                    "large_skew_max_forearm_tilt_deg"
                ).value
            )
        )
        self.large_skew_vertical_rad = math.radians(
            float(
                self.get_parameter(
                    "large_skew_max_gripper_vertical_error_deg"
                ).value
            )
        )
        self.large_skew_template_rad = math.radians(
            float(
                self.get_parameter(
                    "large_skew_max_template_rmse_deg"
                ).value
            )
        )

        max_outset = float(
            self.get_parameter(
                "large_skew_adaptive_max_outset_m"
            ).value
        )
        cross_outset = float(
            self.get_parameter(
                "large_skew_adaptive_cross_side_max_m"
            ).value
        )
        self.large_skew_seed_limit = int(
            self.get_parameter(
                "large_skew_seed_limit"
            ).value
        )
        reach_iterations = int(
            self.get_parameter(
                "large_skew_reach_iterations"
            ).value
        )
        refine_iterations = int(
            self.get_parameter(
                "large_skew_refine_iterations"
            ).value
        )
        line_search_steps = int(
            self.get_parameter(
                "large_skew_line_search_steps"
            ).value
        )

        # --------------------------------------------------------------
        # 参数验证。
        # --------------------------------------------------------------
        if not 35.0 <= max_rotation_deg < 80.0:
            raise RuntimeError(
                "large_skew_max_side_axis_rotation_deg "
                "必须在 [35,80)deg"
            )
        if not 0.25 <= max_dx <= 0.80:
            raise RuntimeError(
                "large_skew_max_target_x_difference_m "
                "必须在 [0.25,0.80]m"
            )
        if not 0.0 <= centerline_margin <= 0.020:
            raise RuntimeError(
                "large_skew_centerline_margin_m "
                "必须在 [0,0.020]m"
            )
        if not math.radians(10.0) <= (
            self.large_skew_trigger_rotation_rad
        ) <= math.radians(45.0):
            raise RuntimeError(
                "large_skew_trigger_rotation_deg "
                "必须在 [10,45]deg"
            )
        if not 0.05 <= self.large_skew_trigger_dx_m <= 0.30:
            raise RuntimeError(
                "large_skew_trigger_dx_m 必须在 [0.05,0.30]m"
            )
        if not 0.010 <= self.large_skew_xy_error_m <= 0.025:
            raise RuntimeError(
                "large_skew_max_xy_error_cm 必须在 [1.0,2.5]cm"
            )
        if not 0.004 <= self.large_skew_lateral_error_m <= 0.012:
            raise RuntimeError(
                "large_skew_max_lateral_error_cm "
                "必须在 [0.4,1.2]cm"
            )
        if not math.radians(5.0) <= self.large_skew_orientation_rad <= math.radians(10.0):
            raise RuntimeError(
                "large_skew_max_orientation_error_deg "
                "必须在 [5,10]deg"
            )
        if not math.radians(3.0) <= self.large_skew_forearm_rad <= math.radians(8.0):
            raise RuntimeError(
                "large_skew_max_forearm_tilt_deg "
                "必须在 [3,8]deg"
            )
        if not math.radians(4.0) <= self.large_skew_vertical_rad <= math.radians(8.0):
            raise RuntimeError(
                "large_skew_max_gripper_vertical_error_deg "
                "必须在 [4,8]deg"
            )
        if not math.radians(20.0) <= self.large_skew_template_rad <= math.radians(40.0):
            raise RuntimeError(
                "large_skew_max_template_rmse_deg "
                "必须在 [20,40]deg"
            )
        if not 0.024 <= max_outset <= 0.030:
            raise RuntimeError(
                "large_skew_adaptive_max_outset_m "
                "必须在 [0.024,0.030]m"
            )
        if not 0.0 <= cross_outset <= min(0.016, max_outset):
            raise RuntimeError(
                "large_skew_adaptive_cross_side_max_m 超出允许范围"
            )
        if not 6 <= self.large_skew_seed_limit <= 14:
            raise RuntimeError(
                "large_skew_seed_limit 必须在 [6,14]"
            )
        if not 120 <= reach_iterations <= 250:
            raise RuntimeError(
                "large_skew_reach_iterations 必须在 [120,250]"
            )
        if not 260 <= refine_iterations <= 500:
            raise RuntimeError(
                "large_skew_refine_iterations 必须在 [260,500]"
            )
        if not 6 <= line_search_steps <= 8:
            raise RuntimeError(
                "large_skew_line_search_steps 必须在 [6,8]"
            )

        # --------------------------------------------------------------
        # 仅几何入口范围是全局扩大。
        # 抓取精度阈值不在这里全局放宽，而是在 large-skew 规划期间临时切换。
        # --------------------------------------------------------------
        self.max_side_axis_rotation_rad = math.radians(
            max_rotation_deg
        )
        self.max_target_x_difference_m = max(
            float(self.max_target_x_difference_m),
            max_dx,
        )
        self.min_centerline_side_margin_m = min(
            float(self.min_centerline_side_margin_m),
            centerline_margin,
        )
        self.require_targets_straddle_centerline = require_straddle

        # 原 common-height / adaptive offset 逻辑保留。
        self.adaptive_max_outset_m = max(
            float(self.adaptive_max_outset_m),
            max_outset,
        )
        self.adaptive_cross_side_max_m = max(
            float(self.adaptive_cross_side_max_m),
            cross_outset,
        )

        # 允许我们的 _seed_pairs 返回更多候选。
        self.offset_ik_seed_branches = max(
            int(self.offset_ik_seed_branches),
            self.large_skew_seed_limit,
        )
        self.offset_ik_reach_iterations = max(
            int(self.offset_ik_reach_iterations),
            reach_iterations,
        )
        self.offset_ik_refine_iterations = max(
            int(self.offset_ik_refine_iterations),
            refine_iterations,
        )
        self.offset_ik_line_search_steps = max(
            int(self.offset_ik_line_search_steps),
            line_search_steps,
        )

        # 当前规划上下文。
        self._large_skew_active = False
        self._large_skew_rotation_rad = 0.0
        self._large_skew_axis_xy = np.array(
            [0.0, 1.0],
            dtype=np.float64,
        )
        self._large_skew_dx_m = 0.0
        self._large_skew_center_y_m = 0.0

        self.get_logger().warning(
            "Large-skew V2 已启用："
            "普通目标保持原严格抓取；困难目标才进入位置优先模式。 "
            f"trigger_rotation="
            f"{math.degrees(self.large_skew_trigger_rotation_rad):.1f}deg, "
            f"trigger_dX={self.large_skew_trigger_dx_m:.3f}m, "
            f"geometry_limit={max_rotation_deg:.1f}deg, "
            f"large_XY<={self.large_skew_xy_error_m*100:.2f}cm, "
            f"large_lateral<={self.large_skew_lateral_error_m*100:.2f}cm, "
            f"large_forearm<={math.degrees(self.large_skew_forearm_rad):.1f}deg, "
            f"large_template<={math.degrees(self.large_skew_template_rad):.1f}deg"
        )

    # ==================================================================
    # Large-skew context
    # ==================================================================
    def _update_large_skew_context(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> None:
        left = np.asarray(left_base, dtype=np.float64)
        right = np.asarray(right_base, dtype=np.float64)

        delta_xy = left[:2] - right[:2]
        distance_xy = float(np.linalg.norm(delta_xy))

        if distance_xy < 1e-9:
            self._large_skew_active = False
            return

        axis = delta_xy / distance_xy
        rotation = math.atan2(
            abs(float(axis[0])),
            abs(float(axis[1])),
        )
        dx = abs(float(delta_xy[0]))
        center_y = 0.5 * float(left[1] + right[1])

        self._large_skew_axis_xy = axis.copy()
        self._large_skew_rotation_rad = float(rotation)
        self._large_skew_dx_m = dx
        self._large_skew_center_y_m = center_y

        self._large_skew_active = bool(
            rotation >= self.large_skew_trigger_rotation_rad
            or dx >= self.large_skew_trigger_dx_m
        )

        self.get_logger().info(
            "箱体抓取轴诊断: "
            f"axis_xy={np.round(axis,5)}, "
            f"rotation={math.degrees(rotation):.2f}deg, "
            f"dX={dx:.4f}m, "
            f"dY={abs(float(delta_xy[1])):.4f}m, "
            f"raw_dZ={abs(float(left[2]-right[2])):.4f}m, "
            f"center_y={center_y:+.4f}m, "
            f"large_skew={self._large_skew_active}"
        )

    def _large_skew_threshold_snapshot(self):
        return (
            float(self.max_xy_error_m),
            float(self.max_lateral_error_m),
            float(self.max_orientation_error_rad),
            float(self.max_forearm_tilt_rad),
            float(self.max_gripper_vertical_error_rad),
            float(self.max_template_rmse_rad),
        )

    def _apply_large_skew_thresholds(self) -> None:
        self.max_xy_error_m = max(
            float(self.max_xy_error_m),
            self.large_skew_xy_error_m,
        )
        self.max_lateral_error_m = max(
            float(self.max_lateral_error_m),
            self.large_skew_lateral_error_m,
        )
        self.max_orientation_error_rad = max(
            float(self.max_orientation_error_rad),
            self.large_skew_orientation_rad,
        )
        self.max_forearm_tilt_rad = max(
            float(self.max_forearm_tilt_rad),
            self.large_skew_forearm_rad,
        )
        self.max_gripper_vertical_error_rad = max(
            float(self.max_gripper_vertical_error_rad),
            self.large_skew_vertical_rad,
        )
        self.max_template_rmse_rad = max(
            float(self.max_template_rmse_rad),
            self.large_skew_template_rad,
        )

    def _restore_thresholds(self, snapshot) -> None:
        (
            self.max_xy_error_m,
            self.max_lateral_error_m,
            self.max_orientation_error_rad,
            self.max_forearm_tilt_rad,
            self.max_gripper_vertical_error_rad,
            self.max_template_rmse_rad,
        ) = snapshot

    # ==================================================================
    # V2 核心：large-skew 时 position-first residual
    # ==================================================================
    def _stage_residual(
        self,
        state: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
        reach_stage: bool,
    ) -> np.ndarray:
        if not self._large_skew_active:
            return super()._stage_residual(
                state,
                left_target,
                right_target,
                reach_stage,
            )

        # 原 residual 每臂：
        # XYZ(3), orientation(3), forearm(1), template(7)
        residual = self._joint_xyz_residual(
            state,
            left_target,
            right_target,
        ).copy()

        for start in (0, 14):
            if reach_stage:
                # 第一阶段：尽快把 TCP 拉到真实抓取点附近。
                # base 中 lateral 本身已有 lateral_weight=4，因此这里不需要夸张倍增。
                residual[start + 0] *= 6.0   # forward X
                residual[start + 1] *= 3.0   # lateral Y (base residual 已×4)
                residual[start + 2] *= 3.0   # Z (base residual 已×2)

                residual[start + 3 : start + 6] *= 0.20
                residual[start + 6] *= 0.25
                residual[start + 7 : start + 14] *= 0.05
            else:
                # 第二阶段仍保持“位置优先”，但恢复一部分姿态/前臂约束。
                # 关键变化：不再恢复到旧版强模板权重。
                residual[start + 0] *= 3.0
                residual[start + 1] *= 2.0
                residual[start + 2] *= 2.0

                residual[start + 3 : start + 6] *= 0.65
                residual[start + 6] *= 0.75
                residual[start + 7 : start + 14] *= 0.20

        return residual

    # ==================================================================
    # V2 seed：24deg 起就启用，而不是等到 >35deg
    # ==================================================================
    def _seed_pairs(
        self,
        left_seed: np.ndarray,
        right_seed: np.ndarray,
        left_lower: np.ndarray,
        left_upper: np.ndarray,
        right_lower: np.ndarray,
        right_upper: np.ndarray,
        center_y: float,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        seeds = list(
            super()._seed_pairs(
                left_seed,
                right_seed,
                left_lower,
                left_upper,
                right_lower,
                right_upper,
                center_y,
            )
        )

        if not self._large_skew_active:
            return seeds[: self.large_skew_seed_limit]

        current_l = np.clip(
            np.asarray(left_seed, dtype=np.float64),
            left_lower,
            left_upper,
        )
        current_r = np.clip(
            np.asarray(right_seed, dtype=np.float64),
            right_lower,
            right_upper,
        )
        template_l = np.clip(
            np.asarray(self.left_template_q, dtype=np.float64),
            left_lower,
            left_upper,
        )
        template_r = np.clip(
            np.asarray(self.right_template_q, dtype=np.float64),
            right_lower,
            right_upper,
        )

        rotation_deg = math.degrees(
            self._large_skew_rotation_rad
        )
        max_rotation_deg = max(
            rotation_deg + 1.0,
            math.degrees(self.max_side_axis_rotation_rad),
        )
        severity = float(
            np.clip(
                (
                    rotation_deg
                    - math.degrees(
                        self.large_skew_trigger_rotation_rad
                    )
                )
                / max(
                    1.0,
                    max_rotation_deg
                    - math.degrees(
                        self.large_skew_trigger_rotation_rad
                    ),
                ),
                0.0,
                1.0,
            )
        )

        axis_x_sign = (
            1.0
            if float(self._large_skew_axis_xy[0]) >= 0.0
            else -1.0
        )

        # 三种“当前姿态 ↔ 模板姿态”混合中心。
        for blend in (0.25, 0.55, 0.80):
            base_l = np.clip(
                blend * current_l
                + (1.0 - blend) * template_l,
                left_lower,
                left_upper,
            )
            base_r = np.clip(
                blend * current_r
                + (1.0 - blend) * template_r,
                right_lower,
                right_upper,
            )

            root_delta = math.radians(
                5.0 + 7.0 * severity
            )
            elbow_delta = math.radians(
                3.0 + 5.0 * severity
            )
            mid_delta = math.radians(
                5.0 + 6.0 * severity
            )

            for branch_sign in (+1.0, -1.0):
                signed = branch_sign * axis_x_sign

                l = base_l.copy()
                r = base_r.copy()

                # 只扰动 root / q3 / q4，避免无意义地破坏腕部。
                l[0] += signed * root_delta
                r[0] -= signed * root_delta

                l[2] -= signed * elbow_delta
                r[2] += signed * elbow_delta

                l[3] += signed * mid_delta
                r[3] -= signed * mid_delta

                seeds.append(
                    (
                        np.clip(l, left_lower, left_upper),
                        np.clip(r, right_lower, right_upper),
                    )
                )

        # 再加两个困难侧 shoulder 边界附近的候选。
        for margin_deg in (1.0, 4.0):
            l = current_l.copy()
            r = current_r.copy()
            margin = math.radians(margin_deg)

            if center_y >= 0.0:
                # 当前日志 center_y>0 时右臂是困难侧。
                r[1] = max(
                    float(right_lower[1]),
                    float(right_upper[1]) - margin,
                )
                r[0] += math.radians(8.0 + 4.0 * severity)
                r[2] -= math.radians(5.0 + 3.0 * severity)
                r[3] -= math.radians(8.0 + 4.0 * severity)
            else:
                l[1] = min(
                    float(left_upper[1]),
                    float(left_lower[1]) + margin,
                )
                l[0] -= math.radians(8.0 + 4.0 * severity)
                l[2] += math.radians(5.0 + 3.0 * severity)
                l[3] += math.radians(8.0 + 4.0 * severity)

            seeds.append(
                (
                    np.clip(l, left_lower, left_upper),
                    np.clip(r, right_lower, right_upper),
                )
            )

        # 去重。
        unique: List[Tuple[np.ndarray, np.ndarray]] = []
        for seed_l, seed_r in seeds:
            duplicate = any(
                np.allclose(
                    seed_l,
                    old_l,
                    atol=1e-7,
                )
                and np.allclose(
                    seed_r,
                    old_r,
                    atol=1e-7,
                )
                for old_l, old_r in unique
            )
            if duplicate:
                continue
            unique.append(
                (
                    np.asarray(seed_l, dtype=np.float64).copy(),
                    np.asarray(seed_r, dtype=np.float64).copy(),
                )
            )

        used = unique[: self.large_skew_seed_limit]

        self.get_logger().info(
            "Large-skew V2 IK seeds: "
            f"rotation={rotation_deg:.2f}deg, "
            f"center_y={center_y:+.4f}m, "
            f"generated={len(unique)}, "
            f"used={len(used)}"
        )
        return used

    # ==================================================================
    # 规划入口：仅规划期间切换 large-skew 阈值。
    # common-height / adaptive outset / wide descend / post-lift 都由父类继续执行。
    # ==================================================================
    def _plan_sequence(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> dict:
        self._update_large_skew_context(
            left_base,
            right_base,
        )

        snapshot = self._large_skew_threshold_snapshot()

        if self._large_skew_active:
            self._apply_large_skew_thresholds()
            self.get_logger().warning(
                "进入 Large-skew V2 位置优先模式："
                f"XY<={self.max_xy_error_m*100:.2f}cm, "
                f"lateral<={self.max_lateral_error_m*100:.2f}cm, "
                f"orientation<="
                f"{math.degrees(self.max_orientation_error_rad):.1f}deg, "
                f"forearm<="
                f"{math.degrees(self.max_forearm_tilt_rad):.1f}deg, "
                f"vertical<="
                f"{math.degrees(self.max_gripper_vertical_error_rad):.1f}deg, "
                f"template<="
                f"{math.degrees(self.max_template_rmse_rad):.1f}deg"
            )
        else:
            self.get_logger().info(
                "未触发 large-skew："
                "继续使用原抓取代码的严格 IK/安全阈值"
            )

        try:
            sequence = super()._plan_sequence(
                left_base,
                right_base,
            )

            sequence["large_skew_v2"] = bool(
                self._large_skew_active
            )
            sequence[
                "large_skew_rotation_deg"
            ] = math.degrees(
                self._large_skew_rotation_rad
            )
            sequence[
                "large_skew_dx_m"
            ] = float(
                self._large_skew_dx_m
            )

            if self._large_skew_active:
                final_plan = sequence["plans"][-1]
                lm = final_plan["metrics"]["left_final"]
                rm = final_plan["metrics"]["right_final"]

                self.get_logger().warning(
                    "Large-skew V2 最终抓取检查通过: "
                    f"L[XY={lm['xy_error_m']*100:.2f}cm, "
                    f"lat={lm['lateral_error_m']*100:.2f}cm, "
                    f"forearm={math.degrees(lm['forearm_tilt_rad']):.2f}deg, "
                    f"vertical="
                    f"{math.degrees(lm['gripper_vertical_error_rad']):.2f}deg, "
                    f"template={math.degrees(lm['template_rmse_rad']):.2f}deg]; "
                    f"R[XY={rm['xy_error_m']*100:.2f}cm, "
                    f"lat={rm['lateral_error_m']*100:.2f}cm, "
                    f"forearm={math.degrees(rm['forearm_tilt_rad']):.2f}deg, "
                    f"vertical="
                    f"{math.degrees(rm['gripper_vertical_error_rad']):.2f}deg, "
                    f"template={math.degrees(rm['template_rmse_rad']):.2f}deg]"
                )

            return sequence
        finally:
            # 不污染后续普通箱子的严格阈值。
            self._restore_thresholds(snapshot)


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[LargeSkewV2Bridge] = None
    try:
        node = LargeSkewV2Bridge()
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
                f"节点创建失败: "
                f"{type(exc).__name__}: {exc}"
            )
    finally:
        if node is not None:
            try:
                node.grasp_controller.cancel_active_goals()
            except Exception:
                pass
            try:
                node.grasp_controller.destroy_node()
            except Exception:
                pass
            try:
                node.destroy_node()
            except Exception:
                pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
