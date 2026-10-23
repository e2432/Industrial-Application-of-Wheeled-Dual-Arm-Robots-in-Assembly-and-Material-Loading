#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot A 双臂抓箱：
Large-Skew V7 / Far-Continuity

基于：
  bridge_xyz_offset_box_wide_descend_postlift_large_skew_v6_far_posture_fast.py

V7 针对 V6 dry-run 暴露出的明确问题：

  接近路点 1/2 关节跳变 22~53deg > 20deg

根因
====
底层 OffsetBoxReachBridge 的 seed 顺序原本为：
  1. template
  2. current / previous-waypoint
  3. template-current blend

而 staged path 虽然会把上一 waypoint 解传入下一 waypoint，
但下一次 IK 仍先尝试 template seed。
对于远目标，template 分支和远距离可达分支可能相差几十度，
于是出现“几何上可达，但相邻 waypoint 突然换支”的现象。

V7
==
仅 Far 模式：
  1. seed 按“距离当前/上一 waypoint 解”从近到远排序；
  2. transition waypoint 只使用最连续的 2 个 seed；
  3. approach_waypoints 临时增至 12（原来通常为 7）；
  4. Far 低位接近的左右外扩由 6cm/side 缩小到 1.5cm/side，
     避免远目标最终点可达、但最外侧 transition waypoint 不可达；
  5. Far forward retreat 最大值由 3cm 扩到 4cm，
     为更远目标增加一个更靠近机器人的候选；
  6. shoulder_clearance 在 V7 中改为 0.070rad（约 4.01deg），
     保留适量初始肩部避障抬举，不再抬得过高；
  7. max_waypoint_joint_delta_deg 仍保持原来的 20deg，不放宽；
  8. V6 的 Far template<=50deg、TCP/姿态/forearm/vertical/
     URDF limits/shared lift 等检查全部保留；
  9. 普通目标、large-skew 目标完全沿用 V6/V4/V3/V2。

注意：
  没有完整三维 collision scene。
  第一次必须 enable_motion:=false。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy

from bridge_xyz_offset_box_wide_descend_postlift_large_skew_v6_far_posture_fast import (
    FarPostureFastV6Bridge,
)


class FarContinuityV7Bridge(FarPostureFastV6Bridge):
    """V6 + Far waypoint continuity."""

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter(
            "v7_far_approach_waypoints",
            12,
        )
        self.declare_parameter(
            "v7_far_transition_seed_limit",
            2,
        )
        self.declare_parameter(
            "v7_transition_span_extra_cm",
            0.5,
        )

        # Far 目标专用：减小低位左右外扩，避免远目标 transition 过宽。
        self.declare_parameter(
            "v7_far_outward_offset_m",
            0.040,
        )
        # Far 目标专用：增加一个 4cm 的向机器人方向回退候选。
        self.declare_parameter(
            "v7_far_forward_retreat_max_m",
            0.040,
        )
        # 初始肩部只做适量抬举，约 4.01deg。
        self.declare_parameter(
            "v7_shoulder_clearance_rad",
            0.070,
        )
        # Far 目标：下降前两只夹爪在 base X 方向先向机器人侧退约 3cm。
        # 下降后随着 inward waypoint 逐步把这段 X 退让收回到 0。
        self.declare_parameter(
            "v7_far_pre_descend_x_retreat_m",
            0.0,
        )
        # 最终抓取姿态默认使用视觉原始点，不再永久加宽。高位接近阶段
        # 高位左右各外扩4cm由 v7_far_outward_offset_m 控制。
        self.declare_parameter(
            "v7_final_grasp_spacing_increase_m",
            0.0,
        )

        self.v7_far_approach_waypoints = int(
            self.get_parameter(
                "v7_far_approach_waypoints"
            ).value
        )
        self.v7_far_transition_seed_limit = int(
            self.get_parameter(
                "v7_far_transition_seed_limit"
            ).value
        )
        self.v7_transition_span_extra_m = (
            float(
                self.get_parameter(
                    "v7_transition_span_extra_cm"
                ).value
            )
            / 100.0
        )

        self.v7_far_outward_offset_m = float(
            self.get_parameter(
                "v7_far_outward_offset_m"
            ).value
        )
        self.v7_far_forward_retreat_max_m = float(
            self.get_parameter(
                "v7_far_forward_retreat_max_m"
            ).value
        )
        self.v7_shoulder_clearance_rad = float(
            self.get_parameter(
                "v7_shoulder_clearance_rad"
            ).value
        )
        self.v7_far_pre_descend_x_retreat_m = float(
            self.get_parameter(
                "v7_far_pre_descend_x_retreat_m"
            ).value
        )
        self.v7_final_grasp_spacing_increase_m = float(
            self.get_parameter(
                "v7_final_grasp_spacing_increase_m"
            ).value
        )

        if abs(self.v7_far_pre_descend_x_retreat_m) > 1e-9:
            self.get_logger().warning(
                "本版禁用 base-X pre-descend retreat，忽略该参数传入值"
            )
        self.v7_far_pre_descend_x_retreat_m = 0.0

        if not 8 <= self.v7_far_approach_waypoints <= 12:
            raise RuntimeError(
                "v7_far_approach_waypoints 必须在 [8,12]"
            )
        if not 1 <= self.v7_far_transition_seed_limit <= 3:
            raise RuntimeError(
                "v7_far_transition_seed_limit 必须在 [1,3]"
            )
        if not 0.002 <= self.v7_transition_span_extra_m <= 0.015:
            raise RuntimeError(
                "v7_transition_span_extra_cm 必须在 [0.2,1.5]cm"
            )
        if not 0.010 <= self.v7_far_outward_offset_m <= 0.040:
            raise RuntimeError(
                "v7_far_outward_offset_m 必须在 [0.010,0.040]m"
            )
        if not 0.030 <= self.v7_far_forward_retreat_max_m <= 0.040:
            raise RuntimeError(
                "v7_far_forward_retreat_max_m 必须在 [0.030,0.040]m"
            )
        if not 0.030 <= self.v7_shoulder_clearance_rad <= 0.070:
            raise RuntimeError(
                "v7_shoulder_clearance_rad 必须在 [0.030,0.070]rad"
            )
        if not 0.0 <= self.v7_far_pre_descend_x_retreat_m <= 0.050:
            raise RuntimeError(
                "v7_far_pre_descend_x_retreat_m 必须在 [0.0,0.050]m"
            )
        if not 0.0 <= self.v7_final_grasp_spacing_increase_m <= 0.120:
            raise RuntimeError(
                "v7_final_grasp_spacing_increase_m 必须在 [0.0,0.120]m"
            )

        # 父类初始化时会先读取/打印原 shoulder_clearance=0.070。
        # 从这里开始，实际 V7 规划与执行使用较温和的 0.070rad。
        self.shoulder_clearance = self.v7_shoulder_clearance_rad

        # 本版统一采用左右各外扩 4cm：只增加最高位临时外扩，
        # 最终抓取点仍使用视觉原始位置。
        self.outward_offset_m = self.v7_far_outward_offset_m

        # V5 默认最大 forward retreat=3cm；V7 为 far 目标扩到 4cm。
        self.far_forward_retreat_max_m = (
            self.v7_far_forward_retreat_max_m
        )

        self._v7_transition_solving = False
        self._v7_plan_call_count = 0

        # Far 路径反向 warm-start：
        # staged 基类虽然先求 final_preplan，但第一条最外侧 waypoint 原本仍从
        # “机器人当前姿态/current_lift”起算。远目标时很容易重新掉回错误 IK 分支。
        # V7 只缓存最近一次成功的 Far 最终点左右臂关节解。
        # 第一条 transition 仅 warm-start q；Z/lift 参考严格沿用原 staged planner。
        self._v7_far_final_warm_left = None
        self._v7_far_final_warm_right = None
        self._v7_transition_chain_started = False

        self.get_logger().warning(
            "Large-skew V7 Lateral-Out-3cm / Descend-Final-Z 已启用："
            f"Far approach waypoints={self.v7_far_approach_waypoints}, "
            f"transition seeds={self.v7_far_transition_seed_limit}, "
            f"Far outward={self.v7_far_outward_offset_m*100:.1f}cm/side, "
            f"forward retreat<={self.far_forward_retreat_max_m*100:.1f}cm, "
            f"shoulder_clearance={self.shoulder_clearance:.3f}rad, "
            f"pre-descend X retreat="
            f"{self.v7_far_pre_descend_x_retreat_m*100:.1f}cm；"
            f"相邻关节跳变上限仍保持 "
            f"{math.degrees(self.max_waypoint_joint_delta_rad):.1f}deg，"
            "不放宽安全门槛"
        )

    # ==================================================================
    # Far transition detection
    #
    # V4 自带 _is_transition_target() 只对 large-skew 生效。
    # Far 模式单独按左右目标 span 判断。
    # ==================================================================
    def _v7_is_far_transition_target(
        self,
        left_target: np.ndarray,
        right_target: np.ndarray,
    ) -> bool:
        if not self._far_target_active:
            return False

        nominal = float(
            getattr(
                self,
                "_v4_nominal_final_span_xy",
                0.0,
            )
        )
        if nominal <= 1e-9:
            return False

        span = float(
            np.linalg.norm(
                np.asarray(
                    left_target,
                    dtype=np.float64,
                )[:2]
                - np.asarray(
                    right_target,
                    dtype=np.float64,
                )[:2]
            )
        )

        return bool(
            span
            >= nominal + self.v7_transition_span_extra_m
        )

    # ==================================================================
    # 在 _solve_asymmetric 周围标记“当前是否是 Far transition”。
    # ==================================================================
    def _solve_asymmetric(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
        left_seed: np.ndarray,
        right_seed: np.ndarray,
        current_lift: float,
        current_waist: float,
        limits,
    ) -> dict:
        old_flag = bool(self._v7_transition_solving)

        is_transition = self._v7_is_far_transition_target(
            left_base,
            right_base,
        )
        self._v7_transition_solving = bool(is_transition)

        if self._v7_transition_solving:
            span = float(
                np.linalg.norm(
                    np.asarray(left_base)[:2]
                    - np.asarray(right_base)[:2]
                )
            )
            self.get_logger().info(
                "V7 Far 连续路点求解: "
                f"span={span:.4f}m, "
                f"nominal="
                f"{float(getattr(self, '_v4_nominal_final_span_xy', 0.0)):.4f}m"
            )

        # --------------------------------------------------------------
        # 本版不做 base-X 前后退让。
        # 左右外扩完全由 outward_offset_m 实现。
        # --------------------------------------------------------------
        solve_left_seed = np.asarray(left_seed, dtype=np.float64)
        solve_right_seed = np.asarray(right_seed, dtype=np.float64)

        # Z-SAFE:
        # current_lift 是原始 staged planner 传进来的真实 Z/lift 参考。
        # warm-start 绝不覆盖它，只改变左右臂关节 seed。
        solve_lift = float(current_lift)

        # 关键修复：
        # staged 基类的路径顺序是 outer -> final，但它在进入路径前已经成功
        # 求过 final_preplan。第一条 outer waypoint 不应该再从机器人当前状态/
        # 当前 lift 重新起算，而应从这个已验证 final 解向外小步反推。
        if (
            self._far_target_active
            and self._v7_transition_solving
            and not self._v7_transition_chain_started
            and self._v7_far_final_warm_left is not None
            and self._v7_far_final_warm_right is not None
        ):
            solve_left_seed = np.asarray(
                self._v7_far_final_warm_left,
                dtype=np.float64,
            ).copy()
            solve_right_seed = np.asarray(
                self._v7_far_final_warm_right,
                dtype=np.float64,
            ).copy()
            self.get_logger().warning(
                "V7 Far transition 首路点 Z-SAFE warm-start："
                "只复用已验证 final-preplan 的左右臂关节 seed；"
                f"Z/lift reference 保持原 staged current_lift={solve_lift:+.4f}m，"
                "绝不使用 final-preplan lift 覆盖"
            )

        try:
            plan = super()._solve_asymmetric(
                left_base,
                right_base,
                solve_left_seed,
                solve_right_seed,
                solve_lift,
                current_waist,
                limits,
            )

            if self._far_target_active:
                if self._v7_transition_solving:
                    # 第一条 transition 成功后，staged 基类会把这个 plan 作为
                    # 下一 waypoint 的 left_seed/right_seed；后续不再强制 final cache。
                    self._v7_transition_chain_started = True
                else:
                    # final_preplan / nominal final 成功：缓存它作为下一条
                    # transition chain 的起点。
                    self._v7_far_final_warm_left = np.asarray(
                        plan["left"],
                        dtype=np.float64,
                    ).copy()
                    self._v7_far_final_warm_right = np.asarray(
                        plan["right"],
                        dtype=np.float64,
                    ).copy()
                    self._v7_transition_chain_started = False

                    self.get_logger().info(
                        "V7 Far final-preplan 关节分支已缓存："
                        f"planned_lift="
                        f"{float(plan['body']['base_elevator_joint']):+.4f}m；"
                        "仅缓存左右臂 q，Z/lift reference 不缓存、不覆盖"
                    )

            return plan

        finally:
            self._v7_transition_solving = old_flag

    # ==================================================================
    # Seed continuity
    #
    # 父类先生成合法 seed；V7 不发明越界关节值，只重新排序。
    # 排序指标 = 左右臂相对传入 seed 的最大关节变化。
    #
    # 对 staged path：
    #   left_seed/right_seed 正是上一 waypoint 的 plan，
    # 因此排序后 current/previous branch 会优先于 template branch。
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
    ):
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

        if not self._far_target_active:
            return seeds

        reference_l = np.asarray(
            left_seed,
            dtype=np.float64,
        )
        reference_r = np.asarray(
            right_seed,
            dtype=np.float64,
        )

        def distance(pair) -> float:
            seed_l, seed_r = pair
            return max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(seed_l) - reference_l
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            np.asarray(seed_r) - reference_r
                        )
                    )
                ),
            )

        seeds.sort(key=distance)

        if self._v7_transition_solving:
            used = seeds[
                : self.v7_far_transition_seed_limit
            ]
        else:
            used = seeds[: self.far_seed_limit]

        distances_deg = [
            round(math.degrees(distance(pair)), 2)
            for pair in used
        ]

        self.get_logger().info(
            "V7 Far seed continuity: "
            f"transition={self._v7_transition_solving}, "
            f"generated={len(seeds)}, "
            f"used={len(used)}, "
            f"seed_delta_deg={distances_deg}"
        )

        return used

    # ==================================================================
    # Far path densification
    #
    # V5 会在这里先识别 far_target，再进入每个 forward candidate。
    # V7 在进入父规划前临时把 approach_waypoints 提高到 12。
    # 20deg hard gate 不改变。
    # ==================================================================
    def _plan_sequence(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> dict:
        self._v7_plan_call_count += 1
        plan_call = self._v7_plan_call_count
        self.get_logger().warning(
            "[GRASP_PLANNING] ENTER "
            f"call={plan_call}, "
            f"left_raw={np.round(np.asarray(left_base, dtype=float), 5)}, "
            f"right_raw={np.round(np.asarray(right_base, dtype=float), 5)}"
        )
        # 最终抓取姿态以视觉点中点为中心对称加宽。使用两点在 base XY
        # 平面内的连线作为夹紧轴，避免把“左右”硬编码成 base Y；箱体有
        # 偏航时仍能保证每侧精确外移总增量的一半。
        left_base = np.asarray(left_base, dtype=float).copy()
        right_base = np.asarray(right_base, dtype=float).copy()
        original_center = 0.5 * (left_base + right_base)
        side_delta_xy = left_base[:2] - right_base[:2]
        original_spacing_xy = float(np.linalg.norm(side_delta_xy))
        if original_spacing_xy <= 1e-6:
            raise RuntimeError(
                "左右抓取点水平间距过小，无法计算对称外扩方向"
            )
        side_axis_xy = side_delta_xy / original_spacing_xy
        per_side_offset = 0.5 * self.v7_final_grasp_spacing_increase_m
        left_base[:2] += side_axis_xy * per_side_offset
        right_base[:2] -= side_axis_xy * per_side_offset
        widened_center = 0.5 * (left_base + right_base)
        widened_spacing_xy = float(
            np.linalg.norm(left_base[:2] - right_base[:2])
        )
        self.get_logger().warning(
            "V7 最终抓取姿态对称加宽："
            f"total=+{self.v7_final_grasp_spacing_increase_m*100:.1f}cm, "
            f"per_side={per_side_offset*100:.1f}cm, "
            f"spacing_xy={original_spacing_xy*100:.1f}"
            f"->{widened_spacing_xy*100:.1f}cm, "
            f"center_shift="
            f"{np.linalg.norm(widened_center-original_center)*1000:.3f}mm"
        )

        # 先计算一次 context，便于决定是否使用 Far 细分。
        self._update_far_context(
            left_base,
            right_base,
        )

        if not self._far_target_active:
            try:
                sequence = super()._plan_sequence(
                    left_base,
                    right_base,
                )
            except Exception as exc:
                self.get_logger().error(
                    "[GRASP_PLANNING] FAILED "
                    f"call={plan_call}, mode=normal, "
                    f"error={type(exc).__name__}: {exc}"
                )
                raise
            self.get_logger().warning(
                "[GRASP_PLANNING] PASSED "
                f"call={plan_call}, mode=normal, "
                f"waypoints={len(sequence.get('plans', []))}"
            )
            return sequence

        old_waypoints = int(
            self.approach_waypoints
        )
        old_outward = float(
            self.outward_offset_m
        )

        try:
            self.approach_waypoints = max(
                old_waypoints,
                self.v7_far_approach_waypoints,
            )

            # 当前执行要求：左右各外扩 4cm，最终抓取点本身不变。
            self.outward_offset_m = self.v7_far_outward_offset_m

            self.get_logger().warning(
                "V7 Far 接近路径连续化："
                f"approach_waypoints "
                f"{old_waypoints} -> {self.approach_waypoints}, "
                f"outward {old_outward*100:.1f} -> "
                f"{self.outward_offset_m*100:.1f}cm/side；"
                f"joint jump hard limit 仍为 "
                f"{math.degrees(self.max_waypoint_joint_delta_rad):.1f}deg"
            )

            sequence = super()._plan_sequence(
                left_base,
                right_base,
            )

            # 最终额外统计整条路径真实最大关节步长。
            plans = sequence.get("plans", [])
            max_delta = 0.0
            max_index = -1

            for index in range(1, len(plans)):
                delta = max(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(plans[index]["left"])
                                - np.asarray(plans[index - 1]["left"])
                            )
                        )
                    ),
                    float(
                        np.max(
                            np.abs(
                                np.asarray(plans[index]["right"])
                                - np.asarray(plans[index - 1]["right"])
                            )
                        )
                    ),
                )
                if delta > max_delta:
                    max_delta = delta
                    max_index = index

            sequence["v7_max_waypoint_joint_delta_rad"] = float(
                max_delta
            )

            self.get_logger().warning(
                "V7 Far 路径连续性通过："
                f"waypoints={len(plans)}, "
                f"max_joint_step="
                f"{math.degrees(max_delta):.2f}deg "
                f"@ waypoint {max_index}, "
                f"limit="
                f"{math.degrees(self.max_waypoint_joint_delta_rad):.2f}deg"
            )

            self.get_logger().warning(
                "[GRASP_PLANNING] PASSED "
                f"call={plan_call}, mode=far, waypoints={len(plans)}, "
                f"max_joint_step_deg={math.degrees(max_delta):.2f}"
            )

            return sequence

        except Exception as exc:
            self.get_logger().error(
                "[GRASP_PLANNING] FAILED "
                f"call={plan_call}, mode=far, "
                f"error={type(exc).__name__}: {exc}"
            )
            raise

        finally:
            self.approach_waypoints = old_waypoints
            self.outward_offset_m = old_outward



    def _execute_sequence(self, sequence: dict) -> None:
        """新执行顺序：
        lift最高 -> 左右各外扩4cm -> lift下降最终抓取高度
        -> 双臂内收到最终抓取点 -> 两边夹爪闭合。
        """
        controller = self.grasp_controller
        log = self.get_logger().info

        plans = list(sequence.get("plans", []))
        if not plans:
            raise RuntimeError("执行序列为空")

        outer_plan = plans[0]
        final_plan = plans[-1]
        final_lift = float(
            final_plan["body"]["base_elevator_joint"]
        )

        # 1) lift 先升到最高避障位（当前基类默认 0.25m）。
        log(
            "[1/8] lift 先上升到最高避障位: "
            f"{self.prelift_target_m:+.4f}m"
        )
        if not controller.animate(
            {"base_elevator_joint": self.prelift_target_m},
            duration=self.prelift_duration_s,
            steps=80,
        ):
            raise RuntimeError("lift 抬升失败")

        # 2) 夹爪先张开。
        log("[2/8] 张开左右夹爪")
        if not controller.command_grippers(0.025):
            raise RuntimeError("夹爪张开失败")

        # 3) 高位肩部避障。
        left_outer_shoulder = float(outer_plan["left"][1])
        right_outer_shoulder = float(outer_plan["right"][1])
        left_clearance = (
            left_outer_shoulder + abs(self.shoulder_clearance)
        )
        right_clearance = (
            right_outer_shoulder - abs(self.shoulder_clearance)
        )

        for name, value in (
            ("arm_left1_arm_left2_joint", left_clearance),
            ("arm_right1_arm_right2_joint", right_clearance),
        ):
            lower, upper, _ = controller._limits[name]
            if not lower <= value <= upper:
                raise RuntimeError(
                    f"{name} 避障角越界: {value:+.4f}, "
                    f"limit=[{lower:+.4f}, {upper:+.4f}]"
                )

        log("[3/8] lift 保持最高位，肩部抬起避障")
        if not controller.animate(
            {
                "arm_left1_arm_left2_joint": left_clearance,
                "arm_right1_arm_right2_joint": right_clearance,
            },
            duration=1.2,
            steps=60,
        ):
            raise RuntimeError("肩部避障动作失败")

        # 4) 高位时先让两侧腕部按规划距离向外移动。
        outer_targets = self._full_joint_targets(
            outer_plan,
            include_lift=False,
        )
        outer_targets.pop("arm_left1_arm_left2_joint", None)
        outer_targets.pop("arm_right1_arm_right2_joint", None)

        log(
            "[4/8] lift 保持最高位，左右腕部沿夹紧轴各向外 "
            f"{self.outward_offset_m*100:.1f}cm"
        )
        if not controller.animate(
            outer_targets,
            duration=self.outer_move_duration_s,
            steps=120,
        ):
            raise RuntimeError(
                "高位左右各外扩"
                f"{self.outward_offset_m*100:.1f}cm失败"
            )

        # 5) 肩部落到外扩规划姿态。
        log(
            "[5/8] 高位肩部落到 "
            f"{self.outward_offset_m*100:.1f}cm 外扩规划位"
        )
        if not controller.animate(
            {
                "arm_left1_arm_left2_joint": left_outer_shoulder,
                "arm_right1_arm_right2_joint": right_outer_shoulder,
            },
            duration=1.2,
            steps=60,
        ):
            raise RuntimeError("肩部落到外扩位失败")

        # 6) 保持左右外扩，lift 直接降到最终抓取高度。
        log(
            "[6/8] 保持左右各外扩"
            f"{self.outward_offset_m*100:.1f}cm，"
            "lift 下降到最终抓取点高度: "
            f"{final_lift:+.4f}m"
        )
        if not controller.animate(
            {"base_elevator_joint": final_lift},
            duration=self.descend_duration_s,
            steps=100,
        ):
            raise RuntimeError("lift 下降到最终抓取高度失败")

        # 7) lift 不再变化，左右臂从外侧内收到对称加宽后的抓取点。
        log(
            "[7/8] lift 保持最终高度，双臂由外侧逐段内收到"
            "对称加宽后的最终抓取点"
        )
        inward_plans = plans[1:]
        for index, plan in enumerate(inward_plans, start=1):
            targets = self._full_joint_targets(
                plan,
                include_lift=False,
            )
            if not controller.animate(
                targets,
                duration=self.inward_step_duration_s,
                steps=60,
            ):
                raise RuntimeError(
                    f"内收路点 {index}/{len(inward_plans)} 执行失败"
                )

        # 8) 到最终抓取位后，两边夹爪同时闭合。
        log("[8/8] 到达对称加宽后的最终抓取点，两边夹爪同时闭合")
        if not controller.command_grippers(0.0):
            raise RuntimeError("夹爪闭合失败")

        log(
            "✅ 新顺序完成：lift最高 -> 左右各外扩"
            f"{self.outward_offset_m*100:.1f}cm -> "
            "下降最终Z -> 内收 -> 两边闭合"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[FarContinuityV7Bridge] = None

    try:
        node = FarContinuityV7Bridge()
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
