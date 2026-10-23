#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot A 双臂抓箱：
Large-Skew V5 / Far-Target Fast Reach

目标
====
在 V4 全部能力基础上，专门解决：
  1. 箱体整体离机器人更远 / 偏向一侧时，某一只手差约 1~2cm 无法通过；
  2. 非 large-skew 目标仍继承 V2 的 12 seeds + 180/360 iterations，
     再叠加 adaptive 17 candidates，导致规划非常慢；
  3. 旧 fallback 主要沿箱体 side-axis 移动，不能有效解决“前后距离太远”。

V5 不修改视觉节点、不修改手眼标定，也不改变执行动作顺序：
  预抬升 -> 张开 -> 外扩 -> 下降 -> 多 waypoint 内收
  -> 闭合验证 -> post-lift / 必要时二次闭合。

Far 模式
========
仅当目标不是 large-skew，且满足以下任一条件时启用：
  - |center_y| >= far_trigger_center_y_m（默认 0.10m）
  - center_x >= far_trigger_center_x_m（默认 0.85m）

Far 模式：
  A. 位置优先 IK；
  B. 最终 XY 默认允许 1.70cm；
  C. 真正夹紧方向 lateral 默认仍限制 0.90cm；
  D. template 默认 <=30deg，forearm<=5deg，vertical<=5deg；
  E. 每个 IK 只保留 4 个高价值 seed，90/160 iterations；
  F. 禁用原 17 路 adaptive side-outset 穷举；
  G. 禁用旧单侧 side-outset fallback；
  H. 只尝试 3 个 forward capture 候选：
       0cm, 1.5cm, 3.0cm
     两只 TCP 一起沿箱体 forward 轴“向机器人”平移，
     左右抓取间距保持不变。

如果普通目标或 large-skew 目标，则完全沿用 V4。

安全
====
本系统仍没有完整 3D collision scene。
第一次必须 enable_motion:=false。
forward retreat 会让最终接触点沿箱体长边方向偏离视觉中点，默认最大 3cm；
必须确认夹爪仍落在箱体侧壁有效接触区域。
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import rclpy

from bridge_xyz_offset_box_wide_descend_postlift_large_skew_v4_transition_relaxed import (
    LargeSkewTransitionV4Bridge,
)


class FarTargetFastV5Bridge(LargeSkewTransitionV4Bridge):
    """V4 + 远目标快速位置优先抓取。"""

    def __init__(self) -> None:
        super().__init__()

        # --------------------------------------------------------------
        # Far 模式触发。
        # --------------------------------------------------------------
        self.declare_parameter(
            "far_trigger_center_y_m",
            0.10,
        )
        self.declare_parameter(
            "far_trigger_center_x_m",
            0.85,
        )

        # --------------------------------------------------------------
        # Far 最终抓取阈值。
        # 不把 lateral 一起粗暴放到 1.7cm。
        # --------------------------------------------------------------
        self.declare_parameter(
            "far_max_xy_error_cm",
            1.70,
        )
        self.declare_parameter(
            "far_max_lateral_error_cm",
            0.90,
        )
        self.declare_parameter(
            "far_max_orientation_error_deg",
            6.0,
        )
        self.declare_parameter(
            "far_max_forearm_tilt_deg",
            5.0,
        )
        self.declare_parameter(
            "far_max_vertical_error_deg",
            5.0,
        )
        self.declare_parameter(
            "far_max_template_rmse_deg",
            30.0,
        )

        # --------------------------------------------------------------
        # Far 快速 IK。
        # --------------------------------------------------------------
        self.declare_parameter(
            "far_seed_limit",
            4,
        )
        self.declare_parameter(
            "far_reach_iterations",
            90,
        )
        self.declare_parameter(
            "far_refine_iterations",
            160,
        )
        self.declare_parameter(
            "far_line_search_steps",
            5,
        )

        # --------------------------------------------------------------
        # 两只 TCP 一起沿 box-forward 向机器人回退。
        # --------------------------------------------------------------
        self.declare_parameter(
            "far_forward_retreat_step_m",
            0.015,
        )
        self.declare_parameter(
            "far_forward_retreat_max_m",
            0.030,
        )

        self.far_trigger_center_y_m = float(
            self.get_parameter(
                "far_trigger_center_y_m"
            ).value
        )
        self.far_trigger_center_x_m = float(
            self.get_parameter(
                "far_trigger_center_x_m"
            ).value
        )

        self.far_max_xy_error_m = (
            float(
                self.get_parameter(
                    "far_max_xy_error_cm"
                ).value
            )
            / 100.0
        )
        self.far_max_lateral_error_m = (
            float(
                self.get_parameter(
                    "far_max_lateral_error_cm"
                ).value
            )
            / 100.0
        )
        self.far_max_orientation_rad = math.radians(
            float(
                self.get_parameter(
                    "far_max_orientation_error_deg"
                ).value
            )
        )
        self.far_max_forearm_rad = math.radians(
            float(
                self.get_parameter(
                    "far_max_forearm_tilt_deg"
                ).value
            )
        )
        self.far_max_vertical_rad = math.radians(
            float(
                self.get_parameter(
                    "far_max_vertical_error_deg"
                ).value
            )
        )
        self.far_max_template_rad = math.radians(
            float(
                self.get_parameter(
                    "far_max_template_rmse_deg"
                ).value
            )
        )

        self.far_seed_limit = int(
            self.get_parameter(
                "far_seed_limit"
            ).value
        )
        self.far_reach_iterations = int(
            self.get_parameter(
                "far_reach_iterations"
            ).value
        )
        self.far_refine_iterations = int(
            self.get_parameter(
                "far_refine_iterations"
            ).value
        )
        self.far_line_search_steps = int(
            self.get_parameter(
                "far_line_search_steps"
            ).value
        )

        self.far_forward_retreat_step_m = float(
            self.get_parameter(
                "far_forward_retreat_step_m"
            ).value
        )
        self.far_forward_retreat_max_m = float(
            self.get_parameter(
                "far_forward_retreat_max_m"
            ).value
        )

        # --------------------------------------------------------------
        # 参数范围。
        # --------------------------------------------------------------
        if not 0.05 <= self.far_trigger_center_y_m <= 0.25:
            raise RuntimeError(
                "far_trigger_center_y_m 必须在 [0.05,0.25]m"
            )
        if not 0.50 <= self.far_trigger_center_x_m <= 1.20:
            raise RuntimeError(
                "far_trigger_center_x_m 必须在 [0.50,1.20]m"
            )
        if not 0.010 <= self.far_max_xy_error_m <= 0.022:
            raise RuntimeError(
                "far_max_xy_error_cm 必须在 [1.0,2.2]cm"
            )
        if not 0.005 <= self.far_max_lateral_error_m <= 0.012:
            raise RuntimeError(
                "far_max_lateral_error_cm 必须在 [0.5,1.2]cm"
            )
        if not math.radians(4.0) <= self.far_max_orientation_rad <= math.radians(10.0):
            raise RuntimeError(
                "far_max_orientation_error_deg 必须在 [4,10]deg"
            )
        if not math.radians(3.0) <= self.far_max_forearm_rad <= math.radians(8.0):
            raise RuntimeError(
                "far_max_forearm_tilt_deg 必须在 [3,8]deg"
            )
        if not math.radians(3.0) <= self.far_max_vertical_rad <= math.radians(8.0):
            raise RuntimeError(
                "far_max_vertical_error_deg 必须在 [3,8]deg"
            )
        if not math.radians(20.0) <= self.far_max_template_rad <= math.radians(35.0):
            raise RuntimeError(
                "far_max_template_rmse_deg 必须在 [20,35]deg"
            )
        if not 2 <= self.far_seed_limit <= 6:
            raise RuntimeError(
                "far_seed_limit 必须在 [2,6]"
            )
        if not 50 <= self.far_reach_iterations <= 160:
            raise RuntimeError(
                "far_reach_iterations 必须在 [50,160]"
            )
        if not 80 <= self.far_refine_iterations <= 260:
            raise RuntimeError(
                "far_refine_iterations 必须在 [80,260]"
            )
        if not 3 <= self.far_line_search_steps <= 6:
            raise RuntimeError(
                "far_line_search_steps 必须在 [3,6]"
            )
        if not 0.005 <= self.far_forward_retreat_step_m <= 0.020:
            raise RuntimeError(
                "far_forward_retreat_step_m 必须在 [0.005,0.020]m"
            )
        if not 0.010 <= self.far_forward_retreat_max_m <= 0.040:
            raise RuntimeError(
                "far_forward_retreat_max_m 必须在 [0.010,0.040]m"
            )

        self._far_target_active = False
        self._far_center_x_m = 0.0
        self._far_center_y_m = 0.0
        self._far_side_axis_xy = np.array(
            [0.0, 1.0],
            dtype=np.float64,
        )
        self._far_forward_axis_xy = np.array(
            [1.0, 0.0],
            dtype=np.float64,
        )

        self.get_logger().warning(
            "Large-skew V5/Far-Fast 已启用："
            f"far trigger |center_y|>={self.far_trigger_center_y_m:.3f}m "
            f"或 center_x>={self.far_trigger_center_x_m:.3f}m；"
            f"far XY<={self.far_max_xy_error_m*100:.2f}cm, "
            f"lateral<={self.far_max_lateral_error_m*100:.2f}cm；"
            f"IK seeds={self.far_seed_limit}, "
            f"iterations={self.far_reach_iterations}/"
            f"{self.far_refine_iterations}；"
            f"forward retreat<={self.far_forward_retreat_max_m*100:.1f}cm"
        )

    # ==================================================================
    # Far context
    # ==================================================================
    def _update_far_context(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> None:
        left = np.asarray(left_base, dtype=np.float64)
        right = np.asarray(right_base, dtype=np.float64)

        center_x = 0.5 * float(left[0] + right[0])
        center_y = 0.5 * float(left[1] + right[1])

        delta_xy = left[:2] - right[:2]
        side_norm = float(np.linalg.norm(delta_xy))
        if side_norm < 1e-9:
            side_axis = np.array(
                [0.0, 1.0],
                dtype=np.float64,
            )
        else:
            side_axis = delta_xy / side_norm

        forward_axis = np.array(
            [side_axis[1], -side_axis[0]],
            dtype=np.float64,
        )
        # 统一让 box-forward 的 X 分量指向机器人前方 +X。
        if float(forward_axis[0]) < 0.0:
            forward_axis *= -1.0

        rotation = math.atan2(
            abs(float(side_axis[0])),
            abs(float(side_axis[1])),
        )
        dx = abs(float(left[0] - right[0]))

        would_large_skew = bool(
            rotation >= self.large_skew_trigger_rotation_rad
            or dx >= self.large_skew_trigger_dx_m
        )

        self._far_center_x_m = center_x
        self._far_center_y_m = center_y
        self._far_side_axis_xy = side_axis.copy()
        self._far_forward_axis_xy = forward_axis.copy()

        self._far_target_active = bool(
            not would_large_skew
            and (
                abs(center_y) >= self.far_trigger_center_y_m
                or center_x >= self.far_trigger_center_x_m
            )
        )

        self.get_logger().info(
            "V5 reach 诊断: "
            f"center_x={center_x:+.4f}m, "
            f"center_y={center_y:+.4f}m, "
            f"rotation={math.degrees(rotation):.2f}deg, "
            f"box_forward={np.round(forward_axis, 5)}, "
            f"far_target={self._far_target_active}, "
            f"would_large_skew={would_large_skew}"
        )

    # ==================================================================
    # Far mode thresholds
    # ==================================================================
    def _far_threshold_snapshot(self):
        return (
            float(self.max_xy_error_m),
            float(self.max_lateral_error_m),
            float(self.max_orientation_error_rad),
            float(self.max_forearm_tilt_rad),
            float(self.max_gripper_vertical_error_rad),
            float(self.max_template_rmse_rad),
        )

    def _apply_far_thresholds(self) -> None:
        self.max_xy_error_m = max(
            float(self.max_xy_error_m),
            self.far_max_xy_error_m,
        )
        self.max_lateral_error_m = max(
            float(self.max_lateral_error_m),
            self.far_max_lateral_error_m,
        )
        self.max_orientation_error_rad = max(
            float(self.max_orientation_error_rad),
            self.far_max_orientation_rad,
        )
        self.max_forearm_tilt_rad = max(
            float(self.max_forearm_tilt_rad),
            self.far_max_forearm_rad,
        )
        self.max_gripper_vertical_error_rad = max(
            float(self.max_gripper_vertical_error_rad),
            self.far_max_vertical_rad,
        )
        self.max_template_rmse_rad = max(
            float(self.max_template_rmse_rad),
            self.far_max_template_rad,
        )

    def _restore_far_thresholds(self, snapshot) -> None:
        (
            self.max_xy_error_m,
            self.max_lateral_error_m,
            self.max_orientation_error_rad,
            self.max_forearm_tilt_rad,
            self.max_gripper_vertical_error_rad,
            self.max_template_rmse_rad,
        ) = snapshot

    # ==================================================================
    # Far mode position-first residual.
    #
    # 与 V2 large-skew 类似，但专门对非 large-skew 的远目标启用。
    # ==================================================================
    def _stage_residual(
        self,
        state: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
        reach_stage: bool,
    ) -> np.ndarray:
        if not self._far_target_active:
            return super()._stage_residual(
                state,
                left_target,
                right_target,
                reach_stage,
            )

        residual = self._joint_xyz_residual(
            state,
            left_target,
            right_target,
        ).copy()

        for start in (0, 14):
            if reach_stage:
                # 先把 TCP 拉到目标附近，模板/姿态只做弱引导。
                residual[start + 0] *= 6.0
                residual[start + 1] *= 3.0
                residual[start + 2] *= 3.0
                residual[start + 3 : start + 6] *= 0.25
                residual[start + 6] *= 0.30
                residual[start + 7 : start + 14] *= 0.08
            else:
                # refine 阶段恢复一部分姿态，但仍以位置为主。
                residual[start + 0] *= 3.5
                residual[start + 1] *= 2.2
                residual[start + 2] *= 2.2
                residual[start + 3 : start + 6] *= 0.70
                residual[start + 6] *= 0.80
                residual[start + 7 : start + 14] *= 0.25

        return residual

    # ==================================================================
    # Far mode seed 数量限制。
    # 父类仍负责构造 template/current/困难侧肩边界等高价值 seed。
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

        used = seeds[: self.far_seed_limit]
        self.get_logger().info(
            "V5 Far 快速 seed: "
            f"generated={len(seeds)}, used={len(used)}"
        )
        return used

    # ==================================================================
    # Forward retreat candidates.
    # 0 -> step -> ... -> max
    # ==================================================================
    def _far_retreat_values(self) -> List[float]:
        values = [0.0]
        value = self.far_forward_retreat_step_m
        while value < self.far_forward_retreat_max_m - 1e-9:
            values.append(float(value))
            value += self.far_forward_retreat_step_m
        values.append(float(self.far_forward_retreat_max_m))
        return sorted(
            set(round(v, 9) for v in values)
        )

    # ==================================================================
    # Far 最终复核。
    # ==================================================================
    def _verify_far_final(self, sequence: dict) -> None:
        final_plan = sequence["plans"][-1]
        lm = final_plan["metrics"]["left_final"]
        rm = final_plan["metrics"]["right_final"]

        failures = []

        for label, metrics in (
            ("L", lm),
            ("R", rm),
        ):
            if metrics["xy_error_m"] > self.far_max_xy_error_m:
                failures.append(
                    f"{label}.XY="
                    f"{metrics['xy_error_m']*100:.2f}cm"
                )
            if (
                metrics["lateral_error_m"]
                > self.far_max_lateral_error_m
            ):
                failures.append(
                    f"{label}.lateral="
                    f"{metrics['lateral_error_m']*100:.2f}cm"
                )
            if (
                metrics["orientation_error_rad"]
                > self.far_max_orientation_rad
            ):
                failures.append(
                    f"{label}.ori="
                    f"{math.degrees(metrics['orientation_error_rad']):.2f}deg"
                )
            if (
                metrics["forearm_tilt_rad"]
                > self.far_max_forearm_rad
            ):
                failures.append(
                    f"{label}.forearm="
                    f"{math.degrees(metrics['forearm_tilt_rad']):.2f}deg"
                )
            if (
                metrics["gripper_vertical_error_rad"]
                > self.far_max_vertical_rad
            ):
                failures.append(
                    f"{label}.vertical="
                    f"{math.degrees(metrics['gripper_vertical_error_rad']):.2f}deg"
                )
            if (
                metrics["template_rmse_rad"]
                > self.far_max_template_rad
            ):
                failures.append(
                    f"{label}.template="
                    f"{math.degrees(metrics['template_rmse_rad']):.2f}deg"
                )

        if failures:
            raise RuntimeError(
                "V5 Far 最终严格复核失败: "
                + ", ".join(failures)
            )

        self.get_logger().warning(
            "V5 Far 最终抓取复核通过: "
            f"L[XY={lm['xy_error_m']*100:.2f}cm, "
            f"lat={lm['lateral_error_m']*100:.2f}cm, "
            f"template={math.degrees(lm['template_rmse_rad']):.2f}deg]; "
            f"R[XY={rm['xy_error_m']*100:.2f}cm, "
            f"lat={rm['lateral_error_m']*100:.2f}cm, "
            f"template={math.degrees(rm['template_rmse_rad']):.2f}deg]"
        )

    # ==================================================================
    # Planning
    # ==================================================================
    def _plan_sequence(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> dict:
        left = np.asarray(
            left_base,
            dtype=np.float64,
        ).copy()
        right = np.asarray(
            right_base,
            dtype=np.float64,
        ).copy()

        self._update_far_context(left, right)

        if not self._far_target_active:
            return super()._plan_sequence(
                left_base,
                right_base,
            )

        self.get_logger().warning(
            "进入 V5 Far-Fast 模式："
            "关闭 17 路 adaptive side-outset 穷举，"
            "使用位置优先 IK + forward capture 快速候选"
        )

        # 快速模式快照。
        threshold_snapshot = self._far_threshold_snapshot()

        old_seed_branches = int(
            self.offset_ik_seed_branches
        )
        old_reach_iterations = int(
            self.offset_ik_reach_iterations
        )
        old_refine_iterations = int(
            self.offset_ik_refine_iterations
        )
        old_line_search = int(
            self.offset_ik_line_search_steps
        )
        old_adaptive = bool(
            self.adaptive_asymmetric_search
        )
        old_final_outset = bool(
            self.enable_final_tcp_outset_fallback
        )

        self._apply_far_thresholds()

        # V2 曾把全局搜索量升到 12 seeds / 180/360。
        # Far 模式单独降回高价值小搜索集。
        self.offset_ik_seed_branches = self.far_seed_limit
        self.offset_ik_reach_iterations = self.far_reach_iterations
        self.offset_ik_refine_iterations = self.far_refine_iterations
        self.offset_ik_line_search_steps = self.far_line_search_steps

        # 避免嵌套：
        # forward候选 × 17 adaptive候选 × 旧side fallback。
        self.adaptive_asymmetric_search = False
        self.enable_final_tcp_outset_fallback = False

        last_error: Optional[Exception] = None
        values = self._far_retreat_values()

        try:
            for index, retreat in enumerate(values, start=1):
                left_target = left.copy()
                right_target = right.copy()

                # +forward 表示箱体朝机器人前方(+X)。
                # 向机器人回退 = -forward。
                delta_xy = (
                    self._far_forward_axis_xy
                    * float(retreat)
                )
                left_target[:2] -= delta_xy
                right_target[:2] -= delta_xy

                self.get_logger().warning(
                    "V5 Far forward candidate "
                    f"{index}/{len(values)}: "
                    f"retreat={retreat*100:.1f}cm, "
                    f"axis={np.round(self._far_forward_axis_xy,5)}, "
                    f"L={np.round(left_target,4)}, "
                    f"R={np.round(right_target,4)}"
                )

                try:
                    sequence = super()._plan_sequence(
                        left_target,
                        right_target,
                    )

                    self._verify_far_final(sequence)

                    sequence["far_fast_v5"] = True
                    sequence["far_forward_retreat_m"] = float(
                        retreat
                    )
                    sequence["far_original_left"] = left.copy()
                    sequence["far_original_right"] = right.copy()

                    self.get_logger().warning(
                        "V5 Far-Fast 路径通过: "
                        f"candidate={index}/{len(values)}, "
                        f"forward_retreat={retreat*100:.1f}cm"
                    )

                    return sequence

                except RuntimeError as exc:
                    last_error = exc
                    self.get_logger().warning(
                        "V5 Far candidate 失败: "
                        f"{index}/{len(values)}, "
                        f"retreat={retreat*100:.1f}cm; "
                        f"{exc}"
                    )

            raise RuntimeError(
                "V5 Far-Fast 全部快速候选失败；"
                f"tried={len(values)}, "
                f"center_x={self._far_center_x_m:+.4f}m, "
                f"center_y={self._far_center_y_m:+.4f}m, "
                f"最后错误: {last_error}"
            )

        finally:
            self._restore_far_thresholds(
                threshold_snapshot
            )
            self.offset_ik_seed_branches = old_seed_branches
            self.offset_ik_reach_iterations = old_reach_iterations
            self.offset_ik_refine_iterations = old_refine_iterations
            self.offset_ik_line_search_steps = old_line_search
            self.adaptive_asymmetric_search = old_adaptive
            self.enable_final_tcp_outset_fallback = old_final_outset


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[FarTargetFastV5Bridge] = None

    try:
        node = FarTargetFastV5Bridge()
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
