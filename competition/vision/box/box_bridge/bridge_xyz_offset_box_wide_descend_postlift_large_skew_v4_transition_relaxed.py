#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot A 双臂抓箱：
Large-Skew V4 / Transition-Relaxed + Tilted-3D

依赖：
  bridge_xyz_offset_box_wide_descend_postlift_large_skew_v3_tilted3d.py
  bridge_xyz_offset_box_wide_descend_postlift_large_skew_v2.py
  以及原 box_bridge 全部依赖。

V4 解决 V3 实机 dry-run 暴露出的两个问题：

1. Tilted-3D 触发边界抖动
------------------------
V3 默认：
  raw_dZ >= 2.0cm 且 slope >= 3.0deg 才保留真实左右 Z。

实际稳定帧：
  raw_dZ 从 2.04cm 漂到 1.88cm，
导致前面通过 Tilted-3D 门控，最终锁定规划时却又回到 common-height。

V4 默认改为：
  raw_dZ >= 1.5cm
  slope  >= 2.0deg

普通、小倾角箱子仍保留原 common-height。

2. 外扩/下降路点被“最终抓取 template 阈值”误杀
------------------------------------------------
日志证明最终抓取点本身可以达到：
  L template ≈ 22~27deg
  R template ≈ 14~15deg

但是外扩 waypoint 为了避障，需要左臂暂时偏离标准模板到约 47deg。
旧逻辑所有 waypoint 都使用同一个 template<=30deg，导致整条路径失败。

V4：
  - 最终抓取姿态：仍要求 large-skew template <= 30deg；
  - 外扩/下降/内收过渡路点：允许 template <= 50deg；
  - 到最后 factor=0 抓取目标时，必须重新满足严格 30deg；
  - XY / box-lateral / Z / orientation / gripper vertical /
    URDF joint limits / waypoint joint jump / shared lift 全部继续检查；
  - 不改变动作顺序；
  - 不修改视觉节点；
  - 不修改手眼标定。

注意：
  没有完整 3D collision scene。
  第一次必须 enable_motion:=false。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy

from bridge_xyz_offset_box_wide_descend_postlift_large_skew_v3_tilted3d import (
    LargeSkewTilted3DV3Bridge,
)


class LargeSkewTransitionV4Bridge(LargeSkewTilted3DV3Bridge):
    """V3 + large-skew 过渡路点 template 临时放宽。"""

    def __init__(self) -> None:
        super().__init__()

        # V4 独立参数，避免重新 declare V3 同名 ROS parameter。
        self.declare_parameter(
            "v4_tilted3d_raw_z_trigger_cm",
            1.5,
        )
        self.declare_parameter(
            "v4_tilted3d_min_slope_deg",
            2.0,
        )

        self.declare_parameter(
            "v4_transition_template_rmse_deg",
            50.0,
        )

        # target span 比“原始最终左右抓取点距离”多出多少，认为正在解
        # 外扩/内收过渡 waypoint。
        # 7 个 waypoint + outward=6cm/side 时，最靠近最终的非零 factor
        # 仍约增加 2cm 总 span，因此默认 1.0cm 可以稳定识别过渡路点。
        self.declare_parameter(
            "v4_transition_span_extra_cm",
            1.0,
        )

        raw_z_trigger_cm = float(
            self.get_parameter(
                "v4_tilted3d_raw_z_trigger_cm"
            ).value
        )
        min_slope_deg = float(
            self.get_parameter(
                "v4_tilted3d_min_slope_deg"
            ).value
        )
        transition_template_deg = float(
            self.get_parameter(
                "v4_transition_template_rmse_deg"
            ).value
        )
        transition_span_extra_cm = float(
            self.get_parameter(
                "v4_transition_span_extra_cm"
            ).value
        )

        if not 1.0 <= raw_z_trigger_cm <= 3.0:
            raise RuntimeError(
                "v4_tilted3d_raw_z_trigger_cm 必须在 [1.0,3.0]cm"
            )
        if not 1.0 <= min_slope_deg <= 5.0:
            raise RuntimeError(
                "v4_tilted3d_min_slope_deg 必须在 [1.0,5.0]deg"
            )
        if not 35.0 <= transition_template_deg <= 55.0:
            raise RuntimeError(
                "v4_transition_template_rmse_deg 必须在 [35,55]deg"
            )
        if not 0.5 <= transition_span_extra_cm <= 3.0:
            raise RuntimeError(
                "v4_transition_span_extra_cm 必须在 [0.5,3.0]cm"
            )

        # 覆盖 V3 内部触发值，不动视觉/TF。
        self.tilted3d_raw_z_trigger_m = (
            raw_z_trigger_cm / 100.0
        )
        self.tilted3d_min_slope_rad = math.radians(
            min_slope_deg
        )

        self.v4_transition_template_rad = math.radians(
            transition_template_deg
        )
        self.v4_transition_span_extra_m = (
            transition_span_extra_cm / 100.0
        )

        # 每次最上层规划时记录原始最终抓取点的 XY span。
        self._v4_nominal_final_span_xy = 0.0
        self._v4_transition_active = False

        self.get_logger().warning(
            "Large-skew V4 已启用："
            f"Tilted-3D raw_dZ trigger="
            f"{self.tilted3d_raw_z_trigger_m*100:.1f}cm, "
            f"slope trigger="
            f"{math.degrees(self.tilted3d_min_slope_rad):.1f}deg；"
            f"最终 template<="
            f"{math.degrees(self.large_skew_template_rad):.1f}deg，"
            f"过渡 waypoint template<="
            f"{math.degrees(self.v4_transition_template_rad):.1f}deg"
        )

    # ------------------------------------------------------------------
    # 记录这一次视觉锁定目标的“最终左右点 XY 距离”。
    # Adaptive outset 会略改变最终 span；V4 最终仍会做严格复核。
    # ------------------------------------------------------------------
    def _plan_sequence(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> dict:
        left = np.asarray(left_base, dtype=np.float64)
        right = np.asarray(right_base, dtype=np.float64)

        self._v4_nominal_final_span_xy = float(
            np.linalg.norm(left[:2] - right[:2])
        )

        self.get_logger().info(
            "V4 路点判定基准: "
            f"nominal_final_span="
            f"{self._v4_nominal_final_span_xy:.4f}m, "
            f"transition_extra="
            f"{self.v4_transition_span_extra_m:.4f}m"
        )

        sequence = super()._plan_sequence(
            left_base,
            right_base,
        )

        # --------------------------------------------------------------
        # 最终抓取点必须重新满足严格 large-skew 条件。
        # 过渡路点允许 50deg，不代表最终也可以 50deg。
        # --------------------------------------------------------------
        if self._large_skew_active:
            final_plan = sequence["plans"][-1]
            lm = final_plan["metrics"]["left_final"]
            rm = final_plan["metrics"]["right_final"]

            failures = []

            for label, metrics in (
                ("L", lm),
                ("R", rm),
            ):
                if (
                    metrics["xy_error_m"]
                    > self.large_skew_xy_error_m
                ):
                    failures.append(
                        f"{label}.XY="
                        f"{metrics['xy_error_m']*100:.2f}cm"
                    )

                # V3 中 metrics["lateral_error_m"] 已是 box-axis lateral。
                if (
                    metrics["lateral_error_m"]
                    > self.large_skew_lateral_error_m
                ):
                    failures.append(
                        f"{label}.boxLat="
                        f"{metrics['lateral_error_m']*100:.2f}cm"
                    )

                if (
                    metrics["orientation_error_rad"]
                    > self.large_skew_orientation_rad
                ):
                    failures.append(
                        f"{label}.ori="
                        f"{math.degrees(metrics['orientation_error_rad']):.2f}deg"
                    )

                if (
                    metrics["forearm_tilt_rad"]
                    > self.large_skew_forearm_rad
                ):
                    failures.append(
                        f"{label}.forearm="
                        f"{math.degrees(metrics['forearm_tilt_rad']):.2f}deg"
                    )

                if (
                    metrics["gripper_vertical_error_rad"]
                    > self.large_skew_vertical_rad
                ):
                    failures.append(
                        f"{label}.vertical="
                        f"{math.degrees(metrics['gripper_vertical_error_rad']):.2f}deg"
                    )

                if (
                    metrics["template_rmse_rad"]
                    > self.large_skew_template_rad
                ):
                    failures.append(
                        f"{label}.template="
                        f"{math.degrees(metrics['template_rmse_rad']):.2f}deg"
                    )

            if failures:
                raise RuntimeError(
                    "V4 过渡路径虽可达，但最终抓取点未重新收回严格阈值: "
                    + ", ".join(failures)
                )

            self.get_logger().warning(
                "V4 最终抓取严格复核通过: "
                f"L[XY={lm['xy_error_m']*100:.2f}cm, "
                f"boxLat={lm['lateral_error_m']*100:.2f}cm, "
                f"template="
                f"{math.degrees(lm['template_rmse_rad']):.2f}deg]; "
                f"R[XY={rm['xy_error_m']*100:.2f}cm, "
                f"boxLat={rm['lateral_error_m']*100:.2f}cm, "
                f"template="
                f"{math.degrees(rm['template_rmse_rad']):.2f}deg]"
            )

        return sequence

    # ------------------------------------------------------------------
    # 判断当前 _solve_asymmetric 是最终抓取点，还是“更宽”的外扩/内收路点。
    #
    # 原 staged path:
    #   left  += side_axis*outward*factor
    #   right -= side_axis*outward*factor
    # 因此 factor>0 时左右 span 必然比最终目标更大。
    # ------------------------------------------------------------------
    def _is_transition_target(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> bool:
        if not self._large_skew_active:
            return False

        if self._v4_nominal_final_span_xy <= 1e-9:
            return False

        span = float(
            np.linalg.norm(
                np.asarray(left_base, dtype=np.float64)[:2]
                - np.asarray(right_base, dtype=np.float64)[:2]
            )
        )

        return bool(
            span
            >= self._v4_nominal_final_span_xy
            + self.v4_transition_span_extra_m
        )

    # ------------------------------------------------------------------
    # 关键改动：
    #
    # _solve_asymmetric 内部不仅 DLS _metrics_pass 会检查 template，
    # 最后 verify_grasp_solution 也会使用 self.max_template_rmse_rad。
    #
    # 因此过渡 waypoint 必须在整个 solve 期间临时放宽该值，
    # solve 完立即恢复。
    # ------------------------------------------------------------------
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
        transition = self._is_transition_target(
            left_base,
            right_base,
        )

        if not transition:
            return super()._solve_asymmetric(
                left_base,
                right_base,
                left_seed,
                right_seed,
                current_lift,
                current_waist,
                limits,
            )

        old_template = float(
            self.max_template_rmse_rad
        )
        old_flag = bool(
            self._v4_transition_active
        )

        span = float(
            np.linalg.norm(
                np.asarray(left_base)[:2]
                - np.asarray(right_base)[:2]
            )
        )

        try:
            self._v4_transition_active = True
            self.max_template_rmse_rad = max(
                old_template,
                self.v4_transition_template_rad,
            )

            self.get_logger().info(
                "V4 large-skew 过渡路点："
                f"span={span:.4f}m "
                f"(final≈{self._v4_nominal_final_span_xy:.4f}m)，"
                f"临时 template<="
                f"{math.degrees(self.max_template_rmse_rad):.1f}deg"
            )

            return super()._solve_asymmetric(
                left_base,
                right_base,
                left_seed,
                right_seed,
                current_lift,
                current_waist,
                limits,
            )

        finally:
            self.max_template_rmse_rad = old_template
            self._v4_transition_active = old_flag


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[LargeSkewTransitionV4Bridge] = None

    try:
        node = LargeSkewTransitionV4Bridge()
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
