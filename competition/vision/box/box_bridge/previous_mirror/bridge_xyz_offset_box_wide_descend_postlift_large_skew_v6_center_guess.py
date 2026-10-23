#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Robot A 双臂抓箱：V6 基于目标中心横向偏移的 outset 快速首猜。

V6 继承 V5。它不直接使用像素距离，而使用左右 3D 抓取点在 base_link
中的中心横向坐标 center_y；该量已经包含相机深度和手眼变换，比像素偏移
更适合直接生成以米为单位的 TCP outset。

流程：
  1. 根据 center_y 判断困难侧；
  2. 预测一个有限、量化到 adaptive step 的单侧 outset；
  3. 暂时关闭嵌套枚举，只规划该快速候选；
  4. 快速候选失败时，自动恢复 V5 的完整安全候选搜索。

最终 IK、安全门限、关节限位和路径检查均由 V5/V4/V3/V2 保留。
第一次必须 enable_motion:=false。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import rclpy

from bridge_xyz_offset_box_wide_descend_postlift_large_skew_v5_force_workspace import (
    ForceLargeSkewWorkspaceV5Bridge,
)


class CenterGuessLargeSkewV6Bridge(ForceLargeSkewWorkspaceV5Bridge):
    """V5 + center_y 预测单侧 outset，命中失败才执行完整枚举。"""

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("v6_enable_center_y_initial_guess", True)

        # predicted_outset = gain * (abs(center_y) - deadband)。
        # center_y≈0.13m、deadband=0.015m、gain=0.07 时约为 8mm。
        self.declare_parameter("v6_center_y_to_outset_gain", 0.070)
        self.declare_parameter("v6_min_initial_outset_m", 0.004)
        self.declare_parameter("v6_max_initial_outset_m", 0.016)
        self.declare_parameter("v6_quantize_to_adaptive_step", True)
        self.declare_parameter("v6_full_search_on_guess_failure", True)

        self.v6_enable_center_y_initial_guess = bool(
            self.get_parameter(
                "v6_enable_center_y_initial_guess"
            ).value
        )
        self.v6_center_y_to_outset_gain = float(
            self.get_parameter(
                "v6_center_y_to_outset_gain"
            ).value
        )
        self.v6_min_initial_outset_m = float(
            self.get_parameter(
                "v6_min_initial_outset_m"
            ).value
        )
        self.v6_max_initial_outset_m = float(
            self.get_parameter(
                "v6_max_initial_outset_m"
            ).value
        )
        self.v6_quantize_to_adaptive_step = bool(
            self.get_parameter(
                "v6_quantize_to_adaptive_step"
            ).value
        )
        self.v6_full_search_on_guess_failure = bool(
            self.get_parameter(
                "v6_full_search_on_guess_failure"
            ).value
        )

        if not 0.0 <= self.v6_center_y_to_outset_gain <= 0.30:
            raise RuntimeError(
                "v6_center_y_to_outset_gain 必须在 [0,0.30] 内"
            )
        if not 0.0 <= self.v6_min_initial_outset_m <= 0.030:
            raise RuntimeError(
                "v6_min_initial_outset_m 必须在 [0,0.030]m 内"
            )
        if not (
            self.v6_min_initial_outset_m
            <= self.v6_max_initial_outset_m
            <= 0.030
        ):
            raise RuntimeError(
                "v6_max_initial_outset_m 必须在 "
                "[v6_min_initial_outset_m,0.030]m 内"
            )

        self.get_logger().warning(
            "Large-skew V6 center_y 快速首猜已启用："
            f"gain={self.v6_center_y_to_outset_gain:.3f}, "
            f"initial_outset="
            f"[{self.v6_min_initial_outset_m*1000:.1f},"
            f"{self.v6_max_initial_outset_m*1000:.1f}]mm, "
            f"fallback_full_search="
            f"{self.v6_full_search_on_guess_failure}"
        )

    def _center_y_initial_guess(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> Tuple[str, float, float]:
        left = np.asarray(left_base, dtype=np.float64).copy()
        right = np.asarray(right_base, dtype=np.float64).copy()

        delta_xy = left[:2] - right[:2]
        separation = float(np.linalg.norm(delta_xy))
        if separation <= 1e-9:
            return "none", 0.0, 0.0

        side_axis = delta_xy / separation

        # 与 AdaptiveOffsetBoxReachBridge 使用完全相同的 nominal center_y。
        left[:2] -= side_axis * self.left_grasp_inset_m
        right[:2] += side_axis * self.right_grasp_inset_m
        center_y = 0.5 * float(left[1] + right[1])

        if center_y > self.offset_side_deadband_m:
            side = "right"
        elif center_y < -self.offset_side_deadband_m:
            side = "left"
        else:
            return "none", 0.0, center_y

        excess = max(
            0.0,
            abs(center_y) - self.offset_side_deadband_m,
        )
        outset = self.v6_center_y_to_outset_gain * excess
        outset = float(
            np.clip(
                outset,
                self.v6_min_initial_outset_m,
                min(
                    self.v6_max_initial_outset_m,
                    float(self.adaptive_max_outset_m),
                ),
            )
        )

        if (
            self.v6_quantize_to_adaptive_step
            and self.adaptive_outset_step_m > 1e-9
        ):
            step = float(self.adaptive_outset_step_m)
            outset = round(outset / step) * step
            outset = float(
                np.clip(
                    outset,
                    self.v6_min_initial_outset_m,
                    min(
                        self.v6_max_initial_outset_m,
                        float(self.adaptive_max_outset_m),
                    ),
                )
            )

        return side, outset, center_y

    @staticmethod
    def _apply_single_side_outset(
        left_base: np.ndarray,
        right_base: np.ndarray,
        side: str,
        outset: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        left = np.asarray(left_base, dtype=np.float64).copy()
        right = np.asarray(right_base, dtype=np.float64).copy()
        delta_xy = left[:2] - right[:2]
        side_axis = delta_xy / max(
            float(np.linalg.norm(delta_xy)),
            1e-9,
        )

        if side == "left":
            left[:2] += side_axis * outset
        elif side == "right":
            right[:2] -= side_axis * outset

        return left, right

    def _plan_sequence(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> dict:
        if not self.v6_enable_center_y_initial_guess:
            return super()._plan_sequence(left_base, right_base)

        side, outset, center_y = self._center_y_initial_guess(
            left_base,
            right_base,
        )
        if side == "none" or outset <= 1e-9:
            return super()._plan_sequence(left_base, right_base)

        guess_left, guess_right = self._apply_single_side_outset(
            left_base,
            right_base,
            side,
            outset,
        )

        self.get_logger().warning(
            "V6 center_y 快速首猜："
            f"center_y={center_y:+.4f}m, "
            f"side={side}, outset={outset*1000:.1f}mm；"
            "先只计算该候选"
        )

        old_adaptive = bool(self.adaptive_asymmetric_search)
        old_final_outset = bool(
            self.enable_final_tcp_outset_fallback
        )

        try:
            # 避免快速首猜内部再次枚举 17 个候选；只验证预测值一次。
            self.adaptive_asymmetric_search = False
            self.enable_final_tcp_outset_fallback = False
            sequence = super()._plan_sequence(
                guess_left,
                guess_right,
            )
            sequence["v6_center_y_m"] = float(center_y)
            sequence["v6_initial_outset_side"] = side
            sequence["v6_initial_outset_m"] = float(outset)
            sequence["v6_initial_guess_succeeded"] = True
            self.get_logger().warning(
                "V6 center_y 快速首猜通过："
                f"side={side}, outset={outset*1000:.1f}mm"
            )
            return sequence
        except RuntimeError as exc:
            self.get_logger().warning(
                "V6 center_y 快速首猜失败："
                f"side={side}, outset={outset*1000:.1f}mm；"
                f"{exc}"
            )
            if not self.v6_full_search_on_guess_failure:
                raise
        finally:
            self.adaptive_asymmetric_search = old_adaptive
            self.enable_final_tcp_outset_fallback = old_final_outset

        self.get_logger().warning(
            "V6 恢复完整安全候选搜索"
        )
        sequence = super()._plan_sequence(
            left_base,
            right_base,
        )
        sequence["v6_center_y_m"] = float(center_y)
        sequence["v6_initial_outset_side"] = side
        sequence["v6_initial_outset_m"] = float(outset)
        sequence["v6_initial_guess_succeeded"] = False
        return sequence


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[CenterGuessLargeSkewV6Bridge] = None

    try:
        node = CenterGuessLargeSkewV6Bridge()
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
