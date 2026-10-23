#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""偏置箱体抓取：水平箱沿共同高度修正版。

把本文件与 bridge_xyz_offset_box.py 及其依赖文件放在同一目录，运行本文件。

用途：
视觉点已经发布，但同一个水平箱口的左右中点在 base_link 中出现约 4~7cm
的虚假高度差时，不直接放宽最终抓取安全阈值，而是只对 Z 做共同水平面投影：

    common_z = (left_z + right_z) / 2

X/Y、非对称位置、关节限位、夹爪姿态、共享 lift、分段路径和稳定性检查均保留。
此修正仅适用于箱子确实水平放置的场景。
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy

from bridge_xyz_offset_box import OffsetBoxReachBridge
from bridge_xyz_guard import GripperBalancedSharedLiftBridge


class AdaptiveOffsetBoxReachBridge(OffsetBoxReachBridge):
    """扩大非对称目标的安全 IK/最终 TCP 捕获搜索覆盖范围。"""

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("adaptive_asymmetric_search", True)
        self.declare_parameter("adaptive_outset_step_m", 0.004)
        self.declare_parameter("adaptive_max_outset_m", 0.024)
        self.declare_parameter("adaptive_cross_side_max_m", 0.012)

        self.adaptive_asymmetric_search = bool(
            self.get_parameter("adaptive_asymmetric_search").value
        )
        self.adaptive_outset_step_m = float(
            self.get_parameter("adaptive_outset_step_m").value
        )
        self.adaptive_max_outset_m = float(
            self.get_parameter("adaptive_max_outset_m").value
        )
        self.adaptive_cross_side_max_m = float(
            self.get_parameter("adaptive_cross_side_max_m").value
        )

        if not 0.002 <= self.adaptive_outset_step_m <= 0.008:
            raise RuntimeError(
                "adaptive_outset_step_m 必须在 [0.002, 0.008]m 内"
            )
        if not 0.012 <= self.adaptive_max_outset_m <= 0.030:
            raise RuntimeError(
                "adaptive_max_outset_m 必须在 [0.012, 0.030]m 内"
            )
        if not 0.0 <= self.adaptive_cross_side_max_m <= 0.016:
            raise RuntimeError(
                "adaptive_cross_side_max_m 必须在 [0, 0.016]m 内"
            )
        if self.adaptive_cross_side_max_m > self.adaptive_max_outset_m:
            raise RuntimeError(
                "adaptive_cross_side_max_m 不能大于 adaptive_max_outset_m"
            )

        # 原算法虽然允许 seed_branches=6，但实际只生成 4 个种子；这里补足
        # 两个靠近困难侧肩关节边界的确定性分支，并给每个分支更多收敛机会。
        self.offset_ik_seed_branches = 6
        self.offset_ik_reach_iterations = max(
            int(self.offset_ik_reach_iterations),
            120,
        )
        self.offset_ik_refine_iterations = max(
            int(self.offset_ik_refine_iterations),
            260,
        )
        self.offset_ik_line_search_steps = max(
            int(self.offset_ik_line_search_steps),
            6,
        )

        self.get_logger().warning(
            "自适应非对称搜索已启用: "
            f"seeds={self.offset_ik_seed_branches}, "
            f"iterations={self.offset_ik_reach_iterations}/"
            f"{self.offset_ik_refine_iterations}, "
            f"outset_step={self.adaptive_outset_step_m*1000:.1f}mm, "
            f"max_outset={self.adaptive_max_outset_m*1000:.1f}mm, "
            f"cross_side={self.adaptive_cross_side_max_m*1000:.1f}mm"
        )

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

        current_l = np.clip(left_seed, left_lower, left_upper)
        current_r = np.clip(right_seed, right_lower, right_upper)
        template_l = np.clip(
            self.left_template_q,
            left_lower,
            left_upper,
        )
        template_r = np.clip(
            self.right_template_q,
            right_lower,
            right_upper,
        )

        # 两个不同混合比例避免 DLS 总从同一局部极小值开始。扰动遵循左右
        # 镜像符号，最终仍会经过真实 URDF 限位、姿态阈值和路径跳变检查。
        for blend, shoulder_margin_deg in ((0.25, 1.0), (0.55, 3.0)):
            seed_l = np.clip(
                blend * current_l + (1.0 - blend) * template_l,
                left_lower,
                left_upper,
            )
            seed_r = np.clip(
                blend * current_r + (1.0 - blend) * template_r,
                right_lower,
                right_upper,
            )
            margin = math.radians(shoulder_margin_deg)

            if center_y >= 0.0:
                seed_r[1] = max(
                    float(right_lower[1]),
                    float(right_upper[1]) - margin,
                )
                seed_r[0] += math.radians(7.0 + 5.0 * blend)
                seed_r[2] -= math.radians(5.0 + 4.0 * blend)
                seed_r[3] -= math.radians(9.0 + 6.0 * blend)
            else:
                seed_l[1] = min(
                    float(left_upper[1]),
                    float(left_lower[1]) + margin,
                )
                seed_l[0] -= math.radians(7.0 + 5.0 * blend)
                seed_l[2] += math.radians(5.0 + 4.0 * blend)
                seed_l[3] += math.radians(9.0 + 6.0 * blend)

            seeds.append(
                (
                    np.clip(seed_l, left_lower, left_upper),
                    np.clip(seed_r, right_lower, right_upper),
                )
            )

        # 去掉数值上重复的种子，否则只会增加耗时而不会扩大搜索空间。
        unique: List[Tuple[np.ndarray, np.ndarray]] = []
        for seed_l, seed_r in seeds:
            if any(
                np.allclose(seed_l, old_l, atol=1e-7)
                and np.allclose(seed_r, old_r, atol=1e-7)
                for old_l, old_r in unique
            ):
                continue
            unique.append((seed_l.copy(), seed_r.copy()))
        return unique[: self.offset_ik_seed_branches]

    def _adaptive_outset_values(self) -> List[float]:
        values = [0.0]
        value = self.adaptive_outset_step_m
        while value < self.adaptive_max_outset_m - 1e-9:
            values.append(float(value))
            value += self.adaptive_outset_step_m
        values.append(float(self.adaptive_max_outset_m))
        return sorted(set(round(value, 9) for value in values))

    def _plan_sequence(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> dict:
        if not self.adaptive_asymmetric_search:
            return super()._plan_sequence(left_base, right_base)

        left_nominal = np.asarray(left_base, dtype=np.float64).copy()
        right_nominal = np.asarray(right_base, dtype=np.float64).copy()
        side_delta = left_nominal[:2] - right_nominal[:2]
        side_distance = float(np.linalg.norm(side_delta))
        if side_distance < 1e-9:
            raise RuntimeError("左右抓取点重合，无法计算箱体横向轴")
        side_axis = side_delta / side_distance

        # 与原 OffsetBoxReachBridge 相同：左右夹紧补偿只应用一次。
        left_nominal[:2] -= side_axis * self.left_grasp_inset_m
        right_nominal[:2] += side_axis * self.right_grasp_inset_m
        center_y = 0.5 * float(left_nominal[1] + right_nominal[1])

        if center_y > self.offset_side_deadband_m:
            preferred_side = "right"
        elif center_y < -self.offset_side_deadband_m:
            preferred_side = "left"
        else:
            preferred_side = "both"
        opposite_side = (
            "left" if preferred_side == "right" else "right"
        )

        outset_values = self._adaptive_outset_values()
        candidates: List[Tuple[str, float]] = [("none", 0.0)]
        if preferred_side in ("left", "right"):
            candidates.extend(
                (preferred_side, value)
                for value in outset_values[1:]
            )
        candidates.extend(
            ("both", value)
            for value in outset_values[1:]
            if value <= self.adaptive_cross_side_max_m + 1e-9
        )
        if preferred_side in ("left", "right"):
            candidates.extend(
                (opposite_side, value)
                for value in outset_values[1:]
                if value <= self.adaptive_cross_side_max_m + 1e-9
            )

        last_error: Optional[Exception] = None
        attempted = 0
        for mode, outset in candidates:
            left_target = left_nominal.copy()
            right_target = right_nominal.copy()
            if mode == "left":
                left_target[:2] += side_axis * outset
            elif mode == "right":
                right_target[:2] -= side_axis * outset
            elif mode == "both":
                # outset 表示总捕获余量，两侧各承担一半，箱体中心不漂移。
                left_target[:2] += side_axis * (0.5 * outset)
                right_target[:2] -= side_axis * (0.5 * outset)

            attempted += 1
            try:
                # 跳过原单侧粗步进版本，直接调用其后的安全分段规划层。
                sequence = super(
                    GripperBalancedSharedLiftBridge,
                    self,
                )._plan_sequence(left_target, right_target)
                sequence["final_tcp_outset_m"] = float(outset)
                sequence["final_tcp_outset_side"] = mode
                sequence["adaptive_candidate_count"] = attempted
                self.get_logger().warning(
                    "自适应非对称路径通过: "
                    f"candidate={attempted}/{len(candidates)}, "
                    f"mode={mode}, outset={outset*1000:.1f}mm, "
                    f"center_y={center_y:+.4f}m"
                )
                return sequence
            except RuntimeError as exc:
                last_error = exc
                self.get_logger().warning(
                    "自适应候选失败: "
                    f"candidate={attempted}/{len(candidates)}, "
                    f"mode={mode}, outset={outset*1000:.1f}mm; {exc}"
                )

        raise RuntimeError(
            "自适应非对称搜索已遍历全部安全候选，仍无完整路径；"
            f"attempted={attempted}, center_y={center_y:+.4f}m, "
            f"最后错误: {last_error}"
        )


class CommonRimHeightOffsetBoxBridge(AdaptiveOffsetBoxReachBridge):
    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter(
            "enable_common_rim_height_projection",
            True,
        )
        self.declare_parameter(
            "max_raw_rim_height_difference_m",
            0.070,
        )
        self.declare_parameter(
            "max_per_point_height_correction_m",
            0.035,
        )
        self.declare_parameter(
            "common_height_log_period_s",
            2.0,
        )

        self.enable_common_rim_height_projection = bool(
            self.get_parameter(
                "enable_common_rim_height_projection"
            ).value
        )
        self.max_raw_rim_height_difference_m = float(
            self.get_parameter(
                "max_raw_rim_height_difference_m"
            ).value
        )
        self.max_per_point_height_correction_m = float(
            self.get_parameter(
                "max_per_point_height_correction_m"
            ).value
        )
        self.common_height_log_period_s = max(
            0.5,
            float(
                self.get_parameter(
                    "common_height_log_period_s"
                ).value
            ),
        )

        if not 0.030 <= self.max_raw_rim_height_difference_m <= 0.080:
            raise RuntimeError(
                "max_raw_rim_height_difference_m 必须在 [0.030, 0.080]m 内"
            )
        if not 0.015 <= self.max_per_point_height_correction_m <= 0.040:
            raise RuntimeError(
                "max_per_point_height_correction_m 必须在 [0.015, 0.040]m 内"
            )

        self._last_common_height_log = 0.0
        self.get_logger().warning(
            "水平箱沿共同高度投影已启用: "
            f"raw_dz_limit={self.max_raw_rim_height_difference_m:.3f}m, "
            f"per_point_limit={self.max_per_point_height_correction_m:.3f}m"
        )

    @staticmethod
    def _project_common_height(
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        left = np.asarray(left_base, dtype=np.float64).copy()
        right = np.asarray(right_base, dtype=np.float64).copy()
        raw_dz = abs(float(left[2] - right[2]))
        common_z = 0.5 * float(left[2] + right[2])
        max_correction = max(
            abs(common_z - float(left[2])),
            abs(common_z - float(right[2])),
        )
        left[2] = common_z
        right[2] = common_z
        return left, right, raw_dz, max_correction

    def _target_geometry_problem(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> Optional[str]:
        reason = super()._target_geometry_problem(left_base, right_base)
        if reason is None or not self.enable_common_rim_height_projection:
            return reason

        (
            corrected_left,
            corrected_right,
            raw_dz,
            max_correction,
        ) = self._project_common_height(left_base, right_base)

        # 只允许共同高度投影修复“高度差”这一项。若 X/Y、中心位置、间距、
        # 旋转角等仍有问题，父类会继续拒绝，不能被本补丁绕过。
        if raw_dz > self.max_raw_rim_height_difference_m:
            return reason
        if max_correction > self.max_per_point_height_correction_m:
            return reason

        corrected_reason = super()._target_geometry_problem(
            corrected_left,
            corrected_right,
        )
        if corrected_reason is not None:
            return reason

        now = time.monotonic()
        if now - self._last_common_height_log >= self.common_height_log_period_s:
            self._last_common_height_log = now
            self.get_logger().warning(
                "检测到水平箱沿虚假高差，允许进入稳定性统计并将在规划前投影: "
                f"raw_dz={raw_dz:.4f}m, "
                f"common_z={corrected_left[2]:+.4f}m, "
                f"per_point_correction={max_correction:.4f}m"
            )
        return None

    def _plan_sequence(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> dict:
        if not self.enable_common_rim_height_projection:
            return super()._plan_sequence(left_base, right_base)

        (
            corrected_left,
            corrected_right,
            raw_dz,
            max_correction,
        ) = self._project_common_height(left_base, right_base)

        if raw_dz > self.max_raw_rim_height_difference_m:
            raise RuntimeError(
                "原始左右箱沿高度差过大，拒绝共同高度投影: "
                f"raw_dz={raw_dz:.4f}m > "
                f"{self.max_raw_rim_height_difference_m:.4f}m"
            )
        if max_correction > self.max_per_point_height_correction_m:
            raise RuntimeError(
                "共同高度对单点修正过大: "
                f"correction={max_correction:.4f}m > "
                f"{self.max_per_point_height_correction_m:.4f}m"
            )

        self.get_logger().warning(
            "规划使用水平箱沿共同高度: "
            f"Lz {float(left_base[2]):+.4f} -> {corrected_left[2]:+.4f}m, "
            f"Rz {float(right_base[2]):+.4f} -> {corrected_right[2]:+.4f}m; "
            "X/Y 保持原始非对称视觉值"
        )
        sequence = super()._plan_sequence(
            corrected_left,
            corrected_right,
        )
        sequence["raw_rim_height_difference_m"] = float(raw_dz)
        sequence["common_rim_height_m"] = float(corrected_left[2])
        return sequence


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[CommonRimHeightOffsetBoxBridge] = None
    try:
        node = CommonRimHeightOffsetBoxBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(
                f"节点异常退出: {type(exc).__name__}: {exc}"
            )
        else:
            print(f"节点创建失败: {type(exc).__name__}: {exc}")
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
