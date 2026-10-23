#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Robot A 双臂抓箱：V5 强制大偏置求解 + 明确工作空间失败。

依赖（放在同一目录）：
  bridge_xyz_offset_box_wide_descend_postlift_large_skew_v4_transition_relaxed.py
  bridge_xyz_offset_box_wide_descend_postlift_large_skew_v3_tilted3d.py
  bridge_xyz_offset_box_wide_descend_postlift_large_skew_v2.py
  bridge_xyz_offset_box_wide_descend_postlift.py
  以及原 box_bridge 的全部依赖。

V5 修复的逻辑问题：

1. V2 原来仅在 rotation/dX 达到触发阈值时启用 large-skew。目标虽然可抓，
   但未达到“几何形状触发阈值”时，会退回严格 IK，V4 的过渡放宽也随之失效。
2. V3 的 tilted-3D raw-Z 保留同样依赖旧 large-skew 触发条件。
3. 规划无解时，上层只能看到零散的 IK 门限错误，不容易区分工作空间失败。

本版对所有非退化、通过几何门控的双抓取点启用 large-skew 求解；过渡点
是否放宽只由其相对最终抓取点的跨度判断，不再依赖旧 rotation/dX 触发器。
最终抓取点仍必须满足 V2/V3/V4 的有限误差、URDF 关节限位、路点跳变、
共享 lift 和最终 template 门限。无解时统一抛出“抓取点超出工作空间限制”，
并保留原始失败原因用于调试。

第一次必须使用 enable_motion:=false 检查完整轨迹。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy

from bridge_xyz_offset_box_wide_descend_postlift_large_skew_v4_transition_relaxed import (
    LargeSkewTransitionV4Bridge,
)


class ForceLargeSkewWorkspaceV5Bridge(LargeSkewTransitionV4Bridge):
    """V4 + 所有有效目标自动进入 large-skew + 统一工作空间错误。"""

    WORKSPACE_ERROR_PREFIX = "抓取点超出工作空间限制"

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("v5_force_large_skew_mode", True)
        self.declare_parameter(
            "v5_relax_all_transition_waypoints",
            True,
        )
        self.declare_parameter(
            "v5_preserve_tilted_z_without_skew_trigger",
            True,
        )

        self.v5_force_large_skew_mode = bool(
            self.get_parameter("v5_force_large_skew_mode").value
        )
        self.v5_relax_all_transition_waypoints = bool(
            self.get_parameter(
                "v5_relax_all_transition_waypoints"
            ).value
        )
        self.v5_preserve_tilted_z_without_skew_trigger = bool(
            self.get_parameter(
                "v5_preserve_tilted_z_without_skew_trigger"
            ).value
        )

        self._v5_forced_this_plan = False

        self.get_logger().warning(
            "Large-skew V5 已启用："
            f"force_large_skew={self.v5_force_large_skew_mode}, "
            "relax_all_transitions="
            f"{self.v5_relax_all_transition_waypoints}, "
            "preserve_tilted_z_without_old_trigger="
            f"{self.v5_preserve_tilted_z_without_skew_trigger}；"
            "最终抓取安全门限和关节限位保持有效"
        )

    # ------------------------------------------------------------------
    # 核心修复 1：不再把 rotation/dX 当作是否允许 large-skew 求解的门票。
    # 父类仍负责计算 axis、rotation、dX、center_y；V5 只改 active 决策。
    # ------------------------------------------------------------------
    def _update_large_skew_context(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> None:
        super()._update_large_skew_context(left_base, right_base)

        left = np.asarray(left_base, dtype=np.float64)
        right = np.asarray(right_base, dtype=np.float64)
        separation_xy = float(
            np.linalg.norm(left[:2] - right[:2])
        )

        old_trigger_result = bool(self._large_skew_active)
        can_force = bool(
            self.v5_force_large_skew_mode
            and np.isfinite(left).all()
            and np.isfinite(right).all()
            and separation_xy > 1e-6
        )

        if can_force:
            self._large_skew_active = True

        self._v5_forced_this_plan = bool(
            can_force and not old_trigger_result
        )

        if self._v5_forced_this_plan:
            self.get_logger().warning(
                "V5 自动升级到大偏置抓取："
                f"rotation={math.degrees(self._large_skew_rotation_rad):.2f}deg "
                f"< trigger="
                f"{math.degrees(self.large_skew_trigger_rotation_rad):.2f}deg, "
                f"dX={self._large_skew_dx_m:.4f}m "
                f"< trigger={self.large_skew_trigger_dx_m:.4f}m；"
                "不再退回严格普通模式"
            )

    # ------------------------------------------------------------------
    # 核心修复 2：真实左右 Z 是否保留只看 dZ/slope，不再额外要求旧的
    # rotation/dX large-skew 触发。最大 raw-dZ 安全限制仍由 V3 检查。
    # ------------------------------------------------------------------
    def _should_preserve_raw_z(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> bool:
        inherited = super()._should_preserve_raw_z(
            left_base,
            right_base,
        )
        if (
            inherited
            or not self.v5_preserve_tilted_z_without_skew_trigger
        ):
            return inherited

        (
            _side_axis,
            _forward_axis,
            _rotation,
            raw_dz,
            slope,
        ) = self._geometry_values(left_base, right_base)

        preserve = bool(
            raw_dz >= self.tilted3d_raw_z_trigger_m
            and slope >= self.tilted3d_min_slope_rad
        )

        self._tilted3d_preserve_raw_z = preserve
        self._tilted3d_raw_dz_m = raw_dz
        self._tilted3d_slope_rad = slope

        if preserve:
            self.get_logger().warning(
                "V5 Tilted-3D 自动触发："
                f"raw_dZ={raw_dz*100:.2f}cm, "
                f"slope={math.degrees(slope):.2f}deg；"
                "不再受 rotation/dX large-skew 入口阈值限制"
            )

        return preserve

    # ------------------------------------------------------------------
    # 核心修复 3：只根据 span 判断过渡点。最终 factor=0 的抓取点不满足
    # span-extra 条件，仍使用严格 large-skew 最终门限。
    # ------------------------------------------------------------------
    def _is_transition_target(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> bool:
        if not self.v5_relax_all_transition_waypoints:
            return super()._is_transition_target(
                left_base,
                right_base,
            )

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

    @staticmethod
    def _target_text(point: np.ndarray) -> str:
        value = np.asarray(point, dtype=np.float64).reshape(-1)
        if len(value) < 3:
            return str(np.round(value, 4))
        return (
            f"({value[0]:+.4f}, "
            f"{value[1]:+.4f}, "
            f"{value[2]:+.4f})m"
        )

    def _workspace_failure_message(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
        reason: str,
    ) -> str:
        try:
            (
                _side_axis,
                _forward_axis,
                rotation,
                raw_dz,
                slope,
            ) = self._geometry_values(left_base, right_base)
            geometry = (
                f"rotation={math.degrees(rotation):.2f}deg, "
                f"raw_dZ={raw_dz:.4f}m, "
                f"slope={math.degrees(slope):.2f}deg"
            )
        except Exception:
            geometry = "geometry=unavailable"

        return (
            f"{self.WORKSPACE_ERROR_PREFIX}: "
            f"L={self._target_text(left_base)}, "
            f"R={self._target_text(right_base)}, "
            f"{geometry}; 原始规划失败原因: {reason}"
        )

    # 几何门控失败发生在 IK 规划之前，也统一给出明确的工作空间错误语义。
    def _target_geometry_problem(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> Optional[str]:
        reason = super()._target_geometry_problem(
            left_base,
            right_base,
        )
        if reason is None:
            return None
        if str(reason).startswith(self.WORKSPACE_ERROR_PREFIX):
            return reason
        return self._workspace_failure_message(
            left_base,
            right_base,
            str(reason),
        )

    # IK、路点、最终安全复核任何一项仍无解时，不再只留下一条模糊门限错误。
    def _plan_sequence(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> dict:
        try:
            sequence = super()._plan_sequence(
                left_base,
                right_base,
            )
        except RuntimeError as exc:
            text = str(exc)
            if text.startswith(self.WORKSPACE_ERROR_PREFIX):
                raise
            raise RuntimeError(
                self._workspace_failure_message(
                    left_base,
                    right_base,
                    text,
                )
            ) from exc

        sequence["v5_force_large_skew"] = bool(
            self.v5_force_large_skew_mode
        )
        sequence["v5_auto_promoted"] = bool(
            self._v5_forced_this_plan
        )
        return sequence


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[ForceLargeSkewWorkspaceV5Bridge] = None

    try:
        node = ForceLargeSkewWorkspaceV5Bridge()
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
