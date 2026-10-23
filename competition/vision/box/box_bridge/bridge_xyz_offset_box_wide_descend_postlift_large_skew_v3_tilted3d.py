#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot A 双臂抓箱：Large-Skew V3 / Tilted-3D 增强版

依赖：
  bridge_xyz_offset_box_wide_descend_postlift_large_skew_v2.py
  以及 V2 原有的全部依赖文件。

放置位置：
  与 V2 及原 box_bridge 文件放在同一目录。

核心原则：
  - 普通箱子：完全沿用 V2/原抓取逻辑；
  - large-skew 箱子：仍保持原动作顺序
      预抬升 -> 张开 -> 外扩 -> 下降 -> 低位内收 -> 闭合验证 -> post-lift；
  - 不修改视觉节点；
  - 不修改手眼标定；
  - 不取消 URDF 限位、waypoint 跳变、夹爪验证、共享 Lift。

V3 针对两类旧逻辑在“大歪箱子”上的结构问题：

1) Common-height 误用
--------------------
旧 common-height 把左右抓取点强制投影到同一个 Z。
对于真正倾斜的箱子，这会抹掉有效的 3D 高度差。

V3：
  仅当 large-skew 且 raw dZ 达到阈值时，
  临时关闭 common-height，保留左右真实 Z。
  普通箱子仍保留原 common-height 修正。

2) lateral 坐标系错误
---------------------
旧 grasp_pose_metrics 把世界坐标 Y 误差固定视为“夹紧方向误差”。
这只在箱体左右轴接近机器人 Y 轴时成立。

V3：
  large-skew 时使用真实左右抓取点连线 side_axis_xy：
      side axis    = right -> left
      forward axis = 与 side axis 垂直
  将 TCP XY 误差投影到箱体自身坐标系：
      box_lateral = dot(error_xy, side_axis_xy)
      box_forward = dot(error_xy, forward_axis_xy)

  优化与安全判定都使用 box_lateral；
  总 XY 误差仍保留；
  downstream 旧 verifier 的世界-Y lateral 门槛仅放宽到总 XY 门槛，
  但 V3 自己仍用更严格的 box_lateral 门槛做真正夹紧方向检查。

注意：
  本版本没有完整 3D collision scene。
  第一次必须 enable_motion:=false。
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import rclpy

from bridge_xyz_offset_box_wide_descend_postlift_large_skew_v2 import (
    LargeSkewV2Bridge,
)

from box_dual_arm_grasp_v9_template10_shoulder007_prelift025 import (
    LEFT_ARM_PARAMS,
    RIGHT_ARM_PARAMS,
    arm_fk_frames,
    grasp_tcp_offset_hand,
    rotation_vector,
    transl_mat,
)


class LargeSkewTilted3DV3Bridge(LargeSkewV2Bridge):
    """V2 + 保留真实 dZ + 箱体坐标系 lateral。"""

    def __init__(self) -> None:
        super().__init__()

        # --------------------------------------------------------------
        # 真实倾斜 3D 模式触发条件。
        # 当前实机困难样本：
        #   rotation ~29.5deg
        #   dX ~0.167m
        #   raw dZ ~0.035m
        # 因此会明确进入 preserve-raw-Z 模式。
        # --------------------------------------------------------------
        self.declare_parameter(
            "tilted3d_preserve_raw_z_trigger_cm",
            2.0,
        )
        self.declare_parameter(
            "tilted3d_max_raw_z_difference_cm",
            8.0,
        )
        self.declare_parameter(
            "tilted3d_min_slope_deg",
            3.0,
        )

        self.tilted3d_raw_z_trigger_m = (
            float(
                self.get_parameter(
                    "tilted3d_preserve_raw_z_trigger_cm"
                ).value
            )
            / 100.0
        )
        self.tilted3d_max_raw_z_difference_m = (
            float(
                self.get_parameter(
                    "tilted3d_max_raw_z_difference_cm"
                ).value
            )
            / 100.0
        )
        self.tilted3d_min_slope_rad = math.radians(
            float(
                self.get_parameter(
                    "tilted3d_min_slope_deg"
                ).value
            )
        )

        if not 0.010 <= self.tilted3d_raw_z_trigger_m <= 0.040:
            raise RuntimeError(
                "tilted3d_preserve_raw_z_trigger_cm "
                "必须在 [1.0,4.0]cm"
            )
        if not 0.050 <= self.tilted3d_max_raw_z_difference_m <= 0.120:
            raise RuntimeError(
                "tilted3d_max_raw_z_difference_cm "
                "必须在 [5.0,12.0]cm"
            )
        if not math.radians(1.0) <= self.tilted3d_min_slope_rad <= math.radians(10.0):
            raise RuntimeError(
                "tilted3d_min_slope_deg 必须在 [1,10]deg"
            )

        self._tilted3d_preserve_raw_z = False
        self._tilted3d_raw_dz_m = 0.0
        self._tilted3d_slope_rad = 0.0

        self.get_logger().warning(
            "Large-skew V3/Tilted-3D 已启用："
            "large-skew 时使用箱体自身 side-axis 计算夹紧误差；"
            f"raw_dZ>={self.tilted3d_raw_z_trigger_m*100:.1f}cm "
            f"且 slope>={math.degrees(self.tilted3d_min_slope_rad):.1f}deg "
            "时保留左右真实 Z，不做 common-height；"
            f"允许 raw_dZ 最大 {self.tilted3d_max_raw_z_difference_m*100:.1f}cm"
        )

    # ==================================================================
    # Geometry helpers
    # ==================================================================
    @staticmethod
    def _geometry_values(
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        left = np.asarray(left_base, dtype=np.float64)
        right = np.asarray(right_base, dtype=np.float64)

        delta_xy = left[:2] - right[:2]
        separation_xy = float(np.linalg.norm(delta_xy))

        if separation_xy < 1e-9:
            side_axis = np.array([0.0, 1.0], dtype=np.float64)
            forward_axis = np.array([1.0, 0.0], dtype=np.float64)
            rotation = 0.0
        else:
            side_axis = delta_xy / separation_xy

            # 选择标准姿态时 forward=[+X,0]。
            # side=[0,+Y] -> forward=[+X,0]
            forward_axis = np.array(
                [side_axis[1], -side_axis[0]],
                dtype=np.float64,
            )

            rotation = math.atan2(
                abs(float(side_axis[0])),
                abs(float(side_axis[1])),
            )

        raw_dz = abs(float(left[2] - right[2]))
        slope = math.atan2(raw_dz, max(separation_xy, 1e-9))

        return (
            side_axis,
            forward_axis,
            float(rotation),
            float(raw_dz),
            float(slope),
        )

    def _should_preserve_raw_z(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> bool:
        (
            _side_axis,
            _forward_axis,
            rotation,
            raw_dz,
            slope,
        ) = self._geometry_values(left_base, right_base)

        delta_xy = (
            np.asarray(left_base, dtype=np.float64)[:2]
            - np.asarray(right_base, dtype=np.float64)[:2]
        )
        dx = abs(float(delta_xy[0]))

        large_skew = bool(
            rotation >= self.large_skew_trigger_rotation_rad
            or dx >= self.large_skew_trigger_dx_m
        )

        preserve = bool(
            large_skew
            and raw_dz >= self.tilted3d_raw_z_trigger_m
            and slope >= self.tilted3d_min_slope_rad
        )

        self._tilted3d_preserve_raw_z = preserve
        self._tilted3d_raw_dz_m = raw_dz
        self._tilted3d_slope_rad = slope

        return preserve

    # ==================================================================
    # Geometry gate
    #
    # 在“真正 tilted3d”情况下，不让 CommonRimHeightOffsetBoxBridge
    # 用 common-Z 修复掉 raw dZ。
    # 同时只临时放宽 max_target_z_difference_m，其他几何门槛全部保留。
    # ==================================================================
    def _target_geometry_problem(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> Optional[str]:
        preserve = self._should_preserve_raw_z(
            left_base,
            right_base,
        )

        if not preserve:
            return super()._target_geometry_problem(
                left_base,
                right_base,
            )

        raw_dz = abs(
            float(
                np.asarray(left_base)[2]
                - np.asarray(right_base)[2]
            )
        )
        if raw_dz > self.tilted3d_max_raw_z_difference_m:
            return (
                "Large-skew tilted3d 原始左右高度差过大: "
                f"raw_dZ={raw_dz:.4f}m > "
                f"{self.tilted3d_max_raw_z_difference_m:.4f}m"
            )

        old_common = bool(
            self.enable_common_rim_height_projection
        )
        old_max_dz = float(
            self.max_target_z_difference_m
        )

        try:
            self.enable_common_rim_height_projection = False
            self.max_target_z_difference_m = max(
                old_max_dz,
                self.tilted3d_max_raw_z_difference_m,
            )
            reason = super()._target_geometry_problem(
                left_base,
                right_base,
            )
        finally:
            self.enable_common_rim_height_projection = old_common
            self.max_target_z_difference_m = old_max_dz

        if reason is None:
            self.get_logger().warning(
                "Tilted-3D 目标通过几何门控："
                f"保留真实 raw_dZ={raw_dz:.4f}m，"
                f"slope={math.degrees(self._tilted3d_slope_rad):.2f}deg；"
                "本目标不会做 common-height 投影"
            )

        return reason

    # ==================================================================
    # V3 position residual in box-aligned coordinates.
    #
    # 普通箱子：完全走 V2/原 residual。
    # large-skew：位置前后/夹紧方向改为箱体自身坐标系。
    # hand orientation、forearm、template 权重结构不变；
    # V2 的 stage-residual 仍会对它继续做 position-first 调权。
    # ==================================================================
    def _box_aligned_tcp_residual(
        self,
        q: np.ndarray,
        target_xyz: np.ndarray,
        template_q: np.ndarray,
        lift: float,
        params,
    ) -> np.ndarray:
        transform, frames = arm_fk_frames(
            q,
            lift,
            0.0,
            params,
        )
        template_transform, _ = arm_fk_frames(
            template_q,
            lift,
            0.0,
            params,
        )

        tcp_offset = grasp_tcp_offset_hand(params)
        tcp_transform = transform @ transl_mat(
            float(tcp_offset[0]),
            float(tcp_offset[1]),
            float(tcp_offset[2]),
        )

        position_error = (
            tcp_transform[:3, 3]
            - np.asarray(target_xyz, dtype=np.float64)
        )

        side_axis = np.asarray(
            self._large_skew_axis_xy,
            dtype=np.float64,
        )
        side_norm = float(np.linalg.norm(side_axis))
        if side_norm < 1e-9:
            side_axis = np.array([0.0, 1.0], dtype=np.float64)
        else:
            side_axis = side_axis / side_norm

        forward_axis = np.array(
            [side_axis[1], -side_axis[0]],
            dtype=np.float64,
        )

        box_forward_error = float(
            np.dot(position_error[:2], forward_axis)
        )
        box_lateral_error = float(
            np.dot(position_error[:2], side_axis)
        )

        orientation_error = rotation_vector(
            transform[:3, :3]
            @ template_transform[:3, :3].T
        )

        forearm = (
            frames[6][:3, 3]
            - frames[3][:3, 3]
        )
        forearm_norm = float(np.linalg.norm(forearm))
        forearm_vertical = (
            1.0
            if forearm_norm < 1e-9
            else float(forearm[2] / forearm_norm)
        )

        return np.concatenate(
            [
                np.array(
                    [
                        box_forward_error,
                        float(self.lateral_weight)
                        * box_lateral_error,
                        2.0 * float(position_error[2]),
                    ],
                    dtype=np.float64,
                ),
                0.30 * orientation_error,
                np.array(
                    [0.40 * forearm_vertical],
                    dtype=np.float64,
                ),
                0.015 * (
                    np.asarray(q, dtype=np.float64)
                    - np.asarray(template_q, dtype=np.float64)
                ),
            ]
        )

    def _joint_xyz_residual(
        self,
        state: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
    ) -> np.ndarray:
        if not self._large_skew_active:
            return super()._joint_xyz_residual(
                state,
                left_target,
                right_target,
            )

        shared_lift = float(state[14])

        return np.concatenate(
            [
                self._box_aligned_tcp_residual(
                    state[:7],
                    left_target,
                    self.left_template_q,
                    shared_lift,
                    LEFT_ARM_PARAMS,
                ),
                self._box_aligned_tcp_residual(
                    state[7:14],
                    right_target,
                    self.right_template_q,
                    shared_lift,
                    RIGHT_ARM_PARAMS,
                ),
            ]
        )

    # ==================================================================
    # Metrics: large-skew 时把 lateral/forward 改成 box frame 投影。
    # ==================================================================
    def _state_metrics(
        self,
        state: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        left_metrics, right_metrics = super()._state_metrics(
            state,
            left_target,
            right_target,
        )

        if not self._large_skew_active:
            return left_metrics, right_metrics

        side_axis = np.asarray(
            self._large_skew_axis_xy,
            dtype=np.float64,
        )
        side_norm = float(np.linalg.norm(side_axis))
        if side_norm < 1e-9:
            side_axis = np.array([0.0, 1.0], dtype=np.float64)
        else:
            side_axis = side_axis / side_norm

        forward_axis = np.array(
            [side_axis[1], -side_axis[0]],
            dtype=np.float64,
        )

        def convert(
            metrics: Dict[str, float],
            target: np.ndarray,
        ) -> Dict[str, float]:
            out = dict(metrics)

            actual_xy = np.array(
                [
                    float(out["x_m"]),
                    float(out["y_m"]),
                ],
                dtype=np.float64,
            )
            target_xy = np.asarray(
                target,
                dtype=np.float64,
            )[:2]

            error_xy = actual_xy - target_xy

            out["world_x_error_m"] = abs(
                float(error_xy[0])
            )
            out["world_y_error_m"] = abs(
                float(error_xy[1])
            )

            out["forward_error_m"] = abs(
                float(np.dot(error_xy, forward_axis))
            )
            out["lateral_error_m"] = abs(
                float(np.dot(error_xy, side_axis))
            )
            out["box_forward_error_m"] = out["forward_error_m"]
            out["box_lateral_error_m"] = out["lateral_error_m"]

            return out

        return (
            convert(left_metrics, left_target),
            convert(right_metrics, right_target),
        )

    # ==================================================================
    # Large-skew safety gate:
    # - 总 XY 使用 V2 large-skew 门槛；
    # - 真正夹紧方向使用 box-axis lateral 门槛；
    # - Z / orientation / forearm / vertical / template 继续原逻辑。
    # ==================================================================
    def _metrics_pass(
        self,
        left_metrics: Mapping[str, float],
        right_metrics: Mapping[str, float],
    ) -> bool:
        if not self._large_skew_active:
            return super()._metrics_pass(
                left_metrics,
                right_metrics,
            )

        xyz_z_tolerance = min(
            self.max_z_error_m,
            0.45 * self.max_shared_lift_mismatch_m,
        )

        for metrics in (left_metrics, right_metrics):
            if metrics["xy_error_m"] > self.max_xy_error_m:
                return False

            # V3 真正安全夹紧方向：箱体自身 side axis。
            if (
                metrics["lateral_error_m"]
                > self.large_skew_lateral_error_m
            ):
                return False

            if metrics["z_error_m"] > xyz_z_tolerance:
                return False

            if (
                metrics["orientation_error_rad"]
                > self.max_orientation_error_rad
            ):
                return False

            if (
                metrics["forearm_tilt_rad"]
                > self.max_forearm_tilt_rad
            ):
                return False

            if (
                metrics["gripper_vertical_error_rad"]
                > self.max_gripper_vertical_error_rad
            ):
                return False

            if (
                metrics["template_rmse_rad"]
                > self.max_template_rmse_rad
            ):
                return False

        return True

    # ==================================================================
    # V2 会把 max_lateral_error_m 设到 large-skew lateral=0.8cm。
    #
    # 但 inherited _solve_asymmetric() 最后还会调用旧 verify_grasp_solution，
    # 该函数仍用“世界 Y”作为 lateral。
    #
    # 为避免旧坐标系重复误拒绝：
    # - downstream 旧 lateral 门槛放宽到总 XY 门槛；
    # - V3 自己 _metrics_pass 仍用 box-axis 0.8cm 严格检查。
    # ==================================================================
    def _apply_large_skew_thresholds(self) -> None:
        super()._apply_large_skew_thresholds()

        self.max_lateral_error_m = max(
            float(self.max_lateral_error_m),
            float(self.max_xy_error_m),
        )

    # ==================================================================
    # Planning:
    # tilted3d 模式下临时关闭 common-height，保留真实左右 Z。
    # Wide-descend / AdaptiveOffset / post-lift 仍全部经过原 MRO。
    # ==================================================================
    def _plan_sequence(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> dict:
        preserve = self._should_preserve_raw_z(
            left_base,
            right_base,
        )

        if not preserve:
            return super()._plan_sequence(
                left_base,
                right_base,
            )

        raw_dz = abs(
            float(
                np.asarray(left_base)[2]
                - np.asarray(right_base)[2]
            )
        )

        if raw_dz > self.tilted3d_max_raw_z_difference_m:
            raise RuntimeError(
                "Tilted-3D raw dZ 超过安全范围: "
                f"{raw_dz:.4f}m > "
                f"{self.tilted3d_max_raw_z_difference_m:.4f}m"
            )

        old_common = bool(
            self.enable_common_rim_height_projection
        )
        old_max_dz = float(
            self.max_target_z_difference_m
        )

        self.get_logger().warning(
            "Tilted-3D 规划：关闭 common-height，"
            "保留视觉/标定后的真实左右 Z："
            f"Lz={float(np.asarray(left_base)[2]):+.4f}m, "
            f"Rz={float(np.asarray(right_base)[2]):+.4f}m, "
            f"raw_dZ={raw_dz:.4f}m, "
            f"slope={math.degrees(self._tilted3d_slope_rad):.2f}deg"
        )

        try:
            self.enable_common_rim_height_projection = False
            self.max_target_z_difference_m = max(
                old_max_dz,
                self.tilted3d_max_raw_z_difference_m,
            )

            sequence = super()._plan_sequence(
                left_base,
                right_base,
            )

            sequence["tilted3d_preserve_raw_z"] = True
            sequence["tilted3d_raw_dz_m"] = raw_dz
            sequence["tilted3d_slope_deg"] = math.degrees(
                self._tilted3d_slope_rad
            )

            final_plan = sequence["plans"][-1]
            lm = final_plan["metrics"]["left_final"]
            rm = final_plan["metrics"]["right_final"]

            self.get_logger().warning(
                "Tilted-3D V3 最终箱体坐标系检查: "
                f"L[XY={lm['xy_error_m']*100:.2f}cm, "
                f"boxLat={lm['lateral_error_m']*100:.2f}cm, "
                f"boxFwd={lm['forward_error_m']*100:.2f}cm, "
                f"Z={lm['z_error_m']*100:.2f}cm]; "
                f"R[XY={rm['xy_error_m']*100:.2f}cm, "
                f"boxLat={rm['lateral_error_m']*100:.2f}cm, "
                f"boxFwd={rm['forward_error_m']*100:.2f}cm, "
                f"Z={rm['z_error_m']*100:.2f}cm]"
            )

            return sequence

        finally:
            self.enable_common_rim_height_projection = old_common
            self.max_target_z_difference_m = old_max_dz


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[LargeSkewTilted3DV3Bridge] = None

    try:
        node = LargeSkewTilted3DV3Bridge()
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
