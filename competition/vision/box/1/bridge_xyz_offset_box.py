#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
偏置箱体双臂抓取增强版。

把本文件与当前正在使用的以下文件放在同一目录：
  bridge_xyz_guard.py
  box_grasp_bridge_asymmetric_staged_sharedlift_xyzfix.py
  box_grasp_bridge_asymmetric_staged_avoidance.py
  box_grasp_bridge_mirror_vision_raw_prelift025.py
  box_dual_arm_grasp_v9_template10_shoulder007_prelift025.py

相对 bridge_xyz_guard.py 的变化：
1. 共享 lift XYZ IK 增加“先到位置、再恢复姿态”的两阶段搜索；
2. 每一步采用回退线搜索，避免一次较大的 DLS 步长把右臂推入错误分支；
3. 精确视觉中点确实受单侧肩关节硬限位约束时，允许困难一侧的最终
   TCP 向箱体外侧移动少量距离。这个移动不是要求底盘重新对齐；夹爪仍从
   外侧下降并向内闭合。

默认 enable_motion=false。必须先完成干运行，确认日志中的 final_tcp_outset、
最终关节角和全部接近路点，再低速实机测试。
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import rclpy

from bridge_xyz_guard import GripperBalancedSharedLiftBridge
from box_dual_arm_grasp_v9_template10_shoulder007_prelift025 import (
    arm_joint_limit_arrays,
)


class OffsetBoxReachBridge(GripperBalancedSharedLiftBridge):
    """保持非对称目标，并为偏置箱体增加单侧可达性回退。"""

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("offset_ik_reach_iterations", 80)
        self.declare_parameter("offset_ik_refine_iterations", 180)
        self.declare_parameter("offset_ik_line_search_steps", 5)
        self.declare_parameter("offset_ik_seed_branches", 4)
        self.declare_parameter("enable_final_tcp_outset_fallback", True)
        self.declare_parameter("max_final_tcp_outset_m", 0.018)
        self.declare_parameter("final_tcp_outset_step_m", 0.006)
        self.declare_parameter("offset_side_deadband_m", 0.015)

        self.offset_ik_reach_iterations = int(
            self.get_parameter("offset_ik_reach_iterations").value
        )
        self.offset_ik_refine_iterations = int(
            self.get_parameter("offset_ik_refine_iterations").value
        )
        self.offset_ik_line_search_steps = int(
            self.get_parameter("offset_ik_line_search_steps").value
        )
        self.offset_ik_seed_branches = int(
            self.get_parameter("offset_ik_seed_branches").value
        )
        self.enable_final_tcp_outset_fallback = bool(
            self.get_parameter("enable_final_tcp_outset_fallback").value
        )
        self.max_final_tcp_outset_m = float(
            self.get_parameter("max_final_tcp_outset_m").value
        )
        self.final_tcp_outset_step_m = float(
            self.get_parameter("final_tcp_outset_step_m").value
        )
        self.offset_side_deadband_m = float(
            self.get_parameter("offset_side_deadband_m").value
        )

        if not 20 <= self.offset_ik_reach_iterations <= 250:
            raise RuntimeError("offset_ik_reach_iterations 必须在 [20, 250] 内")
        if not 50 <= self.offset_ik_refine_iterations <= 500:
            raise RuntimeError("offset_ik_refine_iterations 必须在 [50, 500] 内")
        if not 2 <= self.offset_ik_line_search_steps <= 8:
            raise RuntimeError("offset_ik_line_search_steps 必须在 [2, 8] 内")
        if not 2 <= self.offset_ik_seed_branches <= 6:
            raise RuntimeError("offset_ik_seed_branches 必须在 [2, 6] 内")
        if not 0.0 <= self.max_final_tcp_outset_m <= 0.020:
            raise RuntimeError("max_final_tcp_outset_m 必须在 [0, 0.020]m 内")
        if not 0.001 <= self.final_tcp_outset_step_m <= 0.010:
            raise RuntimeError("final_tcp_outset_step_m 必须在 [0.001, 0.010]m 内")
        if self.offset_side_deadband_m < 0.0:
            raise RuntimeError("offset_side_deadband_m 必须 >= 0")

        self.get_logger().warning(
            "偏置箱体模式已启用: 两阶段 IK + 单侧最终 TCP 外移回退；"
            f"max_outset={self.max_final_tcp_outset_m*1000:.1f}mm"
        )

    def _stage_residual(
        self,
        state: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
        reach_stage: bool,
    ) -> np.ndarray:
        residual = self._joint_xyz_residual(
            state,
            left_target,
            right_target,
        ).copy()
        if not reach_stage:
            return residual

        # 每只手臂的 residual 排列：XYZ(3), orientation(3), forearm(1),
        # template(7)，共 14 项。第一阶段优先逃离错误局部解；第二阶段恢复原始
        # 严格姿态权重，最终是否通过仍由原安全阈值决定。
        for start in (0, 14):
            residual[start + 0] *= 4.0
            residual[start + 1] *= 2.5
            residual[start + 2] *= 2.5
            residual[start + 3 : start + 6] *= 0.35
            residual[start + 6] *= 0.40
            residual[start + 7 : start + 14] *= 0.20
        return residual

    def _run_dls_stage(
        self,
        initial_state: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
        *,
        reach_stage: bool,
        iterations: int,
    ) -> np.ndarray:
        state = np.clip(initial_state.copy(), lower, upper)
        damping = max(0.003, float(self.xyz_solver_damping))
        finite_delta = 1e-5
        stalled_iterations = 0

        for _ in range(iterations):
            residual = self._stage_residual(
                state,
                left_target,
                right_target,
                reach_stage,
            )
            old_cost = float(residual @ residual)
            jacobian = np.zeros(
                (len(residual), len(state)),
                dtype=np.float64,
            )

            for variable_index in range(len(state)):
                trial = state.copy()
                trial[variable_index] = min(
                    float(upper[variable_index]),
                    float(state[variable_index] + finite_delta),
                )
                actual_delta = trial[variable_index] - state[variable_index]
                if abs(actual_delta) < 1e-12:
                    trial[variable_index] = max(
                        float(lower[variable_index]),
                        float(state[variable_index] - finite_delta),
                    )
                    actual_delta = (
                        trial[variable_index] - state[variable_index]
                    )
                if abs(actual_delta) < 1e-12:
                    continue
                jacobian[:, variable_index] = (
                    self._stage_residual(
                        trial,
                        left_target,
                        right_target,
                        reach_stage,
                    )
                    - residual
                ) / actual_delta

            normal = (
                jacobian.T @ jacobian
                + damping * damping
                * np.eye(len(state), dtype=np.float64)
            )
            try:
                delta_state = -np.linalg.solve(
                    normal,
                    jacobian.T @ residual,
                )
            except np.linalg.LinAlgError:
                break

            largest_joint_step = float(
                np.max(np.abs(delta_state[:14]))
            )
            if largest_joint_step > self.xyz_solver_max_joint_step_rad:
                delta_state *= (
                    self.xyz_solver_max_joint_step_rad
                    / largest_joint_step
                )
            # lift 单次最多变化 20mm，防止数值病态时突然撞到行程边界。
            delta_state[14] = float(
                np.clip(delta_state[14], -0.020, 0.020)
            )

            accepted = False
            best_trial = state
            best_cost = old_cost
            for line_index in range(self.offset_ik_line_search_steps):
                scale = 0.5 ** line_index
                trial = np.clip(
                    state + scale * delta_state,
                    lower,
                    upper,
                )
                trial_residual = self._stage_residual(
                    trial,
                    left_target,
                    right_target,
                    reach_stage,
                )
                trial_cost = float(trial_residual @ trial_residual)
                if trial_cost < best_cost - 1e-12:
                    best_trial = trial
                    best_cost = trial_cost
                    accepted = True
                    break

            if accepted:
                movement = float(np.linalg.norm(best_trial - state))
                state = best_trial
                damping = max(0.002, damping * 0.75)
                stalled_iterations = 0
                if movement < 1e-8:
                    break
            else:
                damping = min(0.20, damping * 2.0)
                stalled_iterations += 1
                if stalled_iterations >= 8:
                    break

            left_metrics, right_metrics = self._state_metrics(
                state,
                left_target,
                right_target,
            )
            if not reach_stage and self._metrics_pass(
                left_metrics,
                right_metrics,
            ):
                break
            if reach_stage:
                position_ready = all(
                    metrics["xy_error_m"] <= 1.5 * self.max_xy_error_m
                    and metrics["z_error_m"] <= 1.5 * self.max_z_error_m
                    for metrics in (left_metrics, right_metrics)
                )
                if position_ready:
                    break

        return state

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
        current_l = np.clip(left_seed, left_lower, left_upper)
        current_r = np.clip(right_seed, right_lower, right_upper)
        template_l = np.clip(self.left_template_q, left_lower, left_upper)
        template_r = np.clip(self.right_template_q, right_lower, right_upper)

        seeds: List[Tuple[np.ndarray, np.ndarray]] = [
            (template_l.copy(), template_r.copy()),
            (current_l.copy(), current_r.copy()),
            (
                np.clip(0.65 * template_l + 0.35 * current_l, left_lower, left_upper),
                np.clip(0.65 * template_r + 0.35 * current_r, right_lower, right_upper),
            ),
        ]

        # 箱体偏左时重点扩展左臂分支；偏右时扩展右臂分支。
        branch_l = template_l.copy()
        branch_r = template_r.copy()
        if center_y >= 0.0:
            branch_r[1] = right_upper[1] - min(
                math.radians(0.2),
                0.1 * max(0.0, right_upper[1] - right_lower[1]),
            )
            branch_r[0] += math.radians(12.0)
            branch_r[2] -= math.radians(8.0)
            branch_r[3] -= math.radians(15.0)
            branch_r = np.clip(branch_r, right_lower, right_upper)
        else:
            branch_l[1] = left_lower[1] + min(
                math.radians(0.2),
                0.1 * max(0.0, left_upper[1] - left_lower[1]),
            )
            branch_l[0] -= math.radians(12.0)
            branch_l[2] += math.radians(8.0)
            branch_l[3] += math.radians(15.0)
            branch_l = np.clip(branch_l, left_lower, left_upper)
        seeds.append((branch_l, branch_r))
        return seeds[: self.offset_ik_seed_branches]

    def _optimize_shared_lift_xyz(
        self,
        left_target: np.ndarray,
        right_target: np.ndarray,
        left_seed: np.ndarray,
        right_seed: np.ndarray,
        current_lift: float,
        limits: Mapping[str, Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
        left_lower, left_upper = arm_joint_limit_arrays(limits, "left")
        right_lower, right_upper = arm_joint_limit_arrays(limits, "right")
        lift_lower, lift_upper, _ = limits["base_elevator_joint"]
        waist_lower, waist_upper, _ = limits["elevator_pitch_joint"]
        if not waist_lower - 1e-12 <= 0.0 <= waist_upper + 1e-12:
            raise RuntimeError(
                "偏置箱体 XYZ 求解要求 elevator_pitch_joint 可取 0rad；"
                f"当前限位=[{waist_lower:+.4f}, {waist_upper:+.4f}]"
            )

        lower = np.concatenate(
            [left_lower, right_lower, np.array([lift_lower])]
        )
        upper = np.concatenate(
            [left_upper, right_upper, np.array([lift_upper])]
        )
        center_y = 0.5 * float(left_target[1] + right_target[1])
        seed_pairs = self._seed_pairs(
            np.asarray(left_seed, dtype=np.float64),
            np.asarray(right_seed, dtype=np.float64),
            left_lower,
            left_upper,
            right_lower,
            right_upper,
            center_y,
        )

        self.get_logger().info(
            "偏置 IK 实际肩关节限位: "
            f"Lq2=[{math.degrees(left_lower[1]):+.1f},"
            f"{math.degrees(left_upper[1]):+.1f}]deg, "
            f"Rq2=[{math.degrees(right_lower[1]):+.1f},"
            f"{math.degrees(right_upper[1]):+.1f}]deg, "
            f"target_center_y={center_y:+.4f}m"
        )

        best_state: Optional[np.ndarray] = None
        best_left: Optional[Dict[str, float]] = None
        best_right: Optional[Dict[str, float]] = None
        best_score = float("inf")

        for seed_index, (seed_l, seed_r) in enumerate(seed_pairs):
            lift_guess = self._lift_guess(
                seed_l,
                seed_r,
                left_target,
                right_target,
                current_lift,
                self.left_template_q,
                self.right_template_q,
            )
            initial_state = np.concatenate(
                [
                    seed_l,
                    seed_r,
                    np.array([float(np.clip(lift_guess, lift_lower, lift_upper))]),
                ]
            )
            reached_state = self._run_dls_stage(
                initial_state,
                lower,
                upper,
                left_target,
                right_target,
                reach_stage=True,
                iterations=self.offset_ik_reach_iterations,
            )
            state = self._run_dls_stage(
                reached_state,
                lower,
                upper,
                left_target,
                right_target,
                reach_stage=False,
                iterations=self.offset_ik_refine_iterations,
            )
            left_metrics, right_metrics = self._state_metrics(
                state,
                left_target,
                right_target,
            )
            score = self._candidate_score(left_metrics, right_metrics)
            if score < best_score:
                best_score = score
                best_state = state.copy()
                best_left = dict(left_metrics)
                best_right = dict(right_metrics)

            if self._metrics_pass(left_metrics, right_metrics):
                self.get_logger().info(
                    "偏置箱体共享 lift XYZ IK 通过: "
                    f"seed={seed_index}, lift={state[14]:+.4f}m, "
                    f"Lerr=({left_metrics['xy_error_m']*100:.2f},"
                    f"{left_metrics['z_error_m']*100:.2f})cm, "
                    f"Rerr=({right_metrics['xy_error_m']*100:.2f},"
                    f"{right_metrics['z_error_m']*100:.2f})cm"
                )
                return state, left_metrics, right_metrics

        if best_state is None or best_left is None or best_right is None:
            raise RuntimeError("偏置箱体 IK 未产生候选解")

        def summary(metrics: Mapping[str, float]) -> str:
            return (
                f"XY={metrics['xy_error_m']*100:.2f}cm, "
                f"Z={metrics['z_error_m']*100:.2f}cm, "
                f"vertical={math.degrees(metrics['gripper_vertical_error_rad']):.2f}deg, "
                f"forearm={math.degrees(metrics['forearm_tilt_rad']):.2f}deg, "
                f"template={math.degrees(metrics['template_rmse_rad']):.2f}deg"
            )

        raise RuntimeError(
            "偏置箱体共享 lift XYZ IK 未通过安全阈值。最接近结果: "
            f"lift={best_state[14]:+.4f}m; "
            f"L[{summary(best_left)}]; R[{summary(best_right)}]"
        )

    def _outset_values(self) -> List[float]:
        values = [0.0]
        if not self.enable_final_tcp_outset_fallback:
            return values
        value = self.final_tcp_outset_step_m
        while value < self.max_final_tcp_outset_m - 1e-9:
            values.append(float(value))
            value += self.final_tcp_outset_step_m
        if self.max_final_tcp_outset_m > 0.0:
            values.append(float(self.max_final_tcp_outset_m))
        return sorted(set(round(v, 9) for v in values))

    def _plan_sequence(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> dict:
        left_nominal = np.asarray(left_base, dtype=np.float64).copy()
        right_nominal = np.asarray(right_base, dtype=np.float64).copy()
        side_delta = left_nominal[:2] - right_nominal[:2]
        side_distance = float(np.linalg.norm(side_delta))
        if side_distance < 1e-9:
            raise RuntimeError("左右抓取点重合，无法计算箱体横向轴")
        side_axis = side_delta / side_distance  # 右 -> 左

        # 保留上一版的左右独立夹紧补偿，只应用一次。
        left_nominal[:2] -= side_axis * self.left_grasp_inset_m
        right_nominal[:2] += side_axis * self.right_grasp_inset_m
        center_y = 0.5 * float(left_nominal[1] + right_nominal[1])

        if center_y > self.offset_side_deadband_m:
            difficult_side = "right"
        elif center_y < -self.offset_side_deadband_m:
            difficult_side = "left"
        else:
            difficult_side = "none"

        last_error: Optional[Exception] = None
        for outset in self._outset_values():
            if outset > 0.0 and difficult_side == "none":
                break

            left_target = left_nominal.copy()
            right_target = right_nominal.copy()
            if difficult_side == "right":
                right_target[:2] -= side_axis * outset
            elif difficult_side == "left":
                left_target[:2] += side_axis * outset

            try:
                # 跳过 GripperBalancedSharedLiftBridge._plan_sequence，避免
                # left/right_grasp_inset 被重复应用；其后的分段抬升、外扩、下降、
                # 路点跳变检查和执行锁全部原样保留。
                sequence = super(
                    GripperBalancedSharedLiftBridge,
                    self,
                )._plan_sequence(left_target, right_target)
                sequence["final_tcp_outset_m"] = float(outset)
                sequence["final_tcp_outset_side"] = difficult_side
                if outset > 0.0:
                    self.get_logger().warning(
                        "精确视觉中点受单侧肩关节限位约束，已采用夹爪捕获回退: "
                        f"side={difficult_side}, outset={outset*1000:.1f}mm, "
                        f"L={np.round(left_target, 5)}, "
                        f"R={np.round(right_target, 5)}"
                    )
                else:
                    self.get_logger().info("精确视觉抓取点可达，无需最终 TCP 外移")
                return sequence
            except RuntimeError as exc:
                last_error = exc
                if outset <= 0.0 and "IK" not in str(exc):
                    # 非 IK 错误（几何、预抬升净空或路点跳变）不能靠移动 TCP 掩盖。
                    raise
                self.get_logger().warning(
                    f"偏置抓取候选失败: side={difficult_side}, "
                    f"outset={outset*1000:.1f}mm; {exc}"
                )

        raise RuntimeError(
            "偏置箱体在允许的最终 TCP 捕获范围内仍无安全路径；"
            f"最后错误: {last_error}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[OffsetBoxReachBridge] = None
    try:
        node = OffsetBoxReachBridge()
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
