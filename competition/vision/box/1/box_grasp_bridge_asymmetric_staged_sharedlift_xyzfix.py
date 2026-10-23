#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
非对称箱体分阶段抓取：共享 lift + 左右臂 XYZ 联合求解修正版。

请把本文件与以下三个文件放在同一目录：
  box_grasp_bridge_asymmetric_staged_avoidance.py
  box_grasp_bridge_mirror_vision_raw_prelift025.py
  box_dual_arm_grasp_v9_template10_shoulder007_prelift025.py

原非对称版本仍调用 dual_ik_xy_then_lift：左右臂只求 XY，最后各自计算
所需 lift。非对称姿态下两个 lift 需求可能相差数厘米，而机器人只有一个
共享升降轴，因此规划必然失败。

本版本把变量改为：
  [左臂 7 关节, 右臂 7 关节, 共享 lift]
在 waist=0 的条件下同时优化左右 TCP 的 XYZ、模板姿态、前臂水平度和
夹爪竖直度。waist 固定 0 可保留两侧镜像安装的竖直夹爪姿态，同时允许
左右臂关节分别偏离镜像模板以到达非对称坐标。
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import rclpy

from box_grasp_bridge_asymmetric_staged_avoidance import (
    AsymmetricStagedBoxGraspBridge,
)
from box_dual_arm_grasp_v9_template10_shoulder007_prelift025 import (
    LEFT_ARM_PARAMS,
    RIGHT_ARM_PARAMS,
    arm_fk_frames,
    arm_joint_limit_arrays,
    grasp_pose_metrics,
    grasp_tcp_offset_hand,
    rotation_vector,
    transl_mat,
    validate_solution,
    verify_grasp_solution,
)


class SharedLiftXYZBoxGraspBridge(AsymmetricStagedBoxGraspBridge):
    """用共享 lift 和两套独立关节联合求解两个非对称 XYZ 目标。"""

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("xyz_solver_max_iterations", 260)
        self.declare_parameter("xyz_solver_damping", 0.010)
        self.declare_parameter("xyz_solver_max_joint_step_deg", 5.73)
        self.declare_parameter("xyz_max_approach_lift_variation_m", 0.040)

        self.xyz_solver_max_iterations = int(
            self.get_parameter("xyz_solver_max_iterations").value
        )
        self.xyz_solver_damping = float(
            self.get_parameter("xyz_solver_damping").value
        )
        self.xyz_solver_max_joint_step_rad = math.radians(
            float(
                self.get_parameter("xyz_solver_max_joint_step_deg").value
            )
        )
        self.max_approach_lift_variation_m = float(
            self.get_parameter(
                "xyz_max_approach_lift_variation_m"
            ).value
        )

        if not 50 <= self.xyz_solver_max_iterations <= 1000:
            raise RuntimeError(
                "xyz_solver_max_iterations 必须在 [50, 1000] 内"
            )
        if not 1e-5 <= self.xyz_solver_damping <= 0.2:
            raise RuntimeError("xyz_solver_damping 必须在 [1e-5, 0.2] 内")
        if not 0.01 <= self.xyz_solver_max_joint_step_rad <= 0.30:
            raise RuntimeError(
                "xyz_solver_max_joint_step_deg 必须约在 [0.57, 17.2]deg 内"
            )
        if not 0.0 <= self.max_approach_lift_variation_m <= 0.10:
            raise RuntimeError(
                "xyz_max_approach_lift_variation_m 必须在 [0, 0.10]m 内"
            )

        self.get_logger().warning(
            "已启用共享 lift + 双臂 XYZ 联合 IK；waist 固定为 0rad。"
        )

    @staticmethod
    def _tcp_residual(
        q: np.ndarray,
        target_xyz: np.ndarray,
        template_q: np.ndarray,
        lift: float,
        params,
        lateral_weight: float,
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
        position_error = tcp_transform[:3, 3] - target_xyz

        orientation_error = rotation_vector(
            transform[:3, :3]
            @ template_transform[:3, :3].T
        )

        forearm = frames[6][:3, 3] - frames[3][:3, 3]
        forearm_norm = float(np.linalg.norm(forearm))
        forearm_vertical = (
            1.0
            if forearm_norm < 1e-9
            else float(forearm[2] / forearm_norm)
        )

        # 位置使用米，姿态使用弧度。权重沿用原模板 IK 的量级；Z 明确进入
        # 优化，且左右臂共享同一个 lift 变量。
        return np.concatenate(
            [
                np.array(
                    [
                        position_error[0],
                        float(lateral_weight) * position_error[1],
                        2.0 * position_error[2],
                    ],
                    dtype=np.float64,
                ),
                0.30 * orientation_error,
                np.array(
                    [0.40 * forearm_vertical],
                    dtype=np.float64,
                ),
                0.015 * (q - template_q),
            ]
        )

    def _joint_xyz_residual(
        self,
        state: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
    ) -> np.ndarray:
        left_q = state[:7]
        right_q = state[7:14]
        shared_lift = float(state[14])

        return np.concatenate(
            [
                self._tcp_residual(
                    left_q,
                    left_target,
                    self.left_template_q,
                    shared_lift,
                    LEFT_ARM_PARAMS,
                    self.lateral_weight,
                ),
                self._tcp_residual(
                    right_q,
                    right_target,
                    self.right_template_q,
                    shared_lift,
                    RIGHT_ARM_PARAMS,
                    self.lateral_weight,
                ),
            ]
        )

    def _state_metrics(
        self,
        state: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        lift = float(state[14])
        left_metrics = grasp_pose_metrics(
            state[:7],
            left_target,
            self.left_template_q,
            lift,
            0.0,
            LEFT_ARM_PARAMS,
        )
        right_metrics = grasp_pose_metrics(
            state[7:14],
            right_target,
            self.right_template_q,
            lift,
            0.0,
            RIGHT_ARM_PARAMS,
        )
        return left_metrics, right_metrics

    def _metrics_pass(
        self,
        left_metrics: Mapping[str, float],
        right_metrics: Mapping[str, float],
    ) -> bool:
        # 为保证“左右所需 lift 差”必定小于共享 lift 阈值，单臂 Z 求解
        # 使用更紧的内部阈值；最终仍执行原 verify_grasp_solution 全量检查。
        xyz_z_tolerance = min(
            self.max_z_error_m,
            0.45 * self.max_shared_lift_mismatch_m,
        )

        for metrics in (left_metrics, right_metrics):
            if metrics["xy_error_m"] > self.max_xy_error_m:
                return False
            if metrics["lateral_error_m"] > self.max_lateral_error_m:
                return False
            if metrics["z_error_m"] > xyz_z_tolerance:
                return False
            if (
                metrics["orientation_error_rad"]
                > self.max_orientation_error_rad
            ):
                return False
            if metrics["forearm_tilt_rad"] > self.max_forearm_tilt_rad:
                return False
            if (
                metrics["gripper_vertical_error_rad"]
                > self.max_gripper_vertical_error_rad
            ):
                return False
            if metrics["template_rmse_rad"] > self.max_template_rmse_rad:
                return False
        return True

    @staticmethod
    def _candidate_score(
        left_metrics: Mapping[str, float],
        right_metrics: Mapping[str, float],
    ) -> float:
        return float(
            left_metrics["xy_error_m"]
            + right_metrics["xy_error_m"]
            + 2.0 * left_metrics["z_error_m"]
            + 2.0 * right_metrics["z_error_m"]
            + 0.04 * left_metrics["orientation_error_rad"]
            + 0.04 * right_metrics["orientation_error_rad"]
            + 0.02 * left_metrics["template_rmse_rad"]
            + 0.02 * right_metrics["template_rmse_rad"]
        )

    @staticmethod
    def _lift_guess(
        left_q: np.ndarray,
        right_q: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
        reference_lift: float,
        left_template_q: np.ndarray,
        right_template_q: np.ndarray,
    ) -> float:
        left_metrics = grasp_pose_metrics(
            left_q,
            left_target,
            left_template_q,
            reference_lift,
            0.0,
            LEFT_ARM_PARAMS,
        )
        right_metrics = grasp_pose_metrics(
            right_q,
            right_target,
            right_template_q,
            reference_lift,
            0.0,
            RIGHT_ARM_PARAMS,
        )
        left_required = (
            reference_lift
            + float(left_target[2])
            - float(left_metrics["z_m"])
        )
        right_required = (
            reference_lift
            + float(right_target[2])
            - float(right_metrics["z_m"])
        )
        return 0.5 * (left_required + right_required)

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
                "共享 XYZ 求解要求 waist=0，但当前限位不包含 0rad: "
                f"[{waist_lower:+.4f}, {waist_upper:+.4f}]"
            )

        lower = np.concatenate(
            [
                left_lower,
                right_lower,
                np.array([lift_lower], dtype=np.float64),
            ]
        )
        upper = np.concatenate(
            [
                left_upper,
                right_upper,
                np.array([lift_upper], dtype=np.float64),
            ]
        )

        clipped_left_seed = np.clip(left_seed, left_lower, left_upper)
        clipped_right_seed = np.clip(right_seed, right_lower, right_upper)
        template_left = np.clip(
            self.left_template_q,
            left_lower,
            left_upper,
        )
        template_right = np.clip(
            self.right_template_q,
            right_lower,
            right_upper,
        )

        seed_pairs = [
            (clipped_left_seed, clipped_right_seed),
            (template_left, template_right),
            (
                np.clip(
                    0.65 * template_left + 0.35 * clipped_left_seed,
                    left_lower,
                    left_upper,
                ),
                np.clip(
                    0.65 * template_right + 0.35 * clipped_right_seed,
                    right_lower,
                    right_upper,
                ),
            ),
        ]

        best_state: Optional[np.ndarray] = None
        best_left_metrics: Optional[Dict[str, float]] = None
        best_right_metrics: Optional[Dict[str, float]] = None
        best_score = float("inf")
        finite_delta = 1e-5

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
            state = np.concatenate(
                [
                    seed_l,
                    seed_r,
                    np.array(
                        [
                            float(
                                np.clip(
                                    lift_guess,
                                    lift_lower,
                                    lift_upper,
                                )
                            )
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
            state = np.clip(state, lower, upper)

            for iteration in range(self.xyz_solver_max_iterations):
                residual = self._joint_xyz_residual(
                    state,
                    left_target,
                    right_target,
                )
                left_metrics, right_metrics = self._state_metrics(
                    state,
                    left_target,
                    right_target,
                )
                score = self._candidate_score(
                    left_metrics,
                    right_metrics,
                )
                if score < best_score:
                    best_score = score
                    best_state = state.copy()
                    best_left_metrics = dict(left_metrics)
                    best_right_metrics = dict(right_metrics)

                if self._metrics_pass(left_metrics, right_metrics):
                    self.get_logger().info(
                        "共享 lift XYZ IK 通过: "
                        f"seed={seed_index}, iter={iteration}, "
                        f"lift={state[14]:+.4f}m, waist=0rad, "
                        f"Lxyz_err=({left_metrics['xy_error_m']*100:.2f},"
                        f"{left_metrics['z_error_m']*100:.2f})cm, "
                        f"Rxyz_err=({right_metrics['xy_error_m']*100:.2f},"
                        f"{right_metrics['z_error_m']*100:.2f})cm"
                    )
                    return state, left_metrics, right_metrics

                jacobian = np.zeros(
                    (len(residual), len(state)),
                    dtype=np.float64,
                )
                for variable_index in range(len(state)):
                    trial = state.copy()
                    trial[variable_index] = min(
                        float(upper[variable_index]),
                        float(trial[variable_index] + finite_delta),
                    )
                    if trial[variable_index] == state[variable_index]:
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
                        self._joint_xyz_residual(
                            trial,
                            left_target,
                            right_target,
                        )
                        - residual
                    ) / actual_delta

                normal = (
                    jacobian.T @ jacobian
                    + (self.xyz_solver_damping ** 2)
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

                next_state = np.clip(
                    state + delta_state,
                    lower,
                    upper,
                )
                if float(np.linalg.norm(next_state - state)) < 1e-8:
                    break
                state = next_state

        if (
            best_state is None
            or best_left_metrics is None
            or best_right_metrics is None
        ):
            raise RuntimeError("共享 lift XYZ 优化未产生任何候选解")

        def summary(metrics: Mapping[str, float]) -> str:
            return (
                f"XY={metrics['xy_error_m']*100:.2f}cm, "
                f"Z={metrics['z_error_m']*100:.2f}cm, "
                f"vertical={math.degrees(metrics['gripper_vertical_error_rad']):.2f}deg, "
                f"forearm={math.degrees(metrics['forearm_tilt_rad']):.2f}deg, "
                f"template={math.degrees(metrics['template_rmse_rad']):.2f}deg"
            )

        raise RuntimeError(
            "共享 lift XYZ IK 未通过安全阈值。最接近结果: "
            f"lift={best_state[14]:+.4f}m; "
            f"L[{summary(best_left_metrics)}]; "
            f"R[{summary(best_right_metrics)}]"
        )

    def _solve_asymmetric(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
        left_seed: np.ndarray,
        right_seed: np.ndarray,
        current_lift: float,
        current_waist: float,
        limits: Dict[str, Tuple[float, float, float]],
    ) -> dict:
        del current_waist  # 本修正版固定 waist=0，避免夹爪竖直姿态被破坏。

        left_target = np.asarray(left_base, dtype=np.float64)
        right_target = np.asarray(right_base, dtype=np.float64)
        state, left_metrics, right_metrics = (
            self._optimize_shared_lift_xyz(
                left_target,
                right_target,
                np.asarray(left_seed, dtype=np.float64),
                np.asarray(right_seed, dtype=np.float64),
                float(current_lift),
                limits,
            )
        )

        shared_lift = float(state[14])
        left_required_lift = (
            shared_lift
            + float(left_target[2])
            - float(left_metrics["z_m"])
        )
        right_required_lift = (
            shared_lift
            + float(right_target[2])
            - float(right_metrics["z_m"])
        )
        lift_mismatch = abs(
            left_required_lift - right_required_lift
        )

        plan = {
            "left": state[:7].copy(),
            "right": state[7:14].copy(),
            "body": {
                "base_elevator_joint": shared_lift,
                "elevator_pitch_joint": 0.0,
            },
            "xy_stage": {
                "reference_lift": float(current_lift),
                "left_gripper_z": float(left_metrics["z_m"]),
                "right_gripper_z": float(right_metrics["z_m"]),
                "left_required_lift": left_required_lift,
                "right_required_lift": right_required_lift,
                "shared_lift_mismatch": lift_mismatch,
            },
            "metrics": {
                "left_xy": dict(left_metrics),
                "right_xy": dict(right_metrics),
                "left_final": dict(left_metrics),
                "right_final": dict(right_metrics),
            },
        }

        validate_solution(plan, limits)
        verify_grasp_solution(
            plan,
            tuple(float(v) for v in left_target),
            tuple(float(v) for v in right_target),
            self.left_template_q,
            self.right_template_q,
            self.max_xy_error_m,
            self.max_z_error_m,
            self.max_shared_lift_mismatch_m,
            self.max_orientation_error_rad,
            self.max_forearm_tilt_rad,
            self.max_gripper_vertical_error_rad,
            self.max_template_rmse_rad,
            self.max_lateral_error_m,
        )
        return plan


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[SharedLiftXYZBoxGraspBridge] = None
    try:
        node = SharedLiftXYZBoxGraspBridge()
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
