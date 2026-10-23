#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
非对称箱体双臂抓取 bridge（高位外扩 -> 下降 -> 分段向内 -> 夹紧）。

请把本文件放到以下两个原脚本的同一目录再运行：
  box_grasp_bridge_mirror_vision_raw_prelift025.py
  box_dual_arm_grasp_v9_template10_shoulder007_prelift025.py

本文件复用原脚本的 TF/标定、FK/IK、关节限位和 ros2_control Action，
但不再把视觉点投影成关于 base_link Y=0 的对称点，也不再强制
q_left == -q_right。左右臂仍分别使用互为镜像的标准模板和相反方向的
肩部避障量；一自由度夹爪仍向左右控制器发送相同开度，由镜像安装关系
产生相向开合。

执行顺序：
  0. 冻结稳定的原始左右视觉抓取点；
  1. 对最终点和整条“外侧 -> 最终点”路径做 IK 与限位检查；
  2. lift 先升到 prelift_target_m；
  3. 张开夹爪；
  4. 在高位把双臂放到抓取点外侧 outward_offset_m；
  5. 保持外扩状态下降到接近高度；
  6. 沿两个视觉点连线方向，用多个 IK 路点逐步向箱体靠拢；
  7. 两侧夹爪同时闭合。

重要：本脚本没有完整三维碰撞模型，不能数学保证“绝不碰撞”。它通过
高位横移、外扩下降、分段内收、工作范围检查和关节跳变检查降低风险。
首次必须低速、空载、急停可触达，并先用 enable_motion:=false 检查日志。
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy

try:
    from box_grasp_bridge_mirror_vision_raw_prelift025 import (
        BoxPoseGraspBridge,
        stamp_to_seconds,
    )
    from box_dual_arm_grasp_v9_template10_shoulder007_prelift025 import (
        LEFT_ARM_JOINTS,
        RIGHT_ARM_JOINTS,
        dual_ik_xy_then_lift,
        validate_solution,
        verify_grasp_solution,
    )
except ImportError as exc:
    raise RuntimeError(
        "请把本脚本放到两个原抓取脚本的同一目录后再运行"
    ) from exc


class AsymmetricStagedBoxGraspBridge(BoxPoseGraspBridge):
    """保留原通信层，替换镜像目标、规划与执行逻辑。"""

    def __init__(self) -> None:
        super().__init__()

        # 外扩距离：沿“右抓取点 -> 左抓取点”的箱体横向轴分别向外移动。
        self.declare_parameter("outward_offset_m", 0.060)
        self.declare_parameter("approach_waypoints", 5)
        self.declare_parameter("prelift_target_m", 0.250)
        self.declare_parameter("min_preapproach_clearance_m", 0.060)

        # 箱体/视觉几何保护。超过范围时拒绝锁定，不尝试强行抓取。
        self.declare_parameter("max_box_center_y_m", 0.250)
        self.declare_parameter("require_targets_straddle_centerline", True)
        self.declare_parameter("min_centerline_side_margin_m", 0.020)
        self.declare_parameter("max_target_x_difference_m", 0.250)
        self.declare_parameter("max_target_z_difference_m", 0.030)
        self.declare_parameter("max_side_axis_rotation_deg", 35.0)

        # 路点连续性保护。
        self.declare_parameter("max_waypoint_joint_delta_deg", 20.0)
        self.declare_parameter("max_approach_lift_variation_m", 0.015)

        # 轨迹时长。实际最短时长仍受原控制器速度限位约束。
        self.declare_parameter("prelift_duration_s", 2.0)
        self.declare_parameter("outer_move_duration_s", 3.0)
        self.declare_parameter("descend_duration_s", 2.0)
        self.declare_parameter("inward_step_duration_s", 0.8)

        # 实机默认强制人工确认，即使旧参数 require_execute_confirmation=false。
        self.declare_parameter("force_execute_confirmation", True)

        self.outward_offset_m = float(
            self.get_parameter("outward_offset_m").value
        )
        self.approach_waypoints = int(
            self.get_parameter("approach_waypoints").value
        )
        self.prelift_target_m = float(
            self.get_parameter("prelift_target_m").value
        )
        self.min_preapproach_clearance_m = float(
            self.get_parameter("min_preapproach_clearance_m").value
        )
        self.max_box_center_y_m = float(
            self.get_parameter("max_box_center_y_m").value
        )
        self.require_targets_straddle_centerline = bool(
            self.get_parameter("require_targets_straddle_centerline").value
        )
        self.min_centerline_side_margin_m = float(
            self.get_parameter("min_centerline_side_margin_m").value
        )
        self.max_target_x_difference_m = float(
            self.get_parameter("max_target_x_difference_m").value
        )
        self.max_target_z_difference_m = float(
            self.get_parameter("max_target_z_difference_m").value
        )
        self.max_side_axis_rotation_rad = math.radians(
            float(self.get_parameter("max_side_axis_rotation_deg").value)
        )
        self.max_waypoint_joint_delta_rad = math.radians(
            float(self.get_parameter("max_waypoint_joint_delta_deg").value)
        )
        self.max_approach_lift_variation_m = float(
            self.get_parameter("max_approach_lift_variation_m").value
        )
        self.prelift_duration_s = float(
            self.get_parameter("prelift_duration_s").value
        )
        self.outer_move_duration_s = float(
            self.get_parameter("outer_move_duration_s").value
        )
        self.descend_duration_s = float(
            self.get_parameter("descend_duration_s").value
        )
        self.inward_step_duration_s = float(
            self.get_parameter("inward_step_duration_s").value
        )
        self.force_execute_confirmation = bool(
            self.get_parameter("force_execute_confirmation").value
        )

        self._validate_new_parameters()

        self.get_logger().warning(
            "旧 bridge 的镜像提示已被本脚本覆盖：本次保留原始非对称视觉点，"
            "不再强制 XL=XR、YL=-YR、q_left=-q_right。"
        )
        self.get_logger().info(
            "非对称分段路径参数: "
            f"outward={self.outward_offset_m:.3f}m, "
            f"waypoints={self.approach_waypoints}, "
            f"prelift={self.prelift_target_m:.3f}m"
        )

    def _validate_new_parameters(self) -> None:
        if not 0.010 <= self.outward_offset_m <= 0.150:
            raise RuntimeError("outward_offset_m 必须在 [0.010, 0.150]m 内")
        if not 3 <= self.approach_waypoints <= 12:
            raise RuntimeError("approach_waypoints 必须在 [3, 12] 内")
        if self.min_preapproach_clearance_m < 0.020:
            raise RuntimeError("min_preapproach_clearance_m 不得小于 0.020m")
        if self.max_box_center_y_m <= 0.0:
            raise RuntimeError("max_box_center_y_m 必须 > 0")
        if self.min_centerline_side_margin_m < 0.0:
            raise RuntimeError("min_centerline_side_margin_m 必须 >= 0")
        if self.max_target_x_difference_m <= 0.0:
            raise RuntimeError("max_target_x_difference_m 必须 > 0")
        if self.max_target_z_difference_m <= 0.0:
            raise RuntimeError("max_target_z_difference_m 必须 > 0")
        if not 0.0 < self.max_side_axis_rotation_rad < math.pi / 2:
            raise RuntimeError("max_side_axis_rotation_deg 必须在 (0, 90) 内")
        if self.max_waypoint_joint_delta_rad <= 0.0:
            raise RuntimeError("max_waypoint_joint_delta_deg 必须 > 0")
        if self.max_approach_lift_variation_m < 0.0:
            raise RuntimeError("max_approach_lift_variation_m 必须 >= 0")
        for name, value in (
            ("prelift_duration_s", self.prelift_duration_s),
            ("outer_move_duration_s", self.outer_move_duration_s),
            ("descend_duration_s", self.descend_duration_s),
            ("inward_step_duration_s", self.inward_step_duration_s),
        ):
            if value <= 0.0:
                raise RuntimeError(f"{name} 必须 > 0")

        lower, upper, _ = self.grasp_controller._limits[
            "base_elevator_joint"
        ]
        if not lower <= self.prelift_target_m <= upper:
            raise RuntimeError(
                "prelift_target_m 越过 lift 限位: "
                f"target={self.prelift_target_m:+.4f}, "
                f"limit=[{lower:+.4f}, {upper:+.4f}]"
            )

    # ------------------------------------------------------------------
    # 视觉点：保留实际左右点，不投影到机器人中心线镜像位置。
    # ------------------------------------------------------------------
    def _try_process_pair(self) -> None:
        if (
            self.latest_p1 is None
            or self.latest_p2 is None
            or self.target_locked
        ):
            return

        t1 = stamp_to_seconds(self.latest_p1.header.stamp)
        t2 = stamp_to_seconds(self.latest_p2.header.stamp)
        if abs(t1 - t2) > self.max_pair_dt_s:
            return

        key = (int(t1 * 1e9), int(t2 * 1e9))
        if key == self.last_pair_key:
            return
        self.last_pair_key = key

        try:
            p1_base = self._point_to_base(self.latest_p1)
            p2_base = self._point_to_base(self.latest_p2)
        except Exception as exc:
            self.get_logger().error(
                "3D 抓取点坐标转换失败: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        # base_link +Y 为机器人左侧。
        if float(p1_base[1]) >= float(p2_base[1]):
            left_base = np.asarray(p1_base, dtype=np.float64)
            right_base = np.asarray(p2_base, dtype=np.float64)
            left_stamp = self.latest_p1.header.stamp
            right_stamp = self.latest_p2.header.stamp
        else:
            left_base = np.asarray(p2_base, dtype=np.float64)
            right_base = np.asarray(p1_base, dtype=np.float64)
            left_stamp = self.latest_p2.header.stamp
            right_stamp = self.latest_p1.header.stamp

        reason = self._target_geometry_problem(left_base, right_base)
        if reason is not None:
            self.get_logger().warning(f"视觉抓取点被安全检查拒绝: {reason}")
            self.pair_samples.clear()
            return

        self._publish_base_point(self.left_base_pub, left_base, left_stamp)
        self._publish_base_point(self.right_base_pub, right_base, right_stamp)

        sample = np.concatenate([left_base, right_base])
        self.pair_samples.append(sample)
        if len(self.pair_samples) < self.stability_frames:
            self.get_logger().info(
                "等待非对称双抓取点稳定: "
                f"{len(self.pair_samples)}/{self.stability_frames}"
            )
            return

        samples = np.asarray(self.pair_samples, dtype=np.float64)
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        left_mean, right_mean = mean[:3], mean[3:]
        left_std, right_std = std[:3], std[3:]

        stable = (
            np.all(left_std[:2] <= self.stability_xy_m)
            and left_std[2] <= self.stability_z_m
            and np.all(right_std[:2] <= self.stability_xy_m)
            and right_std[2] <= self.stability_z_m
        )
        if not stable:
            self.get_logger().warning(
                "非对称双抓取点仍不稳定: "
                f"Lstd(mm)={np.round(left_std * 1000.0, 2)}, "
                f"Rstd(mm)={np.round(right_std * 1000.0, 2)}"
            )
            return

        self._lock_plan_and_maybe_execute(
            left_mean,
            right_mean,
            left_std,
            right_std,
        )

    def _target_geometry_problem(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> Optional[str]:
        if not np.all(np.isfinite(left_base)) or not np.all(
            np.isfinite(right_base)
        ):
            return "点包含 NaN/Inf"

        delta = left_base - right_base
        separation_xy = float(np.linalg.norm(delta[:2]))
        if separation_xy < self.min_pair_separation_y_m:
            return f"两侧点 XY 间距过小: {separation_xy:.4f}m"

        center_y = 0.5 * float(left_base[1] + right_base[1])
        if abs(center_y) > self.max_box_center_y_m:
            return (
                f"箱体中心 Y={center_y:+.4f}m 超过 "
                f"{self.max_box_center_y_m:.4f}m"
            )

        if self.require_targets_straddle_centerline:
            if float(left_base[1]) < self.min_centerline_side_margin_m:
                return (
                    f"左抓取点 Y={left_base[1]:+.4f}m 未留在机器人左侧；"
                    f"要求 >= {self.min_centerline_side_margin_m:.4f}m"
                )
            if float(right_base[1]) > -self.min_centerline_side_margin_m:
                return (
                    f"右抓取点 Y={right_base[1]:+.4f}m 未留在机器人右侧；"
                    f"要求 <= {-self.min_centerline_side_margin_m:.4f}m"
                )

        dx = abs(float(delta[0]))
        if dx > self.max_target_x_difference_m:
            return (
                f"左右点前后差 |dX|={dx:.4f}m 超过 "
                f"{self.max_target_x_difference_m:.4f}m"
            )

        dz = abs(float(delta[2]))
        if dz > self.max_target_z_difference_m:
            return (
                f"左右点高度差 |dZ|={dz:.4f}m 超过 "
                f"{self.max_target_z_difference_m:.4f}m"
            )

        axis = delta[:2] / separation_xy
        rotation = math.atan2(abs(float(axis[0])), abs(float(axis[1])))
        if rotation > self.max_side_axis_rotation_rad:
            return (
                "箱体横向轴相对机器人左右方向旋转过大: "
                f"{math.degrees(rotation):.1f}deg > "
                f"{math.degrees(self.max_side_axis_rotation_rad):.1f}deg"
            )
        return None

    # ------------------------------------------------------------------
    # 非对称 IK 与外扩接近路点。
    # ------------------------------------------------------------------
    def _current_seed(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        controller = self.grasp_controller
        left_q = np.array(
            [
                controller.get_pos("pitch_arm_left1_joint"),
                *[controller.get_pos(name) for name in LEFT_ARM_JOINTS],
            ],
            dtype=np.float64,
        )
        right_q = np.array(
            [
                controller.get_pos("pitch_arm_right1_joint"),
                *[controller.get_pos(name) for name in RIGHT_ARM_JOINTS],
            ],
            dtype=np.float64,
        )
        return (
            left_q,
            right_q,
            controller.get_pos("base_elevator_joint"),
            controller.get_pos("elevator_pitch_joint"),
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
        plan = dual_ik_xy_then_lift(
            tuple(float(v) for v in left_base),
            tuple(float(v) for v in right_base),
            limits,
            self.left_template_q,
            self.right_template_q,
            left_seed,
            right_seed,
            current_lift,
            current_waist,
            self.max_xy_error_m,
            self.max_z_error_m,
            self.max_shared_lift_mismatch_m,
            self.max_orientation_error_rad,
            self.max_forearm_tilt_rad,
            self.max_gripper_vertical_error_rad,
            self.max_template_rmse_rad,
            self.lateral_weight,
            self.max_lateral_error_m,
            self.max_lateral_joint_correction_rad,
        )
        if plan is None:
            raise RuntimeError(
                "非对称 XY 模板 IK / 共享 lift 求解无解: "
                f"left={np.round(left_base, 5)}, "
                f"right={np.round(right_base, 5)}"
            )

        validate_solution(plan, limits)
        verify_grasp_solution(
            plan,
            tuple(float(v) for v in left_base),
            tuple(float(v) for v in right_base),
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

    def _plan_sequence(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> dict:
        reason = self._target_geometry_problem(left_base, right_base)
        if reason is not None:
            raise RuntimeError(reason)

        self._refresh_joint_states()
        left_seed, right_seed, current_lift, current_waist = (
            self._current_seed()
        )
        real_limits = dict(self.grasp_controller._limits)

        # 先在最终目标上选择一个可行的共享 waist，再把整条接近路径固定在
        # 该 waist，避免内收过程中本体俯仰跳变。
        final_preplan = self._solve_asymmetric(
            left_base,
            right_base,
            left_seed,
            right_seed,
            current_lift,
            current_waist,
            real_limits,
        )
        fixed_waist = float(final_preplan["body"]["elevator_pitch_joint"])
        waist_velocity = float(real_limits["elevator_pitch_joint"][2])
        path_limits = dict(real_limits)
        path_limits["elevator_pitch_joint"] = (
            fixed_waist,
            fixed_waist,
            waist_velocity,
        )

        delta_xy = np.asarray(left_base[:2] - right_base[:2], dtype=float)
        side_axis_xy = delta_xy / float(np.linalg.norm(delta_xy))
        factors = np.linspace(1.0, 0.0, self.approach_waypoints)

        plans: List[dict] = []
        targets: List[Tuple[np.ndarray, np.ndarray]] = []
        seed_l = left_seed
        seed_r = right_seed
        previous_plan: Optional[dict] = None

        for index, factor in enumerate(factors):
            left_target = np.asarray(left_base, dtype=float).copy()
            right_target = np.asarray(right_base, dtype=float).copy()
            offset_xy = side_axis_xy * self.outward_offset_m * float(factor)
            left_target[:2] += offset_xy
            right_target[:2] -= offset_xy

            plan = self._solve_asymmetric(
                left_target,
                right_target,
                seed_l,
                seed_r,
                current_lift,
                fixed_waist,
                path_limits,
            )

            if previous_plan is not None:
                joint_delta = max(
                    float(
                        np.max(
                            np.abs(plan["left"] - previous_plan["left"])
                        )
                    ),
                    float(
                        np.max(
                            np.abs(plan["right"] - previous_plan["right"])
                        )
                    ),
                )
                if joint_delta > self.max_waypoint_joint_delta_rad:
                    raise RuntimeError(
                        f"接近路点 {index} 关节跳变 "
                        f"{math.degrees(joint_delta):.1f}deg 超过 "
                        f"{math.degrees(self.max_waypoint_joint_delta_rad):.1f}deg"
                    )

            plans.append(plan)
            targets.append((left_target, right_target))
            seed_l = np.asarray(plan["left"], dtype=float)
            seed_r = np.asarray(plan["right"], dtype=float)
            previous_plan = plan

        lift_values = np.array(
            [float(plan["body"]["base_elevator_joint"]) for plan in plans]
        )
        lift_variation = float(np.ptp(lift_values))
        if lift_variation > self.max_approach_lift_variation_m:
            raise RuntimeError(
                "内收路点所需 lift 变化过大: "
                f"{lift_variation:.4f}m > "
                f"{self.max_approach_lift_variation_m:.4f}m"
            )

        highest_approach_lift = float(np.max(lift_values))
        clearance = self.prelift_target_m - highest_approach_lift
        if clearance < self.min_preapproach_clearance_m:
            raise RuntimeError(
                "预抬升高度相对接近路径不够: "
                f"clearance={clearance:.4f}m < "
                f"{self.min_preapproach_clearance_m:.4f}m"
            )

        return {
            "plans": plans,
            "targets": targets,
            "factors": factors,
            "side_axis_xy": side_axis_xy,
            "fixed_waist": fixed_waist,
            "lift_variation": lift_variation,
        }

    def _log_sequence(
        self,
        title: str,
        sequence: dict,
        left_std: np.ndarray,
        right_std: np.ndarray,
    ) -> None:
        final_plan = sequence["plans"][-1]
        final_left, final_right = sequence["targets"][-1]
        self._log_plan(
            title,
            final_plan,
            final_left,
            final_right,
            left_std,
            right_std,
            self._base_to_isaac(final_left),
            self._base_to_isaac(final_right),
        )
        lines = []
        for index, (factor, plan, targets) in enumerate(
            zip(
                sequence["factors"],
                sequence["plans"],
                sequence["targets"],
            )
        ):
            left_target, right_target = targets
            lines.append(
                f"  [{index}] factor={float(factor):.2f}, "
                f"L={np.round(left_target, 4)}, "
                f"R={np.round(right_target, 4)}, "
                f"lift={plan['body']['base_elevator_joint']:+.4f}m"
            )
        self.get_logger().info(
            "外扩到内收的已验证路点:\n"
            + "\n".join(lines)
            + "\nside_axis_xy="
            + str(np.round(sequence["side_axis_xy"], 5))
            + f", fixed_waist={math.degrees(sequence['fixed_waist']):+.2f}deg"
            + f", lift_variation={sequence['lift_variation']:.4f}m"
        )

    # ------------------------------------------------------------------
    # 锁定、二次规划与真实执行。
    # ------------------------------------------------------------------
    def _lock_plan_and_maybe_execute(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
        left_std: np.ndarray,
        right_std: np.ndarray,
    ) -> None:
        if self.target_locked or self.execution_started:
            return

        try:
            sequence = self._plan_sequence(left_base, right_base)
        except Exception as exc:
            self.get_logger().error(
                "第一次非对称分段规划失败，不锁定/不运动: "
                f"{type(exc).__name__}: {exc}"
            )
            self.pair_samples.clear()
            return

        self.target_locked = True
        self.locked_left_base = np.asarray(left_base, dtype=float).copy()
        self.locked_right_base = np.asarray(right_base, dtype=float).copy()
        self._log_sequence(
            "非对称双视觉点已锁定 / 第一次整条路径规划通过",
            sequence,
            left_std,
            right_std,
        )

        if not self.enable_motion:
            self.get_logger().warning(
                "enable_motion=false：只完成视觉、非对称 IK 和整条接近路径检查，"
                "机器人没有运动。"
            )
            return

        must_confirm = (
            self.force_execute_confirmation or self.require_execute_confirmation
        )
        if must_confirm:
            expected = self.execute_confirmation
            self.get_logger().warning(
                "即将执行真实双臂非对称抓箱。请确认抓取点、外扩方向、"
                "所有路点、现场障碍物和急停。\n"
                f"确认后精确输入: {expected}"
            )
            try:
                answer = input(f"\nType {expected} to execute > ").strip()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer != expected:
                self.get_logger().warning("未确认，取消执行；机器人没有运动。")
                return

        try:
            final_sequence = self._plan_sequence(
                self.locked_left_base,
                self.locked_right_base,
            )
        except Exception as exc:
            self.get_logger().error(
                "第二次最终整条路径规划失败，取消运动: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        self._log_sequence(
            "第二次整条路径规划通过，即将执行",
            final_sequence,
            left_std,
            right_std,
        )

        self.execution_started = True
        self.grasp_controller.arm_motion()
        try:
            self.get_logger().warning(
                "REAL ASYMMETRIC DUAL-ARM STAGED BOX GRASP START"
            )
            self._execute_sequence(final_sequence)
            self.execution_finished = True
            self.get_logger().info("非对称双臂抓箱流程完成，左右夹爪已闭合。")
        except Exception as exc:
            self.get_logger().error(
                "实机非对称抓箱失败: "
                f"{type(exc).__name__}: {exc}"
            )
            try:
                self.grasp_controller.cancel_active_goals()
            except Exception:
                pass
        finally:
            self.grasp_controller.disarm_motion()
            self.get_logger().warning("双臂运动接口已重新 DISARMED")

    @staticmethod
    def _full_joint_targets(plan: dict, *, include_lift: bool) -> dict:
        targets = {
            "elevator_pitch_joint": float(
                plan["body"]["elevator_pitch_joint"]
            ),
            "pitch_arm_left1_joint": float(plan["left"][0]),
            "pitch_arm_right1_joint": float(plan["right"][0]),
        }
        for name, value in zip(LEFT_ARM_JOINTS, plan["left"][1:]):
            targets[name] = float(value)
        for name, value in zip(RIGHT_ARM_JOINTS, plan["right"][1:]):
            targets[name] = float(value)
        if include_lift:
            targets["base_elevator_joint"] = float(
                plan["body"]["base_elevator_joint"]
            )
        return targets

    def _execute_sequence(self, sequence: dict) -> None:
        controller = self.grasp_controller
        log = self.get_logger().info
        outer_plan = sequence["plans"][0]

        log(
            "[1/8] 所有机械臂动作前先抬升 lift: "
            f"{self.prelift_target_m:+.4f}m"
        )
        if not controller.animate(
            {"base_elevator_joint": self.prelift_target_m},
            duration=self.prelift_duration_s,
            steps=80,
        ):
            raise RuntimeError("预抬升失败")

        log("[2/8] 张开左右一自由度夹爪")
        if not controller.command_grippers(0.025):
            raise RuntimeError("夹爪张开失败")
        time.sleep(0.2)

        left_outer_shoulder = float(outer_plan["left"][1])
        right_outer_shoulder = float(outer_plan["right"][1])
        left_clearance = left_outer_shoulder + abs(self.shoulder_clearance)
        right_clearance = right_outer_shoulder - abs(self.shoulder_clearance)

        for name, value in (
            ("arm_left1_arm_left2_joint", left_clearance),
            ("arm_right1_arm_right2_joint", right_clearance),
        ):
            lower, upper, _ = controller._limits[name]
            if not lower <= value <= upper:
                raise RuntimeError(
                    f"{name} 高位肩部避障角越界: {value:+.4f}, "
                    f"limit=[{lower:+.4f}, {upper:+.4f}]"
                )

        log("[3/8] 高位先按镜像方向抬肩")
        if not controller.animate(
            {
                "arm_left1_arm_left2_joint": left_clearance,
                "arm_right1_arm_right2_joint": right_clearance,
            },
            duration=1.2,
            steps=60,
        ):
            raise RuntimeError("高位肩部避障动作失败")

        log("[4/8] 高位运动到视觉抓取点外侧，肩部保持避障角")
        outer_targets = self._full_joint_targets(
            outer_plan,
            include_lift=False,
        )
        outer_targets.pop("arm_left1_arm_left2_joint")
        outer_targets.pop("arm_right1_arm_right2_joint")
        if not controller.animate(
            outer_targets,
            duration=self.outer_move_duration_s,
            steps=120,
        ):
            raise RuntimeError("高位外扩位姿运动失败")

        log("[5/8] 高位将肩部落到外扩接近位")
        if not controller.animate(
            {
                "arm_left1_arm_left2_joint": left_outer_shoulder,
                "arm_right1_arm_right2_joint": right_outer_shoulder,
            },
            duration=1.2,
            steps=60,
        ):
            raise RuntimeError("高位肩部落到外扩位失败")

        outer_lift = float(outer_plan["body"]["base_elevator_joint"])
        log(f"[6/8] 保持外扩状态下降到接近高度: {outer_lift:+.4f}m")
        if not controller.animate(
            {"base_elevator_joint": outer_lift},
            duration=self.descend_duration_s,
            steps=100,
        ):
            raise RuntimeError("外扩状态下降失败")

        log("[7/8] 按已验证 IK 路点逐段向箱体中间靠拢")
        for index, plan in enumerate(sequence["plans"][1:], start=1):
            self.get_logger().info(
                f"  inward waypoint {index}/"
                f"{len(sequence['plans']) - 1}"
            )
            if not controller.animate(
                self._full_joint_targets(plan, include_lift=True),
                duration=self.inward_step_duration_s,
                steps=60,
            ):
                raise RuntimeError(f"内收路点 {index} 执行失败")

        log("[8/8] 左右夹爪同时闭合夹紧")
        if not controller.command_grippers(0.0):
            raise RuntimeError("夹爪闭合失败")


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[AsymmetricStagedBoxGraspBridge] = None
    try:
        node = AsymmetricStagedBoxGraspBridge()
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
