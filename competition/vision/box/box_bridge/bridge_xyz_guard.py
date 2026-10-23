#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
共享 lift XYZ 非对称抓箱：夹爪状态保护 + 左右独立压紧补偿。

与以下文件放在同一目录：
  box_grasp_bridge_asymmetric_staged_sharedlift_xyzfix.py
  box_grasp_bridge_asymmetric_staged_avoidance.py
  box_grasp_bridge_mirror_vision_raw_prelift025.py
  box_dual_arm_grasp_v9_template10_shoulder007_prelift025.py

修复：
1. 原控制器只看 GripperCommand Action 是否 SUCCEEDED，没有检查
   reached_goal/stalled/实际位置，因此“左夹爪停在 0.025m 最大开度”也会被
   误报为抓取完成。本版本会拒绝这种结果。
2. 增加 left_grasp_inset_m / right_grasp_inset_m，把对应 TCP 沿箱体两侧
   连线向内移动少量距离。默认均为 0，必须先确认夹爪硬件闭合正常，再从
   0.003m 开始小步标定。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand

import box_grasp_bridge_mirror_vision_raw_prelift025 as legacy_bridge


class VerifiedGripperGuardedController(
    legacy_bridge.GuardedDualArmGraspController
):
    """保留原运动锁，额外验证左右夹爪的实际到位/接触位置。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.left_gripper_effort_scale = 1.0
        self.right_gripper_effort_scale = 1.0
        self.gripper_position_tolerance_m = 0.002
        self.close_contact_max_position_m = 0.015
        self.last_gripper_results = {}

    def command_grippers(self, position: float) -> bool:
        self._require_armed()
        position = float(position)
        if not 0.0 <= position <= 0.025:
            raise RuntimeError(
                f"夹爪位置越界: {position}; 允许范围 [0.0, 0.025]"
            )

        efforts = {
            "左夹爪": self._gripper_effort
            * float(self.left_gripper_effort_scale),
            "右夹爪": self._gripper_effort
            * float(self.right_gripper_effort_scale),
        }
        clients = {
            "左夹爪": self._left_gripper,
            "右夹爪": self._right_gripper,
        }

        send_futures = {}
        for label, client in clients.items():
            goal = GripperCommand.Goal()
            goal.command.position = position
            goal.command.max_effort = float(efforts[label])
            send_futures[label] = client.send_goal_async(goal)

        if not self._wait_all(send_futures.values(), timeout=10.0):
            self.get_logger().error("夹爪目标发送超时")
            return False

        handles = {}
        for label, future in send_futures.items():
            handle = future.result()
            if handle is None or not handle.accepted:
                self.get_logger().error(f"{label}: 目标被拒绝")
            else:
                handles[label] = handle
        if len(handles) != 2:
            return False

        result_futures = {
            label: handle.get_result_async()
            for label, handle in handles.items()
        }
        if not self._wait_all(result_futures.values(), timeout=12.0):
            self.get_logger().error("夹爪执行超时")
            return False

        all_ok = True
        self.last_gripper_results = {}

        for label, future in result_futures.items():
            wrapped = future.result()
            if (
                wrapped is None
                or wrapped.status != GoalStatus.STATUS_SUCCEEDED
            ):
                self.get_logger().error(f"{label}: Action 执行失败")
                all_ok = False
                continue

            result = wrapped.result
            actual = float(result.position)
            reached = bool(result.reached_goal)
            stalled = bool(result.stalled)
            self.last_gripper_results[label] = {
                "position": actual,
                "reached_goal": reached,
                "stalled": stalled,
            }

            # 张开命令：必须接近最大开度；闭合命令允许在物体接触位置 stall，
            # 但不能仍停在接近最大开度的位置。
            if position >= 0.024:
                position_ok = (
                    actual
                    >= position - self.gripper_position_tolerance_m
                )
                reason = "张开位置未到达"
            elif position <= 0.001:
                reached_closed = (
                    reached
                    and actual
                    <= position + self.gripper_position_tolerance_m
                )
                valid_contact = (
                    stalled
                    and actual <= self.close_contact_max_position_m
                )
                position_ok = reached_closed or valid_contact
                reason = (
                    "闭合失败：既未到达闭合目标，也未在合理接触开度内停止"
                )
            else:
                position_ok = (
                    reached
                    or abs(actual - position)
                    <= self.gripper_position_tolerance_m
                )
                reason = "中间位置未到达"

            if position_ok:
                self.get_logger().info(
                    f"{label}: verified position={actual:.5f}, "
                    f"reached_goal={reached}, stalled={stalled}, "
                    f"effort={efforts[label]:.1f}"
                )
            else:
                self.get_logger().error(
                    f"{label}: {reason}; requested={position:.5f}, "
                    f"actual={actual:.5f}, reached_goal={reached}, "
                    f"stalled={stalled}, effort={efforts[label]:.1f}"
                )
                all_ok = False

        return all_ok


# 必须在导入上层 bridge 前替换原模块中的控制器类。
legacy_bridge.GuardedDualArmGraspController = (
    VerifiedGripperGuardedController
)

from box_grasp_bridge_asymmetric_staged_sharedlift_xyzfix import (  # noqa: E402
    SharedLiftXYZBoxGraspBridge,
)


class GripperBalancedSharedLiftBridge(SharedLiftXYZBoxGraspBridge):
    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("left_grasp_inset_m", 0.0)
        self.declare_parameter("right_grasp_inset_m", 0.0)
        self.declare_parameter("left_gripper_effort_scale", 1.0)
        self.declare_parameter("right_gripper_effort_scale", 1.0)
        self.declare_parameter("close_contact_max_position_m", 0.015)
        self.declare_parameter("gripper_position_tolerance_m", 0.002)

        self.left_grasp_inset_m = float(
            self.get_parameter("left_grasp_inset_m").value
        )
        self.right_grasp_inset_m = float(
            self.get_parameter("right_grasp_inset_m").value
        )

        for name, value in (
            ("left_grasp_inset_m", self.left_grasp_inset_m),
            ("right_grasp_inset_m", self.right_grasp_inset_m),
        ):
            if not 0.0 <= value <= 0.020:
                raise RuntimeError(f"{name} 必须在 [0, 0.020]m 内")

        controller = self.grasp_controller
        controller.left_gripper_effort_scale = float(
            self.get_parameter("left_gripper_effort_scale").value
        )
        controller.right_gripper_effort_scale = float(
            self.get_parameter("right_gripper_effort_scale").value
        )
        controller.close_contact_max_position_m = float(
            self.get_parameter("close_contact_max_position_m").value
        )
        controller.gripper_position_tolerance_m = float(
            self.get_parameter("gripper_position_tolerance_m").value
        )

        if not 0.1 <= controller.left_gripper_effort_scale <= 2.0:
            raise RuntimeError("left_gripper_effort_scale 必须在 [0.1, 2.0] 内")
        if not 0.1 <= controller.right_gripper_effort_scale <= 2.0:
            raise RuntimeError("right_gripper_effort_scale 必须在 [0.1, 2.0] 内")
        if not 0.002 <= controller.close_contact_max_position_m <= 0.020:
            raise RuntimeError(
                "close_contact_max_position_m 必须在 [0.002, 0.020]m 内"
            )
        if not 0.0002 <= controller.gripper_position_tolerance_m <= 0.005:
            raise RuntimeError(
                "gripper_position_tolerance_m 必须在 [0.0002, 0.005]m 内"
            )

        self.get_logger().warning(
            "夹爪验证/独立压紧补偿已启用: "
            f"left_inset={self.left_grasp_inset_m*1000:.1f}mm, "
            f"right_inset={self.right_grasp_inset_m*1000:.1f}mm"
        )

    def _plan_sequence(
        self,
        left_base: np.ndarray,
        right_base: np.ndarray,
    ) -> dict:
        left_target = np.asarray(left_base, dtype=np.float64).copy()
        right_target = np.asarray(right_base, dtype=np.float64).copy()
        side_delta = left_target[:2] - right_target[:2]
        side_distance = float(np.linalg.norm(side_delta))
        if side_distance < 1e-9:
            raise RuntimeError("左右抓取点重合，无法计算压紧方向")
        side_axis = side_delta / side_distance

        # side_axis 从右点指向左点；左点向内是 -axis，右点向内是 +axis。
        left_target[:2] -= side_axis * self.left_grasp_inset_m
        right_target[:2] += side_axis * self.right_grasp_inset_m

        if self.left_grasp_inset_m > 0.0 or self.right_grasp_inset_m > 0.0:
            self.get_logger().info(
                "应用独立向内压紧补偿: "
                f"axis={np.round(side_axis, 5)}, "
                f"L {np.round(left_base, 5)} -> "
                f"{np.round(left_target, 5)}, "
                f"R {np.round(right_base, 5)} -> "
                f"{np.round(right_target, 5)}"
            )

        return super()._plan_sequence(left_target, right_target)


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[GripperBalancedSharedLiftBridge] = None
    try:
        node = GripperBalancedSharedLiftBridge()
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
