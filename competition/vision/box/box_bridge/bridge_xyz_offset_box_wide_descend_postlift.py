#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""偏置箱体抓取：宽位下降、低位内收、抓后提升版。

依赖关系（全部放在同一目录）：
  bridge_xyz_offset_box_wide_descend_postlift.py  <- 运行本文件
  bridge_xyz_offset_box_common_height.py
  bridge_xyz_offset_box.py
  以及原抓取脚本的其他依赖文件

动作顺序：
  1. 双臂预抬升；
  2. 夹爪保持张开，左右 TCP 各向箱体外侧扩开；
  3. 保持外扩姿态下降到接近抓取高度；
  4. 在低位沿已验证路点逐步向内收拢到最终抓取点；
  5. 闭合双夹爪并验证到位/接触结果；
  6. 若首次闭合验证失败，保持当前夹持状态，先抬升箱体，再尝试闭合一次；
  7. 首次闭合成功时直接抓后抬升；二次闭合成功后发布抓取完成。

首次必须使用 enable_motion=false 检查整条规划路径。
"""

from __future__ import annotations

import time
from typing import Optional

import rclpy
from std_msgs.msg import Bool, String

from bridge_xyz_offset_box_common_height import (
    CommonRimHeightOffsetBoxBridge,
)


class WideDescendPostLiftBridge(CommonRimHeightOffsetBoxBridge):
    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("descent_outward_offset_m", 0.050)
        self.declare_parameter("minimum_inward_waypoints", 7)
        self.declare_parameter("post_grasp_lift_m", 0.100)
        self.declare_parameter("post_grasp_lift_duration_s", 2.5)
        self.declare_parameter("post_grasp_lift_steps", 100)
        self.declare_parameter("minimum_effective_post_lift_m", 0.030)
        self.declare_parameter(
            "retry_close_after_failed_first_attempt",
            True,
        )
        self.declare_parameter("post_lift_reclose_settle_s", 0.50)

        self.descent_outward_offset_m = float(
            self.get_parameter("descent_outward_offset_m").value
        )
        self.minimum_inward_waypoints = int(
            self.get_parameter("minimum_inward_waypoints").value
        )
        self.post_grasp_lift_m = float(
            self.get_parameter("post_grasp_lift_m").value
        )
        self.post_grasp_lift_duration_s = float(
            self.get_parameter("post_grasp_lift_duration_s").value
        )
        self.post_grasp_lift_steps = int(
            self.get_parameter("post_grasp_lift_steps").value
        )
        self.minimum_effective_post_lift_m = float(
            self.get_parameter("minimum_effective_post_lift_m").value
        )
        self.retry_close_after_failed_first_attempt = bool(
            self.get_parameter(
                "retry_close_after_failed_first_attempt"
            ).value
        )
        self.post_lift_reclose_settle_s = float(
            self.get_parameter("post_lift_reclose_settle_s").value
        )

        if not 0.030 <= self.descent_outward_offset_m <= 0.080:
            raise RuntimeError(
                "descent_outward_offset_m 必须在 [0.030, 0.080]m 内"
            )
        if not 5 <= self.minimum_inward_waypoints <= 12:
            raise RuntimeError(
                "minimum_inward_waypoints 必须在 [5, 12] 内"
            )
        if not 0.030 <= self.post_grasp_lift_m <= 0.180:
            raise RuntimeError(
                "post_grasp_lift_m 必须在 [0.030, 0.180]m 内"
            )
        if not 1.0 <= self.post_grasp_lift_duration_s <= 8.0:
            raise RuntimeError(
                "post_grasp_lift_duration_s 必须在 [1.0, 8.0]s 内"
            )
        if not 40 <= self.post_grasp_lift_steps <= 240:
            raise RuntimeError(
                "post_grasp_lift_steps 必须在 [40, 240] 内"
            )
        if not 0.010 <= self.minimum_effective_post_lift_m <= 0.080:
            raise RuntimeError(
                "minimum_effective_post_lift_m 必须在 [0.010, 0.080]m 内"
            )
        if not 0.0 <= self.post_lift_reclose_settle_s <= 3.0:
            raise RuntimeError(
                "post_lift_reclose_settle_s 必须在 [0.0, 3.0]s 内"
            )

        old_outward = float(self.outward_offset_m)
        old_waypoints = int(self.approach_waypoints)

        # 分段父类使用 outward_offset_m 生成第一个外扩下降点，以及随后由外到内的
        # 全部低位路点。这里只提高最小值，不会减小用户给出的更大安全距离。
        self.outward_offset_m = max(
            old_outward,
            self.descent_outward_offset_m,
        )
        self.approach_waypoints = max(
            old_waypoints,
            self.minimum_inward_waypoints,
        )

        self.post_grasp_lift_done = False
        self.post_grasp_lift_executed = False
        self.grasp_completed_pub = self.create_publisher(
            Bool,
            "/box_grasp/completed",
            10,
        )
        self.grasp_status_pub = self.create_publisher(
            String,
            "/box_grasp/status",
            10,
        )

        # command_grippers(0.0) 是现有执行流程中唯一的最终闭合点。保存原来的
        # 已绑定方法，再包一层：首次验证失败时，不立即让整个流程退出；保持
        # 夹爪当前施力状态，先执行一次抓后 lift，然后再次发送同一闭合目标。
        controller = self.grasp_controller
        original_command_grippers = controller.command_grippers

        def command_grippers_then_postlift(position: float) -> bool:
            first_succeeded = bool(original_command_grippers(position))
            is_final_close = float(position) <= 0.001

            # 张开夹爪等非最终闭合指令完全沿用原控制器结果。
            if not is_final_close:
                return first_succeeded

            if self.post_grasp_lift_done:
                return first_succeeded

            recovery_needed = not first_succeeded
            if (
                recovery_needed
                and not self.retry_close_after_failed_first_attempt
            ):
                return False

            try:
                self._execute_post_grasp_lift(
                    recovery_after_failed_close=recovery_needed,
                )
            except Exception as exc:
                self.grasp_status_pub.publish(
                    String(data=f"POST_LIFT_FAILED: {exc}")
                )
                self.get_logger().error(
                    "抓后提升失败；不会重试夹爪，也不会发布抓取完成: "
                    f"{type(exc).__name__}: {exc}"
                )
                return False

            self.post_grasp_lift_executed = True

            if recovery_needed:
                self.grasp_status_pub.publish(
                    String(data="LIFTED_RETRYING_GRIPPER_CLOSE")
                )
                self.get_logger().warning(
                    "首次夹爪闭合验证失败，但已完成抬升；"
                    f"等待 {self.post_lift_reclose_settle_s:.2f}s 后"
                    "再次闭合夹爪"
                )
                if self.post_lift_reclose_settle_s > 0.0:
                    time.sleep(self.post_lift_reclose_settle_s)

                retry_succeeded = bool(
                    original_command_grippers(position)
                )
                if not retry_succeeded:
                    self.grasp_status_pub.publish(
                        String(data="GRIPPER_RECLOSE_FAILED_AFTER_LIFT")
                    )
                    self.get_logger().error(
                        "抬升后的第二次夹爪闭合验证仍失败；"
                        "保持当前升降高度并停止，不发布抓取完成"
                    )
                    return False

                self.get_logger().info(
                    "抬升后的第二次夹爪闭合验证通过"
                )

            self.post_grasp_lift_done = True
            self.grasp_status_pub.publish(String(data="SUCCEEDED"))
            self.grasp_completed_pub.publish(Bool(data=True))
            self.get_logger().info(
                "抓取闭合与抓后提升均完成；已发布 "
                "/box_grasp/completed=true"
            )
            return True

        controller.command_grippers = command_grippers_then_postlift

        self.get_logger().warning(
            "宽位下降/低位内收/抓后提升已启用: "
            f"outward {old_outward:.3f} -> {self.outward_offset_m:.3f}m/side, "
            f"waypoints {old_waypoints} -> {self.approach_waypoints}, "
            f"post_lift={self.post_grasp_lift_m:.3f}m, "
            "failed_close_recovery="
            f"{self.retry_close_after_failed_first_attempt}, "
            f"reclose_settle={self.post_lift_reclose_settle_s:.2f}s"
        )

    def _plan_sequence(self, left_base, right_base) -> dict:
        sequence = super()._plan_sequence(left_base, right_base)
        if not sequence.get("plans"):
            raise RuntimeError("分段规划没有返回任何路点")

        final_plan = sequence["plans"][-1]
        final_lift = float(
            final_plan["body"]["base_elevator_joint"]
        )
        lower, upper, _ = self.grasp_controller._limits[
            "base_elevator_joint"
        ]
        requested_post_lift = final_lift + self.post_grasp_lift_m
        planned_post_lift = min(float(upper), requested_post_lift)
        effective_lift = planned_post_lift - final_lift

        if final_lift < float(lower) - 1e-9:
            raise RuntimeError(
                f"最终抓取 lift={final_lift:+.4f}m 低于限位"
            )
        if effective_lift < self.minimum_effective_post_lift_m:
            raise RuntimeError(
                "规划阶段发现抓后上升行程不足: "
                f"final={final_lift:+.4f}m, upper={float(upper):+.4f}m, "
                f"effective={effective_lift:.4f}m"
            )

        sequence["post_grasp_lift_target_m"] = planned_post_lift
        sequence["post_grasp_lift_effective_m"] = effective_lift
        self.get_logger().info(
            "抓后上升预检查通过: "
            f"final_lift={final_lift:+.4f}m, "
            f"post_lift={planned_post_lift:+.4f}m, "
            f"rise={effective_lift:.4f}m"
        )
        return sequence

    def _execute_post_grasp_lift(
        self,
        recovery_after_failed_close: bool = False,
    ) -> None:
        if self.post_grasp_lift_executed:
            raise RuntimeError("抓后抬升已经执行，拒绝重复抬升")

        controller = self.grasp_controller
        current_lift = float(controller.get_pos("base_elevator_joint"))
        lower, upper, _ = controller._limits["base_elevator_joint"]
        requested_target = current_lift + self.post_grasp_lift_m
        target_lift = min(float(upper), requested_target)
        effective_lift = target_lift - current_lift

        if target_lift < float(lower) - 1e-9:
            raise RuntimeError(
                f"抓后 lift 目标低于限位: {target_lift:+.4f}m"
            )
        if effective_lift < self.minimum_effective_post_lift_m:
            raise RuntimeError(
                "升降轴剩余上行行程不足: "
                f"current={current_lift:+.4f}m, upper={float(upper):+.4f}m, "
                f"effective={effective_lift:.4f}m"
            )

        if target_lift < requested_target - 1e-9:
            self.get_logger().warning(
                "抓后提升目标受升降轴上限裁剪: "
                f"requested={requested_target:+.4f}m -> "
                f"target={target_lift:+.4f}m"
            )

        lift_reason = (
            "首次闭合验证失败，保持当前夹持/施力状态并恢复抬升"
            if recovery_after_failed_close
            else "夹爪闭合验证通过，执行抓后抬升"
        )
        self.get_logger().warning(
            f"{lift_reason}: "
            f"lift {current_lift:+.4f} -> {target_lift:+.4f}m "
            f"(rise={effective_lift:.4f}m)"
        )
        ok = controller.animate(
            {"base_elevator_joint": target_lift},
            duration=self.post_grasp_lift_duration_s,
            steps=self.post_grasp_lift_steps,
        )
        if not ok:
            raise RuntimeError("base_elevator_joint 抓后上升轨迹失败")


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[WideDescendPostLiftBridge] = None
    try:
        node = WideDescendPostLiftBridge()
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
