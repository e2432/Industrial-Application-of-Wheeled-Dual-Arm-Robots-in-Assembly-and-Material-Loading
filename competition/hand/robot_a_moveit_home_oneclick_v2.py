#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot A MoveIt2 归位脚本

归位顺序：
  1) 先通过 MoveIt 将 base_elevator_joint 归零
  2) 再通过 MoveIt 将双臂/躯干相关关节归零

默认直接真实执行“一键归位”；加 --dry-run 时只做 MoveIt 规划检查。
"""

import argparse
import math
import time
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Set, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rcl_interfaces.srv import GetParameters

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes


MOVE_ACTION = "/move_action"
MOVE_GROUP_NODE = "/move_group"

LIFT_JOINT = "base_elevator_joint"

ARM_ZERO_JOINTS = [
    "elevator_pitch_joint",

    "pitch_arm_left1_joint",
    "arm_left1_arm_left2_joint",
    "arm_left2_arm_left3_joint",
    "arm_left3_arm_left4_joint",
    "arm_left4_arm_left5_joint",
    "arm_left5_arm_left6_joint",
    "arm_left6_arm_left7_joint",

    "pitch_arm_right1_joint",
    "arm_right1_arm_right2_joint",
    "arm_right2_arm_right3_joint",
    "arm_right3_arm_right4_joint",
    "arm_right4_arm_right5_joint",
    "arm_right5_arm_right6_joint",
    "arm_right6_arm_right7_joint",
]


def wait_future(node: Node, future, timeout: float):
    deadline = time.monotonic() + timeout
    while rclpy.ok() and not future.done():
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("等待 ROS future 超时")
        rclpy.spin_once(
            node,
            timeout_sec=min(0.1, remaining),
        )

    if not future.done():
        raise TimeoutError("等待 ROS future 超时")

    return future.result()


def parse_srdf_groups(srdf_text: str):
    root = ET.fromstring(srdf_text)
    groups = {}
    for group in root.findall("group"):
        name = group.get("name")
        if name:
            groups[name] = group
    return root, groups


def parse_urdf_parent_map(urdf_text: str):
    root = ET.fromstring(urdf_text)
    child_to_joint = {}

    for joint in root.findall("joint"):
        name = joint.get("name")
        parent = joint.find("parent")
        child = joint.find("child")
        jtype = joint.get("type", "")

        if (
            not name
            or parent is None
            or child is None
            or not parent.get("link")
            or not child.get("link")
        ):
            continue

        info = {
            "name": name,
            "parent": parent.get("link"),
            "child": child.get("link"),
            "type": jtype,
        }
        child_to_joint[info["child"]] = info

    return child_to_joint


def chain_joints(
    base_link: str,
    tip_link: str,
    child_to_joint: Dict[str, dict],
) -> List[str]:
    result_reversed = []
    current = tip_link
    visited = set()

    while current != base_link:
        if current in visited:
            raise RuntimeError(
                f"URDF chain 出现环: {base_link} -> {tip_link}"
            )
        visited.add(current)

        info = child_to_joint.get(current)
        if info is None:
            raise RuntimeError(
                f"无法从 {tip_link} 回溯到 {base_link}，"
                f"卡在 link={current}"
            )

        if info["type"] != "fixed":
            result_reversed.append(info["name"])

        current = info["parent"]

    return list(reversed(result_reversed))


def expand_group_members(
    group_name: str,
    groups: Dict[str, ET.Element],
    child_to_joint: Dict[str, dict],
    stack: Optional[Set[str]] = None,
) -> Set[str]:

    if stack is None:
        stack = set()

    if group_name in stack:
        raise RuntimeError(
            f"SRDF subgroup 循环引用: {group_name}"
        )

    group = groups.get(group_name)
    if group is None:
        raise RuntimeError(
            f"SRDF 中不存在 planning group: {group_name}"
        )

    stack = set(stack)
    stack.add(group_name)

    members: Set[str] = set()

    for elem in group:
        if elem.tag == "joint":
            name = elem.get("name")
            if name:
                members.add(name)

        elif elem.tag == "chain":
            base_link = elem.get("base_link")
            tip_link = elem.get("tip_link")
            if base_link and tip_link:
                members.update(
                    chain_joints(
                        base_link,
                        tip_link,
                        child_to_joint,
                    )
                )

        elif elem.tag == "group":
            subgroup = elem.get("name")
            if subgroup:
                members.update(
                    expand_group_members(
                        subgroup,
                        groups,
                        child_to_joint,
                        stack,
                    )
                )

    return members


def find_smallest_group(
    group_members: Dict[str, Set[str]],
    required: Set[str],
) -> Optional[str]:

    candidates = [
        (name, members)
        for name, members in group_members.items()
        if required.issubset(members)
    ]

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            len(item[1]),
            item[0],
        )
    )

    return candidates[0][0]


class MoveItHome(Node):
    def __init__(
        self,
        execute: bool,
        velocity_scale: float,
        acceleration_scale: float,
        planning_time: float,
        tolerance: float,
        action_name: str,
    ):
        super().__init__("robot_a_moveit_home")

        self.execute = bool(execute)
        self.velocity_scale = float(velocity_scale)
        self.acceleration_scale = float(acceleration_scale)
        self.planning_time = float(planning_time)
        self.tolerance = float(tolerance)

        self.current_positions: Dict[str, float] = {}

        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            20,
        )

        self.move_client = ActionClient(
            self,
            MoveGroup,
            action_name,
        )

    def _joint_state_cb(self, msg: JointState):
        for name, position in zip(
            msg.name,
            msg.position,
        ):
            if math.isfinite(float(position)):
                self.current_positions[name] = float(position)

    def wait_for_joint_states(
        self,
        required: Iterable[str],
        timeout: float = 8.0,
    ):
        required = list(required)
        deadline = time.monotonic() + timeout

        while rclpy.ok():
            missing = [
                j for j in required
                if j not in self.current_positions
            ]

            if not missing:
                return

            if time.monotonic() > deadline:
                raise RuntimeError(
                    "等待 /joint_states 超时，缺少: "
                    + ", ".join(missing)
                )

            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

    def get_moveit_descriptions(
        self,
        timeout: float = 8.0,
    ) -> Tuple[str, str]:
        """
        直接通过 /move_group/get_parameters 读取 URDF / SRDF。
        使用 ROS2 标准 GetParameters 服务，兼容当前 Robot A 的 rclpy。
        """
        service_name = (
            f"{MOVE_GROUP_NODE}/get_parameters"
        )

        client = self.create_client(
            GetParameters,
            service_name,
        )

        self.get_logger().info(
            f"等待参数服务: {service_name}"
        )

        if not client.wait_for_service(
            timeout_sec=timeout
        ):
            raise RuntimeError(
                f"找不到 {service_name}。"
                "请确认 MoveIt move_group 已启动。"
            )

        request = GetParameters.Request()
        request.names = [
            "robot_description",
            "robot_description_semantic",
        ]

        future = client.call_async(request)
        result = wait_future(
            self,
            future,
            timeout,
        )

        if (
            result is None
            or len(result.values) != 2
        ):
            raise RuntimeError(
                "读取 MoveIt robot_description/SRDF 失败"
            )

        urdf = result.values[0].string_value
        srdf = result.values[1].string_value

        if not urdf.strip():
            raise RuntimeError(
                "/move_group 的 robot_description 为空"
            )

        if not srdf.strip():
            raise RuntimeError(
                "/move_group 的 robot_description_semantic 为空"
            )

        self.get_logger().info(
            "✅ 已从 /move_group 读取 URDF/SRDF"
        )

        return urdf, srdf

    def wait_for_move_action(
        self,
        timeout: float = 10.0,
    ):
        self.get_logger().info(
            "等待 MoveIt /move_action ..."
        )

        if not self.move_client.wait_for_server(
            timeout_sec=timeout
        ):
            raise RuntimeError(
                "MoveIt /move_action 不可用。"
                "请确认 move_group 已启动。"
            )

        self.get_logger().info(
            "✅ MoveIt /move_action 已就绪"
        )

    def send_joint_goal(
        self,
        group_name: str,
        group_members: Set[str],
        target_overrides: Dict[str, float],
        label: str,
    ):
        self.wait_for_joint_states(
            group_members,
            timeout=8.0,
        )

        missing_target = [
            j for j in target_overrides
            if j not in group_members
        ]

        if missing_target:
            raise RuntimeError(
                f"{label}: planning group '{group_name}' "
                "不包含目标关节: "
                + ", ".join(missing_target)
            )

        constraints = Constraints()
        constraints.name = label

        targets_for_log = {}

        # group 内不需要归零的关节保持当前值，
        # 避免 phase-1 lift 归零时其它关节被规划器改变。
        for joint_name in sorted(group_members):
            if joint_name not in self.current_positions:
                continue

            target = target_overrides.get(
                joint_name,
                self.current_positions[joint_name],
            )

            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(target)
            jc.tolerance_above = self.tolerance
            jc.tolerance_below = self.tolerance
            jc.weight = 1.0

            constraints.joint_constraints.append(jc)

            if joint_name in target_overrides:
                targets_for_log[joint_name] = float(target)

        if not constraints.joint_constraints:
            raise RuntimeError(
                f"{label}: 没有可用的 joint constraints"
            )

        goal = MoveGroup.Goal()

        goal.request.group_name = group_name
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = self.planning_time
        goal.request.max_velocity_scaling_factor = self.velocity_scale
        goal.request.max_acceleration_scaling_factor = self.acceleration_scale
        goal.request.goal_constraints = [
            constraints
        ]

        # 使用 MoveIt PlanningScene 的当前状态。
        goal.request.start_state.is_diff = True

        goal.planning_options.plan_only = not self.execute
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3
        goal.planning_options.replan_delay = 0.2

        self.get_logger().info(
            "=" * 70
        )
        self.get_logger().info(label)
        self.get_logger().info(
            f"MoveIt group: {group_name}"
        )
        self.get_logger().info(
            f"目标关节: {targets_for_log}"
        )
        self.get_logger().info(
            "模式: "
            + (
                "REAL EXECUTE"
                if self.execute
                else "DRY-RUN / PLAN ONLY"
            )
        )

        send_future = self.move_client.send_goal_async(
            goal
        )

        goal_handle = wait_future(
            self,
            send_future,
            timeout=10.0,
        )

        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(
                f"{label}: MoveIt goal 被拒绝"
            )

        result_future = goal_handle.get_result_async()

        wrapped_result = wait_future(
            self,
            result_future,
            timeout=max(
                30.0,
                self.planning_time * 8.0,
            ),
        )

        if wrapped_result is None:
            raise RuntimeError(
                f"{label}: MoveIt 没有返回结果"
            )

        result = wrapped_result.result
        error_code = int(
            result.error_code.val
        )

        if error_code != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(
                f"{label}: MoveIt 失败，"
                f"error_code={error_code}, "
                f"state='{result.state}'"
            )

        self.get_logger().info(
            f"✅ {label}: "
            + (
                "MoveIt 规划并执行成功"
                if self.execute
                else "MoveIt 规划成功（未执行）"
            )
        )

    def verify_zero(
        self,
        joints: Iterable[str],
        tolerance: float,
        timeout: float = 4.0,
    ):
        if not self.execute:
            return

        joints = list(joints)
        deadline = time.monotonic() + timeout

        while rclpy.ok():
            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

            errors = {}

            for joint in joints:
                if joint not in self.current_positions:
                    errors[joint] = float("inf")
                else:
                    errors[joint] = abs(
                        self.current_positions[joint]
                    )

            if errors and max(errors.values()) <= tolerance:
                self.get_logger().info(
                    "✅ /joint_states 实际归零验证通过: "
                    f"max_error={max(errors.values()):.6f}"
                )
                return

            if time.monotonic() > deadline:
                bad = {
                    name: err
                    for name, err in errors.items()
                    if err > tolerance
                }

                raise RuntimeError(
                    "MoveIt 返回成功，但 /joint_states "
                    "实际未归零到允许误差内: "
                    + str(bad)
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Robot A MoveIt2 一键归位："
            "默认真实执行，严格先 lift=0，再双臂/躯干关节=0。"
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅规划检查，不真实执行；默认直接执行归位",
    )

    parser.add_argument(
        "--lift-group",
        default=None,
        help=(
            "包含 base_elevator_joint 的 MoveIt planning group；"
            "默认自动检测"
        ),
    )

    parser.add_argument(
        "--arms-group",
        default=None,
        help=(
            "包含全部双臂归零关节的 MoveIt planning group；"
            "默认自动检测"
        ),
    )

    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=0.20,
        help="MoveIt 最大速度比例，默认 0.20",
    )

    parser.add_argument(
        "--acceleration-scale",
        type=float,
        default=0.20,
        help="MoveIt 最大加速度比例，默认 0.20",
    )

    parser.add_argument(
        "--planning-time",
        type=float,
        default=8.0,
        help="单阶段 MoveIt 规划时间，默认 8 秒",
    )

    parser.add_argument(
        "--goal-tolerance",
        type=float,
        default=0.005,
        help="关节目标容差，默认 0.005 rad/m",
    )

    parser.add_argument(
        "--verify-tolerance",
        type=float,
        default=0.01,
        help="实机 /joint_states 最终验证容差，默认 0.01 rad/m",
    )

    parser.add_argument(
        "--action-name",
        default=MOVE_ACTION,
        help="MoveIt MoveGroup action，默认 /move_action",
    )

    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="只打印 SRDF planning groups 及其关节成员，然后退出",
    )

    args = parser.parse_args()

    if not 0.0 < args.velocity_scale <= 1.0:
        parser.error("--velocity-scale 必须在 (0,1] 内")

    if not 0.0 < args.acceleration_scale <= 1.0:
        parser.error("--acceleration-scale 必须在 (0,1] 内")

    rclpy.init()

    node = MoveItHome(
        execute=(not args.dry_run),
        velocity_scale=args.velocity_scale,
        acceleration_scale=args.acceleration_scale,
        planning_time=args.planning_time,
        tolerance=args.goal_tolerance,
        action_name=args.action_name,
    )

    try:
        urdf_text, srdf_text = (
            node.get_moveit_descriptions()
        )

        _, srdf_groups = parse_srdf_groups(
            srdf_text
        )

        child_to_joint = (
            parse_urdf_parent_map(
                urdf_text
            )
        )

        expanded: Dict[str, Set[str]] = {}

        for group_name in srdf_groups:
            try:
                expanded[group_name] = (
                    expand_group_members(
                        group_name,
                        srdf_groups,
                        child_to_joint,
                    )
                )
            except Exception as exc:
                node.get_logger().warning(
                    f"无法展开 group '{group_name}': {exc}"
                )
                expanded[group_name] = set()

        if args.list_groups:
            print("\nMoveIt SRDF planning groups:")
            for name in sorted(expanded):
                print(f"\n[{name}]")
                for joint in sorted(expanded[name]):
                    print(f"  {joint}")
            return 0

        lift_required = {LIFT_JOINT}
        arms_required = set(ARM_ZERO_JOINTS)

        lift_group = (
            args.lift_group
            or find_smallest_group(
                expanded,
                lift_required,
            )
        )

        arms_group = (
            args.arms_group
            or find_smallest_group(
                expanded,
                arms_required,
            )
        )

        if lift_group is None:
            raise RuntimeError(
                "无法自动找到包含 base_elevator_joint 的 "
                "MoveIt planning group。"
                "先运行 --list-groups 查看。"
            )

        if arms_group is None:
            ranked = sorted(
                (
                    (
                        len(members & arms_required),
                        name,
                    )
                    for name, members in expanded.items()
                ),
                reverse=True,
            )

            detail = ", ".join(
                f"{name}:{count}/{len(arms_required)}"
                for count, name in ranked[:5]
            )

            raise RuntimeError(
                "无法自动找到一个同时包含全部双臂归零关节的 "
                "MoveIt planning group。"
                "先运行 --list-groups 查看，"
                "再用 --arms-group 指定。"
                f" 最接近的 group: {detail}"
            )

        if lift_group not in expanded:
            raise RuntimeError(
                f"--lift-group '{lift_group}' 不在 SRDF 中"
            )

        if arms_group not in expanded:
            raise RuntimeError(
                f"--arms-group '{arms_group}' 不在 SRDF 中"
            )

        node.get_logger().info(
            "=" * 70
        )
        node.get_logger().info(
            "Robot A MoveIt2 一键归位"
        )
        node.get_logger().warning(
            "执行顺序固定：① Lift -> 0；② 双臂/躯干 -> 0。"
            "不会先动手臂。"
        )
        node.get_logger().info(
            f"第一阶段 Lift group : {lift_group}"
        )
        node.get_logger().info(
            f"第二阶段 Arms group : {arms_group}"
        )
        node.get_logger().info(
            "顺序: Lift -> 0，然后双臂/躯干 -> 0"
        )
        node.get_logger().warning(
            "模式: "
            + (
                "REAL EXECUTE / ONE-CLICK HOME"
                if not args.dry_run
                else "DRY-RUN；机器人不会运动"
            )
        )
        node.get_logger().info(
            "=" * 70
        )

        node.wait_for_move_action()

        # ----------------------------------------------------------
        # Phase 1: Lift first.
        # ----------------------------------------------------------
        node.get_logger().info(
            "▶ [1/2] Lift 归零"
        )

        node.send_joint_goal(
            group_name=lift_group,
            group_members=expanded[lift_group],
            target_overrides={
                LIFT_JOINT: 0.0,
            },
            label="Lift -> 0",
        )

        node.verify_zero(
            [LIFT_JOINT],
            tolerance=args.verify_tolerance,
        )

        # ----------------------------------------------------------
        # Phase 2: Arms second.
        # ----------------------------------------------------------
        node.get_logger().info(
            "▶ [2/2] 双臂/躯干关节归零"
        )

        arm_targets = {
            joint: 0.0
            for joint in ARM_ZERO_JOINTS
        }

        node.send_joint_goal(
            group_name=arms_group,
            group_members=expanded[arms_group],
            target_overrides=arm_targets,
            label="Arms -> 0",
        )

        node.verify_zero(
            ARM_ZERO_JOINTS,
            tolerance=args.verify_tolerance,
        )

        if not args.dry_run:
            node.get_logger().info(
                "✅ Robot A 一键归位完成："
                "Lift 已先归零，随后双臂/躯干关节归零。"
            )
        else:
            node.get_logger().info(
                "✅ Dry-run 完成：两阶段 MoveIt 规划均通过，"
                "机器人没有运动。"
            )

        return 0

    except KeyboardInterrupt:
        node.get_logger().warning(
            "收到 Ctrl+C，程序退出。"
        )
        return 130

    except Exception as exc:
        node.get_logger().error(
            f"❌ 归位失败: {type(exc).__name__}: {exc}"
        )
        return 1

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
