#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Competition route built on the verified navigation/grasp recovery stack.

Route:
  start -> point1 -> point2
  -> [grabN, grasp, lift, dropN, release, reset, grabN retreat] for N=1..4
  -> point2 -> point1 -> start

This module intentionally reuses box_waypoint_mission.py so vision recovery,
IK-driven chassis correction, velocity arbitration, Nav2 cancellation handling,
TF settling and costmap refresh remain identical to the tested task stack.
"""

from __future__ import annotations

import threading
from typing import Optional

import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool

import box_waypoint_mission as mission_base


class CompetitionMission(mission_base.RecoveryBoxWaypointMission):
    GRAB_IDS = frozenset(f"grab{index}" for index in range(1, 5))
    DROP_IDS = frozenset(f"drop{index}" for index in range(1, 5))
    REQUIRED_IDS = frozenset(
        {"start", "point1", "point2"} | set(GRAB_IDS) | set(DROP_IDS)
    )

    def __init__(self, cli) -> None:
        super().__init__(cli)
        self.waypoint_by_id = {
            waypoint.object_id: waypoint for waypoint in self.waypoints
        }
        missing = sorted(self.REQUIRED_IDS - set(self.waypoint_by_id))
        if missing:
            raise RuntimeError(
                "competition waypoint configuration is missing: "
                + ", ".join(missing)
            )
        self.get_logger().warning(
            "Competition route loaded: start -> point1 -> point2 -> "
            "four grab/drop/retreat cycles -> point2 -> point1 -> start"
        )

    def arrival_tolerances(self, point):
        if point.object_id in self.GRAB_IDS:
            return (
                self.cli.grasp_arrival_position_tolerance,
                self.cli.grasp_arrival_yaw_tolerance,
                "competition_grasp",
            )
        if point.object_id in self.DROP_IDS:
            return (
                self.cli.task_arrival_position_tolerance,
                self.cli.task_arrival_yaw_tolerance,
                "competition_drop",
            )
        return (
            self.cli.arrival_position_tolerance,
            self.cli.arrival_yaw_tolerance,
            "competition_transit",
        )

    def requires_precision_alignment(self, point) -> bool:
        return point.object_id in self.GRAB_IDS or point.object_id in self.DROP_IDS

    def _fail(self, detail: str) -> bool:
        self.stop_robot()
        self.state("FAILED", detail)
        return False

    def _go(self, object_id: str, purpose: str) -> bool:
        point = self.waypoint_by_id[object_id]
        self.state("COMPETITION_ROUTE", f"{object_id}: {purpose}")
        if not self.navigate(point):
            return self._fail(f"cannot reach {object_id}: {purpose}")
        settle = max(
            point.dwell_seconds,
            self.cli.arrival_settle
            if object_id in self.GRAB_IDS or object_id in self.DROP_IDS
            else 0.0,
        )
        if not self.sleep(settle):
            return False
        return True

    def _run_transport_cycle(self, index: int) -> bool:
        grab_id = f"grab{index}"
        drop_id = f"drop{index}"

        if not self._go(grab_id, f"cycle {index}/4 pickup"):
            return False
        self.state("COMPETITION_GRASP", f"{grab_id}: cycle {index}/4")
        if not self.grasp_task():
            return self._fail(f"grasp failed at {grab_id}")

        if not self.simple_task(
            f"cycle {index}/4 lift to transport height",
            "lift",
        ):
            return self._fail(f"lift failed after {grab_id}")
        self.clear_navigation_costmaps(
            f"cycle {index}/4 after lift before transport"
        )

        if not self._go(drop_id, f"cycle {index}/4 placement"):
            return False
        if not self.simple_task(f"cycle {index}/4 place and release", "release"):
            return self._fail(f"release failed at {drop_id}")
        if not self.simple_task(f"cycle {index}/4 reset all joints", "reset"):
            return self._fail(f"joint reset failed at {drop_id}")

        # Return to the same pickup pose without grasping. This is the retreat
        # leg confirmed by the operator and establishes the approach corridor
        # for the next pickup point (or the final return route for cycle 4).
        if not self._go(grab_id, f"cycle {index}/4 retreat after placement"):
            return False
        return True

    def run_mission(self) -> bool:
        self.done_pub.publish(Bool(data=False))
        self.state(
            "READY",
            "competition_mission: 4 transport cycles and return to start",
        )

        # start is both the physical initial pose and the final pose. navigate()
        # first checks map->base_link and does not send a zero-length Nav2 goal
        # when the robot is already within tolerance.
        for object_id, purpose in (
            ("start", "confirm initial pose"),
            ("point1", "outbound transit 1/2"),
            ("point2", "outbound transit 2/2"),
        ):
            if not self._go(object_id, purpose):
                return False

        for index in range(1, 5):
            if not self._run_transport_cycle(index):
                return False

        for object_id, purpose in (
            ("point2", "return transit 1/2"),
            ("point1", "return transit 2/2"),
            ("start", "competition finish"),
        ):
            if not self._go(object_id, purpose):
                return False

        self.stop_robot()
        self.done_pub.publish(Bool(data=True))
        self.state("COMPLETED", "four cycles complete; robot returned to start")
        return True


def main() -> int:
    cli = mission_base.parse_args()
    rclpy.init(args=[])
    node: Optional[CompetitionMission] = None
    executor: Optional[MultiThreadedExecutor] = None
    thread: Optional[threading.Thread] = None
    try:
        node = CompetitionMission(cli)
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        return 0 if node.run_mission() else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if node is not None:
            node.get_logger().error(
                f"competition mission exception: {type(exc).__name__}: {exc}"
            )
        else:
            print(
                f"competition mission creation failed: {type(exc).__name__}: {exc}"
            )
        return 1
    finally:
        if node is not None:
            node.shutdown()
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if thread is not None:
            thread.join(timeout=2.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
