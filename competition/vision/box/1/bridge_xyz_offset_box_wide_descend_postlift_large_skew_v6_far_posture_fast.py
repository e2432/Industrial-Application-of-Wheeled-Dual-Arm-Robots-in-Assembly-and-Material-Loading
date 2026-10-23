#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot A 双臂抓箱：
Large-Skew V6 / Far-Posture Fast

基于：
  bridge_xyz_offset_box_wide_descend_postlift_large_skew_v5_far_fast.py

V6 针对 V5 实机日志中的明确瓶颈：

V5 在 far target 上已经得到非常好的 TCP 几何结果，例如：
  L: XY≈0.00~0.01cm, Z≈0.00~0.01cm
  R: XY≈0.39~0.74cm, Z≈0.06~0.10cm
  forearm / vertical 都很小

但最终仍失败，主要因为：
  L template RMSE ≈ 46deg
  R template RMSE ≈ 36~39deg
而 V5 far 模式仍要求 template <= 30deg。

对于远目标，标准近距离 template 不应继续作为 30deg 的硬拒绝门槛。
V6 做法：
  1) 仅 Far 模式把最终 template 上限从 30deg 提到默认 50deg；
  2) template 仍保留在 DLS residual 中作为软正则，不是完全取消；
  3) TCP XY / lateral / Z / orientation / gripper vertical /
     forearm / URDF limits / waypoint joint jump 全部继续检查；
  4) Large-skew V2/V3/V4 的逻辑不改变；
  5) Far 默认搜索量从 V5 的 4 seeds, 90/160 降到
     3 seeds, 70/120，line-search 4；
  6) 候选仍按 0cm -> 1.5cm -> 3.0cm forward retreat，
     一旦原始目标通过会立即返回，不再继续后两个候选。

注意：
  本系统没有完整 3D collision scene。
  第一次必须 enable_motion:=false 检查整条路径。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy

from bridge_xyz_offset_box_wide_descend_postlift_large_skew_v5_far_fast import (
    FarTargetFastV5Bridge,
)


class FarPostureFastV6Bridge(FarTargetFastV5Bridge):
    """V5 + Far 专用姿态域 + 更快早停。"""

    def __init__(self) -> None:
        super().__init__()

        # --------------------------------------------------------------
        # 使用 V6 独立参数名，避免与 V5 已声明参数重名。
        # --------------------------------------------------------------
        self.declare_parameter(
            "v6_far_template_rmse_deg",
            50.0,
        )
        self.declare_parameter(
            "v6_far_seed_limit",
            3,
        )
        self.declare_parameter(
            "v6_far_reach_iterations",
            70,
        )
        self.declare_parameter(
            "v6_far_refine_iterations",
            120,
        )
        self.declare_parameter(
            "v6_far_line_search_steps",
            4,
        )

        template_deg = float(
            self.get_parameter(
                "v6_far_template_rmse_deg"
            ).value
        )
        seed_limit = int(
            self.get_parameter(
                "v6_far_seed_limit"
            ).value
        )
        reach_iterations = int(
            self.get_parameter(
                "v6_far_reach_iterations"
            ).value
        )
        refine_iterations = int(
            self.get_parameter(
                "v6_far_refine_iterations"
            ).value
        )
        line_search_steps = int(
            self.get_parameter(
                "v6_far_line_search_steps"
            ).value
        )

        if not 40.0 <= template_deg <= 55.0:
            raise RuntimeError(
                "v6_far_template_rmse_deg 必须在 [40,55]deg"
            )
        if not 2 <= seed_limit <= 5:
            raise RuntimeError(
                "v6_far_seed_limit 必须在 [2,5]"
            )
        if not 50 <= reach_iterations <= 120:
            raise RuntimeError(
                "v6_far_reach_iterations 必须在 [50,120]"
            )
        if not 80 <= refine_iterations <= 200:
            raise RuntimeError(
                "v6_far_refine_iterations 必须在 [80,200]"
            )
        if not 3 <= line_search_steps <= 6:
            raise RuntimeError(
                "v6_far_line_search_steps 必须在 [3,6]"
            )

        # --------------------------------------------------------------
        # 直接覆盖 V5 far-mode 内部值。
        # 普通目标 / large-skew 目标不会使用这些 Far 参数。
        # --------------------------------------------------------------
        self.far_max_template_rad = math.radians(
            template_deg
        )
        self.far_seed_limit = seed_limit
        self.far_reach_iterations = reach_iterations
        self.far_refine_iterations = refine_iterations
        self.far_line_search_steps = line_search_steps

        self.get_logger().warning(
            "Large-skew V6/Far-Posture Fast 已启用："
            "仅 Far 模式允许标准 template 更大偏离；"
            f"far template<={template_deg:.1f}deg, "
            f"seeds={seed_limit}, "
            f"iterations={reach_iterations}/"
            f"{refine_iterations}, "
            f"line_search={line_search_steps}；"
            "TCP/夹紧/姿态/前臂/竖直/URDF/路径连续性检查保持"
        )

    # ------------------------------------------------------------------
    # 扩展 V5 的最终日志，让实机前能直接看到关键姿态指标和最终 q。
    # V5 的 _verify_far_final 会按当前 self.far_max_template_rad
    # （V6 已设为 50deg）执行全部 Far 最终检查。
    # ------------------------------------------------------------------
    def _verify_far_final(self, sequence: dict) -> None:
        super()._verify_far_final(sequence)

        final_plan = sequence["plans"][-1]
        lm = final_plan["metrics"]["left_final"]
        rm = final_plan["metrics"]["right_final"]

        left_q = np.asarray(
            final_plan["left"],
            dtype=np.float64,
        )
        right_q = np.asarray(
            final_plan["right"],
            dtype=np.float64,
        )

        self.get_logger().warning(
            "V6 Far 最终姿态诊断通过:\n"
            f"  L: XY={lm['xy_error_m']*100:.2f}cm, "
            f"lateral={lm['lateral_error_m']*100:.2f}cm, "
            f"Z={lm['z_error_m']*100:.2f}cm, "
            f"orientation="
            f"{math.degrees(lm['orientation_error_rad']):.2f}deg, "
            f"forearm="
            f"{math.degrees(lm['forearm_tilt_rad']):.2f}deg, "
            f"vertical="
            f"{math.degrees(lm['gripper_vertical_error_rad']):.2f}deg, "
            f"template="
            f"{math.degrees(lm['template_rmse_rad']):.2f}deg\n"
            f"  R: XY={rm['xy_error_m']*100:.2f}cm, "
            f"lateral={rm['lateral_error_m']*100:.2f}cm, "
            f"Z={rm['z_error_m']*100:.2f}cm, "
            f"orientation="
            f"{math.degrees(rm['orientation_error_rad']):.2f}deg, "
            f"forearm="
            f"{math.degrees(rm['forearm_tilt_rad']):.2f}deg, "
            f"vertical="
            f"{math.degrees(rm['gripper_vertical_error_rad']):.2f}deg, "
            f"template="
            f"{math.degrees(rm['template_rmse_rad']):.2f}deg\n"
            f"  left q(deg)={np.round(np.degrees(left_q), 2)}\n"
            f"  right q(deg)={np.round(np.degrees(right_q), 2)}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[FarPostureFastV6Bridge] = None

    try:
        node = FarPostureFastV6Bridge()
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
