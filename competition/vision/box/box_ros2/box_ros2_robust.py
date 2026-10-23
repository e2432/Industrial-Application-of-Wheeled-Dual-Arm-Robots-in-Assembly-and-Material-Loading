#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""箱口 3D 抓取点视觉节点：斜视角稳健门限 + 近距居中目标优选。

把本文件与原始
    box_ros2_open_rim_corner_midpoints_3d_geomselect.py
放在同一目录，然后运行本文件。

本启动版保留原脚本的点云同步、RANSAC、深度范围、TF、发布话题和
后续箱口几何处理，只在 YOLO 输出多个箱子时增加目标选择：

1. 优先保留检测框中心位于画面中央带内的箱子；
2. 中央带内优先选择点云估计深度最小的箱子；
3. 点云暂不可用时，以检测框面积作为“越大通常越近”的回退依据；
4. 中央带内没有箱子时，对全部箱子使用同样的近距排序。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy

from box_ros2_open_rim_corner_midpoints_3d_geomselect import (
    BoxPoseShortEdgeDetector,
    pointcloud2_xyz,
    project_plumb_bob,
    stamp_to_seconds,
    transform_points,
)


class _CachedPredictionModel:
    """让父类复用已经完成的 YOLO 推理，避免同一帧重复推理。"""

    def __init__(self, selected_result) -> None:
        self._selected_result = selected_result

    def predict(self, **_kwargs):
        return [self._selected_result]


class RobustBoxPoseShortEdgeDetector(BoxPoseShortEdgeDetector):
    def __init__(self) -> None:
        super().__init__()

        # --------------------------------------------------------------
        # 原有斜视角稳健几何门限
        # --------------------------------------------------------------
        # 只检查排序最靠前的两个主平面。当前相机斜视箱口时，第 3 个
        # RANSAC 候选经常是箱体内侧壁；它的矩形分数偶尔更低，原算法会
        # 在箱口与侧壁之间跳选，导致发布的 3D 抓取点出现厘米级抖动。
        self.declare_parameter("robust_plane_candidate_count", 2)
        self.declare_parameter(
            "robust_max_opposite_edge_rel_error",
            0.35,
        )
        self.declare_parameter(
            "robust_max_corner_orthogonality_cos",
            0.50,
        )

        robust_candidate_count = int(
            self.get_parameter("robust_plane_candidate_count").value
        )
        robust_opposite_error = float(
            self.get_parameter(
                "robust_max_opposite_edge_rel_error"
            ).value
        )
        robust_orthogonality_cos = float(
            self.get_parameter(
                "robust_max_corner_orthogonality_cos"
            ).value
        )

        if not 1 <= robust_candidate_count <= 2:
            raise RuntimeError(
                "robust_plane_candidate_count 必须在 [1, 2] 内"
            )
        if not 0.18 <= robust_opposite_error <= 0.40:
            raise RuntimeError(
                "robust_max_opposite_edge_rel_error 必须在 [0.18, 0.40] 内"
            )
        if not 0.25 <= robust_orthogonality_cos <= 0.55:
            raise RuntimeError(
                "robust_max_corner_orthogonality_cos 必须在 [0.25, 0.55] 内"
            )

        old_candidate_count = int(self.plane_candidate_count)
        old_opposite_error = float(self.max_opposite_edge_rel_error)
        old_orthogonality_cos = float(
            self.max_corner_orthogonality_cos
        )

        # 原类在回调开始前读取这些成员，所以在 super().__init__() 返回后调整
        # 是有效的；不会重复创建 ROS 节点或发布器。
        # 候选数量必须直接覆盖，不能取 max；否则原脚本默认的 3 个候选仍会
        # 把 idx=2 的内侧壁送入最终几何评分。
        self.plane_candidate_count = robust_candidate_count
        self.max_opposite_edge_rel_error = max(
            old_opposite_error,
            robust_opposite_error,
        )
        self.max_corner_orthogonality_cos = max(
            old_orthogonality_cos,
            robust_orthogonality_cos,
        )

        # --------------------------------------------------------------
        # 新增：多个箱子之间的“居中 + 近距”选择参数
        # --------------------------------------------------------------
        # center_error = |box_cx - image_cx| / (image_width / 2)。
        # 0 表示正好在中线；1 表示接近左右边缘。
        self.declare_parameter("target_center_band_ratio", 0.35)

        # 为减小检测框边缘处背景/邻物点云的干扰，只在框中央缩小区域估深。
        self.declare_parameter("target_depth_roi_scale", 0.50)

        # 使用 ROI 内 Z 的较低分位数，降低开口箱底部较远点对估深的影响。
        self.declare_parameter("target_depth_percentile", 30.0)
        self.declare_parameter("target_min_depth_points", 30)

        self.target_center_band_ratio = float(
            self.get_parameter("target_center_band_ratio").value
        )
        self.target_depth_roi_scale = float(
            self.get_parameter("target_depth_roi_scale").value
        )
        self.target_depth_percentile = float(
            self.get_parameter("target_depth_percentile").value
        )
        self.target_min_depth_points = int(
            self.get_parameter("target_min_depth_points").value
        )

        if not 0.0 <= self.target_center_band_ratio <= 1.0:
            raise RuntimeError(
                "target_center_band_ratio 必须在 [0, 1] 内"
            )
        if not 0.10 <= self.target_depth_roi_scale <= 1.0:
            raise RuntimeError(
                "target_depth_roi_scale 必须在 [0.10, 1.0] 内"
            )
        if not 0.0 <= self.target_depth_percentile <= 100.0:
            raise RuntimeError(
                "target_depth_percentile 必须在 [0, 100] 内"
            )
        if self.target_min_depth_points < 3:
            raise RuntimeError(
                "target_min_depth_points 必须 >= 3"
            )

        min_corner_angle_deg = 0.0
        try:
            import math

            min_corner_angle_deg = math.degrees(
                math.acos(self.max_corner_orthogonality_cos)
            )
        except (ValueError, TypeError):
            pass

        self.get_logger().warning(
            "已启用斜视角稳健箱口几何门限: "
            f"candidates {old_candidate_count} -> "
            f"{self.plane_candidate_count}, "
            f"opposite_rel {old_opposite_error:.2f} -> "
            f"{self.max_opposite_edge_rel_error:.2f}, "
            f"orthogonality_cos {old_orthogonality_cos:.2f} -> "
            f"{self.max_corner_orthogonality_cos:.2f} "
            f"(corner_angle >= {min_corner_angle_deg:.1f}deg)"
        )
        self.get_logger().warning(
            "已启用近距居中目标优选: "
            f"center_band={self.target_center_band_ratio:.2f}, "
            f"depth_roi_scale={self.target_depth_roi_scale:.2f}, "
            f"depth_percentile={self.target_depth_percentile:.1f}, "
            f"min_depth_points={self.target_min_depth_points}"
        )

    def _project_cloud_for_target_selection(
        self,
        msg,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """返回与当前图像同步的点云投影 (u, v, z)，失败则允许面积回退。"""
        if (
            self.color_K is None
            or self.color_D is None
            or self.color_width is None
            or self.color_height is None
        ):
            return None

        cloud, sync_dt = self._find_nearest_cloud(
            stamp_to_seconds(msg.header.stamp)
        )
        if cloud is None or sync_dt > self.max_sync_dt_s:
            return None

        try:
            points_cloud = pointcloud2_xyz(cloud)
            valid = np.isfinite(points_cloud).all(axis=1)
            points_cloud = points_cloud[valid]
            if len(points_cloud) < self.target_min_depth_points:
                return None

            cloud_frame = str(cloud.header.frame_id).strip()
            if not cloud_frame:
                return None

            if cloud_frame == self.color_frame:
                points_color = points_cloud
            else:
                transform = self._lookup_matrix(
                    self.color_frame,
                    cloud_frame,
                    cloud.header.stamp,
                )
                points_color = transform_points(transform, points_cloud)

            valid = (
                np.isfinite(points_color).all(axis=1)
                & (points_color[:, 2] >= self.min_depth_m)
                & (points_color[:, 2] <= self.max_depth_m)
            )
            points_color = points_color[valid]
            if len(points_color) < self.target_min_depth_points:
                return None

            u, v = project_plumb_bob(
                points_color,
                self.color_K,
                self.color_D,
            )
            in_image = (
                np.isfinite(u)
                & np.isfinite(v)
                & (u >= 0.0)
                & (v >= 0.0)
                & (u < float(self.color_width))
                & (v < float(self.color_height))
            )
            return (
                u[in_image],
                v[in_image],
                points_color[in_image, 2],
            )
        except Exception as exc:
            # 目标排序失败不应中断原有识别；后续仍可按检测框面积回退。
            if (
                self.processed_frame_count
                % self.log_every_n_frames
                == 0
            ):
                self.get_logger().warning(
                    "目标点云估深失败，使用框面积回退: "
                    f"{type(exc).__name__}: {exc}"
                )
            return None

    def _estimate_box_depths(
        self,
        boxes_xyxy: np.ndarray,
        projected_cloud: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray]
        ],
    ) -> List[Optional[float]]:
        depths: List[Optional[float]] = [None] * len(boxes_xyxy)
        if projected_cloud is None:
            return depths

        u, v, z = projected_cloud
        scale = self.target_depth_roi_scale

        for index, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            cx = 0.5 * (float(x1) + float(x2))
            cy = 0.5 * (float(y1) + float(y2))
            half_w = 0.5 * max(0.0, float(x2) - float(x1)) * scale
            half_h = 0.5 * max(0.0, float(y2) - float(y1)) * scale

            inside = (
                (u >= cx - half_w)
                & (u <= cx + half_w)
                & (v >= cy - half_h)
                & (v <= cy + half_h)
            )
            roi_z = z[inside]
            if len(roi_z) >= self.target_min_depth_points:
                depths[index] = float(
                    np.percentile(
                        roi_z,
                        self.target_depth_percentile,
                    )
                )

        return depths

    def _select_target_index(
        self,
        result,
        msg,
        image_shape: Sequence[int],
    ) -> int:
        boxes_xyxy = (
            result.boxes.xyxy.detach().cpu().numpy()
        )
        confidences = (
            result.boxes.conf.detach().cpu().numpy()
        )
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64)
        confidences = np.asarray(confidences, dtype=np.float64)

        image_width = float(image_shape[1])
        image_center_x = image_width * 0.5
        box_center_x = 0.5 * (
            boxes_xyxy[:, 0] + boxes_xyxy[:, 2]
        )
        center_errors = np.abs(
            box_center_x - image_center_x
        ) / max(image_center_x, 1.0)

        widths = np.maximum(
            0.0,
            boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
        )
        heights = np.maximum(
            0.0,
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
        )
        areas = widths * heights

        central_indices = [
            index
            for index, error in enumerate(center_errors)
            if error <= self.target_center_band_ratio
        ]
        pool = central_indices or list(range(len(boxes_xyxy)))

        projected_cloud = self._project_cloud_for_target_selection(msg)
        depths = self._estimate_box_depths(
            boxes_xyxy,
            projected_cloud,
        )
        depth_pool = [
            index for index in pool if depths[index] is not None
        ]

        if depth_pool:
            # 中央带是第一优先级；带内先选最小 Z，再用中线误差和置信度破同值。
            best_index = min(
                depth_pool,
                key=lambda index: (
                    float(depths[index]),
                    float(center_errors[index]),
                    -float(confidences[index]),
                ),
            )
            mode = "pointcloud-depth"
        else:
            # 点云不足时，较大的同类箱体投影通常更近。
            best_index = min(
                pool,
                key=lambda index: (
                    -float(areas[index]),
                    float(center_errors[index]),
                    -float(confidences[index]),
                ),
            )
            mode = "bbox-area-fallback"

        if (
            self.processed_frame_count
            % self.log_every_n_frames
            == 0
        ):
            depth_text = (
                f"{depths[best_index]:.3f}m"
                if depths[best_index] is not None
                else "n/a"
            )
            self.get_logger().info(
                "目标选择: "
                f"idx={best_index}/{len(boxes_xyxy) - 1}, "
                f"mode={mode}, "
                f"central_pool={len(central_indices)}, "
                f"center_error={center_errors[best_index]:.3f}, "
                f"depth={depth_text}, "
                f"conf={confidences[best_index]:.3f}"
            )

        return int(best_index)

    def _process_image(self, msg) -> None:
        """只改写多目标选择，选定后完整复用父类的原处理流程。"""
        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8",
        )
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        if not results:
            # 极少见的模型空列表情况仍交给父类按原逻辑处理。
            return super()._process_image(msg)

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if (
            boxes is None
            or len(boxes) == 0
            or keypoints is None
            or len(keypoints) == 0
        ):
            selected_result = result
        else:
            best_index = self._select_target_index(
                result,
                msg,
                image.shape,
            )
            # Ultralytics Results 支持切片；保留原置信度和对应关键点，
            # 父类看到单个候选后便会完整执行原有门控与几何流程。
            selected_result = result[best_index : best_index + 1]

        original_model = self.model
        self.model = _CachedPredictionModel(selected_result)
        try:
            super()._process_image(msg)
        finally:
            self.model = original_model


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[RobustBoxPoseShortEdgeDetector] = None
    try:
        node = RobustBoxPoseShortEdgeDetector()
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
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
