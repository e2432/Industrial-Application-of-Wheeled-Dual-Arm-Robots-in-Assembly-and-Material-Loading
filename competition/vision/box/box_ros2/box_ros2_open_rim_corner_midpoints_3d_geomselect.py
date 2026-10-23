#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot A 箱体 YOLO Pose：顶部平面拟合 -> 4角点3D -> 左右3D中点
================================================================

输入:
  /camera_head/color/image_raw
  /camera_head/depth/points
  /camera_head/color/camera_info

模型:
  /home/niic/Desktop/competition/vision/models/box.pt

输出:
  /box_detection/image
  /box_detection/valid
  /box_detection/center_pixel

  /box_detection/corner_0_pixel
  /box_detection/corner_1_pixel
  /box_detection/corner_2_pixel
  /box_detection/corner_3_pixel

  /box_detection/short_edge_midpoint_1_pixel   # 左边中点
  /box_detection/short_edge_midpoint_2_pixel   # 右边中点

  /box_grasp_point/short_edge_1
  /box_grasp_point/short_edge_2
  /box_grasp_point/valid

算法:
  1. YOLO Pose 读取最高置信度箱体的前4个 keypoints。
  2. 根据图像位置将4点标记为 top_left / top_right / bottom_left / bottom_right。
  3. 将深度 PointCloud 投影到 RGB 图像，并选取顶部四边形内部点。
  4. 对顶部内部点做 RANSAC + SVD 平面拟合。
  5. TL/TR/BL/BR 四个像素分别反投影成相机射线，与顶部平面求交得到4个3D角点。
  6. LEFT_3D=(TL_3D+BL_3D)/2；RIGHT_3D=(TR_3D+BR_3D)/2。
  7. 两个3D中点发布在 camera_head_color_optical_frame，后续 bridge 再做手眼标定。

注意:
  - 本节点只做视觉和3D抓取点生成，不控制机器人。
  - 不再用“2D中点附近取一个深度”作为抓取点；最终抓取点严格来自4个3D角点的三维中点。
  - 如果模型不是 pose 模型，或者少于4个关键点，会发布 valid=false。
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener
from ultralytics import YOLO

try:
    from sensor_msgs_py import point_cloud2
except ImportError as exc:
    raise RuntimeError(
        "缺少 sensor_msgs_py。请确认 ROS 2 Humble 环境已 source。"
    ) from exc


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quaternion_to_rotation_matrix(
    x: float,
    y: float,
    z: float,
    w: float,
) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("四元数长度为0")
    x, y, z, w = q / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def transform_stamped_to_matrix(msg: TransformStamped) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_rotation_matrix(
        q.x, q.y, q.z, q.w
    )
    T[:3, 3] = [
        float(t.x),
        float(t.y),
        float(t.z),
    ]
    return T


def transform_points(
    T: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def pointcloud2_xyz(msg: PointCloud2) -> np.ndarray:
    if hasattr(point_cloud2, "read_points_numpy"):
        try:
            arr = point_cloud2.read_points_numpy(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
            )
            arr = np.asarray(arr)
            if arr.dtype.names:
                pts = np.column_stack(
                    [arr["x"], arr["y"], arr["z"]]
                )
            else:
                pts = arr.reshape(-1, 3)
            return np.asarray(pts, dtype=np.float64)
        except Exception:
            pass

    arr = point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )
    if isinstance(arr, np.ndarray):
        if arr.dtype.names:
            pts = np.column_stack(
                [arr["x"], arr["y"], arr["z"]]
            )
        else:
            pts = np.asarray(arr).reshape(-1, 3)
    else:
        pts = np.asarray(
            list(arr),
            dtype=np.float64,
        ).reshape(-1, 3)
    return np.asarray(pts, dtype=np.float64)


def project_plumb_bob(
    points_color: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X = points_color[:, 0]
    Y = points_color[:, 1]
    Z = points_color[:, 2]

    x = X / Z
    y = Y / Z

    d = np.zeros(8, dtype=np.float64)
    n = min(len(D), 8)
    if n > 0:
        d[:n] = D[:n]

    k1, k2, p1, p2, k3, k4, k5, k6 = d

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    radial_num = (
        1.0 + k1 * r2 + k2 * r4 + k3 * r6
    )
    radial_den = (
        1.0 + k4 * r2 + k5 * r4 + k6 * r6
    )
    radial_den = np.where(
        np.abs(radial_den) < 1e-12,
        1.0,
        radial_den,
    )
    radial = radial_num / radial_den

    x_dist = (
        x * radial
        + 2.0 * p1 * x * y
        + p2 * (r2 + 2.0 * x * x)
    )
    y_dist = (
        y * radial
        + p1 * (r2 + 2.0 * y * y)
        + 2.0 * p2 * x * y
    )

    fx = float(K[0, 0])
    skew = float(K[0, 1])
    cx = float(K[0, 2])
    fy = float(K[1, 1])
    cy = float(K[1, 2])

    u = fx * x_dist + skew * y_dist + cx
    v = fy * y_dist + cy
    return u, v



def pixel_to_camera_ray(
    pixel: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    RGB 像素 -> color optical frame 中的一条相机射线。

    返回 [x, y, 1] 形式的归一化像平面方向。
    使用 cv2.undistortPoints 去除 CameraInfo 中的镜头畸变。
    """
    uv = np.asarray(
        pixel,
        dtype=np.float64,
    ).reshape(1, 1, 2)

    dist = np.asarray(
        D,
        dtype=np.float64,
    ).reshape(-1, 1)

    undistorted = cv2.undistortPoints(
        uv,
        np.asarray(K, dtype=np.float64),
        dist if dist.size > 0 else None,
    )

    x = float(undistorted[0, 0, 0])
    y = float(undistorted[0, 0, 1])

    ray = np.array(
        [x, y, 1.0],
        dtype=np.float64,
    )

    if not np.isfinite(ray).all():
        raise RuntimeError(
            "像素反投影得到 NaN/Inf"
        )

    return ray


def fit_plane_svd(
    points: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    最小二乘拟合平面:
        normal · X + d = 0
    """
    pts = np.asarray(
        points,
        dtype=np.float64,
    ).reshape(-1, 3)

    if len(pts) < 3:
        raise RuntimeError(
            "拟合平面至少需要3个点"
        )

    centroid = np.mean(
        pts,
        axis=0,
    )
    centered = pts - centroid

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    normal = np.asarray(
        vh[-1],
        dtype=np.float64,
    )

    norm = float(
        np.linalg.norm(normal)
    )
    if norm < 1e-12:
        raise RuntimeError(
            "平面法向量退化"
        )

    normal /= norm

    # 统一让法向量大致朝向相机原点，便于日志稳定。
    if float(
        np.dot(normal, centroid)
    ) > 0.0:
        normal = -normal

    d = -float(
        np.dot(
            normal,
            centroid,
        )
    )

    return normal, d


def fit_plane_ransac(
    points: np.ndarray,
    *,
    iterations: int,
    distance_threshold_m: float,
    min_inliers: int,
    max_fit_points: int,
) -> Tuple[
    np.ndarray,
    float,
    np.ndarray,
    dict,
]:
    """
    RANSAC 找主平面，随后用全部内点 SVD 重拟合。
    """
    pts_all = np.asarray(
        points,
        dtype=np.float64,
    ).reshape(-1, 3)

    if len(pts_all) < min_inliers:
        raise RuntimeError(
            "顶部平面候选点不足: "
            f"{len(pts_all)} < {min_inliers}"
        )

    rng = np.random.default_rng(20260810)

    if (
        max_fit_points > 0
        and len(pts_all) > max_fit_points
    ):
        sample_idx = rng.choice(
            len(pts_all),
            size=max_fit_points,
            replace=False,
        )
        pts = pts_all[
            sample_idx
        ]
    else:
        pts = pts_all

    best_count = 0
    best_error = float("inf")
    best_normal = None
    best_d = None

    for _ in range(
        max(20, int(iterations))
    ):
        idx = rng.choice(
            len(pts),
            size=3,
            replace=False,
        )
        a, b, c = pts[idx]

        normal = np.cross(
            b - a,
            c - a,
        )
        n_norm = float(
            np.linalg.norm(normal)
        )
        if n_norm < 1e-9:
            continue

        normal = normal / n_norm
        d = -float(
            np.dot(normal, a)
        )

        distances = np.abs(
            pts @ normal + d
        )
        inliers = (
            distances
            <= distance_threshold_m
        )
        count = int(
            np.count_nonzero(inliers)
        )

        if count == 0:
            continue

        median_error = float(
            np.median(
                distances[inliers]
            )
        )

        if (
            count > best_count
            or (
                count == best_count
                and median_error
                < best_error
            )
        ):
            best_count = count
            best_error = median_error
            best_normal = normal
            best_d = d

    if (
        best_normal is None
        or best_count < min_inliers
    ):
        raise RuntimeError(
            "RANSAC 未找到足够可靠的顶部平面: "
            f"best_inliers={best_count}, "
            f"required={min_inliers}"
        )

    # 在完整点集上重新判定内点。
    distances_all = np.abs(
        pts_all @ best_normal
        + best_d
    )
    inlier_mask = (
        distances_all
        <= distance_threshold_m
    )

    if (
        int(
            np.count_nonzero(
                inlier_mask
            )
        )
        < min_inliers
    ):
        raise RuntimeError(
            "完整点集上的平面内点不足"
        )

    # 第一次 SVD 重拟合。
    normal, d = fit_plane_svd(
        pts_all[inlier_mask]
    )

    # 再做一次残差筛选 + SVD，减少边缘/侧壁污染。
    distances_all = np.abs(
        pts_all @ normal + d
    )
    inlier_mask = (
        distances_all
        <= distance_threshold_m
    )

    inlier_points = pts_all[
        inlier_mask
    ]
    if len(inlier_points) < min_inliers:
        raise RuntimeError(
            "平面精修后内点不足"
        )

    normal, d = fit_plane_svd(
        inlier_points
    )

    residuals = np.abs(
        inlier_points @ normal
        + d
    )
    rms = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    stats = {
        "candidate_points":
            int(len(pts_all)),
        "inlier_points":
            int(len(inlier_points)),
        "inlier_ratio":
            float(
                len(inlier_points)
                / len(pts_all)
            ),
        "rms_m": rms,
        "median_residual_m":
            float(
                np.median(
                    residuals
                )
            ),
    }

    return (
        normal,
        d,
        inlier_mask,
        stats,
    )


def intersect_pixel_ray_with_plane(
    pixel: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    normal: np.ndarray,
    d: float,
) -> np.ndarray:
    """
    像素射线与平面求交，结果位于 camera color optical frame。
    """
    ray = pixel_to_camera_ray(
        pixel,
        K,
        D,
    )

    denom = float(
        np.dot(
            normal,
            ray,
        )
    )
    if abs(denom) < 1e-8:
        raise RuntimeError(
            "像素射线几乎与顶部平面平行"
        )

    scale = -float(d) / denom
    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise RuntimeError(
            "像素射线与顶部平面的交点位于相机后方"
        )

    point = scale * ray

    if not np.isfinite(
        point
    ).all():
        raise RuntimeError(
            "平面交点包含 NaN/Inf"
        )

    return np.asarray(
        point,
        dtype=np.float64,
    )


def label_top_surface_corners(
    corners: np.ndarray,
):
    """
    将顶部表面的4个关键点按图像位置标记为：

        top_left -------- top_right
           |                 |
           |                 |
        bottom_left ----- bottom_right

    规则：
      1) 按图像 v(y) 坐标取上方两个点与下方两个点；
      2) 每一行再按 u(x) 坐标区分 left / right。

    这正对应本项目需要的两个抓取点：
      left_mid  = (top_left  + bottom_left)  / 2
      right_mid = (top_right + bottom_right) / 2
    """
    pts = np.asarray(
        corners,
        dtype=np.float64,
    ).reshape(4, 2)

    if not np.isfinite(pts).all():
        raise ValueError("4个角点包含 NaN/Inf")

    # 先按图像 y(v) 从上到下排序。
    order_y = np.argsort(pts[:, 1])
    top_pair = pts[order_y[:2]]
    bottom_pair = pts[order_y[2:]]

    # 每组再按图像 x(u) 从左到右排序。
    top_pair = top_pair[
        np.argsort(top_pair[:, 0])
    ]
    bottom_pair = bottom_pair[
        np.argsort(bottom_pair[:, 0])
    ]

    top_left = top_pair[0]
    top_right = top_pair[1]
    bottom_left = bottom_pair[0]
    bottom_right = bottom_pair[1]

    # 退化检查：顶部/底部左右点不应重合。
    eps_px = 2.0
    pairs = [
        ("top", top_left, top_right),
        ("bottom", bottom_left, bottom_right),
        ("left", top_left, bottom_left),
        ("right", top_right, bottom_right),
    ]
    for name, a, b in pairs:
        if float(np.linalg.norm(a - b)) < eps_px:
            raise ValueError(
                f"{name} 边长度过小，4角点可能识别异常"
            )

    return (
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    )


def side_midpoints_from_labeled_corners(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
):
    """
    本项目最终需要的两个抓取像素：

      left_mid  = midpoint(top_left,  bottom_left)
      right_mid = midpoint(top_right, bottom_right)
    """
    left_mid = 0.5 * (
        np.asarray(top_left, dtype=np.float64)
        + np.asarray(bottom_left, dtype=np.float64)
    )
    right_mid = 0.5 * (
        np.asarray(top_right, dtype=np.float64)
        + np.asarray(bottom_right, dtype=np.float64)
    )

    return left_mid, right_mid


class BoxPoseShortEdgeDetector(Node):
    def __init__(self) -> None:
        super().__init__(
            "box_pose_short_edge_detector"
        )

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------
        self.declare_parameter(
            "model_path",
            "/home/niic/Desktop/competition/vision/models/box.pt",
        )
        self.declare_parameter(
            "color_topic",
            "/camera_head/color/image_raw",
        )
        self.declare_parameter(
            "points_topic",
            "/camera_head/depth/points",
        )
        self.declare_parameter(
            "color_info_topic",
            "/camera_head/color/camera_info",
        )
        self.declare_parameter(
            "color_frame",
            "camera_head_color_optical_frame",
        )

        self.declare_parameter(
            "debug_image_topic",
            "/box_detection/image",
        )
        self.declare_parameter(
            "valid_topic",
            "/box_detection/valid",
        )
        self.declare_parameter(
            "grasp_valid_topic",
            "/box_grasp_point/valid",
        )
        self.declare_parameter(
            "center_topic",
            "/box_detection/center_pixel",
        )

        self.declare_parameter(
            "short_edge_1_topic",
            "/box_grasp_point/short_edge_1",
        )
        self.declare_parameter(
            "short_edge_2_topic",
            "/box_grasp_point/short_edge_2",
        )
        self.declare_parameter(
            "midpoint_1_pixel_topic",
            "/box_detection/short_edge_midpoint_1_pixel",
        )
        self.declare_parameter(
            "midpoint_2_pixel_topic",
            "/box_detection/short_edge_midpoint_2_pixel",
        )

        self.declare_parameter(
            "confidence",
            0.50,
        )
        self.declare_parameter(
            "keypoint_confidence",
            0.25,
        )
        self.declare_parameter(
            "imgsz",
            640,
        )
        self.declare_parameter(
            "device",
            "0",
        )

        # 开口箱上沿平面拟合参数。
        # 不再使用整个开口内部（容易拟合到箱底），而只使用四条上沿附近的内侧环带。
        # rim_band_width_px 表示从检测到的箱口轮廓向内取多宽的点云带。
        self.declare_parameter(
            "rim_band_width_px",
            10,
        )

        # 环带中可能同时包含箱口、内壁和箱底多个平面。
        # 不再只选“RANSAC内点最多”的平面，而是提取多个候选平面，
        # 再用四个3D角点是否近似构成矩形来选择真正的箱口平面。
        self.declare_parameter(
            "plane_candidate_count",
            3,
        )
        self.declare_parameter(
            "max_opposite_edge_rel_error",
            0.18,
        )
        self.declare_parameter(
            "max_corner_orthogonality_cos",
            0.25,
        )
        self.declare_parameter(
            "plane_min_points",
            80,
        )
        self.declare_parameter(
            "plane_ransac_iterations",
            180,
        )
        self.declare_parameter(
            "plane_ransac_threshold_m",
            0.012,
        )
        self.declare_parameter(
            "plane_min_inlier_ratio",
            0.50,
        )
        self.declare_parameter(
            "plane_max_rms_m",
            0.012,
        )
        self.declare_parameter(
            "plane_max_fit_points",
            5000,
        )

        self.declare_parameter(
            "min_depth_m",
            0.15,
        )
        self.declare_parameter(
            "max_depth_m",
            2.00,
        )
        self.declare_parameter(
            "max_sync_dt_s",
            0.03,
        )
        self.declare_parameter(
            "cloud_buffer_size",
            8,
        )
        self.declare_parameter(
            "tf_timeout_s",
            0.20,
        )
        self.declare_parameter(
            "log_every_n_frames",
            5,
        )

        self.model_path = str(
            self.get_parameter(
                "model_path"
            ).value
        )
        self.color_topic = str(
            self.get_parameter(
                "color_topic"
            ).value
        )
        self.points_topic = str(
            self.get_parameter(
                "points_topic"
            ).value
        )
        self.color_info_topic = str(
            self.get_parameter(
                "color_info_topic"
            ).value
        )
        self.color_frame = str(
            self.get_parameter(
                "color_frame"
            ).value
        )

        self.debug_image_topic = str(
            self.get_parameter(
                "debug_image_topic"
            ).value
        )
        self.valid_topic = str(
            self.get_parameter(
                "valid_topic"
            ).value
        )
        self.grasp_valid_topic = str(
            self.get_parameter(
                "grasp_valid_topic"
            ).value
        )
        self.center_topic = str(
            self.get_parameter(
                "center_topic"
            ).value
        )
        self.short_edge_1_topic = str(
            self.get_parameter(
                "short_edge_1_topic"
            ).value
        )
        self.short_edge_2_topic = str(
            self.get_parameter(
                "short_edge_2_topic"
            ).value
        )
        self.midpoint_1_pixel_topic = str(
            self.get_parameter(
                "midpoint_1_pixel_topic"
            ).value
        )
        self.midpoint_2_pixel_topic = str(
            self.get_parameter(
                "midpoint_2_pixel_topic"
            ).value
        )

        self.confidence = float(
            self.get_parameter(
                "confidence"
            ).value
        )
        self.keypoint_confidence = float(
            self.get_parameter(
                "keypoint_confidence"
            ).value
        )
        self.imgsz = int(
            self.get_parameter(
                "imgsz"
            ).value
        )
        self.device = str(
            self.get_parameter(
                "device"
            ).value
        )

        self.rim_band_width_px = max(
            1,
            int(
                self.get_parameter(
                    "rim_band_width_px"
                ).value
            ),
        )
        self.plane_candidate_count = max(
            1,
            int(
                self.get_parameter(
                    "plane_candidate_count"
                ).value
            ),
        )
        self.max_opposite_edge_rel_error = float(
            self.get_parameter(
                "max_opposite_edge_rel_error"
            ).value
        )
        self.max_corner_orthogonality_cos = float(
            self.get_parameter(
                "max_corner_orthogonality_cos"
            ).value
        )
        self.plane_min_points = max(
            20,
            int(
                self.get_parameter(
                    "plane_min_points"
                ).value
            ),
        )
        self.plane_ransac_iterations = max(
            20,
            int(
                self.get_parameter(
                    "plane_ransac_iterations"
                ).value
            ),
        )
        self.plane_ransac_threshold_m = float(
            self.get_parameter(
                "plane_ransac_threshold_m"
            ).value
        )
        self.plane_min_inlier_ratio = float(
            self.get_parameter(
                "plane_min_inlier_ratio"
            ).value
        )
        self.plane_max_rms_m = float(
            self.get_parameter(
                "plane_max_rms_m"
            ).value
        )
        self.plane_max_fit_points = max(
            0,
            int(
                self.get_parameter(
                    "plane_max_fit_points"
                ).value
            ),
        )

        self.min_depth_m = float(
            self.get_parameter(
                "min_depth_m"
            ).value
        )
        self.max_depth_m = float(
            self.get_parameter(
                "max_depth_m"
            ).value
        )

        if self.plane_ransac_threshold_m <= 0.0:
            raise RuntimeError(
                "plane_ransac_threshold_m 必须 > 0"
            )
        if not 0.0 < self.plane_min_inlier_ratio <= 1.0:
            raise RuntimeError(
                "plane_min_inlier_ratio 必须在 (0,1] 内"
            )
        if self.plane_max_rms_m <= 0.0:
            raise RuntimeError(
                "plane_max_rms_m 必须 > 0"
            )
        self.max_sync_dt_s = float(
            self.get_parameter(
                "max_sync_dt_s"
            ).value
        )
        self.cloud_buffer_size = max(
            2,
            int(
                self.get_parameter(
                    "cloud_buffer_size"
                ).value
            ),
        )
        self.tf_timeout_s = float(
            self.get_parameter(
                "tf_timeout_s"
            ).value
        )
        self.log_every_n_frames = max(
            1,
            int(
                self.get_parameter(
                    "log_every_n_frames"
                ).value
            ),
        )

        # --------------------------------------------------------------
        # YOLO
        # --------------------------------------------------------------
        self.bridge = CvBridge()

        self.get_logger().info(
            f"加载箱体 Pose 模型: {self.model_path}"
        )
        self.model = YOLO(
            self.model_path
        )
        self.get_logger().info(
            f"模型任务: {self.model.task}"
        )
        self.get_logger().info(
            f"模型类别: {self.model.names}"
        )

        if str(
            getattr(
                self.model,
                "task",
                "",
            )
        ) != "pose":
            self.get_logger().warning(
                "⚠️ 当前模型 task 不是 pose。"
                "若 result.keypoints 为空，本节点会拒绝输出抓取点。"
            )

        # --------------------------------------------------------------
        # TF / camera
        # --------------------------------------------------------------
        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=10.0)
        )
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        self.color_K: Optional[
            np.ndarray
        ] = None
        self.color_D: Optional[
            np.ndarray
        ] = None
        self.color_width: Optional[int] = None
        self.color_height: Optional[int] = None

        self.cloud_lock = threading.Lock()
        self.cloud_buffer: Deque[
            PointCloud2
        ] = deque(
            maxlen=self.cloud_buffer_size
        )

        # --------------------------------------------------------------
        # ROS I/O
        # --------------------------------------------------------------
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.color_sub = self.create_subscription(
            Image,
            self.color_topic,
            self._on_image,
            camera_qos,
        )
        self.points_sub = self.create_subscription(
            PointCloud2,
            self.points_topic,
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.color_info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
        )

        self.debug_pub = self.create_publisher(
            Image,
            self.debug_image_topic,
            10,
        )
        self.valid_pub = self.create_publisher(
            Bool,
            self.valid_topic,
            10,
        )
        self.grasp_valid_pub = (
            self.create_publisher(
                Bool,
                self.grasp_valid_topic,
                10,
            )
        )
        self.center_pub = self.create_publisher(
            PointStamped,
            self.center_topic,
            10,
        )

        self.mid1_pixel_pub = (
            self.create_publisher(
                PointStamped,
                self.midpoint_1_pixel_topic,
                10,
            )
        )
        self.mid2_pixel_pub = (
            self.create_publisher(
                PointStamped,
                self.midpoint_2_pixel_topic,
                10,
            )
        )
        self.short1_pub = self.create_publisher(
            PointStamped,
            self.short_edge_1_topic,
            10,
        )
        self.short2_pub = self.create_publisher(
            PointStamped,
            self.short_edge_2_topic,
            10,
        )

        # 额外发布4个顶部角点的3D坐标，便于检查平面反投影是否正确。
        self.corner_3d_pubs = {
            "top_left": self.create_publisher(
                PointStamped,
                "/box_grasp_point/top_left_3d",
                10,
            ),
            "top_right": self.create_publisher(
                PointStamped,
                "/box_grasp_point/top_right_3d",
                10,
            ),
            "bottom_left": self.create_publisher(
                PointStamped,
                "/box_grasp_point/bottom_left_3d",
                10,
            ),
            "bottom_right": self.create_publisher(
                PointStamped,
                "/box_grasp_point/bottom_right_3d",
                10,
            ),
        }

        self.corner_pubs = [
            self.create_publisher(
                PointStamped,
                f"/box_detection/corner_{i}_pixel",
                10,
            )
            for i in range(4)
        ]

        # 同时发布明确语义的4个顶部角点，便于现场核对。
        self.top_left_pub = self.create_publisher(
            PointStamped,
            "/box_detection/top_left_pixel",
            10,
        )
        self.top_right_pub = self.create_publisher(
            PointStamped,
            "/box_detection/top_right_pixel",
            10,
        )
        self.bottom_left_pub = self.create_publisher(
            PointStamped,
            "/box_detection/bottom_left_pixel",
            10,
        )
        self.bottom_right_pub = self.create_publisher(
            PointStamped,
            "/box_detection/bottom_right_pixel",
            10,
        )

        # --------------------------------------------------------------
        # Latest-frame inference worker
        # --------------------------------------------------------------
        self.frame_condition = (
            threading.Condition()
        )
        self.latest_image_msg: Optional[
            Image
        ] = None
        self.worker_stop = False
        self.processed_frame_count = 0

        self.inference_thread = (
            threading.Thread(
                target=self._inference_loop,
                name="box_pose_inference",
                daemon=True,
            )
        )
        self.inference_thread.start()

        self.get_logger().info(
            "===================================================="
        )
        self.get_logger().info(
            "箱体 Pose 顶部平面 -> 4角点3D -> 左右3D中点 节点已启动"
        )
        self.get_logger().info(
            f"RGB: {self.color_topic}"
        )
        self.get_logger().info(
            f"PointCloud: {self.points_topic}"
        )
        self.get_logger().info(
            f"输出3D（1=左边中点，2=右边中点）: {self.short_edge_1_topic}, "
            f"{self.short_edge_2_topic}"
        )
        self.get_logger().info(
            "本节点不发送任何机器人运动命令"
        )
        self.get_logger().info(
            "3D算法: 箱口四边环带点云 RANSAC平面 -> "
            "4角点射线/平面交点 -> "
            "LEFT=(TL3D+BL3D)/2, RIGHT=(TR3D+BR3D)/2"
        )
        self.get_logger().info(
            "===================================================="
        )

    def _on_camera_info(
        self,
        msg: CameraInfo,
    ) -> None:
        self.color_K = np.asarray(
            msg.k,
            dtype=np.float64,
        ).reshape(3, 3)
        self.color_D = np.asarray(
            msg.d,
            dtype=np.float64,
        )
        self.color_width = int(msg.width)
        self.color_height = int(msg.height)

        if self.info_sub is not None:
            self.destroy_subscription(
                self.info_sub
            )
            self.info_sub = None

        self.get_logger().info(
            "✅ 已读取 RGB CameraInfo: "
            f"{self.color_width}x{self.color_height}"
        )

    def _on_cloud(
        self,
        msg: PointCloud2,
    ) -> None:
        if not str(
            msg.header.frame_id
        ).strip():
            return
        with self.cloud_lock:
            self.cloud_buffer.append(msg)

    def _on_image(
        self,
        msg: Image,
    ) -> None:
        with self.frame_condition:
            self.latest_image_msg = msg
            self.frame_condition.notify()

    def _inference_loop(self) -> None:
        while True:
            with self.frame_condition:
                self.frame_condition.wait_for(
                    lambda: (
                        self.worker_stop
                        or self.latest_image_msg
                        is not None
                    )
                )
                if self.worker_stop:
                    return

                msg = self.latest_image_msg
                self.latest_image_msg = None

            if msg is None:
                continue

            try:
                self._process_image(msg)
            except Exception as exc:
                # Ctrl+C 后 rclpy context 可能已经 shutdown，此时不要继续 publish/log。
                if not rclpy.ok():
                    return
                self.get_logger().error(
                    "Pose/3D处理失败: "
                    f"{type(exc).__name__}: {exc}"
                )
                if rclpy.ok():
                    self._publish_valid(False)

    def _find_nearest_cloud(
        self,
        stamp_s: float,
    ) -> Tuple[
        Optional[PointCloud2],
        float,
    ]:
        with self.cloud_lock:
            clouds = list(
                self.cloud_buffer
            )

        if not clouds:
            return None, float("inf")

        best_msg = None
        best_dt = float("inf")

        for cloud in clouds:
            dt = abs(
                stamp_to_seconds(
                    cloud.header.stamp
                )
                - stamp_s
            )
            if dt < best_dt:
                best_dt = dt
                best_msg = cloud

        return best_msg, best_dt

    def _lookup_matrix(
        self,
        target_frame: str,
        source_frame: str,
        stamp,
    ) -> np.ndarray:
        try:
            tf_msg = (
                self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time.from_msg(stamp),
                    timeout=Duration(
                        seconds=self.tf_timeout_s
                    ),
                )
            )
        except TransformException:
            # 相机内部通常是静态 TF，允许 latest fallback。
            tf_msg = (
                self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(
                        seconds=self.tf_timeout_s
                    ),
                )
            )

        return transform_stamped_to_matrix(
            tf_msg
        )

    def _process_image(
        self,
        msg: Image,
    ) -> None:
        self.processed_frame_count += 1

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8",
        )
        annotated = image.copy()

        result = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]

        boxes = result.boxes
        keypoints = result.keypoints

        if (
            boxes is None
            or len(boxes) == 0
            or keypoints is None
            or len(keypoints) == 0
        ):
            self._draw_status(
                annotated,
                "NO POSE",
            )
            self._publish_debug(
                annotated,
                msg,
            )
            self._publish_valid(False)
            return

        confidences = (
            boxes.conf.detach()
            .cpu()
            .numpy()
        )
        best_index = int(
            np.argmax(confidences)
        )
        detection_conf = float(
            confidences[best_index]
        )

        kpts = (
            keypoints.xy[best_index]
            .detach()
            .cpu()
            .numpy()
        )
        kpts = np.asarray(
            kpts,
            dtype=np.float64,
        )

        if (
            kpts.ndim != 2
            or kpts.shape[0] < 4
            or kpts.shape[1] < 2
        ):
            self.get_logger().warning(
                "Pose关键点不足4个: "
                f"shape={kpts.shape}"
            )
            self._publish_valid(False)
            return

        corners_raw = kpts[:4, :2]

        # 如果模型输出每个关键点置信度，则做硬门控。
        kp_conf_obj = getattr(
            keypoints,
            "conf",
            None,
        )
        if kp_conf_obj is not None:
            kp_conf = (
                kp_conf_obj[best_index]
                .detach()
                .cpu()
                .numpy()
            )
            kp_conf = np.asarray(
                kp_conf,
                dtype=np.float64,
            ).reshape(-1)

            if (
                len(kp_conf) >= 4
                and np.any(
                    kp_conf[:4]
                    < self.keypoint_confidence
                )
            ):
                self.get_logger().warning(
                    "4角点中存在低置信度关键点: "
                    f"{np.round(kp_conf[:4],3)}"
                )
                self._publish_valid(False)
                return

        if not np.isfinite(
            corners_raw
        ).all():
            self._publish_valid(False)
            return

        (
            top_left,
            top_right,
            bottom_left,
            bottom_right,
        ) = label_top_surface_corners(
            corners_raw
        )

        left_mid, right_mid = (
            side_midpoints_from_labeled_corners(
                top_left,
                top_right,
                bottom_left,
                bottom_right,
            )
        )

        # 按 TL -> TR -> BR -> BL 组成顶部四边形，
        # 用于可视化与点云 ROI 的箱体内部约束。
        corners = np.asarray(
            [
                top_left,
                top_right,
                bottom_right,
                bottom_left,
            ],
            dtype=np.float64,
        )

        box_center = np.mean(
            corners,
            axis=0,
        )

        # 像素输出
        self._publish_pixel(
            self.center_pub,
            box_center,
            msg,
        )
        # corner_0..3 的固定顺序:
        #   0=top_left, 1=top_right, 2=bottom_right, 3=bottom_left
        for pub, point in zip(
            self.corner_pubs,
            corners,
        ):
            self._publish_pixel(
                pub,
                point,
                msg,
            )

        # 再发布带明确名称的4个角点。
        for pub, point in (
            (self.top_left_pub, top_left),
            (self.top_right_pub, top_right),
            (self.bottom_left_pub, bottom_left),
            (self.bottom_right_pub, bottom_right),
        ):
            self._publish_pixel(
                pub,
                point,
                msg,
            )

        # short_edge_1 明确对应图像左侧：
        # midpoint(top_left, bottom_left)
        self._publish_pixel(
            self.mid1_pixel_pub,
            left_mid,
            msg,
        )

        # short_edge_2 明确对应图像右侧：
        # midpoint(top_right, bottom_right)
        self._publish_pixel(
            self.mid2_pixel_pub,
            right_mid,
            msg,
        )

        # 调试图
        self._draw_pose_geometry(
            annotated,
            top_left,
            top_right,
            bottom_left,
            bottom_right,
            left_mid,
            right_mid,
            detection_conf,
        )

        # 没有 CameraInfo/点云时仍可看2D调试图，但不发布3D有效点。
        if (
            self.color_K is None
            or self.color_D is None
            or self.color_width is None
            or self.color_height is None
        ):
            self._draw_status(
                annotated,
                "WAIT CAMERA_INFO",
            )
            self._publish_debug(
                annotated,
                msg,
            )
            self._publish_valid(False)
            return

        cloud, sync_dt = (
            self._find_nearest_cloud(
                stamp_to_seconds(
                    msg.header.stamp
                )
            )
        )

        if cloud is None:
            self._draw_status(
                annotated,
                "WAIT POINTCLOUD",
            )
            self._publish_debug(
                annotated,
                msg,
            )
            self._publish_valid(False)
            return

        if sync_dt > self.max_sync_dt_s:
            self._draw_status(
                annotated,
                f"CLOUD DT {sync_dt*1000:.0f}ms",
            )
            self._publish_debug(
                annotated,
                msg,
            )
            self._publish_valid(False)
            return

        (
            left_3d,
            right_3d,
            corner_3d,
            plane_stats,
        ) = self._localize_top_plane_corners(
            cloud,
            top_left,
            top_right,
            bottom_left,
            bottom_right,
        )

        out_stamp = cloud.header.stamp

        self._publish_3d(
            self.short1_pub,
            left_3d,
            out_stamp,
        )
        self._publish_3d(
            self.short2_pub,
            right_3d,
            out_stamp,
        )

        for name, point_3d in (
            ("top_left", corner_3d["top_left"]),
            ("top_right", corner_3d["top_right"]),
            ("bottom_left", corner_3d["bottom_left"]),
            ("bottom_right", corner_3d["bottom_right"]),
        ):
            self._publish_3d(
                self.corner_3d_pubs[name],
                point_3d,
                out_stamp,
            )

        self._publish_valid(True)
        self._publish_debug(
            annotated,
            msg,
        )

        if (
            self.processed_frame_count
            % self.log_every_n_frames
            == 0
        ):
            self.get_logger().info(
                "\n"
                "======= BOX TOP-SURFACE GRASP POINTS =======\n"
                f"det conf       : {detection_conf:.3f}\n"
                f"top_left px    : {np.round(top_left,1)}\n"
                f"top_right px   : {np.round(top_right,1)}\n"
                f"bottom_left px : {np.round(bottom_left,1)}\n"
                f"bottom_right px: {np.round(bottom_right,1)}\n"
                f"LEFT midpoint  : {np.round(left_mid,1)} px "
                "= (TL + BL)/2\n"
                f"RIGHT midpoint : {np.round(right_mid,1)} px "
                "= (TR + BR)/2\n"
                f"cloud dt       : {sync_dt*1000.0:.1f}ms\n"
                f"TL 3D          : {np.round(corner_3d['top_left'],4)} m\n"
                f"TR 3D          : {np.round(corner_3d['top_right'],4)} m\n"
                f"BL 3D          : {np.round(corner_3d['bottom_left'],4)} m\n"
                f"BR 3D          : {np.round(corner_3d['bottom_right'],4)} m\n"
                f"LEFT 3D        : {np.round(left_3d,4)} m = (TL3D+BL3D)/2\n"
                f"RIGHT 3D       : {np.round(right_3d,4)} m = (TR3D+BR3D)/2\n"
                f"frame          : {self.color_frame}\n"
                f"plane stats    : {plane_stats}\n"
                "============================================"
            )

    def _localize_top_plane_corners(
        self,
        cloud_msg: PointCloud2,
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
    ):
        """
        新算法：

          PointCloud -> RGB 投影
               ↓
          箱口四边附近内侧环带点
               ↓
          RANSAC + SVD 顶部平面
               ↓
          TL/TR/BL/BR 像素射线与平面求交
               ↓
          TL3D/TR3D/BL3D/BR3D
               ↓
          LEFT  = (TL3D + BL3D) / 2
          RIGHT = (TR3D + BR3D) / 2

        注意：
        这里不再在 2D 中点附近取局部深度，因此透视情况下
        “2D 中点 != 3D 中点”的问题被消除。
        """
        points_cloud = pointcloud2_xyz(
            cloud_msg
        )

        finite = np.isfinite(
            points_cloud
        ).all(axis=1)
        z_ok = (
            (
                points_cloud[:, 2]
                >= self.min_depth_m
            )
            & (
                points_cloud[:, 2]
                <= self.max_depth_m
            )
        )
        points_cloud = points_cloud[
            finite & z_ok
        ]

        if (
            len(points_cloud)
            < self.plane_min_points
        ):
            raise RuntimeError(
                "有效点云数量过少: "
                f"{len(points_cloud)} < "
                f"{self.plane_min_points}"
            )

        cloud_frame = str(
            cloud_msg.header.frame_id
        ).strip()
        if not cloud_frame:
            raise RuntimeError(
                "PointCloud frame_id为空"
            )

        # ----------------------------------------------------------
        # 1) PointCloud 实际 frame -> RGB color optical
        # ----------------------------------------------------------
        if cloud_frame == self.color_frame:
            points_color = (
                points_cloud.copy()
            )
        else:
            T_color_cloud = (
                self._lookup_matrix(
                    self.color_frame,
                    cloud_frame,
                    cloud_msg.header.stamp,
                )
            )
            points_color = (
                transform_points(
                    T_color_cloud,
                    points_cloud,
                )
            )

        front = (
            np.isfinite(
                points_color
            ).all(axis=1)
            & (
                points_color[:, 2]
                > 1e-6
            )
        )
        points_color = points_color[
            front
        ]

        if (
            len(points_color)
            < self.plane_min_points
        ):
            raise RuntimeError(
                "变换到 color optical 后有效点不足"
            )

        # ----------------------------------------------------------
        # 2) 投影到 RGB 图像
        # ----------------------------------------------------------
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
            & (
                u
                < float(self.color_width)
            )
            & (
                v
                < float(self.color_height)
            )
        )

        ui = np.clip(
            np.round(u).astype(np.int64),
            0,
            self.color_width - 1,
        )
        vi = np.clip(
            np.round(v).astype(np.int64),
            0,
            self.color_height - 1,
        )

        # ----------------------------------------------------------
        # 3) 开口箱四条上沿附近的“内侧环带”区域
        # ----------------------------------------------------------
        # polygon 顺序必须沿轮廓一圈:
        # TL -> TR -> BR -> BL
        #
        # 旧实现使用整个四边形内部做平面拟合。对于开口箱，内部大部分
        # 点云来自箱底/内壁，RANSAC 很容易把箱底误认为“顶部平面”。
        #
        # 新实现只保留 detected rim 向箱内侧的一圈窄带：
        #
        #     ################
        #     ##            ##
        #     ##   opening  ##
        #     ##            ##
        #     ################
        #
        # 这样中央箱底不会参与平面拟合，同时不引入多边形外部背景。
        polygon = np.round(
            np.asarray(
                [
                    top_left,
                    top_right,
                    bottom_right,
                    bottom_left,
                ],
                dtype=np.float64,
            )
        ).astype(np.int32)

        full_mask = np.zeros(
            (
                self.color_height,
                self.color_width,
            ),
            dtype=np.uint8,
        )
        cv2.fillPoly(
            full_mask,
            [polygon],
            255,
        )

        # 向内腐蚀后，用 full - inner 得到箱口内侧环带。
        k = 2 * self.rim_band_width_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k),
        )
        inner_mask = cv2.erode(
            full_mask,
            kernel,
        )
        rim_mask = cv2.subtract(
            full_mask,
            inner_mask,
        )

        inside_rim = (
            in_image
            & (
                rim_mask[vi, ui]
                > 0
            )
        )

        top_points = points_color[
            inside_rim
        ]

        if (
            len(top_points)
            < self.plane_min_points
        ):
            raise RuntimeError(
                "箱口环带点云不足: "
                f"{len(top_points)} < "
                f"{self.plane_min_points}; "
                f"rim_band_width_px={self.rim_band_width_px}"
            )

        # ----------------------------------------------------------
        # 4) 环带中提取多个候选平面，再用箱口矩形几何约束选最佳平面
        # ----------------------------------------------------------
        #
        # 开口箱的环带点云通常混有：
        #   - 真正的箱口上沿所在平面
        #   - 内侧壁
        #   - 箱底
        #
        # 单纯选择 RANSAC 内点最多的平面会在这些平面之间跳变。
        # 因此这里最多连续提取 plane_candidate_count 个主平面，
        # 每个平面都把 4 个 YOLO 角点射线投到该平面上，然后检查：
        #   1) 上/下对边长度是否接近
        #   2) 左/右对边长度是否接近
        #   3) 四个角处相邻边是否接近 90°
        #   4) 平面本身的 RMS
        #
        # 真正的箱口四角在 3D 中应该近似构成矩形。
        remaining_points = np.asarray(
            top_points,
            dtype=np.float64,
        )
        plane_candidates = []

        def _corner_intersections_for_plane(
            candidate_normal: np.ndarray,
            candidate_d: float,
        ):
            pts = {
                "top_left":
                    intersect_pixel_ray_with_plane(
                        top_left,
                        self.color_K,
                        self.color_D,
                        candidate_normal,
                        candidate_d,
                    ),
                "top_right":
                    intersect_pixel_ray_with_plane(
                        top_right,
                        self.color_K,
                        self.color_D,
                        candidate_normal,
                        candidate_d,
                    ),
                "bottom_left":
                    intersect_pixel_ray_with_plane(
                        bottom_left,
                        self.color_K,
                        self.color_D,
                        candidate_normal,
                        candidate_d,
                    ),
                "bottom_right":
                    intersect_pixel_ray_with_plane(
                        bottom_right,
                        self.color_K,
                        self.color_D,
                        candidate_normal,
                        candidate_d,
                    ),
            }

            for p3 in pts.values():
                if (
                    float(p3[2]) < self.min_depth_m
                    or float(p3[2]) > self.max_depth_m
                ):
                    raise RuntimeError(
                        "候选平面角点深度越界"
                    )
            return pts

        def _geometry_metrics(pts: dict):
            tl = pts["top_left"]
            tr = pts["top_right"]
            bl = pts["bottom_left"]
            br = pts["bottom_right"]

            v_top = tr - tl
            v_bottom = br - bl
            v_left = bl - tl
            v_right = br - tr

            top_len = float(np.linalg.norm(v_top))
            bottom_len = float(np.linalg.norm(v_bottom))
            left_len = float(np.linalg.norm(v_left))
            right_len = float(np.linalg.norm(v_right))

            eps = 1e-9
            long_rel = abs(top_len - bottom_len) / max(
                0.5 * (top_len + bottom_len),
                eps,
            )
            short_rel = abs(left_len - right_len) / max(
                0.5 * (left_len + right_len),
                eps,
            )

            def _abs_cos(a, b):
                denom = float(
                    np.linalg.norm(a)
                    * np.linalg.norm(b)
                )
                if denom < eps:
                    return 1.0
                return abs(
                    float(np.dot(a, b)) / denom
                )

            corner_cos = [
                _abs_cos(v_top, v_left),
                _abs_cos(-v_top, v_right),
                _abs_cos(v_bottom, v_left),
                _abs_cos(-v_bottom, v_right),
            ]
            max_corner_cos = float(max(corner_cos))

            return {
                "left_edge_m": left_len,
                "right_edge_m": right_len,
                "top_edge_m": top_len,
                "bottom_edge_m": bottom_len,
                "long_opposite_rel_error": float(long_rel),
                "short_opposite_rel_error": float(short_rel),
                "max_corner_orthogonality_cos":
                    max_corner_cos,
            }

        for candidate_index in range(
            self.plane_candidate_count
        ):
            if (
                len(remaining_points)
                < self.plane_min_points
            ):
                break

            try:
                (
                    candidate_normal,
                    candidate_d,
                    candidate_inlier_mask,
                    candidate_stats,
                ) = fit_plane_ransac(
                    remaining_points,
                    iterations=(
                        self.plane_ransac_iterations
                    ),
                    distance_threshold_m=(
                        self.plane_ransac_threshold_m
                    ),
                    min_inliers=(
                        self.plane_min_points
                    ),
                    max_fit_points=(
                        self.plane_max_fit_points
                    ),
                )
            except Exception:
                break

            try:
                candidate_corner_3d = (
                    _corner_intersections_for_plane(
                        candidate_normal,
                        candidate_d,
                    )
                )
                geom = _geometry_metrics(
                    candidate_corner_3d
                )
            except Exception:
                geom = None
                candidate_corner_3d = None

            if geom is not None:
                long_rel = float(
                    geom[
                        "long_opposite_rel_error"
                    ]
                )
                short_rel = float(
                    geom[
                        "short_opposite_rel_error"
                    ]
                )
                ortho_cos = float(
                    geom[
                        "max_corner_orthogonality_cos"
                    ]
                )
                rms = float(
                    candidate_stats["rms_m"]
                )

                geometry_valid = (
                    long_rel
                    <= self.max_opposite_edge_rel_error
                    and short_rel
                    <= self.max_opposite_edge_rel_error
                    and ortho_cos
                    <= self.max_corner_orthogonality_cos
                )

                # 几何优先，RMS 次之。
                # 对边误差和直角误差均为无量纲；RMS 转成米后给较小权重。
                geometry_score = (
                    long_rel
                    + short_rel
                    + 0.75 * ortho_cos
                    + 5.0 * rms
                )

                plane_candidates.append(
                    {
                        "candidate_index":
                            candidate_index,
                        "normal":
                            np.asarray(
                                candidate_normal,
                                dtype=np.float64,
                            ),
                        "d":
                            float(candidate_d),
                        "stats":
                            dict(candidate_stats),
                        "corner_3d":
                            candidate_corner_3d,
                        "geom":
                            geom,
                        "geometry_valid":
                            bool(geometry_valid),
                        "score":
                            float(geometry_score),
                    }
                )

            # 无论该平面几何是否合格，都把其内点去掉，再寻找下一个物理平面。
            candidate_inlier_mask = np.asarray(
                candidate_inlier_mask,
                dtype=bool,
            )
            if (
                len(candidate_inlier_mask)
                != len(remaining_points)
            ):
                break

            next_points = remaining_points[
                ~candidate_inlier_mask
            ]
            if (
                len(next_points)
                >= len(remaining_points)
            ):
                break
            remaining_points = next_points

        valid_candidates = [
            c for c in plane_candidates
            if c["geometry_valid"]
        ]

        if not valid_candidates:
            summary = []
            for c in plane_candidates:
                g = c["geom"]
                summary.append(
                    {
                        "idx": c["candidate_index"],
                        "score": round(
                            c["score"], 4
                        ),
                        "opp_long": round(
                            g[
                                "long_opposite_rel_error"
                            ],
                            3,
                        ),
                        "opp_short": round(
                            g[
                                "short_opposite_rel_error"
                            ],
                            3,
                        ),
                        "ortho_cos": round(
                            g[
                                "max_corner_orthogonality_cos"
                            ],
                            3,
                        ),
                        "normal": np.round(
                            c["normal"],
                            3,
                        ).tolist(),
                    }
                )
            raise RuntimeError(
                "没有候选平面通过箱口矩形几何约束: "
                f"{summary}"
            )

        best = min(
            valid_candidates,
            key=lambda c: c["score"],
        )

        normal = best["normal"]
        d = best["d"]
        stats = best["stats"]
        corner_3d = best["corner_3d"]
        geom = best["geom"]

        # ----------------------------------------------------------
        # 5) 真正的三维左右抓取中点
        # ----------------------------------------------------------
        left_3d = 0.5 * (
            corner_3d["top_left"]
            + corner_3d["bottom_left"]
        )
        right_3d = 0.5 * (
            corner_3d["top_right"]
            + corner_3d["bottom_right"]
        )

        stats.update(
            {
                "roi_mode":
                    "inner_rim_band_multi_plane",
                "rim_band_width_px":
                    int(self.rim_band_width_px),
                "candidate_planes_found":
                    int(len(plane_candidates)),
                "selected_candidate_index":
                    int(best["candidate_index"]),
                "geometry_score":
                    float(best["score"]),
                "plane_normal":
                    np.round(
                        normal,
                        6,
                    ).tolist(),
                "plane_d":
                    float(d),
                **geom,
                "candidate_summary": [
                    {
                        "idx":
                            int(c["candidate_index"]),
                        "valid":
                            bool(c["geometry_valid"]),
                        "score":
                            round(
                                float(c["score"]),
                                4,
                            ),
                        "opp_long":
                            round(
                                float(
                                    c["geom"][
                                        "long_opposite_rel_error"
                                    ]
                                ),
                                3,
                            ),
                        "opp_short":
                            round(
                                float(
                                    c["geom"][
                                        "short_opposite_rel_error"
                                    ]
                                ),
                                3,
                            ),
                        "ortho_cos":
                            round(
                                float(
                                    c["geom"][
                                        "max_corner_orthogonality_cos"
                                    ]
                                ),
                                3,
                            ),
                        "normal":
                            np.round(
                                c["normal"],
                                3,
                            ).tolist(),
                    }
                    for c in plane_candidates
                ],
            }
        )

        return (
            np.asarray(
                left_3d,
                dtype=np.float64,
            ),
            np.asarray(
                right_3d,
                dtype=np.float64,
            ),
            corner_3d,
            stats,
        )


    def _publish_pixel(
        self,
        pub,
        point: np.ndarray,
        source_msg: Image,
    ) -> None:
        msg = PointStamped()
        msg.header = source_msg.header
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = 0.0
        pub.publish(msg)

    def _publish_3d(
        self,
        pub,
        point: np.ndarray,
        stamp,
    ) -> None:
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = (
            self.color_frame
        )
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = float(point[2])
        pub.publish(msg)

    def _publish_valid(
        self,
        valid: bool,
    ) -> None:
        msg = Bool()
        msg.data = bool(valid)
        self.valid_pub.publish(msg)
        self.grasp_valid_pub.publish(msg)

    def _publish_debug(
        self,
        image: np.ndarray,
        source_msg: Image,
    ) -> None:
        try:
            out = (
                self.bridge.cv2_to_imgmsg(
                    image,
                    encoding="bgr8",
                )
            )
            out.header = source_msg.header
            self.debug_pub.publish(out)
        except Exception as exc:
            self.get_logger().error(
                f"发布调试图失败: {exc}"
            )

    @staticmethod
    def _draw_status(
        image: np.ndarray,
        text: str,
    ) -> None:
        cv2.rectangle(
            image,
            (10, 10),
            (
                min(
                    image.shape[1] - 10,
                    760,
                ),
                62,
            ),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            text,
            (22, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_pose_geometry(
        image: np.ndarray,
        top_left: np.ndarray,
        top_right: np.ndarray,
        bottom_left: np.ndarray,
        bottom_right: np.ndarray,
        left_mid: np.ndarray,
        right_mid: np.ndarray,
        conf: float,
    ) -> None:
        """
        调试图:
          红色点: TL/TR/BL/BR
          绿色线: 顶部四边形
          黄色粗线: 左右两条抓取边
          蓝色圆: 左右抓取中点
        """
        points = {
            "TL": np.asarray(top_left),
            "TR": np.asarray(top_right),
            "BL": np.asarray(bottom_left),
            "BR": np.asarray(bottom_right),
        }

        polygon = np.round(
            np.asarray(
                [
                    top_left,
                    top_right,
                    bottom_right,
                    bottom_left,
                ]
            )
        ).astype(np.int32)

        cv2.polylines(
            image,
            [polygon],
            True,
            (0, 255, 0),
            2,
        )

        for label, p in points.items():
            uv = tuple(
                np.round(p).astype(int)
            )
            cv2.circle(
                image,
                uv,
                7,
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                image,
                label,
                (
                    uv[0] + 8,
                    uv[1] - 8,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 用户指定的两条抓取边:
        # left  = TL -> BL
        # right = TR -> BR
        cv2.line(
            image,
            tuple(np.round(top_left).astype(int)),
            tuple(np.round(bottom_left).astype(int)),
            (0, 255, 255),
            4,
        )
        cv2.line(
            image,
            tuple(np.round(top_right).astype(int)),
            tuple(np.round(bottom_right).astype(int)),
            (0, 255, 255),
            4,
        )

        for label, p in (
            ("LEFT_GRASP", left_mid),
            ("RIGHT_GRASP", right_mid),
        ):
            uv = tuple(
                np.round(p).astype(int)
            )
            cv2.circle(
                image,
                uv,
                10,
                (255, 0, 0),
                3,
            )
            cv2.putText(
                image,
                label,
                (
                    uv[0] + 12,
                    uv[1],
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        left_len = float(
            np.linalg.norm(
                np.asarray(top_left)
                - np.asarray(bottom_left)
            )
        )
        right_len = float(
            np.linalg.norm(
                np.asarray(top_right)
                - np.asarray(bottom_right)
            )
        )

        text = (
            f"pose conf={conf:.2f} "
            f"left_edge={left_len:.0f}px "
            f"right_edge={right_len:.0f}px"
        )
        cv2.putText(
            image,
            text,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def destroy_node(self):
        with self.frame_condition:
            self.worker_stop = True
            self.latest_image_msg = None
            self.frame_condition.notify_all()

        thread = getattr(
            self,
            "inference_thread",
            None,
        )
        if (
            thread is not None
            and thread.is_alive()
            and threading.current_thread()
            is not thread
        ):
            thread.join(timeout=2.0)

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[
        BoxPoseShortEdgeDetector
    ] = None

    try:
        node = (
            BoxPoseShortEdgeDetector()
        )
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(
                "节点异常退出: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            print(
                "节点初始化失败: "
                f"{type(exc).__name__}: {exc}"
            )
        raise

    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
