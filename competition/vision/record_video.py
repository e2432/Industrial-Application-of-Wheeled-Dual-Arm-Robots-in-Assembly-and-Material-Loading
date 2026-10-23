#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 2 同步保存彩色图和深度图。

默认输入：
    /camera_head/color/image_raw
    /camera_head/depth/image_raw

默认输出：
    /home/niic/Desktop/competition/vision/rgbd_recordings/
        rgbd_YYYYMMDD_HHMMSS/
            color/
                frame_000001.png
                frame_000002.png
            depth/
                frame_000001.png
                frame_000002.png
            metadata.csv

同一组同步彩色图和深度图使用完全相同的文件名，分别保存到
color 和 depth 目录中。

说明：
1. 彩色图保存为无损 PNG。
2. 深度图保存为 16 位单通道 PNG，数值单位统一为毫米。
3. 32FC1 深度图会从“米”转换为“毫米”后保存。
4. 默认每 5 组同步消息保存 1 组，可通过 ROS 参数调整。
5. 按 Ctrl+C 安全结束。
"""

from __future__ import annotations

import csv
import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


ImagePair = Tuple[Image, Image]


class SynchronizedRgbdRecorder(Node):
    """同步接收彩色图和深度图，并以相同文件名保存。"""

    def __init__(self) -> None:
        super().__init__("synchronized_rgbd_recorder")

        # --------------------------- 参数 ---------------------------
        self.declare_parameter(
            "color_topic",
            "/camera_head/color/image_raw",
        )
        self.declare_parameter(
            "depth_topic",
            "/camera_head/depth/image_raw",
        )
        self.declare_parameter(
            "output_dir",
            "/home/niic/Desktop/competition/vision/rgbd_recordings",
        )

        # 每多少组同步消息保存一组。相机15 FPS时，默认约保存3组/秒。
        self.declare_parameter("save_every_n_pairs", 5)

        # 彩色和深度时间戳允许的最大差值，单位秒。
        self.declare_parameter("sync_slop", 0.08)
        self.declare_parameter("sync_queue_size", 20)

        # PNG压缩等级：0最快、9压缩最高。
        self.declare_parameter("png_compression", 3)

        # 写盘队列上限。队列满时丢弃最旧的待保存图像组。
        self.declare_parameter("writer_queue_size", 8)

        # 0表示不限制，使用Ctrl+C停止。
        self.declare_parameter("max_saved_pairs", 0)
        self.declare_parameter("max_duration_sec", 0.0)

        self.color_topic = str(
            self.get_parameter("color_topic").value
        )
        self.depth_topic = str(
            self.get_parameter("depth_topic").value
        )
        self.output_root = Path(
            str(self.get_parameter("output_dir").value)
        )
        self.save_every_n_pairs = max(
            1,
            int(self.get_parameter("save_every_n_pairs").value),
        )
        self.sync_slop = float(
            self.get_parameter("sync_slop").value
        )
        self.sync_queue_size = max(
            2,
            int(self.get_parameter("sync_queue_size").value),
        )
        self.png_compression = int(
            self.get_parameter("png_compression").value
        )
        self.writer_queue_size = max(
            1,
            int(self.get_parameter("writer_queue_size").value),
        )
        self.max_saved_pairs = max(
            0,
            int(self.get_parameter("max_saved_pairs").value),
        )
        self.max_duration_sec = max(
            0.0,
            float(self.get_parameter("max_duration_sec").value),
        )

        if not 0 <= self.png_compression <= 9:
            raise ValueError("png_compression必须在0到9之间")

        if self.sync_slop <= 0.0:
            raise ValueError("sync_slop必须大于0")

        # --------------------------- 输出目录 ---------------------------
        session_name = datetime.now().strftime("rgbd_%Y%m%d_%H%M%S")
        self.session_dir = self.output_root / session_name
        self.color_dir = self.session_dir / "color"
        self.depth_dir = self.session_dir / "depth"
        self.metadata_path = self.session_dir / "metadata.csv"

        self.color_dir.mkdir(parents=True, exist_ok=False)
        self.depth_dir.mkdir(parents=True, exist_ok=False)

        self.metadata_file = self.metadata_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self.metadata_writer = csv.writer(self.metadata_file)
        self.metadata_writer.writerow(
            [
                "index",
                "filename",
                "color_stamp_sec",
                "color_stamp_nanosec",
                "depth_stamp_sec",
                "depth_stamp_nanosec",
                "time_difference_ms",
                "color_frame_id",
                "depth_frame_id",
                "color_encoding",
                "depth_encoding",
                "color_width",
                "color_height",
                "depth_width",
                "depth_height",
            ]
        )
        self.metadata_file.flush()

        # --------------------------- 状态 ---------------------------
        self.bridge = CvBridge()

        self.synchronized_pair_count = 0
        self.selected_pair_count = 0
        self.saved_pair_count = 0
        self.failed_pair_count = 0
        self.dropped_queue_pair_count = 0

        self.start_time = time.monotonic()
        self.stop_requested = False

        # 同步回调只将完整的一对消息放入有界队列。
        # 图像转换和磁盘写入由独立线程完成，避免阻塞ROS回调。
        self.queue_condition = threading.Condition()
        self.writer_queue: Deque[ImagePair] = deque(
            maxlen=self.writer_queue_size
        )

        # --------------------------- ROS同步订阅 ---------------------------
        self.color_subscriber = Subscriber(
            self,
            Image,
            self.color_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_subscriber = Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.synchronizer = ApproximateTimeSynchronizer(
            [
                self.color_subscriber,
                self.depth_subscriber,
            ],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=False,
        )
        self.synchronizer.registerCallback(
            self.synchronized_callback
        )

        self.writer_thread = threading.Thread(
            target=self.writer_loop,
            name="rgbd_image_writer",
            daemon=True,
        )
        self.writer_thread.start()

        self.status_timer = self.create_timer(
            5.0,
            self.report_status,
        )

        self.duration_timer = None
        if self.max_duration_sec > 0.0:
            self.duration_timer = self.create_timer(
                0.2,
                self.check_duration,
            )

        self.get_logger().info("RGB-D同步保存节点已启动")
        self.get_logger().info(
            f"彩色图话题：{self.color_topic}"
        )
        self.get_logger().info(
            f"深度图话题：{self.depth_topic}"
        )
        self.get_logger().info(
            f"输出目录：{self.session_dir}"
        )
        self.get_logger().info(
            f"彩色图目录：{self.color_dir}"
        )
        self.get_logger().info(
            f"深度图目录：{self.depth_dir}"
        )
        self.get_logger().info(
            f"每{self.save_every_n_pairs}组同步消息保存1组"
        )
        self.get_logger().info(
            f"同步容差：{self.sync_slop * 1000.0:.1f} ms"
        )

        if self.max_saved_pairs > 0:
            self.get_logger().info(
                f"保存达到{self.max_saved_pairs}组后自动停止"
            )

        if self.max_duration_sec > 0.0:
            self.get_logger().info(
                f"运行达到{self.max_duration_sec:.1f}秒后自动停止"
            )
        else:
            self.get_logger().info("按 Ctrl+C 停止保存")

    # ============================================================
    # ROS同步回调
    # ============================================================

    def synchronized_callback(
        self,
        color_msg: Image,
        depth_msg: Image,
    ) -> None:
        """收到一组时间同步的彩色图和深度图。"""
        self.synchronized_pair_count += 1

        if (
            self.synchronized_pair_count
            % self.save_every_n_pairs
            != 0
        ):
            return

        self.selected_pair_count += 1

        with self.queue_condition:
            if len(self.writer_queue) >= self.writer_queue.maxlen:
                # deque(maxlen)直接append时也会丢最旧项，
                # 这里显式弹出以便准确计数。
                self.writer_queue.popleft()
                self.dropped_queue_pair_count += 1

            self.writer_queue.append(
                (color_msg, depth_msg)
            )
            self.queue_condition.notify()

    # ============================================================
    # 独立写盘线程
    # ============================================================

    def writer_loop(self) -> None:
        """从队列中取出完整RGB-D图像组并保存。"""
        while True:
            with self.queue_condition:
                self.queue_condition.wait_for(
                    lambda: (
                        self.stop_requested
                        or len(self.writer_queue) > 0
                    )
                )

                if (
                    self.stop_requested
                    and len(self.writer_queue) == 0
                ):
                    return

                color_msg, depth_msg = self.writer_queue.popleft()

            try:
                color_image = self.bridge.imgmsg_to_cv2(
                    color_msg,
                    desired_encoding="bgr8",
                )
                depth_raw = self.bridge.imgmsg_to_cv2(
                    depth_msg,
                    desired_encoding="passthrough",
                )

                depth_mm = self.convert_depth_to_uint16_mm(
                    depth_raw,
                    depth_msg.encoding,
                )

                self.save_pair(
                    color_msg=color_msg,
                    depth_msg=depth_msg,
                    color_image=color_image,
                    depth_mm=depth_mm,
                )

            except Exception as exc:
                self.failed_pair_count += 1
                self.get_logger().error(
                    "保存RGB-D图像组失败："
                    f"{type(exc).__name__}: {exc}"
                )

    @staticmethod
    def convert_depth_to_uint16_mm(
        depth: np.ndarray,
        encoding: str,
    ) -> np.ndarray:
        """
        将不同ROS深度格式统一转换为16位毫米深度图。

        16UC1 / MONO16：
            默认原始数值已经是毫米。

        32FC1：
            默认原始数值是米，乘1000转换为毫米。

        无效深度：
            保存为0。
        """
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        encoding_upper = encoding.upper()

        if encoding_upper in {"16UC1", "MONO16"}:
            result = depth.astype(np.uint16, copy=True)
            return result

        if encoding_upper in {"32FC1", "32FC"}:
            depth_float = depth.astype(np.float32, copy=False)
            result = np.zeros(
                depth_float.shape,
                dtype=np.uint16,
            )

            valid = (
                np.isfinite(depth_float)
                & (depth_float > 0.0)
            )

            depth_mm = np.clip(
                np.rint(depth_float[valid] * 1000.0),
                0,
                np.iinfo(np.uint16).max,
            )
            result[valid] = depth_mm.astype(np.uint16)
            return result

        # 对不标准encoding按dtype兜底。
        if np.issubdtype(depth.dtype, np.integer):
            return np.clip(
                depth,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)

        if np.issubdtype(depth.dtype, np.floating):
            depth_float = depth.astype(np.float32, copy=False)
            result = np.zeros(
                depth_float.shape,
                dtype=np.uint16,
            )
            valid = (
                np.isfinite(depth_float)
                & (depth_float > 0.0)
            )
            result[valid] = np.clip(
                np.rint(depth_float[valid] * 1000.0),
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            return result

        raise ValueError(
            f"不支持的深度格式：encoding={encoding}, "
            f"dtype={depth.dtype}"
        )

    def save_pair(
        self,
        color_msg: Image,
        depth_msg: Image,
        color_image: np.ndarray,
        depth_mm: np.ndarray,
    ) -> None:
        """使用相同文件名分别保存彩色图和深度图。"""
        next_index = self.saved_pair_count + 1
        filename = f"frame_{next_index:06d}.png"

        color_path = self.color_dir / filename
        depth_path = self.depth_dir / filename

        write_parameters = [
            cv2.IMWRITE_PNG_COMPRESSION,
            self.png_compression,
        ]

        color_success = cv2.imwrite(
            str(color_path),
            color_image,
            write_parameters,
        )
        depth_success = cv2.imwrite(
            str(depth_path),
            depth_mm,
            write_parameters,
        )

        if not color_success or not depth_success:
            # 避免只留下单边文件。
            if color_path.exists():
                color_path.unlink()
            if depth_path.exists():
                depth_path.unlink()

            raise IOError(
                "cv2.imwrite写入失败："
                f"color={color_success}, depth={depth_success}"
            )

        color_time = (
            float(color_msg.header.stamp.sec)
            + float(color_msg.header.stamp.nanosec) * 1e-9
        )
        depth_time = (
            float(depth_msg.header.stamp.sec)
            + float(depth_msg.header.stamp.nanosec) * 1e-9
        )
        difference_ms = abs(
            color_time - depth_time
        ) * 1000.0

        self.saved_pair_count = next_index

        self.metadata_writer.writerow(
            [
                next_index,
                filename,
                color_msg.header.stamp.sec,
                color_msg.header.stamp.nanosec,
                depth_msg.header.stamp.sec,
                depth_msg.header.stamp.nanosec,
                f"{difference_ms:.3f}",
                color_msg.header.frame_id,
                depth_msg.header.frame_id,
                color_msg.encoding,
                depth_msg.encoding,
                color_image.shape[1],
                color_image.shape[0],
                depth_mm.shape[1],
                depth_mm.shape[0],
            ]
        )
        self.metadata_file.flush()

        if (
            self.max_saved_pairs > 0
            and self.saved_pair_count >= self.max_saved_pairs
        ):
            self.get_logger().info(
                f"已经保存{self.saved_pair_count}组RGB-D图像，"
                "达到设定上限"
            )
            rclpy.shutdown()

    # ============================================================
    # 状态和退出
    # ============================================================

    def report_status(self) -> None:
        elapsed = max(
            time.monotonic() - self.start_time,
            1e-6,
        )

        self.get_logger().info(
            "保存状态："
            f"运行={elapsed:.1f}s，"
            f"同步收到={self.synchronized_pair_count}组，"
            f"选中={self.selected_pair_count}组，"
            f"成功保存={self.saved_pair_count}组，"
            f"写盘队列丢弃={self.dropped_queue_pair_count}组，"
            f"失败={self.failed_pair_count}组"
        )

    def check_duration(self) -> None:
        elapsed = time.monotonic() - self.start_time
        if elapsed >= self.max_duration_sec:
            self.get_logger().info(
                "达到设定运行时间，正在结束"
            )
            rclpy.shutdown()

    def destroy_node(self):
        """等待写盘队列处理完成后安全关闭。"""
        with self.queue_condition:
            self.stop_requested = True
            self.queue_condition.notify_all()

        if (
            self.writer_thread.is_alive()
            and threading.current_thread() is not self.writer_thread
        ):
            self.writer_thread.join()

        if not self.metadata_file.closed:
            self.metadata_file.flush()
            self.metadata_file.close()

        elapsed = max(
            time.monotonic() - self.start_time,
            1e-6,
        )

        self.get_logger().info(
            "RGB-D保存结束："
            f"运行={elapsed:.1f}s，"
            f"成功保存={self.saved_pair_count}组"
        )
        self.get_logger().info(
            f"彩色图：{self.color_dir}"
        )
        self.get_logger().info(
            f"深度图：{self.depth_dir}"
        )
        self.get_logger().info(
            f"元数据：{self.metadata_path}"
        )

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[SynchronizedRgbdRecorder] = None

    try:
        node = SynchronizedRgbdRecorder()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(
                f"节点异常：{type(exc).__name__}: {exc}"
            )
        else:
            print(
                f"节点初始化失败："
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
