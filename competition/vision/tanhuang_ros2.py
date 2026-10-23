#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 2 螺钉袋 YOLO 检测节点

默认输入:
  /camera_head/color/image_raw

默认输出:
  /screw_bag_detection/image
  /screw_bag_detection/center_pixel
  /screw_bag_detection/valid

说明:
- 使用 Win11 predict.py 相同的“取最高置信度检测框”逻辑。
- 调试图保留绿色 bbox、中心十字、置信度和中心像素坐标。
- 采用“最新帧覆盖 + 独立推理线程”，避免 YOLO 阻塞 ROS 回调导致延迟积累。
- 这里只发布 2D 中心像素，不计算 3D 坐标。
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from ultralytics import YOLO


class ScrewBagDetector(Node):
    def __init__(self) -> None:
        super().__init__("screw_bag_detector")

        self.declare_parameter(
            "model_path",
            "/home/niic/Desktop/competition/vision/models/tanhuang.pt",
        )
        self.declare_parameter(
            "color_topic",
            "/camera_head/color/image_raw",
        )
        self.declare_parameter(
            "debug_image_topic",
            "/screw_bag_detection/image",
        )
        self.declare_parameter(
            "center_topic",
            "/screw_bag_detection/center_pixel",
        )
        self.declare_parameter(
            "valid_topic",
            "/screw_bag_detection/valid",
        )
        self.declare_parameter("confidence", 0.50)
        self.declare_parameter("imgsz", 1280)
        self.declare_parameter("device", "0")
        self.declare_parameter("log_every_n_frames", 10)

        self.model_path = str(self.get_parameter("model_path").value)
        self.color_topic = str(self.get_parameter("color_topic").value)
        self.debug_image_topic = str(
            self.get_parameter("debug_image_topic").value
        )
        self.center_topic = str(self.get_parameter("center_topic").value)
        self.valid_topic = str(self.get_parameter("valid_topic").value)
        self.confidence = float(self.get_parameter("confidence").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.device = str(self.get_parameter("device").value)
        self.log_every_n_frames = max(
            1, int(self.get_parameter("log_every_n_frames").value)
        )

        self.bridge = CvBridge()

        self.get_logger().info(f"加载模型：{self.model_path}")
        self.model = YOLO(self.model_path)
        self.get_logger().info(f"模型任务：{self.model.task}")
        self.get_logger().info(f"模型类别：{self.model.names}")

        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.color_sub = self.create_subscription(
            Image,
            self.color_topic,
            self.image_callback,
            camera_qos,
        )
        self.debug_image_pub = self.create_publisher(
            Image,
            self.debug_image_topic,
            10,
        )
        self.center_pub = self.create_publisher(
            PointStamped,
            self.center_topic,
            10,
        )
        self.valid_pub = self.create_publisher(
            Bool,
            self.valid_topic,
            10,
        )

        self._frame_condition = threading.Condition()
        self._latest_image_msg: Optional[Image] = None
        self._worker_should_stop = False
        self.processed_frame_count = 0

        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="screw_bag_inference",
            daemon=True,
        )
        self._inference_thread.start()

        self.get_logger().info("螺钉袋检测节点已启动")
        self.get_logger().info(f"彩色图：{self.color_topic}")
        self.get_logger().info(f"检测图：{self.debug_image_topic}")
        self.get_logger().info(f"中心像素：{self.center_topic}")
        self.get_logger().info(f"检测有效：{self.valid_topic}")

    def image_callback(self, msg: Image) -> None:
        with self._frame_condition:
            self._latest_image_msg = msg
            self._frame_condition.notify()

    def _inference_loop(self) -> None:
        while True:
            with self._frame_condition:
                self._frame_condition.wait_for(
                    lambda: (
                        self._worker_should_stop
                        or self._latest_image_msg is not None
                    )
                )

                if self._worker_should_stop:
                    return

                msg = self._latest_image_msg
                self._latest_image_msg = None

            if msg is None:
                continue

            try:
                self._process_image(msg)
            except Exception as exc:
                self.get_logger().error(
                    f"处理图像失败：{type(exc).__name__}: {exc}"
                )

    def _process_image(self, msg: Image) -> None:
        self.processed_frame_count += 1

        try:
            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
        except Exception as exc:
            self.get_logger().error(f"cv_bridge 转换失败：{exc}")
            return

        annotated = image.copy()

        try:
            result = self.model.predict(
                source=image,
                imgsz=self.imgsz,
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )[0]
        except Exception as exc:
            self.get_logger().error(f"YOLO 推理失败：{exc}")
            self._draw_status(annotated, "INFERENCE ERROR")
            self._publish_debug_image(annotated, msg)
            self._publish_valid(False)
            return

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            self._draw_status(annotated, "NO DETECTION")
            self._publish_debug_image(annotated, msg)
            self._publish_valid(False)

            if self.processed_frame_count % self.log_every_n_frames == 0:
                self.get_logger().info("未检测到螺钉袋")
            return

        confidences = boxes.conf.detach().cpu().numpy()
        best_index = int(np.argmax(confidences))

        box = (
            boxes.xyxy[best_index]
            .detach()
            .cpu()
            .numpy()
        )
        conf = float(confidences[best_index])

        class_id = 0
        if boxes.cls is not None and len(boxes.cls) > best_index:
            class_id = int(boxes.cls[best_index].item())

        names = self.model.names
        if isinstance(names, dict):
            class_name = str(names.get(class_id, class_id))
        else:
            class_name = str(names[class_id])

        center_u, center_v = self._draw_detection(
            annotated,
            box,
            conf,
            class_name,
        )

        center_msg = PointStamped()
        center_msg.header = msg.header
        center_msg.point.x = float(center_u)
        center_msg.point.y = float(center_v)
        center_msg.point.z = 0.0
        self.center_pub.publish(center_msg)

        self._publish_valid(True)
        self._publish_debug_image(annotated, msg)

        if self.processed_frame_count % self.log_every_n_frames == 0:
            self.get_logger().info(
                f"检测到 {class_name}: "
                f"conf={conf:.3f}, "
                f"center=({center_u},{center_v}), "
                f"bbox=({box[0]:.1f},{box[1]:.1f},"
                f"{box[2]:.1f},{box[3]:.1f})"
            )

    @staticmethod
    def _draw_detection(
        image: np.ndarray,
        box: np.ndarray,
        conf: float,
        class_name: str,
    ) -> Tuple[int, int]:
        height, width = image.shape[:2]

        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        x1 = int(np.clip(x1, 0, width - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        y2 = int(np.clip(y2, 0, height - 1))

        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))

        green = (0, 255, 0)
        red = (0, 0, 255)
        yellow = (0, 255, 255)
        white = (255, 255, 255)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            green,
            2,
        )

        cross_size = 15
        cv2.line(
            image,
            (cx - cross_size, cy),
            (cx + cross_size, cy),
            red,
            2,
        )
        cv2.line(
            image,
            (cx, cy - cross_size),
            (cx, cy + cross_size),
            red,
            2,
        )
        cv2.circle(image, (cx, cy), 8, red, 2)
        cv2.circle(image, (cx, cy), 3, yellow, -1)

        label = (
            f"{class_name} {conf:.2f} | "
            f"center({cx},{cy})"
        )
        font_scale = 0.7
        thickness = 2
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )

        text_top = max(0, y1 - th - 10)
        text_bottom = max(y1, th + 10)

        cv2.rectangle(
            image,
            (x1, text_top),
            (min(width - 1, x1 + tw + 8), text_bottom),
            green,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 4, max(th + 2, text_bottom - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            white,
            thickness,
            cv2.LINE_AA,
        )

        return cx, cy

    @staticmethod
    def _draw_status(
        image: np.ndarray,
        text: str,
    ) -> None:
        cv2.rectangle(
            image,
            (10, 10),
            (min(image.shape[1] - 10, 700), 62),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            text,
            (22, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    def _publish_debug_image(
        self,
        image: np.ndarray,
        source_msg: Image,
    ) -> None:
        try:
            output_msg = self.bridge.cv2_to_imgmsg(
                image,
                encoding="bgr8",
            )
            output_msg.header = source_msg.header
            self.debug_image_pub.publish(output_msg)
        except Exception as exc:
            self.get_logger().error(
                f"发布检测图失败：{exc}"
            )

    def _publish_valid(self, valid: bool) -> None:
        msg = Bool()
        msg.data = bool(valid)
        self.valid_pub.publish(msg)

    def destroy_node(self):
        with self._frame_condition:
            self._worker_should_stop = True
            self._latest_image_msg = None
            self._frame_condition.notify_all()

        thread = getattr(self, "_inference_thread", None)
        if (
            thread is not None
            and thread.is_alive()
            and threading.current_thread() is not thread
        ):
            thread.join()

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[ScrewBagDetector] = None

    try:
        node = ScrewBagDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(
                f"节点异常退出：{type(exc).__name__}: {exc}"
            )
        else:
            print(
                f"节点初始化失败：{type(exc).__name__}: {exc}"
            )
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
