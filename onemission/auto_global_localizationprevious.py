#!/usr/bin/env python3
import math
from enum import Enum, auto
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener


class State(Enum):
    WAITING_FOR_AMCL = auto()
    REINITIALIZING = auto()
    WAITING_FOR_ODOM = auto()
    ROTATING = auto()
    VERIFYING = auto()
    SUCCESS = auto()
    FAILED = auto()


class AutoGlobalLocalization(Node):
    """
    AMCL automatic global localization - strategy V2.1.

    V2.1 strategy:
        1. Wait only for the global-localization service and odom TF.
           Do NOT require /amcl_pose before global reinitialization.
        2. Call /reinitialize_global_localization exactly ONCE.
        3. Start rotating even if /amcl_pose has not arrived yet.
           Once /amcl_pose arrives, begin covariance evaluation immediately.
        4. Rotate left by one ACTUAL full turn measured from odom -> base_link yaw.
        5. Rotate right by one ACTUAL full turn.
        6. Continue alternating full turns without reinitializing particles.
        7. Continuously monitor /amcl_pose covariance whenever it is available.
        8. If x/y/yaw covariance remain below thresholds for stable_duration,
           stop immediately and report success.

    Important:
        - No repeated global-localization reset.
        - No "60 seconds means one turn" assumption.
        - No automatic forward/lateral translation.
    """

    def __init__(self):
        super().__init__("auto_global_localization")

        # ---------------- Parameters ----------------
        self.declare_parameter(
            "cmd_vel_topic",
            "/mecanum_drive_controller/reference_unstamped",
        )
        self.declare_parameter("amcl_pose_topic", "/amcl_pose")
        self.declare_parameter(
            "global_localization_service",
            "/reinitialize_global_localization",
        )

        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        # Keep the same slow rotation speed that already worked manually.
        self.declare_parameter("angular_speed", 0.40)          # rad/s

        # One direction switch only after the robot has ACTUALLY rotated 2*pi.
        self.declare_parameter("rotation_angle", 2.0 * math.pi)  # rad

        # Thresholds based on the successful manual-localization experiment.
        self.declare_parameter("x_variance_threshold", 0.05)
        self.declare_parameter("y_variance_threshold", 0.05)
        self.declare_parameter("yaw_variance_threshold", 0.10)
        self.declare_parameter("stable_duration", 3.0)         # seconds

        self.declare_parameter("startup_timeout", 30.0)        # seconds
        self.declare_parameter("control_rate", 10.0)           # Hz
        self.declare_parameter("status_log_period", 1.0)       # seconds

        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.amcl_pose_topic = str(self.get_parameter("amcl_pose_topic").value)
        self.global_localization_service = str(
            self.get_parameter("global_localization_service").value
        )
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        self.angular_speed = abs(float(self.get_parameter("angular_speed").value))
        self.rotation_angle = abs(float(self.get_parameter("rotation_angle").value))

        self.x_var_threshold = float(
            self.get_parameter("x_variance_threshold").value
        )
        self.y_var_threshold = float(
            self.get_parameter("y_variance_threshold").value
        )
        self.yaw_var_threshold = float(
            self.get_parameter("yaw_variance_threshold").value
        )
        self.stable_duration = float(self.get_parameter("stable_duration").value)

        self.startup_timeout = float(self.get_parameter("startup_timeout").value)
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.status_log_period = float(
            self.get_parameter("status_log_period").value
        )

        # ---------------- ROS interfaces ----------------
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.amcl_pose_topic,
            self.pose_callback,
            10,
        )

        self.global_localization_client = self.create_client(
            Empty,
            self.global_localization_service,
        )

        # TF is used ONLY to measure the robot's real accumulated rotation.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------- Runtime state ----------------
        self.state = State.WAITING_FOR_AMCL

        self.latest_pose: Optional[PoseWithCovarianceStamped] = None
        self.received_amcl_pose = False

        # Global localization is deliberately allowed only once.
        self.global_localization_called = False
        self.reinit_future = None

        # Rotation strategy: first LEFT, then RIGHT, then LEFT, ...
        self.rotation_direction = 1.0
        self.rotation_index = 1
        self.accumulated_rotation = 0.0
        self.last_odom_yaw: Optional[float] = None

        self.stable_start_time: Optional[float] = None

        self.startup_start_time = self.now_sec()
        self.last_status_log_time = 0.0
        self.last_wait_log_time = 0.0

        timer_period = 1.0 / max(self.control_rate, 1.0)
        self.timer = self.create_timer(timer_period, self.control_loop)

        self.get_logger().info(
            "Auto global localization V2.1 started: "
            "ONE reinitialize, alternating measured 360-degree rotations."
        )
        self.get_logger().info(
            f"cmd_vel_topic={self.cmd_vel_topic}, "
            f"amcl_pose_topic={self.amcl_pose_topic}, "
            f"TF={self.odom_frame}->{self.base_frame}"
        )
        self.get_logger().info(
            "Success thresholds: "
            f"x<{self.x_var_threshold:.3f}, "
            f"y<{self.y_var_threshold:.3f}, "
            f"yaw<{self.yaw_var_threshold:.3f}, "
            f"stable={self.stable_duration:.1f}s"
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi)."""
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        """Return yaw from quaternion without requiring tf_transformations."""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.latest_pose = msg
        self.received_amcl_pose = True

    def get_variances(self):
        if self.latest_pose is None:
            return None

        cov = self.latest_pose.pose.covariance
        # PoseWithCovariance is a 6x6 matrix:
        # x -> [0], y -> [7], yaw -> [35]
        return float(cov[0]), float(cov[7]), float(cov[35])

    def localization_is_good(self) -> bool:
        variances = self.get_variances()
        if variances is None:
            return False

        x_var, y_var, yaw_var = variances
        return (
            math.isfinite(x_var)
            and math.isfinite(y_var)
            and math.isfinite(yaw_var)
            and 0.0 <= x_var < self.x_var_threshold
            and 0.0 <= y_var < self.y_var_threshold
            and 0.0 <= yaw_var < self.yaw_var_threshold
        )

    def get_odom_yaw(self) -> Optional[float]:
        """Read current yaw of base_frame in odom_frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_frame,
                Time(),
            )
        except Exception:
            return None

        q = transform.transform.rotation
        return self.yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def publish_rotation(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.angular.z = self.rotation_direction * self.angular_speed
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        msg = Twist()
        for _ in range(3):
            self.cmd_pub.publish(msg)

    def start_reinitialization(self):
        """Call global localization exactly once."""
        self.stop_robot()

        if self.global_localization_called:
            self.get_logger().error(
                "Internal safety check: global localization was already called. "
                "V2 will NOT call it again."
            )
            self.state = State.FAILED
            return

        if not self.global_localization_client.service_is_ready():
            return

        self.global_localization_called = True
        self.stable_start_time = None

        self.get_logger().info(
            "Calling /reinitialize_global_localization ONCE. "
            "Particles will not be reset again during rotation."
        )

        self.reinit_future = self.global_localization_client.call_async(
            Empty.Request()
        )
        self.state = State.REINITIALIZING

    def update_accumulated_rotation(self, current_yaw: float):
        """
        Accumulate real yaw motion from odometry.

        Each sample contributes the shortest wrapped angular difference.
        At normal control rates and 0.20 rad/s this correctly handles crossing
        +pi/-pi while measuring a complete physical turn.
        """
        if self.last_odom_yaw is None:
            self.last_odom_yaw = current_yaw
            return

        delta = self.normalize_angle(current_yaw - self.last_odom_yaw)
        self.last_odom_yaw = current_yaw
        self.accumulated_rotation += abs(delta)

    def switch_rotation_direction(self, current_yaw: float):
        completed_direction = "LEFT" if self.rotation_direction > 0.0 else "RIGHT"

        # Reverse direction without touching AMCL particles.
        self.rotation_direction *= -1.0
        self.rotation_index += 1
        self.accumulated_rotation = 0.0
        self.last_odom_yaw = current_yaw

        next_direction = "LEFT" if self.rotation_direction > 0.0 else "RIGHT"

        self.get_logger().info(
            f"Completed actual 360-degree {completed_direction} rotation."
        )
        self.get_logger().info(
            f"Particles preserved. Starting rotation #{self.rotation_index}: "
            f"{next_direction}."
        )

    def log_localization_status(self, now: float):
        if now - self.last_status_log_time < self.status_log_period:
            return

        self.last_status_log_time = now

        variances = self.get_variances()
        if variances is None:
            self.get_logger().info(
                "[LOCALIZING] Robot is rotating; waiting for first /amcl_pose ..."
            )
            return

        x_var, y_var, yaw_var = variances

        def mark(value: float, threshold: float) -> str:
            if not math.isfinite(value) or value < 0.0:
                return "WAIT"
            return "OK" if value < threshold else "WAIT"

        direction_text = "LEFT" if self.rotation_direction > 0.0 else "RIGHT"
        degrees = math.degrees(self.accumulated_rotation)

        lines = [
            "[LOCALIZING]",
            f"x_var   = {x_var:.5f}  [{mark(x_var, self.x_var_threshold)}]",
            f"y_var   = {y_var:.5f}  [{mark(y_var, self.y_var_threshold)}]",
            f"yaw_var = {yaw_var:.5f}  [{mark(yaw_var, self.yaw_var_threshold)}]",
            f"Rotation: {direction_text}",
            f"Accumulated: {degrees:.1f} deg / "
            f"{math.degrees(self.rotation_angle):.1f} deg",
        ]

        if self.stable_start_time is not None:
            stable_for = max(0.0, now - self.stable_start_time)
            lines.append(
                f"Verifying: {min(stable_for, self.stable_duration):.1f} / "
                f"{self.stable_duration:.1f} s"
            )
        else:
            waiting = []
            if not (
                math.isfinite(x_var)
                and 0.0 <= x_var < self.x_var_threshold
            ):
                waiting.append("x")
            if not (
                math.isfinite(y_var)
                and 0.0 <= y_var < self.y_var_threshold
            ):
                waiting.append("y")
            if not (
                math.isfinite(yaw_var)
                and 0.0 <= yaw_var < self.yaw_var_threshold
            ):
                waiting.append("yaw")

            if waiting:
                lines.append(
                    "Waiting for covariance: " + ", ".join(waiting)
                )

        self.get_logger().info("\n" + "\n".join(lines))

    def report_success(self):
        self.stop_robot()
        self.state = State.SUCCESS

        pose = self.latest_pose.pose.pose
        q = pose.orientation
        yaw = self.yaw_from_quaternion(q.x, q.y, q.z, q.w)

        x_var, y_var, yaw_var = self.get_variances()

        self.get_logger().info(
            "\n"
            "=============================\n"
            "GLOBAL LOCALIZATION SUCCESS\n"
            f"x = {pose.position.x:.3f}\n"
            f"y = {pose.position.y:.3f}\n"
            f"yaw = {yaw:.3f} rad ({math.degrees(yaw):.1f} deg)\n"
            f"x_var = {x_var:.5f}\n"
            f"y_var = {y_var:.5f}\n"
            f"yaw_var = {yaw_var:.5f}\n"
            "=============================\n"
            "Robot stopped."
        )

    def control_loop(self):
        now = self.now_sec()

        # ------------------------------------------------------------
        # 1. V2.1 startup:
        #    Wait ONLY for the global-localization service + odom TF.
        #
        #    /amcl_pose is deliberately NOT a startup prerequisite.
        #    Some AMCL states may not publish a usable pose until after
        #    global reinitialization / motion has begun.
        # ------------------------------------------------------------
        if self.state == State.WAITING_FOR_AMCL:
            service_ready = self.global_localization_client.service_is_ready()
            odom_yaw = self.get_odom_yaw()
            odom_ready = odom_yaw is not None

            if service_ready and odom_ready:
                self.get_logger().info(
                    "Global localization service and odom TF are ready. "
                    "/amcl_pose is NOT required before reinitialization."
                )
                self.start_reinitialization()
                return

            if now - self.startup_start_time > self.startup_timeout:
                self.stop_robot()
                self.state = State.FAILED
                self.get_logger().error(
                    "Startup timeout: service or odom TF did not become ready. "
                    f"service_ready={service_ready}, "
                    f"odom_tf_ready={odom_ready}, "
                    f"amcl_pose_received={self.received_amcl_pose} "
                    "(AMCL pose is informational only at startup)"
                )
                return

            if now - self.last_wait_log_time > 2.0:
                self.last_wait_log_time = now
                self.get_logger().info(
                    "Waiting before localization: "
                    f"service_ready={service_ready}, "
                    f"odom_tf_ready={odom_ready}, "
                    f"amcl_pose_received={self.received_amcl_pose} "
                    "(not required yet)"
                )
            return

        # ------------------------------------------------------------
        # 2. Wait for the single global-localization service call.
        # ------------------------------------------------------------
        if self.state == State.REINITIALIZING:
            if self.reinit_future is None:
                self.stop_robot()
                self.state = State.FAILED
                self.get_logger().error("Global-localization future is missing.")
                return

            if not self.reinit_future.done():
                return

            try:
                self.reinit_future.result()
            except Exception as exc:
                self.stop_robot()
                self.state = State.FAILED
                self.get_logger().error(
                    f"Global localization service call failed: {exc}"
                )
                return

            # Reset ONLY rotation measurement, never the particles again.
            self.rotation_direction = 1.0
            self.rotation_index = 1
            self.accumulated_rotation = 0.0
            self.last_odom_yaw = None
            self.stable_start_time = None

            self.state = State.WAITING_FOR_ODOM
            self.get_logger().info(
                "Particles reinitialized once. "
                "Waiting for odom yaw, then starting LEFT rotation. "
                "Rotation will begin even if /amcl_pose has not arrived yet."
            )
            return

        # ------------------------------------------------------------
        # 3. Obtain a clean initial odom yaw after reinitialization.
        # ------------------------------------------------------------
        if self.state == State.WAITING_FOR_ODOM:
            current_yaw = self.get_odom_yaw()
            if current_yaw is None:
                self.stop_robot()
                if now - self.last_wait_log_time > 2.0:
                    self.last_wait_log_time = now
                    self.get_logger().warn(
                        f"Waiting for TF {self.odom_frame} -> {self.base_frame} ..."
                    )
                return

            self.last_odom_yaw = current_yaw
            self.accumulated_rotation = 0.0
            self.state = State.ROTATING
            self.get_logger().info(
                f"Starting rotation #1: LEFT at {self.angular_speed:.3f} rad/s."
            )
            return

        # ------------------------------------------------------------
        # 4. Rotate forever (LEFT 360 -> RIGHT 360 -> LEFT 360 -> ...)
        #    until covariance is stable for stable_duration.
        # ------------------------------------------------------------
        if self.state in (State.ROTATING, State.VERIFYING):
            current_yaw = self.get_odom_yaw()

            if current_yaw is None:
                # Do not rotate blindly if odom yaw disappears, because V2's
                # key requirement is measuring the actual completed angle.
                self.stop_robot()
                if now - self.last_wait_log_time > 2.0:
                    self.last_wait_log_time = now
                    self.get_logger().warn(
                        f"Lost TF {self.odom_frame} -> {self.base_frame}; "
                        "rotation paused until TF returns."
                    )
                return

            self.update_accumulated_rotation(current_yaw)
            self.publish_rotation()

            # Check covariance continuously, independent of turn boundaries.
            if self.localization_is_good():
                if self.stable_start_time is None:
                    self.stable_start_time = now
                    self.state = State.VERIFYING
                    self.get_logger().info(
                        "Candidate localization found. "
                        "Covariance is below all thresholds; verifying stability."
                    )

                stable_for = now - self.stable_start_time
                if stable_for >= self.stable_duration:
                    self.report_success()
                    return
            else:
                if self.stable_start_time is not None:
                    self.get_logger().warn(
                        "Candidate lost: covariance rose above a threshold. "
                        "Continuing rotation without reinitializing particles."
                    )
                self.stable_start_time = None
                self.state = State.ROTATING

            self.log_localization_status(now)

            # Switch direction only after an ACTUAL measured full turn.
            if self.accumulated_rotation >= self.rotation_angle:
                self.switch_rotation_direction(current_yaw)

            return

        if self.state == State.SUCCESS:
            self.stop_robot()
            return

        if self.state == State.FAILED:
            self.stop_robot()
            return

    def destroy_node(self):
        self.stop_robot()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AutoGlobalLocalization()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
