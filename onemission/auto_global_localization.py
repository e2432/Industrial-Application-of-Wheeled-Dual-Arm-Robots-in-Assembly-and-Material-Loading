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
    AMCL automatic global localization - strategy V2.3.

    V2.1 strategy:
        1. Wait only for the global-localization service and odom TF.
           Do NOT require /amcl_pose before global reinitialization.
        2. Automatically call /reinitialize_global_localization. If the first
           request returns but no new /amcl_pose arrives, retry it once. This
           replaces the service command that previously had to be run by hand.
        3. Start rotating even if /amcl_pose has not arrived yet.
           Once /amcl_pose arrives, begin covariance evaluation immediately.
        4. Rotate left by one ACTUAL full turn measured from odom -> base_link yaw.
        5. Rotate right by one ACTUAL full turn.
        6. Continue alternating full turns without reinitializing particles.
        7. Continuously monitor /amcl_pose covariance whenever it is available.
        8. If x/y/yaw covariance remain below thresholds for stable_duration,
           stop immediately and report success.

    Important:
        - If localization has not succeeded within reinitialize_interval
          (default: 60 seconds) after the latest global reinitialization, publish
          global particles again and restart the localization cycle.
        - Each localization cycle still allows one fast no-pose recovery retry.
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
        self.declare_parameter("service_call_timeout", 10.0)   # seconds
        self.declare_parameter("pose_after_reinit_timeout", 8.0)  # seconds

        # If one localization cycle still has not succeeded after this interval,
        # republish global particles and start a fresh cycle.
        self.declare_parameter("reinitialize_interval", 60.0)    # seconds

        # Maximum service attempts INSIDE one localization cycle. With the
        # default value 2 this means: initial call + one fast no-pose retry.
        self.declare_parameter("max_reinitialize_attempts", 2)

        # Overall safety timeout. Set to 0.0 to keep localizing indefinitely.
        self.declare_parameter("localization_timeout", 300.0)  # seconds
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
        self.service_call_timeout = float(
            self.get_parameter("service_call_timeout").value
        )
        self.pose_after_reinit_timeout = float(
            self.get_parameter("pose_after_reinit_timeout").value
        )
        self.reinitialize_interval = max(
            0.0, float(self.get_parameter("reinitialize_interval").value)
        )
        self.max_reinitialize_attempts = max(
            1, int(self.get_parameter("max_reinitialize_attempts").value)
        )
        self.localization_timeout = float(
            self.get_parameter("localization_timeout").value
        )
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

        # Number of reinitialization service calls made in the CURRENT
        # localization cycle. It is reset when a new 60-second cycle begins.
        self.reinitialize_attempts = 0
        self.reinit_future = None
        self.reinit_call_start_time: Optional[float] = None
        self.reinit_complete_time: Optional[float] = None
        self.localization_start_time: Optional[float] = None

        # main() watches this flag and exits the Python process after success or
        # a terminal failure. Merely changing the state is not enough because
        # rclpy.spin() otherwise runs forever.
        self.finished = False
        self.exit_code = 1

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
            "Auto global localization V2.3 started: automatic reinitialize, "
            "one fast no-pose recovery retry per cycle, measured 360-degree "
            "rotations, and periodic particle reset if localization is not "
            f"successful within {self.reinitialize_interval:.1f}s."
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

    def report_failure(self, message: str):
        """Stop the robot and make main() leave the ROS spin loop."""
        if self.finished:
            return
        self.stop_robot()
        self.state = State.FAILED
        self.exit_code = 1
        self.finished = True
        self.timer.cancel()
        self.get_logger().error(message)

    def start_reinitialization(self, reason: str, new_cycle: bool = False):
        """Issue the global-localization request and verify its completion.

        new_cycle=True starts a fresh localization cycle and resets the
        per-cycle service-attempt counter. A fast no-pose retry uses
        new_cycle=False so it still obeys max_reinitialize_attempts.
        """
        self.stop_robot()

        if new_cycle:
            self.reinitialize_attempts = 0

        if self.reinitialize_attempts >= self.max_reinitialize_attempts:
            self.report_failure(
                "Global-localization retry limit reached in the current "
                "localization cycle without receiving a new /amcl_pose."
            )
            return

        if not self.global_localization_client.service_is_ready():
            return

        self.reinitialize_attempts += 1
        self.stable_start_time = None
        # Only a pose received after this request counts as confirmation.
        self.latest_pose = None
        self.received_amcl_pose = False

        self.get_logger().info(
            f"Calling {self.global_localization_service} automatically "
            f"(attempt {self.reinitialize_attempts}/"
            f"{self.max_reinitialize_attempts}, reason: {reason})."
        )

        self.reinit_future = self.global_localization_client.call_async(
            Empty.Request()
        )
        self.reinit_call_start_time = self.now_sec()
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
            "Robot stopped. The localization process will now exit automatically."
        )
        self.exit_code = 0
        self.finished = True
        self.timer.cancel()

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
                self.start_reinitialization("startup", new_cycle=True)
                return

            if now - self.startup_start_time > self.startup_timeout:
                self.report_failure(
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
                self.report_failure("Global-localization future is missing.")
                return

            if not self.reinit_future.done():
                if (
                    self.reinit_call_start_time is not None
                    and now - self.reinit_call_start_time
                    > self.service_call_timeout
                ):
                    self.report_failure(
                        "Global-localization service call timed out after "
                        f"{self.service_call_timeout:.1f}s."
                    )
                return

            try:
                self.reinit_future.result()
            except Exception as exc:
                self.report_failure(
                    f"Global localization service call failed: {exc}"
                )
                return

            # Reset ONLY rotation measurement, never the particles again.
            self.rotation_direction = 1.0
            self.rotation_index = 1
            self.accumulated_rotation = 0.0
            self.last_odom_yaw = None
            self.stable_start_time = None
            self.reinit_complete_time = now
            if self.localization_start_time is None:
                self.localization_start_time = now

            self.state = State.WAITING_FOR_ODOM
            self.get_logger().info(
                "Global-localization service call completed successfully. "
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
        # 4. Rotate (LEFT 360 -> RIGHT 360 -> LEFT 360 -> ...)
        #    until covariance is stable for stable_duration.
        #    If one cycle exceeds reinitialize_interval, republish particles
        #    and begin a fresh localization cycle.
        # ------------------------------------------------------------
        if self.state in (State.ROTATING, State.VERIFYING):
            # If the service acknowledged the request but AMCL never published
            # a new pose, automatically perform the manual recovery call once.
            if (
                not self.received_amcl_pose
                and self.reinit_complete_time is not None
                and now - self.reinit_complete_time
                > self.pose_after_reinit_timeout
                and self.reinitialize_attempts < self.max_reinitialize_attempts
            ):
                self.get_logger().warn(
                    "No new /amcl_pose after the reinitialization request; "
                    "performing the fast recovery retry for this localization "
                    "cycle."
                )
                self.start_reinitialization("no AMCL pose after first request")
                return

            if (
                self.localization_timeout > 0.0
                and self.localization_start_time is not None
                and now - self.localization_start_time
                > self.localization_timeout
            ):
                self.report_failure(
                    "Localization timed out after "
                    f"{self.localization_timeout:.1f}s without stable "
                    "covariance. Robot has been stopped."
                )
                return

            # V2.3: if this localization cycle has run for one full interval
            # without reaching SUCCESS, republish the global particles and
            # restart the state machine from REINITIALIZING. This also clears
            # the old AMCL pose, stable timer, and rotation measurement.
            if (
                self.reinitialize_interval > 0.0
                and self.reinit_complete_time is not None
                and now - self.reinit_complete_time
                >= self.reinitialize_interval
            ):
                self.get_logger().warn(
                    "Localization has not succeeded within "
                    f"{self.reinitialize_interval:.1f}s. Republishing global "
                    "particles and starting a fresh localization cycle."
                )
                self.start_reinitialization(
                    "periodic particle reset after localization timeout",
                    new_cycle=True,
                )
                return

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
    exit_code = 1

    try:
        # A bounded spin lets report_success()/report_failure() end the process.
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        exit_code = node.exit_code if node.finished else 1
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
        exit_code = 130
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
