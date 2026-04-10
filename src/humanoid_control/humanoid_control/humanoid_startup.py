import rclpy
import time

from rclpy.node import Node
from controller_manager_msgs.srv import ListControllers
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from enum import Enum



class StartupState(Enum):
    WAIT_FOR_CONTROLLERS = 0
    PAUSE_SIM = 1
    SEND_POSE = 2
    WAIT_POSE = 3
    READY = 4
    ERROR = 5


class HumanoidStartupManager(Node):

    def __init__(self):
        super().__init__('humanoid_startup_manager')

        # ---------------- Parameters ----------------
        self.declare_parameter("controller_name", "effort_position_controller")
        self.declare_parameter("joint_state_broadcaster", "joint_state_broadcaster")
        self.declare_parameter("pose_tolerance", 0.02)
        self.declare_parameter("startup_timeout_sec", 60.0)

        self.controller_name = self.get_parameter("controller_name").value
        self.jsb_name = self.get_parameter("joint_state_broadcaster").value
        self.pose_tolerance = self.get_parameter("pose_tolerance").value
        self.timeout_sec = self.get_parameter("startup_timeout_sec").value

        # ---------------- State ----------------
        self.state = StartupState.WAIT_FOR_CONTROLLERS
        self.start_time = self.get_clock().now()

        # ---------------- Target Pose ----------------
        self.joint_names = [
            "left_hip_roll",
            "left_hip_yaw",
            "left_hip_pitch",
            "left_knee",
            "left_ankle_roll",
            "left_ankle_pitch",
            "right_hip_roll",
            "right_hip_yaw",
            "right_hip_pitch",
            "right_knee",
            "right_ankle_roll",
            "right_ankle_pitch"
        ]

        self.target_positions = [
            0.0,
            0.0,
            0.275,
            -0.5,
            0.0,
            0.25,
            0.0,
            0.0,
            0.275,
            -0.5,
            0.0,
            0.25
        ]

        self.current_joint_state = None
        self.controllers_future = None

        # ---------------- Clients ----------------
        self.cm_list_client = self.create_client(
            ListControllers,
            '/controller_manager/list_controllers'
        )

        self.pause_client = self.create_client(Empty, '/pause_physics')
        self.unpause_client = self.create_client(Empty, '/unpause_physics')

        # ---------------- Publisher ----------------
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            f'/{self.controller_name}/joint_trajectory',
            10
        )

        # ---------------- Subscriber ----------------
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # ---------------- Timer ----------------
        self.timer = self.create_timer(0.1, self.update)

        self.get_logger().info("Humanoid Startup Manager initialized")

    # ============================================================
    # Core Update Loop
    # ============================================================

    def update(self):
        if (self.state != StartupState.READY and self.timeout_exceeded()):
            self.fail("Startup timeout exceeded")
            return

        if self.state == StartupState.WAIT_FOR_CONTROLLERS:
            if self.controllers_active():
                self.get_logger().info("Controllers active")
                self.state = StartupState.PAUSE_SIM
            else:
                self.get_logger().info("Unpausing and pausing simulation to trigger controller activation")
                self.unpause_client.call_async(Empty.Request())
                time.sleep(0.005)
                self.pause_client.call_async(Empty.Request())

        elif self.state == StartupState.PAUSE_SIM:
            self.pause_sim()
            self.state = StartupState.SEND_POSE

        elif self.state == StartupState.SEND_POSE:
            self.send_initial_pose()
            self.unpause_sim()
            self.state = StartupState.WAIT_POSE

        elif self.state == StartupState.WAIT_POSE:
            if self.pose_reached():
                self.get_logger().info("Initial pose reached")
                self.state = StartupState.READY
                self.timer.cancel()
        
    # ============================================================
    # State Checks
    # ============================================================

    def controllers_active(self):
        if not self.cm_list_client.service_is_ready():
            self.get_logger().info("Controller manager service not ready")
            return False

        if self.controllers_future is None:
            self.controllers_future = self.cm_list_client.call_async(
                ListControllers.Request()
            )
            self.get_logger().info("Requested controller list")
            return False

        if not self.controllers_future.done():
            self.get_logger().info("Waiting for controller list response")
            return False

        try:
            response = self.controllers_future.result()
        except Exception as exc:
            self.get_logger().warn(f"ListControllers failed: {exc}")
            self.controllers_future = None
            return False

        self.controllers_future = None

        self.get_logger().info(f"ListControllers response: {response}")
        if response is None:
            self.get_logger().info("Returning False: Future result is None")
            return False

        controllers = response.controller
        self.get_logger().info(f"Controllers: {controllers}")

        active = {c.name: c.state for c in controllers}

        return (
            active.get(self.controller_name) == "active" and
            active.get(self.jsb_name) == "active"
        )

    def pose_reached(self):

        if self.current_joint_state is None:
            return False

        for name, target in zip(self.joint_names, self.target_positions):
            if name not in self.current_joint_state.name:
                return False

            idx = self.current_joint_state.name.index(name)
            current = self.current_joint_state.position[idx]

            if abs(current - target) > self.pose_tolerance:
                return False

        return True

    # ============================================================
    # Actions
    # ============================================================

    def send_initial_pose(self):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = self.target_positions
        point.time_from_start.sec = 2

        msg.points.append(point)
        self.traj_pub.publish(msg)

        self.get_logger().info("Initial pose command sent")

    def pause_sim(self):
        if self.pause_client.wait_for_service(timeout_sec=1.0):
            self.pause_client.call_async(Empty.Request())
            self.get_logger().info("Simulation paused")

    def unpause_sim(self):
        if self.unpause_client.wait_for_service(timeout_sec=1.0):
            self.unpause_client.call_async(Empty.Request())
            self.get_logger().info("Simulation unpaused")

    # ============================================================
    # Utilities
    # ============================================================

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def timeout_exceeded(self):
        elapsed = (
            self.get_clock().now() - self.start_time
        ).nanoseconds * 1e-9
        return elapsed > self.timeout_sec

    def fail(self, reason):
        self.get_logger().error(reason)
        self.state = StartupState.ERROR


# ============================================================
# Main
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidStartupManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()