"""Publish shared world-frame robot measurements and the dynamic world -> base TF."""

import os

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped, WrenchStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from humanoid_control.tools.humanoid_kinematics import HumanoidKinematics
from humanoid_control.tools.joints import JOINT_NAMES
from humanoid_control.tools.state_estimator import RobotStateEstimator
from humanoid_control.tools.state_messages import (
    estimate_to_message, point, pose_to_se3, warn_stale_state, xyz,
)
from humanoid_interfaces.msg import RobotState


class HumanoidEstimator(Node):
    """Own measurement subscriptions, forward kinematics, contact state and base TF."""

    def __init__(self) -> None:
        """Load the robot model and configure the measurement inputs and outputs."""
        super().__init__('humanoid_estimator')
        defaults = {
            'update_rate_hz': 50.0,
            'odom_topic': '/odom',
            'left_ft_topic': '/left_foot_ft',
            'right_ft_topic': '/right_foot_ft',
            'world_frame': 'world',
            'base_frame': 'base',
            'publish_world_to_base_tf': True,
            'ground_height': 0.0,
            'contact_tolerance': 0.005,
            'ankle_to_sole': 0.05,
            'state_timeout_sec': 0.2,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.world_frame = str(self.get_parameter('world_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.publish_tf = bool(self.get_parameter('publish_world_to_base_tf').value)
        self.ground_height = float(self.get_parameter('ground_height').value)
        self.contact_tolerance = float(self.get_parameter('contact_tolerance').value)
        self.ankle_to_sole = float(self.get_parameter('ankle_to_sole').value)
        self.timeout = float(self.get_parameter('state_timeout_sec').value)
        rate = float(self.get_parameter('update_rate_hz').value)
        if rate <= 0.0 or self.timeout <= 0.0:
            raise ValueError('update_rate_hz and state_timeout_sec must be positive')
        self.kinematics = HumanoidKinematics.from_xacro(os.path.join(
            get_package_share_directory('humanoid_description'), 'urdf', 'humanoid.xacro'))
        self.estimator = RobotStateEstimator()
        self.joints = None
        self.base_stamp = None
        self.wrench_stamps = {}
        self.state_pub = self.create_publisher(RobotState, '/humanoid_estimator/state', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value),
            self._odom_cb, qos_profile_sensor_data)
        for side in ('left', 'right'):
            self.create_subscription(
                WrenchStamped, str(self.get_parameter(f'{side}_ft_topic').value),
                lambda msg, side=side: self._wrench_cb(side, msg), qos_profile_sensor_data)
        self.create_timer(1.0 / rate, self._publish_state)

    def _joint_cb(self, msg: JointState) -> None:
        """Cache joint measurements for shared forward kinematics and walking startup."""
        if len(msg.name) != len(msg.position):
            self.get_logger().error('JointState names and positions have different lengths')
            return
        self.joints = msg
        self.estimator.set_joint_state(msg.name, msg.position, msg.velocity, msg.effort)

    def _odom_cb(self, msg: Odometry) -> None:
        """Accept Gazebo p3d odometry, whose twist is expressed in the world frame."""
        if msg.header.frame_id != self.world_frame or msg.child_frame_id != self.base_frame:
            self.get_logger().error(
                f'Expected odometry {self.world_frame} -> {self.base_frame}, got '
                f'{msg.header.frame_id} -> {msg.child_frame_id}',
                throttle_duration_sec=5.0)
            return
        pose = msg.pose.pose
        q = pose.orientation
        if q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w < 1e-12:
            self.get_logger().error('Odometry has an invalid zero quaternion')
            return
        self.base_stamp = msg.header.stamp
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.estimator.set_base(
            pose_to_se3(pose), stamp, xyz(msg.twist.twist.linear), xyz(msg.twist.twist.angular))
        if self.publish_tf:
            transform = TransformStamped()
            transform.header = msg.header
            transform.child_frame_id = self.base_frame
            transform.transform.translation.x = pose.position.x
            transform.transform.translation.y = pose.position.y
            transform.transform.translation.z = pose.position.z
            transform.transform.rotation = pose.orientation
            self.tf_broadcaster.sendTransform(transform)

    def _wrench_cb(self, side: str, msg: WrenchStamped) -> None:
        """Accumulate the ankle wrench samples for the next estimation period."""
        self.wrench_stamps[side] = msg.header.stamp
        self.estimator.set_wrench(side, xyz(msg.wrench.force), xyz(msg.wrench.torque))

    def _publish_state(self) -> None:
        """Publish a complete measurement snapshot at the configured estimator rate."""
        if (self.joints is None
                or not all(n in self.estimator.joint_state for n in JOINT_NAMES)):
            self.get_logger().warn(
                'Waiting for all controlled joint positions',
                throttle_duration_sec=5.0)
            return
        state = self.estimator.estimate(
            self.kinematics, self.ground_height, self.contact_tolerance, self.ankle_to_sole)
        if state is None:
            self.get_logger().warn(
                'No base odometry: publishing base-frame CoM only; world state unavailable',
                throttle_duration_sec=5.0)
            self.kinematics.set_joint_positions(
                self.estimator.joint_state.keys(),
                [values[0] for values in self.estimator.joint_state.values()])
            msg = RobotState(com_base=point(self.kinematics.compute().com))
        else:
            msg = estimate_to_message(state)
            msg.base_stamp = self.base_stamp
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.base_frame_id = self.base_frame
        msg.joint_state.header = self.joints.header
        msg.joint_state.name = list(self.estimator.joint_state)
        msg.joint_state.position = [v[0] for v in self.estimator.joint_state.values()]
        msg.joint_state.velocity = [v[1] for v in self.estimator.joint_state.values()]
        msg.joint_state.effort = [v[2] for v in self.estimator.joint_state.values()]
        for side, stamp in self.wrench_stamps.items():
            setattr(msg, f'{side}_wrench_stamp', stamp)
            setattr(msg, f'{side}_wrench_available', True)
        warn_stale_state(self, msg, self.timeout)
        self.state_pub.publish(msg)


def main(args=None):
    """Run the shared estimator node."""
    rclpy.init(args=args)
    node = HumanoidEstimator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
