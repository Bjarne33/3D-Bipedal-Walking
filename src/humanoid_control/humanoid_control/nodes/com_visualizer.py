"""Render the shared estimated CoM as base-frame and world-frame RViz markers."""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from humanoid_control.tools.state_messages import warn_stale_state, xyz
from humanoid_interfaces.msg import RobotState


class CoMVisualizer(Node):
    """Display measured CoM without owning kinematics, sensor subscriptions or base TF."""

    def __init__(self):
        """Declare marker parameters and subscribe to the shared robot measurements."""
        super().__init__('com_visualizer')

        # --- parameters ---
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("marker_rate_hz", 20.0)
        self.declare_parameter("publish_world_markers", True)
        self.declare_parameter("state_timeout_sec", 0.2)

        self.base_frame = self.get_parameter("base_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        self.marker_rate_hz = float(self.get_parameter("marker_rate_hz").value)
        self.publish_world_markers = bool(self.get_parameter("publish_world_markers").value)
        self.state_timeout = float(self.get_parameter("state_timeout_sec").value)
        if self.state_timeout <= 0.0:
            raise ValueError('state_timeout_sec must be positive')

        # --- ROS interfaces ---
        self.sub = self.create_subscription(
            RobotState,
            "/humanoid_estimator/state",
            self.state_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            Marker,
            "/com_marker",
            10
        )

        self.world_marker_pub = self.create_publisher(
            Marker,
            "/com_world_marker",
            10
        )

        self.latest_state = None
        self.publish_timer = self.create_timer(
            1.0 / max(self.marker_rate_hz, 1.0),
            self.publish_latest_markers
        )

    def state_callback(self, msg: RobotState):
        """Cache the estimated CoM and verify the configured marker frames."""
        if msg.header.frame_id != self.world_frame or msg.base_frame_id != self.base_frame:
            self.get_logger().error(
                'RobotState frames do not match the configured marker frames',
                throttle_duration_sec=5.0)
            return
        self.latest_state = msg

    def publish_latest_markers(self):
        """Publish markers for the most recently computed CoM, if any."""
        if self.latest_state is None:
            return

        warn_stale_state(self, self.latest_state, self.state_timeout)
        self.publish_markers(xyz(self.latest_state.com_base))

    def _new_marker(self, marker_id: int, marker_type: int,
                    now_msg, frame_id: str, namespace: str):
        """Create an ADD marker with the common header fields set.

        Args:
            marker_id: Marker id within the namespace.
            marker_type: Marker shape (`Marker` type constant).
            now_msg: Timestamp for the header.
            frame_id: Frame the marker is expressed in.
            namespace: Marker namespace.

        Returns:
            The new marker.
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = now_msg
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        return marker

    @staticmethod
    def _point(x: float, y: float, z: float) -> Point:
        """Create a `Point` message.

        Args:
            x: X coordinate.
            y: Y coordinate.
            z: Z coordinate.

        Returns:
            The point.
        """
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        return p

    def _publish_marker_set(self, com, frame_id, namespace, marker_pub, marker_id_offset):
        """Publish the CoM sphere and its supporting markers.

        Args:
            com: CoM position (x, y, z) in `frame_id`.
            frame_id: Frame the markers are expressed in.
            namespace: Marker namespace.
            marker_pub: Publisher to publish the markers on.
            marker_id_offset: Offset added to the marker ids to keep sets distinct.
        """
        now_msg = self.get_clock().now().to_msg()

        # CoM sphere
        com_marker = self._new_marker(
            0 + marker_id_offset,
            Marker.SPHERE,
            now_msg,
            frame_id,
            namespace)

        com_marker.pose.position.x = float(com[0])
        com_marker.pose.position.y = float(com[1])
        com_marker.pose.position.z = float(com[2])

        com_marker.pose.orientation.w = 1.0

        com_marker.scale.x = 0.05
        com_marker.scale.y = 0.05
        com_marker.scale.z = 0.05

        com_marker.color.r = 1.0
        com_marker.color.g = 0.0
        com_marker.color.b = 0.0
        com_marker.color.a = 1.0

        # Ground projection sphere
        proj_marker = self._new_marker(
            1 + marker_id_offset,
            Marker.SPHERE,
            now_msg,
            frame_id,
            namespace)

        proj_marker.pose.position.x = float(com[0])
        proj_marker.pose.position.y = float(com[1])
        proj_marker.pose.position.z = 0.0

        proj_marker.pose.orientation.w = 1.0

        proj_marker.scale.x = 0.05
        proj_marker.scale.y = 0.05
        proj_marker.scale.z = 0.05

        proj_marker.color.r = 0.0
        proj_marker.color.g = 0.0
        proj_marker.color.b = 1.0
        proj_marker.color.a = 1.0

        # Vertical line marker
        line_marker = self._new_marker(
            2 + marker_id_offset,
            Marker.LINE_STRIP,
            now_msg,
            frame_id,
            namespace)

        line_marker.scale.x = 0.01

        line_marker.color.r = 1.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0

        line_marker.points = [
            self._point(com[0], com[1], com[2]),
            self._point(com[0], com[1], 0.0),
        ]

        # Publish marker set.
        for marker in (com_marker, proj_marker, line_marker):
            marker_pub.publish(marker)

    def publish_markers(self, com):
        """Publish the CoM marker set in the base frame and, if possible, the world frame.

        Args:
            com: CoM position (x, y, z) in the base frame.
        """
        self._publish_marker_set(
            com,
            frame_id=self.base_frame,
            namespace="com_base",
            marker_pub=self.marker_pub,
            marker_id_offset=0,
        )

        if self.publish_world_markers and self.latest_state.base_valid:
            self._publish_marker_set(
                xyz(self.latest_state.com),
                frame_id=self.world_frame,
                namespace="com_world",
                marker_pub=self.world_marker_pub,
                marker_id_offset=100,
            )


def main():
    """Run the CoM visualizer node."""
    rclpy.init()
    node = CoMVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
