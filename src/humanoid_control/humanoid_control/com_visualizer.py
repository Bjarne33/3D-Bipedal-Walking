#!/usr/bin/env python3

import os
import tempfile

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import QoSHistoryPolicy

import pinocchio as pin
import xacro
from ament_index_python.packages import get_package_share_directory

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from tf2_ros import TransformBroadcaster


class CoMVisualizer(Node):

    def __init__(self):
        super().__init__('com_visualizer')

        # --- parameters ---
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("model_states_topic", "/gazebo/model_states")
        self.declare_parameter("model_name", "humanoid")
        self.declare_parameter("marker_rate_hz", 20.0)
        self.declare_parameter("publish_world_markers", True)
        self.declare_parameter("publish_world_to_base_tf", True)

        self.base_frame = self.get_parameter("base_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        self.model_states_topic = self.get_parameter("model_states_topic").value
        self.model_name = self.get_parameter("model_name").value
        self.marker_rate_hz = float(self.get_parameter("marker_rate_hz").value)
        self.publish_world_markers = bool(self.get_parameter("publish_world_markers").value)
        self.publish_world_to_base_tf = bool(self.get_parameter("publish_world_to_base_tf").value)

        resolved_urdf_path = self._resolve_urdf_path()

        # --- load robot model ---
        self.model = pin.buildModelFromUrdf(resolved_urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        self.get_logger().info(f"Loaded URDF with {self.model.nq} DoF from {resolved_urdf_path}")

        # configuration vector
        self.q = pin.neutral(self.model)

        if self.model.njoints > 1:
            root_nq = self.model.joints[1].nq
            root_nv = self.model.joints[1].nv
            self.get_logger().info(
                f"Floating-base check: root joint '{self.model.names[1]}' has nq={root_nq}, nv={root_nv} (expected nq=7, nv=6 for free-flyer)."
            )
        else:
            self.get_logger().info("Floating-base check: model has no joints beyond universe.")

        # joint name -> id
        self.joint_name_to_id = {name: jid for jid, name in enumerate(self.model.names)}

        self.get_logger().info(f"Joint index mapping: {self.joint_name_to_id}")

        # --- ROS interfaces ---
        self.sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
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

        model_states_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.model_states_sub = self.create_subscription(
            ModelStates,
            self.model_states_topic,
            self.model_states_callback,
            model_states_qos,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.latest_com = None
        self.latest_link_coms = None
        self.latest_base_pose = None
        self._missing_model_warned = False
        self._missing_pose_warned = False
        self.publish_timer = self.create_timer(
            1.0 / max(self.marker_rate_hz, 1.0),
            self.publish_latest_markers
        )

        self.get_logger().info(
            f"World CoM source: topic={self.model_states_topic}, model_name={self.model_name}, "
            f"publish_world_markers={self.publish_world_markers}, publish_world_to_base_tf={self.publish_world_to_base_tf}"
        )

    def _resolve_urdf_path(self) -> str:

        pkg_share = get_package_share_directory("humanoid_description")
        model_path = os.path.join(pkg_share, "urdf", "humanoid.xacro")

        doc = xacro.process_file(model_path)
        urdf_xml = doc.toprettyxml(indent="  ")
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        tmp_file.write(urdf_xml)
        tmp_file.close()
        self.get_logger().info(f"Expanded Xacro to temporary URDF: {tmp_file.name}")
        return tmp_file.name

    def joint_callback(self, msg: JointState):

        # update joint configuration
        for name, pos in zip(msg.name, msg.position):
            jid = self.joint_name_to_id.get(name)
            if jid is None:
                continue

            if self.model.joints[jid].nq == 1:
                self.q[self.model.idx_qs[jid]] = pos

        # compute forward kinematics
        pin.forwardKinematics(self.model, self.data, self.q)

        # compute CoM
        com = pin.centerOfMass(self.model, self.data, self.q)

        self.latest_com = com

        link_coms = []
        for jid in range(1, self.model.njoints):
            joint_name = self.model.names[jid]
            com_local = self.model.inertias[jid].lever
            com_world = self.data.oMi[jid].act(com_local)
            link_coms.append((joint_name, com_world))

        self.latest_link_coms = link_coms

    def publish_latest_markers(self):
        if self.latest_com is None:
            return

        self.publish_markers(self.latest_com)

    def _new_marker(self, marker_id: int, marker_type: int, now_msg, frame_id: str, namespace: str):
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
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        return p

    def model_states_callback(self, msg: ModelStates):
        try:
            idx = msg.name.index(self.model_name)
        except ValueError:
            if not self._missing_model_warned:
                self.get_logger().warn(
                    f"Model '{self.model_name}' not found on {self.model_states_topic}. "
                    f"Available models: {list(msg.name)}"
                )
                self._missing_model_warned = True
            return

        self._missing_model_warned = False
        self.latest_base_pose = msg.pose[idx]

        if self.publish_world_to_base_tf:
            self.publish_world_to_base_transform(self.latest_base_pose)

    def publish_world_to_base_transform(self, pose):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.world_frame
        tf_msg.child_frame_id = self.base_frame

        tf_msg.transform.translation.x = pose.position.x
        tf_msg.transform.translation.y = pose.position.y
        tf_msg.transform.translation.z = pose.position.z

        tf_msg.transform.rotation.x = pose.orientation.x
        tf_msg.transform.rotation.y = pose.orientation.y
        tf_msg.transform.rotation.z = pose.orientation.z
        tf_msg.transform.rotation.w = pose.orientation.w

        self.tf_broadcaster.sendTransform(tf_msg)

    @staticmethod
    def _quat_rotate_vector(quat_xyzw, vec_xyz):
        qx, qy, qz, qw = quat_xyzw
        vx, vy, vz = vec_xyz

        q_norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
        if q_norm < 1e-12:
            return [vx, vy, vz]

        qx /= q_norm
        qy /= q_norm
        qz /= q_norm
        qw /= q_norm

        # v' = v + 2 * cross(q_vec, cross(q_vec, v) + q_w * v)
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)

        vpx = vx + qw * tx + (qy * tz - qz * ty)
        vpy = vy + qw * ty + (qz * tx - qx * tz)
        vpz = vz + qw * tz + (qx * ty - qy * tx)
        return [vpx, vpy, vpz]

    def transform_com_to_world(self, com_base):
        if self.latest_base_pose is None:
            if not self._missing_pose_warned:
                self.get_logger().warn(
                    f"No pose received yet from {self.model_states_topic}; world CoM markers will be skipped."
                )
                self._missing_pose_warned = True
            return None

        self._missing_pose_warned = False

        pose = self.latest_base_pose
        rotated = self._quat_rotate_vector(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            [float(com_base[0]), float(com_base[1]), float(com_base[2])],
        )

        return [
            rotated[0] + pose.position.x,
            rotated[1] + pose.position.y,
            rotated[2] + pose.position.z,
        ]

    def _publish_marker_set(self, com, frame_id, namespace, marker_pub, marker_id_offset):
        now_msg = self.get_clock().now().to_msg()

        # CoM sphere
        com_marker = self._new_marker(0 + marker_id_offset, Marker.SPHERE, now_msg, frame_id, namespace)

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
        proj_marker = self._new_marker(1 + marker_id_offset, Marker.SPHERE, now_msg, frame_id, namespace)

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
        line_marker = self._new_marker(2 + marker_id_offset, Marker.LINE_STRIP, now_msg, frame_id, namespace)

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
        self._publish_marker_set(
            com,
            frame_id=self.base_frame,
            namespace="com_base",
            marker_pub=self.marker_pub,
            marker_id_offset=0,
        )

        if self.publish_world_markers:
            com_world = self.transform_com_to_world(com)
            if com_world is not None:
                self._publish_marker_set(
                    com_world,
                    frame_id=self.world_frame,
                    namespace="com_world",
                    marker_pub=self.world_marker_pub,
                    marker_id_offset=100,
                )

def main():

    rclpy.init()
    node = CoMVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()