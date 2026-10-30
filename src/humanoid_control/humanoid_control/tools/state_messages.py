"""ROS message conversions for the shared measured robot state."""

import numpy as np
import pinocchio as pin
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3

from humanoid_control.tools.state_estimator import RobotEstimate
from humanoid_interfaces.msg import RobotState


def xyz(value) -> np.ndarray:
    """Read a ROS point or vector as an array."""
    return np.array([value.x, value.y, value.z])


def point(value) -> Point:
    """Convert an array to a ROS point."""
    return Point(x=float(value[0]), y=float(value[1]), z=float(value[2]))


def vector(value) -> Vector3:
    """Convert an array to a ROS vector."""
    return Vector3(x=float(value[0]), y=float(value[1]), z=float(value[2]))


def pose_to_se3(pose: Pose) -> pin.SE3:
    """Convert a ROS pose to a rigid transform."""
    q = pose.orientation
    return pin.XYZQUATToSE3(np.array([
        pose.position.x, pose.position.y, pose.position.z, q.x, q.y, q.z, q.w,
    ]))


def se3_to_pose(transform: pin.SE3) -> Pose:
    """Convert a rigid transform to a ROS pose."""
    q = pin.Quaternion(transform.rotation).coeffs()
    return Pose(position=point(transform.translation), orientation=Quaternion(
        x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])))


def estimate_to_message(state: RobotEstimate) -> RobotState:
    """Encode world-frame measurements; the caller supplies headers and source stamps."""
    return RobotState(
        base_valid=True,
        base_pose=se3_to_pose(state.base),
        base_angular_velocity=vector(state.base_omega),
        com=point(state.com),
        com_base=point(state.base.inverse().act(state.com)),
        com_velocity=vector(state.com_vel),
        left_foot_pose=se3_to_pose(state.feet['left']),
        right_foot_pose=se3_to_pose(state.feet['right']),
        left_normal_force=float(state.foot_force['left']),
        right_normal_force=float(state.foot_force['right']),
        left_contact=state.contact['left'],
        right_contact=state.contact['right'],
        support_points=[point(p) for p in state.support_points],
        zmp=point(state.zmp),
        zmp_valid=state.zmp_valid,
    )


def message_to_estimate(msg: RobotState) -> RobotEstimate:
    """Decode measured state without applying a gait frame or a velocity filter."""
    if not msg.base_valid:
        raise ValueError('World-frame robot state is unavailable without base odometry')
    return RobotEstimate(
        base=pose_to_se3(msg.base_pose),
        base_omega=xyz(msg.base_angular_velocity),
        com=xyz(msg.com),
        com_vel=xyz(msg.com_velocity),
        feet={'left': pose_to_se3(msg.left_foot_pose),
              'right': pose_to_se3(msg.right_foot_pose)},
        foot_force={'left': msg.left_normal_force, 'right': msg.right_normal_force},
        contact={'left': msg.left_contact, 'right': msg.right_contact},
        support_points=np.array([xyz(p) for p in msg.support_points]).reshape(-1, 3),
        zmp=xyz(msg.zmp),
        zmp_valid=msg.zmp_valid,
    )


def warn_stale_state(node, msg: RobotState, timeout: float) -> None:
    """Report stale sources while preserving the previous cached-measurement behavior."""
    now = node.get_clock().now().nanoseconds * 1e-9
    sources = [('estimate', msg.header.stamp), ('joints', msg.joint_state.header.stamp)]
    if msg.base_valid:
        sources.append(('base', msg.base_stamp))
    if msg.left_wrench_available:
        sources.append(('left wrench', msg.left_wrench_stamp))
    if msg.right_wrench_available:
        sources.append(('right wrench', msg.right_wrench_stamp))
    stale = [name for name, stamp in sources
             if not 0.0 <= now - (stamp.sec + stamp.nanosec * 1e-9) <= timeout]
    if stale:
        node.get_logger().warn(
            f'Stale robot state ({", ".join(stale)}); retaining cached measurements',
            throttle_duration_sec=5.0,
        )
