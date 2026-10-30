"""Regression tests for shared estimation, frame projection and ROS consumers."""

import os
import time
from types import SimpleNamespace

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped, WrenchStamped
from nav_msgs.msg import Odometry
import numpy as np
import pinocchio as pin
import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.exceptions import InvalidHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker

from humanoid_control.nodes.com_visualizer import CoMVisualizer
from humanoid_control.nodes.humanoid_estimator import HumanoidEstimator
from humanoid_control.nodes.humanoid_walk import HumanoidWalk
from humanoid_control.tools.humanoid_kinematics import HumanoidKinematics
from humanoid_control.tools.joints import JOINT_NAMES, STANDING_POSE
from humanoid_control.tools.state_estimator import RobotStateEstimator, WalkStateProjection
from humanoid_control.tools.state_messages import (
    estimate_to_message, message_to_estimate, pose_to_se3, se3_to_pose, warn_stale_state, xyz,
)
from humanoid_interfaces.action import Walk
from humanoid_interfaces.msg import BalanceState, RobotState


@pytest.fixture(scope='module')
def kinematics():
    """Load the real robot model in the standing pose."""
    kin = HumanoidKinematics.from_xacro(os.path.join(
        get_package_share_directory('humanoid_description'), 'urdf', 'humanoid.xacro'))
    kin.set_joint_positions(STANDING_POSE.keys(), STANDING_POSE.values())
    return kin


def test_world_projection_preserves_reference_filter_and_dcm(kinematics):
    """Keep translated/rotated walk anchoring and per-control-tick filtering unchanged."""
    reference = pin.SE3(pin.exp3(np.array([0.1, -0.2, 0.3])), np.array([1.0, -2.0, 0.4]))
    estimator = RobotStateEstimator()
    assert estimator.estimate(kinematics, 0.0, 0.005, 0.05) is None
    estimator.set_joint_state(list(STANDING_POSE), list(STANDING_POSE.values()))
    estimator.set_base(reference, 1.0, [0.2, -0.1, 0.05], [0.0, 0.0, 0.0])
    state = estimator.estimate(kinematics, -10.0, 0.005, 0.05)

    class Frame:
        forward = np.array([0.0, 1.0, 0.0])
        left = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])

        def to_plane(self, value):
            return np.array([value @ self.forward, value @ self.left, value @ self.up])

    frame = Frame()
    projection = WalkStateProjection(reference, alpha=0.25)
    first = projection.project(state, frame, 3.7)
    np.testing.assert_allclose(
        first.com, frame.to_plane(reference.inverse().act(state.com))[:2])
    np.testing.assert_allclose(
        first.com_vel, frame.to_plane(reference.rotation.T @ state.com_vel)[:2])
    first_velocity = state.com_vel.copy()
    state.com_vel = np.array([-0.3, 0.4, 0.2])
    second = projection.project(state, frame, 3.7)
    filtered = 0.75 * first_velocity + 0.25 * state.com_vel
    np.testing.assert_allclose(
        second.com_vel, frame.to_plane(reference.rotation.T @ filtered)[:2])
    np.testing.assert_allclose(second.dcm, second.com + second.com_vel / 3.7)
    # A new walk must not inherit the previous walk's filter history.
    restarted = WalkStateProjection(reference, 0.25).project(state, frame, 3.7)
    np.testing.assert_allclose(
        restarted.com_vel, frame.to_plane(reference.rotation.T @ state.com_vel)[:2])
    assert second.support_hull.shape == (0, 2)
    assert not second.zmp_valid
    np.testing.assert_array_equal(second.zmp, np.zeros(2))


def test_message_round_trip_preserves_world_geometry_and_validity(kinematics):
    """Transport all state used by the balance controller without losing dimensions."""
    estimator = RobotStateEstimator()
    estimator.set_joint_state(list(STANDING_POSE), list(STANDING_POSE.values()))
    estimator.set_base(
        pin.SE3(pin.exp3(np.array([0.1, 0.05, 0.4])), np.array([1.0, -0.5, 0.2])),
        1.0, [0.1, 0.2, 0.3], [0.2, -0.3, 0.1])
    estimator.set_wrench('left', [10.0, -5.0, 300.0], [1.0, -2.0, 0.0])
    estimator.set_wrench('right', [0.0, 0.0, 5.0], [0.0, 0.0, 0.0])
    original = estimator.estimate(kinematics, 0.0, 0.005, 0.05)
    encoded = estimate_to_message(original)
    decoded = message_to_estimate(encoded)
    for field in ('com', 'com_vel', 'base_omega', 'support_points', 'zmp'):
        np.testing.assert_allclose(getattr(decoded, field), getattr(original, field))
    np.testing.assert_allclose(decoded.base.homogeneous, original.base.homogeneous)
    for side in ('left', 'right'):
        np.testing.assert_allclose(decoded.feet[side].homogeneous,
                                   original.feet[side].homogeneous)
    assert decoded.contact == {'left': True, 'right': False}
    assert decoded.foot_force == original.foot_force
    assert decoded.zmp_valid
    np.testing.assert_allclose(xyz(encoded.com_base), original.base.inverse().act(original.com))
    assert len(decoded.support_points) == 4
    with pytest.raises(ValueError, match='unavailable'):
        message_to_estimate(RobotState())


def test_stale_warning_checks_sources_not_just_publication_time():
    """A freshly published estimate must not hide old sensor inputs."""
    warnings = []
    node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)),
        get_logger=lambda: SimpleNamespace(warn=lambda text, **kw: warnings.append(text)))
    msg = RobotState(base_valid=True)
    msg.header.stamp.sec = 10
    msg.base_stamp.sec = 9
    msg.joint_state.header.stamp.sec = 10
    msg.left_wrench_available = True
    msg.left_wrench_stamp.sec = 8
    warn_stale_state(node, msg, 0.2)
    assert len(warnings) == 1
    assert 'base' in warnings[0] and 'left wrench' in warnings[0]
    assert 'joints' not in warnings[0] and 'estimate,' not in warnings[0]
    warnings.clear()
    msg.base_stamp.sec = msg.left_wrench_stamp.sec = 10
    warn_stale_state(node, msg, 0.2)
    assert not warnings


@pytest.fixture
def ros_stack():
    """Run the estimator, walker and visualizer with synthetic stamped sensor topics."""
    rclpy.init()
    estimator = HumanoidEstimator()
    walker = HumanoidWalk()
    visualizer = CoMVisualizer()
    probe = Node('estimator_test_probe')
    executor = MultiThreadedExecutor(num_threads=4)
    nodes = (estimator, walker, visualizer, probe)
    for node in nodes:
        executor.add_node(node)
    states, transforms, base_markers, world_markers, trajectories, balances = (
        [] for _ in range(6))
    probe.create_subscription(RobotState, '/humanoid_estimator/state', states.append, 10)
    probe.create_subscription(TFMessage, '/tf', transforms.append, 10)
    probe.create_subscription(Marker, '/com_marker', base_markers.append, 100)
    probe.create_subscription(Marker, '/com_world_marker', world_markers.append, 100)
    probe.create_subscription(
        JointTrajectory, '/effort_position_controller/joint_trajectory', trajectories.append, 10)
    probe.create_subscription(BalanceState, '/humanoid_walk/balance_state', balances.append, 10)
    joint_pub = probe.create_publisher(JointState, '/joint_states', 10)
    odom_pub = probe.create_publisher(Odometry, '/odom', qos_profile_sensor_data)
    wrench_pubs = {
        side: probe.create_publisher(WrenchStamped, f'/{side}_foot_ft', qos_profile_sensor_data)
        for side in ('left', 'right')
    }
    inputs = SimpleNamespace(base=True, wrenches=True)
    base = pin.SE3(pin.exp3(np.array([0.02, -0.03, 0.1])), np.array([0.1, -0.2, 0.0]))

    def publish():
        stamp = probe.get_clock().now().to_msg()
        joints = JointState()
        joints.header.stamp = stamp
        joints.name = list(JOINT_NAMES)
        joints.position = [STANDING_POSE[n] for n in JOINT_NAMES]
        joint_pub.publish(joints)
        if inputs.base:
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = 'world'
            odom.child_frame_id = 'base'
            odom.pose.pose = se3_to_pose(base)
            odom_pub.publish(odom)
        if inputs.wrenches:
            for side, pub in wrench_pubs.items():
                wrench = WrenchStamped()
                wrench.header.stamp = stamp
                wrench.header.frame_id = f'{side}_foot_link'
                wrench.wrench.force.z = 300.0 if side == 'left' else 5.0
                pub.publish(wrench)

    probe.create_timer(0.01, publish)

    def wait(predicate, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
            if predicate():
                return
        pytest.fail('Timed out waiting for ROS estimation/consumer result')

    try:
        yield SimpleNamespace(
            estimator=estimator, walker=walker, visualizer=visualizer, probe=probe,
            inputs=inputs, base=base, states=states, transforms=transforms,
            base_markers=base_markers, world_markers=world_markers,
            trajectories=trajectories, balances=balances, wait=wait)
    finally:
        executor.shutdown(timeout_sec=10.0)
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.shutdown()


def test_ros_transport_tf_markers_and_single_sensor_owner(ros_stack):
    """Verify actual ROS publication, consumers, marker IDs and TF pose/timestamp."""
    stack = ros_stack
    stack.wait(lambda: stack.states and stack.transforms
               and len(stack.base_markers) >= 3 and len(stack.world_markers) >= 3)
    state = stack.states[-1]
    assert state.base_valid and state.zmp_valid
    assert state.left_contact and not state.right_contact
    assert state.left_wrench_available and state.right_wrench_available
    assert list(state.joint_state.name) == list(JOINT_NAMES)
    np.testing.assert_allclose(pose_to_se3(state.base_pose).homogeneous, stack.base.homogeneous)
    transform = stack.transforms[-1].transforms[0]
    assert isinstance(transform, TransformStamped)
    assert transform.header.frame_id == 'world' and transform.child_frame_id == 'base'
    assert transform.header.stamp.sec > 0
    np.testing.assert_allclose(xyz(transform.transform.translation), stack.base.translation)
    assert {m.id for m in stack.base_markers} == {0, 1, 2}
    assert {m.id for m in stack.world_markers} == {100, 101, 102}
    base_sphere = next(m for m in reversed(stack.base_markers) if m.id == 0)
    world_sphere = next(m for m in reversed(stack.world_markers) if m.id == 100)
    np.testing.assert_allclose(xyz(base_sphere.pose.position), xyz(state.com_base))
    np.testing.assert_allclose(xyz(world_sphere.pose.position), xyz(state.com))
    for topic in ('/odom', '/joint_states', '/left_foot_ft', '/right_foot_ft'):
        subscribers = stack.probe.get_subscriptions_info_by_topic(topic)
        assert {info.node_name for info in subscribers} == {'humanoid_estimator'}
    stack.wait(lambda: stack.walker.latest_state is not None)
    assert stack.walker._have_all_joints()


def test_base_only_state_and_disabled_tf(ros_stack):
    """Preserve base-only markers and walking readiness without base odometry."""
    stack = ros_stack
    stack.inputs.base = False
    stack.inputs.wrenches = False
    stack.estimator.publish_tf = False
    stack.wait(lambda: stack.states and len(stack.base_markers) >= 3
               and stack.walker._have_all_joints())
    assert not stack.states[-1].base_valid
    assert not stack.states[-1].zmp_valid
    assert not stack.world_markers
    assert not stack.transforms
    stack.inputs.base = True
    stack.wait(lambda: stack.states[-1].base_valid and stack.world_markers)
    assert not stack.transforms  # pinned mode still measures state but never duplicates base TF


@pytest.mark.xfail(
    raises=InvalidHandle,
    reason='Existing per-walk rate-timer destruction race in the Humble executor; see README',
    strict=False,
)
def test_walk_action_finite_cancel_and_stop(ros_stack):
    """Keep finite, canceled and service-stopped walk interfaces working end to end."""
    stack = ros_stack
    walker = stack.walker
    walker.set_parameters([
        Parameter('step_time', value=0.25),
        Parameter('hold_time', value=0.02),
        Parameter('init_time', value=0.1),
        Parameter('final_time', value=0.1),
        Parameter('stop_transition_time', value=0.1),
        Parameter('balance_enabled', value=False),
    ])
    stack.wait(lambda: walker._have_all_joints() and walker.latest_state.base_valid)
    client = ActionClient(stack.probe, Walk, '/humanoid_walk/walk')
    stop = stack.probe.create_client(Trigger, '/humanoid_walk/stop')
    try:
        assert client.wait_for_server(timeout_sec=5.0)
        for mode in ('finite', 'balanced', 'cancel', 'stop'):
            stack.balances.clear()
            walker.set_parameters([Parameter('balance_enabled', value=(mode == 'balanced'))])
            finite = mode in ('finite', 'balanced')
            future = client.send_goal_async(Walk.Goal(num_steps=1 if finite else 0))
            stack.wait(future.done)
            goal = future.result()
            assert goal.accepted
            result = goal.get_result_async()
            if not finite:
                stack.wait(lambda: stack.balances and stack.balances[-1].phase == 'step')
                if mode == 'cancel':
                    canceled = goal.cancel_goal_async()
                    stack.wait(canceled.done)
                    assert canceled.result().goals_canceling
                else:
                    assert stop.wait_for_service(timeout_sec=5.0)
                    stopped = stop.call_async(Trigger.Request())
                    stack.wait(stopped.done)
                    assert stopped.result().success
            stack.wait(result.done, timeout=30.0)
            outcome = result.result()
            assert outcome.result.success == (mode != 'cancel')
            assert outcome.result.steps_completed >= 1
            assert outcome.status == (5 if mode == 'cancel' else 4)
        assert stack.trajectories
        command = stack.trajectories[-1]
        assert list(command.joint_names) == list(JOINT_NAMES)
        assert len(command.points[0].positions) == 12
        assert command.points[0].time_from_start.sec == 0
        assert command.points[0].time_from_start.nanosec == 60_000_000
    finally:
        client.destroy()
        stack.probe.destroy_client(stop)
