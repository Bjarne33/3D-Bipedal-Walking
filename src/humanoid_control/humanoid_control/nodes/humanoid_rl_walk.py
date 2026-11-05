"""RL walking node: deploy the trained PPO policy as joint position targets.

Start a walk with the `humanoid_interfaces/action/RlWalk` action on `/humanoid_rl_walk/rl_walk`
(`forward_speed` [m/s], clamped to the trained range; `duration` [s], 0 = walk until canceled).
Canceling the goal, reaching the duration or calling `/humanoid_rl_walk/stop` ends the walk and
commands the standing pose at once.

One policy step is computed for every `/humanoid_estimator/state` message, whose 50 Hz rate
equals the policy control step `ctrl_dt`. The targets are published as single-point
trajectories to the trajectory controller.
"""

import os
import threading
from typing import Optional

import numpy as np
import rclpy
import xacro
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from humanoid_control.tools.joints import JOINT_NAMES
from humanoid_control.tools.rl_policy import (
    advance_phase, build_observation, clamp_speed, joint_limits_from_urdf,
    quaternion_to_matrix, RlPolicy)
from humanoid_control.tools.state_messages import warn_stale_state
from humanoid_interfaces.action import RlWalk
from humanoid_interfaces.msg import RobotState

# Base-frame forward direction of the robot (walking is along base +Y).
FORWARD_AXIS = 1


def _stamp_to_sec(stamp) -> float:
    """Convert a ROS time stamp to seconds.

    Args:
        stamp: `builtin_interfaces/Time`.

    Returns:
        Time in seconds.
    """
    return stamp.sec + stamp.nanosec * 1e-9


class _Session:
    """Mutable state of one RL walk goal."""

    def __init__(self, command: float, duration: float, action_size: int) -> None:
        """Start a session at clock phase 0 with a zero last action.

        Args:
            command: Forward speed command [m/s].
            duration: Walk duration [s]; 0 walks until stopped.
            action_size: Number of policy actions.
        """
        self.command = command
        self.duration = duration
        self.active = True
        self.phase = 0.0
        self.last_action = np.zeros(action_size)
        self.start_time: Optional[float] = None
        self.last_step_time: Optional[float] = None
        self.elapsed = 0.0
        self.duration_reached = False
        self.base_lost = False
        self.measured_speed = float('nan')


class HumanoidRlWalk(Node):
    """RL policy walking node exposing the rl_walk action and a stop service."""

    def __init__(self) -> None:
        """Load the policy and joint limits and create the action, service and I/O."""
        super().__init__('humanoid_rl_walk')
        share = get_package_share_directory('humanoid_control')
        self.declare_parameter('policy_path', '')
        self.declare_parameter('controller_name', 'effort_position_controller')
        self.declare_parameter('min_forward_speed', 0.05)
        self.declare_parameter('max_forward_speed', 0.2)
        self.declare_parameter('target_time_from_start', 0.0)
        self.declare_parameter('state_timeout_sec', 0.2)

        policy_path = str(self.get_parameter('policy_path').value) or os.path.join(
            share, 'policies', 'policy.npz')
        self.policy = RlPolicy(policy_path)
        if self.policy.joint_names != list(JOINT_NAMES):
            raise ValueError(f'Policy joint order {self.policy.joint_names} differs from '
                             f'{JOINT_NAMES}')
        self.speed_range = (float(self.get_parameter('min_forward_speed').value),
                            float(self.get_parameter('max_forward_speed').value))
        if self.speed_range[0] > self.speed_range[1]:
            raise ValueError('min_forward_speed must not exceed max_forward_speed')
        self.target_time = float(self.get_parameter('target_time_from_start').value)
        if self.target_time < 0.0:
            raise ValueError('target_time_from_start must not be negative')
        self.state_timeout = float(self.get_parameter('state_timeout_sec').value)
        if self.state_timeout <= 0.0:
            raise ValueError('state_timeout_sec must be positive')

        xacro_path = os.path.join(
            get_package_share_directory('humanoid_description'), 'urdf', 'humanoid.xacro')
        urdf = xacro.process_file(xacro_path, mappings={'floating': 'false'}).toxml()
        self.lower, self.upper = joint_limits_from_urdf(urdf, JOINT_NAMES)

        self.lock = threading.Lock()
        self.session: Optional[_Session] = None
        self.stop_requested = threading.Event()
        self.latest_state: Optional[RobotState] = None

        group = ReentrantCallbackGroup()
        self.create_subscription(
            RobotState, '/humanoid_estimator/state', self._state_cb, 10, callback_group=group)
        controller = str(self.get_parameter('controller_name').value)
        self.traj_pub = self.create_publisher(
            JointTrajectory, f'/{controller}/joint_trajectory', 10)
        self.create_service(
            Trigger, '/humanoid_rl_walk/stop', self._stop_cb, callback_group=group)
        self.action_server = ActionServer(
            self, RlWalk, '/humanoid_rl_walk/rl_walk',
            execute_callback=self._execute,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=group,
        )
        self.get_logger().info(
            f'Humanoid RL walk node ready with policy {policy_path} '
            f'(ctrl_dt {self.policy.ctrl_dt:g} s, gait period {self.policy.gait_period:g} s, '
            f'speed range [{self.speed_range[0]:g}, {self.speed_range[1]:g}] m/s): send a goal '
            'to /humanoid_rl_walk/rl_walk (duration = 0 walks until canceled)')

    # ------------------------------------------------------------------ callbacks
    def _stop_cb(self, _req, response):
        """Handle `/humanoid_rl_walk/stop`: end the walk and command the standing pose.

        Args:
            _req: Unused trigger request.
            response: Response to fill in.

        Returns:
            The response; `success` is True if a walk was running.
        """
        with self.lock:
            walking = self.session is not None and self.session.active
        if walking:
            self.stop_requested.set()
        response.success = walking
        response.message = 'stopping' if walking else 'not walking'
        return response

    def _goal_cb(self, goal) -> GoalResponse:
        """Accept a goal unless the state is incomplete, the goal invalid or a walk runs.

        Args:
            goal: RlWalk goal request.

        Returns:
            ACCEPT or REJECT.
        """
        if not np.isfinite(goal.forward_speed) or not np.isfinite(goal.duration) \
                or goal.duration < 0.0:
            self.get_logger().warn('Goal rejected: forward_speed must be finite and '
                                   'duration finite and non-negative')
            return GoalResponse.REJECT
        state = self.latest_state
        if state is None or not state.base_valid:
            self.get_logger().warn('Goal rejected: no robot state with a valid base yet')
            return GoalResponse.REJECT
        if not set(JOINT_NAMES) <= set(state.joint_state.name):
            self.get_logger().warn('Goal rejected: joint states incomplete')
            return GoalResponse.REJECT
        with self.lock:
            if self.session is not None:
                self.get_logger().warn('Goal rejected: already walking')
                return GoalResponse.REJECT
            # Reserve the slot; `_execute` replaces it with the real session.
            self.session = _Session(0.0, 0.0, self.policy.action_size)
            self.session.active = False
        return GoalResponse.ACCEPT

    def _cancel_cb(self, _goal) -> CancelResponse:
        """Accept a cancel request; the walk ends at the next check.

        Args:
            _goal: Unused goal handle.

        Returns:
            Always ACCEPT.
        """
        return CancelResponse.ACCEPT

    def _state_cb(self, msg: RobotState) -> None:
        """Cache the robot state and, while walking, run one policy step.

        Args:
            msg: Shared robot state from the estimator.
        """
        self.latest_state = msg
        with self.lock:
            session = self.session
            if session is None or not session.active:
                return
            now = _stamp_to_sec(msg.header.stamp)
            dt = self.policy.ctrl_dt
            if session.last_step_time is not None:
                period = now - session.last_step_time
                if period < 0.5 * dt:
                    return
                if abs(period - dt) > 0.25 * dt:
                    self.get_logger().warn(
                        f'Robot state period {period:.3f} s differs from the policy step '
                        f'{dt:g} s', throttle_duration_sec=5.0)
            if session.start_time is None:
                session.start_time = now
            session.elapsed = now - session.start_time
            if session.duration > 0.0 and session.elapsed >= session.duration:
                session.duration_reached = True
                return
            if not msg.base_valid:
                session.base_lost = True
                return
            warn_stale_state(self, msg, self.state_timeout)
            index = {name: i for i, name in enumerate(msg.joint_state.name)}
            if any(name not in index for name in JOINT_NAMES) or \
                    len(msg.joint_state.velocity) != len(msg.joint_state.name):
                self.get_logger().warn('Joint state incomplete; skipping policy step',
                                       throttle_duration_sec=5.0)
                return
            positions = [msg.joint_state.position[index[name]] for name in JOINT_NAMES]
            velocities = [msg.joint_state.velocity[index[name]] for name in JOINT_NAMES]
            orientation = msg.base_pose.orientation
            rotation = quaternion_to_matrix(
                (orientation.x, orientation.y, orientation.z, orientation.w))
            omega = msg.base_angular_velocity
            obs = build_observation(
                rotation, (omega.x, omega.y, omega.z), session.command, positions, velocities,
                self.policy.default_pose, session.last_action, session.phase)
            action = self.policy(obs)
            self._publish_targets(self.policy.joint_targets(action, self.lower, self.upper))
            session.last_action = action
            session.phase = advance_phase(session.phase, dt, self.policy.gait_period)
            session.last_step_time = now
            forward = rotation[:, FORWARD_AXIS].copy()
            forward[2] = 0.0
            norm = np.linalg.norm(forward)
            if norm > 1e-6:
                v = msg.com_velocity
                session.measured_speed = float(np.dot(forward / norm, (v.x, v.y, v.z)))

    # ------------------------------------------------------------------ output
    def _publish_targets(self, positions: np.ndarray) -> None:
        """Publish a single-point joint trajectory.

        Args:
            positions: Joint position targets in `JOINT_NAMES` order [rad].
        """
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        sec = int(self.target_time)
        point.time_from_start.sec = sec
        point.time_from_start.nanosec = int(round((self.target_time - sec) * 1e9))
        msg.points = [point]
        self.traj_pub.publish(msg)

    # ------------------------------------------------------------------ action
    def _execute(self, goal_handle):
        """Run the policy until the goal is canceled, stopped or its duration elapses.

        Args:
            goal_handle: Accepted RlWalk goal handle.

        Returns:
            The RlWalk result.
        """
        goal = goal_handle.request
        low, high = self.speed_range
        command, clamped = clamp_speed(goal.forward_speed, low, high)
        if clamped:
            self.get_logger().warn(
                f'forward_speed {goal.forward_speed:g} m/s is outside the trained range '
                f'[{low:g}, {high:g}]; using {command:g} m/s')
        self.stop_requested.clear()
        session = _Session(command, float(goal.duration), self.policy.action_size)
        with self.lock:
            self.session = session
        self.get_logger().info(
            f'RL walk started: {command:g} m/s, '
            + (f'{goal.duration:g} s' if goal.duration > 0.0 else 'until canceled'))

        result = RlWalk.Result()
        feedback = RlWalk.Feedback()
        reason = 'shutdown'
        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    reason = 'canceled'
                    break
                if self.stop_requested.is_set():
                    reason = 'stopped'
                    break
                with self.lock:
                    if session.duration_reached:
                        reason = 'finished'
                        break
                    if session.base_lost:
                        reason = 'base odometry lost'
                        break
                    feedback.elapsed = session.elapsed
                    feedback.forward_speed_command = session.command
                    feedback.forward_speed_measured = session.measured_speed
                    feedback.gait_phase = session.phase
                goal_handle.publish_feedback(feedback)
                self.stop_requested.wait(0.1)
        finally:
            with self.lock:
                session.active = False
                self._publish_targets(self.policy.default_pose)
                self.session = None
        self.get_logger().info(f'RL walk ended ({reason}) after {session.elapsed:.2f} s; '
                               'commanded the standing pose')

        result.elapsed = session.elapsed
        result.message = reason
        if reason == 'canceled':
            goal_handle.canceled()
            result.success = False
        elif reason in ('finished', 'stopped'):
            goal_handle.succeed()
            result.success = True
        else:
            goal_handle.abort()
            result.success = False
        return result


def main(args=None) -> None:
    """Run the RL walking node with a multi-threaded executor.

    Args:
        args: Optional command-line arguments passed to `rclpy.init`.
    """
    rclpy.init(args=args)
    node = HumanoidRlWalk()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
