"""Node that manages the walking action servers and exposes a simple start/stop interface."""

import math
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from humanoid_control.tools.walk_manager import (
    ManagerParams,
    RobotObservation,
    tilt_from_quaternion,
    WalkKind,
    WalkManager,
)
from humanoid_interfaces.action import RlWalk, Walk
from humanoid_interfaces.msg import RobotState, WalkManagerState

STATUS_NAMES = {
    GoalStatus.STATUS_SUCCEEDED: 'succeeded',
    GoalStatus.STATUS_CANCELED: 'canceled',
    GoalStatus.STATUS_ABORTED: 'aborted',
}


class HumanoidWalkManager(Node):
    """Run the walk manager state machine on top of the walk and RL walk action servers."""

    def __init__(self):
        """Declare parameters and create the clients, services, subscriptions and timer.

        Raises:
            ValueError: If a parameter is out of range.
        """
        super().__init__('humanoid_walk_manager')
        self.declare_parameter('update_rate', 20.0)
        self.declare_parameter('publish_period_sec', 1.0)
        self.declare_parameter('state_timeout_sec', 0.5)
        self.declare_parameter('rest_speed', 0.02)
        self.declare_parameter('rest_hold_sec', 1.0)
        self.declare_parameter('fall_tilt', 0.8)
        self.declare_parameter('fall_height_ratio', 0.6)
        self.declare_parameter('walk_num_steps', 0)
        self.declare_parameter('rl_forward_speed', 0.15)
        self.declare_parameter('rl_duration', 0.0)

        update_rate = float(self.get_parameter('update_rate').value)
        if update_rate <= 0.0:
            raise ValueError('update_rate must be positive')
        self.publish_period = float(self.get_parameter('publish_period_sec').value)
        self.state_timeout = float(self.get_parameter('state_timeout_sec').value)
        if self.state_timeout <= 0.0:
            raise ValueError('state_timeout_sec must be positive')
        self.walk_num_steps = int(self.get_parameter('walk_num_steps').value)
        if self.walk_num_steps < 0:
            raise ValueError('walk_num_steps must not be negative')
        self.rl_forward_speed = float(self.get_parameter('rl_forward_speed').value)
        self.rl_duration = float(self.get_parameter('rl_duration').value)
        if not math.isfinite(self.rl_duration) or self.rl_duration < 0.0:
            raise ValueError('rl_duration must be finite and non-negative')
        self.manager = WalkManager(ManagerParams(
            rest_speed=float(self.get_parameter('rest_speed').value),
            rest_hold=float(self.get_parameter('rest_hold_sec').value),
            fall_tilt=float(self.get_parameter('fall_tilt').value),
            fall_height_ratio=float(self.get_parameter('fall_height_ratio').value),
        ))

        self.startup_ready = False
        self.latest_state: Optional[RobotState] = None
        # Kind of the goal currently sent or running; at most one goal at a time.
        self.goal_kind: Optional[WalkKind] = None
        self.goal_handle = None
        self.cancel_pending = False
        self.last_published = None
        self.last_publish_time = None

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/humanoid_startup/ready', self._ready_cb, latched)
        self.create_subscription(
            RobotState, '/humanoid_estimator/state', self._state_cb, 10)
        self.state_pub = self.create_publisher(
            WalkManagerState, '/humanoid_walk_manager/state', latched)
        self.action_clients = {
            WalkKind.WALK: ActionClient(self, Walk, '/humanoid_walk/walk'),
            WalkKind.RL_WALK: ActionClient(self, RlWalk, '/humanoid_rl_walk/rl_walk'),
        }
        self.create_service(
            Trigger, '/humanoid_walk_manager/walk',
            lambda req, res: self._walk_cb(WalkKind.WALK, res))
        self.create_service(
            Trigger, '/humanoid_walk_manager/rl_walk',
            lambda req, res: self._walk_cb(WalkKind.RL_WALK, res))
        self.create_service(Trigger, '/humanoid_walk_manager/stop', self._stop_cb)
        self.create_timer(1.0 / update_rate, self._update)
        self._publish(force=True)
        self.get_logger().info(
            'Humanoid walk manager ready: call /humanoid_walk_manager/{walk,rl_walk,stop}')

    # ------------------------------------------------------------------ inputs
    def _ready_cb(self, msg: Bool) -> None:
        """Store the startup readiness flag.

        Args:
            msg: Latched flag from humanoid_startup.
        """
        self.startup_ready = bool(msg.data)

    def _state_cb(self, msg: RobotState) -> None:
        """Cache the latest shared robot state.

        Args:
            msg: Robot state from humanoid_estimator.
        """
        self.latest_state = msg

    def _observation(self) -> Optional[RobotObservation]:
        """Build the rest/fall observation from a fresh robot state.

        Returns:
            The observation, or None if no fresh state with a valid base exists.
        """
        msg = self.latest_state
        if msg is None or not msg.base_valid:
            return None
        age = (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9
        if age > self.state_timeout:
            return None
        q = msg.base_pose.orientation
        v = msg.com_velocity
        return RobotObservation(
            com_height=msg.com.z,
            com_speed=math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z),
            tilt=tilt_from_quaternion(q.x, q.y, q.z, q.w),
        )

    def _server_ready(self, kind: WalkKind) -> bool:
        """Check whether a walk goal of the given kind can be sent now.

        Args:
            kind: Walking controller.

        Returns:
            True if no goal is outstanding and the action server is available.
        """
        return self.goal_kind is None and self.action_clients[kind].server_is_ready()

    # ------------------------------------------------------------------ main loop
    def _update(self) -> None:
        """Advance the state machine and publish the state."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.manager.update(now, self.startup_ready, self._observation()):
            self.get_logger().warn('Robot fell while walking: canceling the walk')
            self._cancel_goal()
        self._publish()

    def _publish(self, force: bool = False) -> None:
        """Publish the manager state on change and periodically.

        Args:
            force: Publish even if nothing changed.
        """
        walk, rl_walk, stop = self.manager.allowed(
            self._server_ready(WalkKind.WALK), self._server_ready(WalkKind.RL_WALK))
        content = (self.manager.state.value, walk, rl_walk, stop, self.manager.message)
        now = self.get_clock().now()
        periodic = self.last_publish_time is None or \
            (now - self.last_publish_time).nanoseconds * 1e-9 >= self.publish_period
        if not force and content == self.last_published and not periodic:
            return
        if content != self.last_published:
            self.get_logger().info(
                f'State {content[0]}: {content[4]} (walk={walk}, rl_walk={rl_walk}, '
                f'stop={stop})')
        msg = WalkManagerState()
        msg.header.stamp = now.to_msg()
        msg.state, msg.walk_enabled, msg.rl_walk_enabled, msg.stop_enabled, msg.message = \
            content
        self.state_pub.publish(msg)
        self.last_published = content
        self.last_publish_time = now

    # ------------------------------------------------------------------ user requests
    def _walk_cb(self, kind: WalkKind, response):
        """Handle a walk or RL walk request and send the goal if allowed.

        Args:
            kind: Requested walking controller.
            response: Trigger response to fill in.

        Returns:
            The response.
        """
        ok, message = self.manager.request_walk(kind, self._server_ready(kind))
        if ok:
            self._send_goal(kind)
        response.success, response.message = ok, message
        self.get_logger().info(f'{kind.value} request: {message}')
        self._publish()
        return response

    def _stop_cb(self, _req, response):
        """Handle a stop request by canceling the active goal.

        Args:
            _req: Unused trigger request.
            response: Trigger response to fill in.

        Returns:
            The response.
        """
        ok, message = self.manager.request_stop()
        if ok:
            self._cancel_goal()
        response.success, response.message = ok, message
        self.get_logger().info(f'stop request: {message}')
        self._publish()
        return response

    # ------------------------------------------------------------------ action handling
    def _send_goal(self, kind: WalkKind) -> None:
        """Send the configured goal to the selected action server.

        Args:
            kind: Walking controller.
        """
        if kind == WalkKind.WALK:
            goal = Walk.Goal(num_steps=self.walk_num_steps)
        else:
            goal = RlWalk.Goal(forward_speed=self.rl_forward_speed, duration=self.rl_duration)
        self.goal_kind = kind
        self.goal_handle = None
        self.cancel_pending = False
        future = self.action_clients[kind].send_goal_async(goal)
        future.add_done_callback(lambda f: self._goal_response(kind, f))

    def _goal_response(self, kind: WalkKind, future) -> None:
        """Handle acceptance or rejection of a sent goal.

        Args:
            kind: Walking controller of the goal.
            future: Future of `send_goal_async`.
        """
        handle = future.result() if future.exception() is None else None
        if handle is None or not handle.accepted:
            self.get_logger().warn(f'{kind.value} goal rejected')
            self._goal_finished(kind, 'rejected')
            return
        self.goal_handle = handle
        self.manager.goal_accepted(kind)
        handle.get_result_async().add_done_callback(
            lambda f: self._goal_result(kind, f))
        if self.cancel_pending:
            self._cancel_goal()
        self._publish()

    def _goal_result(self, kind: WalkKind, future) -> None:
        """Handle the terminal status of a goal.

        Args:
            kind: Walking controller of the goal.
            future: Future of `get_result_async`.
        """
        if future.exception() is not None:
            detail = f'error: {future.exception()}'
        else:
            result = future.result()
            detail = STATUS_NAMES.get(result.status, f'status {result.status}')
            if result.result.message:
                detail += f': {result.result.message}'
        self.get_logger().info(f'{kind.value} goal ended ({detail})')
        self._goal_finished(kind, detail)

    def _goal_finished(self, kind: WalkKind, detail: str) -> None:
        """Clear the goal bookkeeping and inform the state machine.

        Args:
            kind: Walking controller of the goal.
            detail: Outcome description.
        """
        self.goal_kind = None
        self.goal_handle = None
        self.cancel_pending = False
        self.manager.goal_ended(kind, detail)
        self._publish()

    def _cancel_goal(self) -> None:
        """Cancel the active goal, or cancel it once it is accepted."""
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        elif self.goal_kind is not None:
            self.cancel_pending = True


def main(args=None):
    """Spin the walk manager node until shutdown.

    Args:
        args: Optional ROS command-line arguments.
    """
    rclpy.init(args=args)
    node = HumanoidWalkManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
