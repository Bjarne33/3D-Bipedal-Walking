"""Walking node: footstep plan -> LIPM CoM + swing foot -> Pink IK -> joint trajectory.

Start a walk with the `humanoid_interfaces/action/Walk` action on `/humanoid_walk/walk`
(`num_steps` > 0: that many steps plus a closing step; `num_steps` = 0: walk until canceled).
Canceling the goal or calling `/humanoid_walk/stop` ends the walk gracefully at the next step
boundary by bringing the feet together.
"""

import dataclasses
import os
import threading
from typing import Dict, Optional

import numpy as np
import pinocchio as pin
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from humanoid_control.tools.balance import BalanceController
from humanoid_control.tools.gait_params import GaitParams
from humanoid_control.tools.humanoid_kinematics import HumanoidKinematics
from humanoid_control.tools.joints import JOINT_NAMES
from humanoid_control.tools.plan import plan_walk, rotation_joint_maps
from humanoid_control.tools.state_estimator import WalkStateProjection
from humanoid_control.tools.state_messages import message_to_estimate, warn_stale_state
from humanoid_control.tools.support_polygon import signed_margin
from humanoid_interfaces.action import Walk
from humanoid_interfaces.msg import BalanceState, RobotState


class HumanoidWalk(Node):
    """Model-based walking node exposing the walk action and stop service."""

    def __init__(self) -> None:
        """Declare gait parameters, load the model and create the action, services and I/O."""
        super().__init__('humanoid_walk')

        self.declare_parameter('controller_name', 'effort_position_controller')
        self.declare_parameter('update_rate_hz', 50.0)
        self.declare_parameter('ik_iterations', 2)
        self.declare_parameter('trajectory_lead_sec', 0.06)
        self.declare_parameter('first_swing', 'right')
        # Base-frame direction the robot walks in; the left direction is measured from the feet.
        self.declare_parameter('forward_axis', [0.0, 1.0, 0.0])
        # Distance from the ankle (foot frame) down to the sole, used to derive the CoM height.
        self.declare_parameter('ankle_to_sole', 0.05)
        # <= 0: derive from the standing pose (feet distance / CoM height above the ground).
        self.declare_parameter('step_width', 0.0)
        self.declare_parameter('com_height', 0.0)
        self.declare_parameter('state_timeout_sec', 0.2)
        defaults = GaitParams(zmp_offset_forward=0.05)
        for field in dataclasses.fields(GaitParams):
            if field.name in ('step_width', 'com_height'):
                continue
            self.declare_parameter(field.name, getattr(defaults, field.name))

        self.rate_hz = float(self.get_parameter('update_rate_hz').value)
        self.ik_iterations = int(self.get_parameter('ik_iterations').value)
        self.lead = float(self.get_parameter('trajectory_lead_sec').value)

        self.ankle_to_sole = float(self.get_parameter('ankle_to_sole').value)
        self.state_timeout = float(self.get_parameter('state_timeout_sec').value)
        if self.state_timeout <= 0.0:
            raise ValueError('state_timeout_sec must be positive')
        self.joint_positions: Dict[str, float] = {}
        self.stop_requested = threading.Event()
        self.walking = False
        self.walk_lock = threading.Lock()

        self.latest_state = None

        group = ReentrantCallbackGroup()
        self.create_subscription(
            RobotState, '/humanoid_estimator/state', self._state_cb, 10)
        controller = str(self.get_parameter('controller_name').value)
        self.traj_pub = self.create_publisher(
            JointTrajectory, f'/{controller}/joint_trajectory', 10)
        self.balance_pub = self.create_publisher(BalanceState, '/humanoid_walk/balance_state', 10)
        self.create_service(
            Trigger, '/humanoid_walk/stop', self._stop_cb, callback_group=group)

        self.kinematics = HumanoidKinematics.from_xacro(
            os.path.join(get_package_share_directory(
                'humanoid_description'), 'urdf', 'humanoid.xacro')
        )
        self.action_server = ActionServer(
            self, Walk, '/humanoid_walk/walk',
            execute_callback=self._execute,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=group,
        )
        self.get_logger().info(
            'Humanoid walk node ready: send a goal to /humanoid_walk/walk '
            '(num_steps = 0 walks until canceled)'
        )

    # ------------------------------------------------------------------ callbacks
    def _state_cb(self, msg: RobotState) -> None:
        """Cache the shared measurements and joint positions used to initialise IK."""
        self.latest_state = msg
        for name, position in zip(msg.joint_state.name, msg.joint_state.position):
            self.joint_positions[name] = position

    def _have_all_joints(self) -> bool:
        """Check whether a position is known for every controlled joint.

        Returns:
            True if all joint positions have been received.
        """
        return all(name in self.joint_positions for name in JOINT_NAMES)

    def _stop_cb(self, _req, response):
        """Handle `/humanoid_walk/stop`: request a graceful stop at the next step boundary.

        Args:
            _req: Unused trigger request.
            response: Response to fill in.

        Returns:
            The response; `success` is True if a walk was running.
        """
        self.stop_requested.set()
        response.success = self.walking
        response.message = 'stopping at next step' if self.walking else 'not walking'
        return response

    def _goal_cb(self, _goal) -> GoalResponse:
        """Accept a walk goal unless joint states are incomplete or a walk is running.

        Args:
            _goal: Unused goal request.

        Returns:
            ACCEPT or REJECT.
        """
        if not self._have_all_joints():
            self.get_logger().warn('Goal rejected: joint states incomplete')
            return GoalResponse.REJECT
        with self.walk_lock:
            if self.walking:
                self.get_logger().warn('Goal rejected: already walking')
                return GoalResponse.REJECT
            self.walking = True
        return GoalResponse.ACCEPT

    def _cancel_cb(self, _goal) -> CancelResponse:
        """Accept a cancel request and ask the walk to close the feet at the next step.

        Args:
            _goal: Unused goal handle.

        Returns:
            Always ACCEPT.
        """
        self.stop_requested.set()
        return CancelResponse.ACCEPT

    # ------------------------------------------------------------------ setup
    def _plan_config(self) -> dict:
        """Collect the planning-related ROS parameters.

        Returns:
            Parameter values by name, as expected by `plan_walk`.
        """
        names = ['forward_axis', 'first_swing', 'ankle_to_sole', 'step_width', 'com_height']
        names += [f.name for f in dataclasses.fields(GaitParams)]
        return {name: self.get_parameter(name).value for name in names}

    def _init_kinematics(self):
        """Reset the kinematics to the measured joint positions and the IK state.

        Returns:
            Tuple of the kinematics object and the resulting standing `KinematicsResult`.
        """
        kin = self.kinematics
        kin.q = pin.neutral(kin.model)
        kin.set_joint_positions(self.joint_positions.keys(), self.joint_positions.values())
        kin.reset_ik()
        return kin, kin.compute()

    def _apply_joint_offsets(self, q: np.ndarray, maps, out) -> np.ndarray:
        """Add the balance ankle and torso rotations as joint offsets to an IK solution.

        Args:
            q: Configuration from the inverse kinematics.
            maps: Per side and joint group mapping from body rotation to joint offsets.
            out: Balance output holding the ankle and torso rotations.

        Returns:
            A copy of `q` with the offsets applied.
        """
        offsets = {}
        for side, rot in out.ankle_rotation.items():
            joints, m = maps[side]['ankle']
            offsets.update(zip(joints, m @ rot))
        if out.torso_side is not None:
            joints, m = maps[out.torso_side]['hip']
            offsets.update(zip(joints, m @ out.torso_rotation))
        q = q.copy()
        for name, value in offsets.items():
            jid = self.kinematics.joint_name_to_id[name]
            q[self.kinematics.model.idx_qs[jid]] += float(value)
        return q

    def _publish_balance_state(self, ref, est, out, balance, enabled):
        """Publish reference, measured and commanded balance quantities for debugging.

        Args:
            ref: Walking reference at the current time.
            est: Balance estimate, or None if unavailable.
            out: Balance controller output, or None if balance is inactive.
            balance: Balance controller (used for the reference DCM).
            enabled: Whether balance feedback is active.
        """
        msg = BalanceState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.step_index = int(ref.step_index)
        msg.phase = ref.phase
        msg.enabled = bool(enabled)
        msg.estimate_valid = est is not None
        xi_ref = balance.dcm_reference(ref)
        msg.ref_com_x, msg.ref_com_y = (float(v) for v in ref.com[:2])
        msg.ref_dcm_x, msg.ref_dcm_y = (float(v) for v in xi_ref)
        msg.ref_zmp_x, msg.ref_zmp_y = (float(v) for v in ref.zmp[:2])
        nan2 = (float('nan'), float('nan'))
        msg.des_zmp_x, msg.des_zmp_y = msg.ref_zmp_x, msg.ref_zmp_y
        msg.meas_com_x, msg.meas_com_y = nan2
        msg.meas_dcm_x, msg.meas_dcm_y = nan2
        msg.meas_zmp_x, msg.meas_zmp_y = nan2
        msg.dcm_margin = float('nan')
        if est is not None:
            msg.zmp_valid = bool(est.zmp_valid)
            msg.meas_com_x, msg.meas_com_y = (float(v) for v in est.com)
            msg.meas_dcm_x, msg.meas_dcm_y = (float(v) for v in est.dcm)
            if est.zmp_valid:
                msg.meas_zmp_x, msg.meas_zmp_y = (float(v) for v in est.zmp)
            if len(est.support_hull) >= 3:
                msg.dcm_margin = float(signed_margin(est.dcm, est.support_hull))
            msg.tilt_x, msg.tilt_y = (float(v) for v in est.tilt)
        if out is not None:
            msg.saturated = bool(out.saturated)
            msg.des_zmp_x, msg.des_zmp_y = (float(v) for v in out.zmp_des)
            msg.com_offset_x, msg.com_offset_y = (float(v) for v in out.com_offset)
            if ref.stance_side in out.ankle_rotation:
                ankle = out.ankle_rotation[ref.stance_side]
                msg.ankle_rot_x, msg.ankle_rot_y = (float(v) for v in ankle)
            msg.torso_rot_x, msg.torso_rot_y = (float(v) for v in out.torso_rotation)
            msg.step_adjusted = bool(np.any(out.step_adjust != 0.0))
            msg.step_adjust_x, msg.step_adjust_y = (float(v) for v in out.step_adjust)
        self.balance_pub.publish(msg)

    def _publish_joints(self, q: np.ndarray, q_prev: Optional[np.ndarray]) -> None:
        """Publish a single-point joint trajectory for the actuated joints.

        Args:
            q: Target configuration.
            q_prev: Previous configuration used for finite-difference velocities, or None.
        """
        kin = self.kinematics
        idx = [kin.model.idx_qs[kin.joint_name_to_id[n]] for n in JOINT_NAMES]
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(q[i]) for i in idx]
        if q_prev is not None:
            dt = 1.0 / self.rate_hz
            point.velocities = [float((q[i] - q_prev[i]) / dt) for i in idx]
        point.time_from_start.sec = int(self.lead)
        point.time_from_start.nanosec = int((self.lead - int(self.lead)) * 1e9)
        msg.points = [point]
        self.traj_pub.publish(msg)

    # ------------------------------------------------------------------ walking
    @staticmethod
    def _is_closing(pattern, ref) -> bool:
        """Check whether the reference is in the final (closing) step of a finite pattern.

        Args:
            pattern: Walking pattern.
            ref: Walking reference at the current time.

        Returns:
            True if the pattern is finite and `ref` is in its last step or later.
        """
        return pattern.num_steps is not None and ref.step_index >= pattern.num_steps - 1

    def _execute(self, goal_handle):
        """Execute a walk goal, aborting it on errors and releasing the walking flag.

        Args:
            goal_handle: Action goal handle.

        Returns:
            The `Walk.Result`.
        """
        result = Walk.Result()
        try:
            return self._walk(goal_handle, result)
        except Exception as exc:  # keep the node alive and report the failure to the client
            self.get_logger().error(f'Walk failed: {exc}')
            goal_handle.abort()
            result.success = False
            result.message = str(exc)
            return result
        finally:
            self.stop_requested.clear()
            with self.walk_lock:
                self.walking = False

    def _walk(self, goal_handle, result):
        """Run the walking control loop: plan, sample references, solve IK and publish.

        Args:
            goal_handle: Action goal handle.
            result: Result object to fill in.

        Returns:
            The filled `Walk.Result`.
        """
        self.stop_requested.clear()
        num_steps = int(goal_handle.request.num_steps)
        kin, standing = self._init_kinematics()
        frame, params, planner, pattern = plan_walk(standing, self._plan_config(), num_steps)
        initial_state = self.latest_state
        projection = None
        if initial_state is not None and initial_state.base_valid:
            projection = WalkStateProjection(
                message_to_estimate(initial_state).base, params.balance_state_alpha)
        else:
            self.get_logger().warn(
                'No base pose received: balance feedback disabled for this walk')
        balance = BalanceController(params)
        joint_maps = rotation_joint_maps(kin, frame)
        rotations = {s: standing.feet[s].rotation for s in ('left', 'right')}
        frames = kin.foot_frames
        self.get_logger().info(
            f'Walking {"continuously" if num_steps == 0 else f"{num_steps} steps"}: '
            f'step_length={params.step_length} width={params.step_width:.3f} '
            f'time={params.step_time} com_height={params.com_height:.3f}'
        )

        clock = self.get_clock()
        rate = self.create_rate(self.rate_hz)
        dt = 1.0 / self.rate_hz
        t0 = clock.now()
        q_prev = kin.configuration.q.copy()
        last_step = -1
        stopping = False
        steps_before_stop = 0
        feedback_count = 0
        feedback = Walk.Feedback()

        while rclpy.ok():
            t = (clock.now() - t0).nanoseconds * 1e-9
            ref = pattern.sample(t)

            if ref.phase == 'step' and ref.step_index != last_step:
                stop_now = self.stop_requested.is_set() and not stopping
                if stop_now and not self._is_closing(pattern, ref):
                    boundary = pattern.step_start_time(ref.step_index)
                    pattern = pattern.stop_pattern(ref.step_index)
                    t0 = t0 + Duration(seconds=boundary)
                    stopping = True
                    steps_before_stop = ref.step_index
                    self.get_logger().info('Stop requested: closing the feet')
                    t = (clock.now() - t0).nanoseconds * 1e-9
                    ref = pattern.sample(t)
                    last_step = -1
                else:
                    last_step = ref.step_index

            ref_feet = {side: np.array(ref.feet[side], dtype=float) for side in ('left', 'right')}
            com_plane = np.array(ref.com, dtype=float)
            # Stabiliser stage: measured state -> corrected CoM target and joint offsets.
            state_msg = self.latest_state
            est = None
            if projection is not None and state_msg is not None and state_msg.base_valid:
                warn_stale_state(self, state_msg, self.state_timeout)
                est = projection.project(message_to_estimate(state_msg), frame, balance.omega)
            out = balance.update(ref, est, dt) if est is not None else None
            if out is not None:
                com_plane[:2] += out.com_offset

            poses = {
                s: pin.SE3(rotations[s], frame.to_base(ref_feet[s])) for s in ('left', 'right')
            }
            com_target = frame.to_base(com_plane)
            for _ in range(self.ik_iterations):
                q = kin.compute_ik_step(
                    frames['left'], poses['left'], frames['right'], poses['right'], com_target, dt
                )
            q_cmd = self._apply_joint_offsets(q, joint_maps, out) if out is not None else q
            self._publish_joints(q_cmd, q_prev)
            q_prev = q_cmd.copy()
            self._publish_balance_state(ref, est, out, balance, params.balance_enabled)

            feedback_count += 1
            if feedback_count % max(1, int(self.rate_hz / 10)) == 0:
                feedback.step_index = max(ref.step_index, 0)
                feedback.phase = ref.phase
                feedback.swing_side = ref.swing_side or ''
                feedback.com.x, feedback.com.y, feedback.com.z = (
                    float(v) for v in frame.to_base(ref.com))
                goal_handle.publish_feedback(feedback)

            if ref.finished:
                break
            rate.sleep()

        self.destroy_rate(rate)
        result.steps_completed = steps_before_stop + 1 if stopping else max(num_steps, 0)
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = 'canceled, feet closed'
        else:
            goal_handle.succeed()
            result.success = True
            result.message = 'walk finished'
        return result


def main(args=None) -> None:
    """Run the walking node with a multi-threaded executor.

    Args:
        args: Optional command-line arguments passed to `rclpy.init`.
    """
    rclpy.init(args=args)
    node = HumanoidWalk()
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
