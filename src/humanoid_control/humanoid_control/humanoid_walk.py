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
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from humanoid_control.gait import FixedFootstepPlanner, GaitParams, WalkingPattern
from humanoid_control.humanoid_kinematics import HumanoidKinematics
from humanoid_interfaces.action import Walk

JOINT_NAMES = [
    'left_hip_roll', 'left_hip_yaw', 'left_hip_pitch', 'left_knee', 'left_ankle_roll',
    'left_ankle_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_hip_pitch', 'right_knee',
    'right_ankle_roll', 'right_ankle_pitch',
]


class PlaneFrame:
    """Maps between the base frame and the gait plane frame (x forward, y left, z up)."""

    def __init__(self, forward, left) -> None:
        self.forward = np.asarray(forward, float) / np.linalg.norm(forward)
        self.left = np.asarray(left, float) / np.linalg.norm(left)
        self.up = np.array([0.0, 0.0, 1.0])

    def to_plane(self, p) -> np.ndarray:
        return np.array([p @ self.forward, p @ self.left, p @ self.up])

    def to_base(self, p) -> np.ndarray:
        return p[0] * self.forward + p[1] * self.left + p[2] * self.up


class HumanoidWalk(Node):
    def __init__(self) -> None:
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
        defaults = GaitParams(zmp_offset_forward=0.05)
        for field in dataclasses.fields(GaitParams):
            if field.name in ('step_width', 'com_height'):
                continue
            self.declare_parameter(field.name, getattr(defaults, field.name))

        self.rate_hz = float(self.get_parameter('update_rate_hz').value)
        self.ik_iterations = int(self.get_parameter('ik_iterations').value)
        self.lead = float(self.get_parameter('trajectory_lead_sec').value)

        self.joint_positions: Dict[str, float] = {}
        self.stop_requested = threading.Event()
        self.walking = False
        self.walk_lock = threading.Lock()

        group = ReentrantCallbackGroup()
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)
        controller = str(self.get_parameter('controller_name').value)
        self.traj_pub = self.create_publisher(JointTrajectory, f'/{controller}/joint_trajectory', 10)
        self.create_service(
            Trigger, '/humanoid_walk/stop', self._stop_cb, callback_group=group)

        self.kinematics = HumanoidKinematics.from_xacro(
            os.path.join(get_package_share_directory('humanoid_description'), 'urdf', 'humanoid.xacro')
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
    def _joint_state_cb(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            self.joint_positions[name] = position

    def _have_all_joints(self) -> bool:
        return all(name in self.joint_positions for name in JOINT_NAMES)

    def _stop_cb(self, _req, response):
        self.stop_requested.set()
        response.success = self.walking
        response.message = 'stopping at next step' if self.walking else 'not walking'
        return response

    def _goal_cb(self, _goal) -> GoalResponse:
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
        self.stop_requested.set()
        return CancelResponse.ACCEPT

    # ------------------------------------------------------------------ setup
    def _gait_params(self, standing, frame: PlaneFrame) -> GaitParams:
        values = {
            f.name: self.get_parameter(f.name).value
            for f in dataclasses.fields(GaitParams)
            if f.name not in ('step_width', 'com_height')
        }
        left = standing.feet['left'].position
        right = standing.feet['right'].position
        width = float(self.get_parameter('step_width').value)
        if width <= 0.0:
            width = abs((left - right) @ frame.left)
        height = float(self.get_parameter('com_height').value)
        if height <= 0.0:
            ground_z = 0.5 * (left[2] + right[2]) - float(self.get_parameter('ankle_to_sole').value)
            height = standing.com[2] - ground_z
        params = GaitParams(step_width=width, com_height=height, **values)
        params.validate()
        return params

    def _init_kinematics(self):
        kin = self.kinematics
        kin.q = pin.neutral(kin.model)
        kin.set_joint_positions(self.joint_positions.keys(), self.joint_positions.values())
        kin.reset_ik()
        return kin, kin.compute()

    def _publish_joints(self, q: np.ndarray, q_prev: Optional[np.ndarray]) -> None:
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
        return pattern.num_steps is not None and ref.step_index >= pattern.num_steps - 1

    def _execute(self, goal_handle):
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
        self.stop_requested.clear()
        num_steps = int(goal_handle.request.num_steps)
        kin, standing = self._init_kinematics()
        left0, right0 = standing.feet['left'], standing.feet['right']

        lateral = (left0.position - right0.position).copy()
        lateral[2] = 0.0
        frame = PlaneFrame(np.array(self.get_parameter('forward_axis').value, float), lateral)
        params = self._gait_params(standing, frame)

        feet0 = {s: frame.to_plane(standing.feet[s].position) for s in ('left', 'right')}
        planner = FixedFootstepPlanner(
            params, feet0['left'][:2], feet0['right'][:2],
            first_swing=str(self.get_parameter('first_swing').value),
            num_steps=num_steps if num_steps > 0 else None,
        )
        com0 = frame.to_plane(standing.com)
        zero = np.zeros(2)
        pattern = WalkingPattern(
            params, planner, feet0['left'], feet0['right'],
            com_state=(com0[:2], zero, zero), com_z=com0[2],
        )
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

            poses = {
                s: pin.SE3(rotations[s], frame.to_base(ref.feet[s])) for s in ('left', 'right')
            }
            com_target = frame.to_base(ref.com)
            for _ in range(self.ik_iterations):
                q = kin.compute_ik_step(
                    frames['left'], poses['left'], frames['right'], poses['right'], com_target, dt
                )
            self._publish_joints(q, q_prev)
            q_prev = q.copy()

            feedback_count += 1
            if feedback_count % max(1, int(self.rate_hz / 10)) == 0:
                feedback.step_index = max(ref.step_index, 0)
                feedback.phase = ref.phase
                feedback.swing_side = ref.swing_side or ''
                feedback.com.x, feedback.com.y, feedback.com.z = (float(v) for v in frame.to_base(ref.com))
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
