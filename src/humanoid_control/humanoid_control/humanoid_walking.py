import math
import os
from enum import Enum
from typing import Dict, List

import numpy as np
import pinocchio as pin
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from humanoid_control.humanoid_kinematics import HumanoidKinematics
from humanoid_interfaces.srv import StepN


class StepPhase(Enum):
    IDLE = 0
    SHIFT_TO_LEFT_SUPPORT = 1
    SWING_RIGHT_FOOT = 2
    PLACE_RIGHT_FOOT = 3
    SHIFT_TO_RIGHT_SUPPORT = 4
    SWING_LEFT_FOOT = 5
    PLACE_LEFT_FOOT = 6
    SETTLE_DOUBLE_SUPPORT = 7


class HumanoidWalking(Node):
    def __init__(self) -> None:
        super().__init__('humanoid_walking')

        self.declare_parameter('controller_name', 'effort_position_controller')
        self.declare_parameter('pose_tolerance', 0.08)
        self.declare_parameter('update_rate_hz', 50.0)
        self.declare_parameter('phase_timeout_margin_sec', 3.0)

        self.declare_parameter('shift_duration_sec', 3)
        self.declare_parameter('swing_duration_sec', 3)
        self.declare_parameter('place_duration_sec', 3)
        self.declare_parameter('settle_duration_sec', 1.8)

        self.declare_parameter('left_support_hip_roll', 0.18)
        self.declare_parameter('left_support_ankle_roll', -0.18)

        self.declare_parameter('swing_hip_pitch_delta', 0.4)
        self.declare_parameter('swing_knee_delta', -0.42)
        self.declare_parameter('swing_ankle_pitch_delta', 0.16)

        self.declare_parameter('place_hip_pitch_delta', 0.22)
        self.declare_parameter('place_knee_delta', -0.08)
        self.declare_parameter('place_ankle_pitch_delta', -0.08)

        # Inverse kinematics test (/humanoid_walking/test_ik)
        self.declare_parameter('ik_test_swing_offset', [0.05, 0.0, 0.04])
        self.declare_parameter('ik_test_com_offset', [0.0, 0.03, 0.0])
        self.declare_parameter('ik_test_iterations', 100)
        self.declare_parameter('ik_test_dt', 0.02)
        self.declare_parameter('ik_test_tolerance', 1e-3)
        self.declare_parameter('ik_test_execute', True)
        self.declare_parameter('ik_test_duration_sec', 3.0)

        self.controller_name = str(self.get_parameter('controller_name').value)
        self.pose_tolerance = float(self.get_parameter('pose_tolerance').value)
        self.update_rate_hz = float(self.get_parameter('update_rate_hz').value)
        self.phase_timeout_margin_sec = float(self.get_parameter('phase_timeout_margin_sec').value)

        self.shift_duration_sec = float(self.get_parameter('shift_duration_sec').value)
        self.swing_duration_sec = float(self.get_parameter('swing_duration_sec').value)
        self.place_duration_sec = float(self.get_parameter('place_duration_sec').value)
        self.settle_duration_sec = float(self.get_parameter('settle_duration_sec').value)

        self.left_support_hip_roll = float(self.get_parameter('left_support_hip_roll').value)
        self.left_support_ankle_roll = float(self.get_parameter('left_support_ankle_roll').value)

        self.swing_hip_pitch_delta = float(self.get_parameter('swing_hip_pitch_delta').value)
        self.swing_knee_delta = float(self.get_parameter('swing_knee_delta').value)
        self.swing_ankle_pitch_delta = float(self.get_parameter('swing_ankle_pitch_delta').value)

        self.place_hip_pitch_delta = float(self.get_parameter('place_hip_pitch_delta').value)
        self.place_knee_delta = float(self.get_parameter('place_knee_delta').value)
        self.place_ankle_pitch_delta = float(self.get_parameter('place_ankle_pitch_delta').value)

        self.joint_names = [
            'left_hip_roll',
            'left_hip_yaw',
            'left_hip_pitch',
            'left_knee',
            'left_ankle_roll',
            'left_ankle_pitch',
            'right_hip_roll',
            'right_hip_yaw',
            'right_hip_pitch',
            'right_knee',
            'right_ankle_roll',
            'right_ankle_pitch',
        ]

        self.neutral_pose = {
            'left_hip_roll': 0.0,
            'left_hip_yaw': 0.0,
            'left_hip_pitch': 0.275,
            'left_knee': -0.5,
            'left_ankle_roll': 0.0,
            'left_ankle_pitch': 0.25,
            'right_hip_roll': 0.0,
            'right_hip_yaw': 0.0,
            'right_hip_pitch': 0.275,
            'right_knee': -0.5,
            'right_ankle_roll': 0.0,
            'right_ankle_pitch': 0.25,
        }

        self.current_joint_positions: Dict[str, float] = {}
        self.phase = StepPhase.IDLE
        self.phase_started_at = self.get_clock().now()
        self.active_target_pose: Dict[str, float] = {}
        self.active_phase_duration_sec = 0.0
        self.phase_queue: List[tuple[StepPhase, Dict[str, float], float]] = []

        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            f'/{self.controller_name}/joint_trajectory',
            10,
        )

        self.step_service = self.create_service(
            Trigger,
            '/humanoid_walking/step_once',
            self.step_once_callback,
        )
        self.step_n_service = self.create_service(
            StepN,
            '/humanoid_walking/step_n',
            self.step_n_callback,
        )
        self.kinematics = HumanoidKinematics.from_xacro(
            os.path.join(get_package_share_directory('humanoid_description'), 'urdf', 'humanoid.xacro')
        )
        self.ik_service = self.create_service(
            Trigger,
            '/humanoid_walking/test_ik',
            self.test_ik_callback,
        )
        self.timer = self.create_timer(1.0 / max(self.update_rate_hz, 1.0), self.update)

        self.get_logger().info(
            'Humanoid walking node ready: call /humanoid_walking/step_once (or /humanoid_walking/step_n) '
            'for test shift-swing-place-settle behavior'
        )

    def joint_state_callback(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            self.current_joint_positions[name] = position

    def step_once_callback(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.get_logger().info('Received /humanoid_walking/step_once request')

        if self.phase != StepPhase.IDLE:
            response.success = False
            response.message = 'Step already running'
            self.get_logger().warn(f'Step request rejected: {response.message}')
            return response

        if not self.have_all_joints():
            response.success = False
            response.message = 'Joint states incomplete, cannot step safely yet'
            self.get_logger().warn(f'Step request rejected: {response.message}')
            return response

        self.phase_queue = self.build_phase_queue()
        self.start_next_phase()

        response.success = True
        response.message = (
            'Test step started '
            '(shift-left support -> swing right foot -> place right foot -> centered settle)'
        )
        self.get_logger().info(response.message)
        return response

    def step_n_callback(self, request: StepN.Request, response: StepN.Response) -> StepN.Response:
        num_steps = int(request.num_steps)
        self.get_logger().info(f'Received /humanoid_walking/step_n request (num_steps={num_steps})')

        if num_steps < 1:
            response.success = False
            response.message = 'num_steps must be >= 1'
        elif self.phase != StepPhase.IDLE:
            response.success = False
            response.message = 'Step already running'
        elif not self.have_all_joints():
            response.success = False
            response.message = 'Joint states incomplete, cannot step safely yet'
        else:
            right_swing = self.build_phase_queue()
            left_swing = self.mirror_phase_queue(right_swing)
            self.phase_queue = []
            for i in range(num_steps):
                self.phase_queue.extend(right_swing if i % 2 == 0 else left_swing)
            self.start_next_phase()
            response.success = True
            response.message = f'Started {num_steps} alternating test steps (right foot first)'

        if response.success:
            self.get_logger().info(response.message)
        else:
            self.get_logger().warn(f'Step request rejected: {response.message}')
        return response

    def test_ik_callback(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Check IK: left foot stays put, right foot and CoM move by the configured offsets."""
        if not self.have_all_joints():
            response.success = False
            response.message = 'Joint states incomplete, cannot run IK test'
            return response

        iterations = int(self.get_parameter('ik_test_iterations').value)
        dt = float(self.get_parameter('ik_test_dt').value)
        tolerance = float(self.get_parameter('ik_test_tolerance').value)
        swing_offset = np.array(self.get_parameter('ik_test_swing_offset').value, dtype=float)
        com_offset = np.array(self.get_parameter('ik_test_com_offset').value, dtype=float)
        if swing_offset.shape != (3,) or com_offset.shape != (3,):
            response.success = False
            response.message = 'ik_test_swing_offset and ik_test_com_offset need 3 values'
            return response

        kin = self.kinematics
        # FK is computed in the base frame: floating base stays at identity, joints from /joint_states.
        kin.q = pin.neutral(kin.model)
        kin.set_joint_positions(self.current_joint_positions.keys(), self.current_joint_positions.values())
        kin.reset_ik()
        start = kin.compute()

        stance_frame = kin.foot_frames['left']
        swing_frame = kin.foot_frames['right']
        stance_pose = pin.SE3(start.feet['left'].rotation, start.feet['left'].position)
        swing_pose = pin.SE3(start.feet['right'].rotation, start.feet['right'].position + swing_offset)
        com_target = start.com + com_offset

        try:
            for _ in range(iterations):
                kin.compute_ik_step(stance_frame, stance_pose, swing_frame, swing_pose, com_target, dt)
        except ValueError as exc:
            response.success = False
            response.message = f'IK test failed: {exc}'
            return response

        result = kin.compute()
        stance_err = float(np.linalg.norm(result.feet['left'].position - stance_pose.translation))
        swing_err = float(np.linalg.norm(result.feet['right'].position - swing_pose.translation))
        com_err = float(np.linalg.norm(result.com - com_target))
        # Rotation error of the swing foot (angle of the relative rotation).
        rot_err = float(np.linalg.norm(pin.log3(swing_pose.rotation.T @ result.feet['right'].rotation)))

        q = kin.configuration.q
        joint_deltas = {
            name: float(q[kin.model.idx_qs[kin.joint_name_to_id[name]]] - self.current_joint_positions[name])
            for name in self.joint_names
        }
        lower, upper = kin.model.lowerPositionLimit, kin.model.upperPositionLimit
        limit_violations = [
            name for name in self.joint_names
            if not lower[kin.model.idx_qs[kin.joint_name_to_id[name]]] - 1e-6
            <= q[kin.model.idx_qs[kin.joint_name_to_id[name]]]
            <= upper[kin.model.idx_qs[kin.joint_name_to_id[name]]] + 1e-6
        ]

        ok = max(stance_err, swing_err, com_err, rot_err) < tolerance
        lines = [
            f'IK test {"PASSED" if ok else "FAILED"} ({iterations} steps, dt={dt}, solver={kin.ik_solver})',
            f'errors [m/rad]: stance_pos={stance_err:.2e} swing_pos={swing_err:.2e} '
            f'swing_rot={rot_err:.2e} com={com_err:.2e} (tol {tolerance:.1e})',
            'joint deltas [rad]: ' + ', '.join(f'{n}={d:+.3f}' for n, d in joint_deltas.items()),
        ]
        if limit_violations:
            ok = False
            lines.append(f'joint limit violations: {limit_violations}')

        if ok and bool(self.get_parameter('ik_test_execute').value):
            if self.phase != StepPhase.IDLE:
                lines.append('not executed: step running')
            else:
                target = {name: float(q[kin.model.idx_qs[kin.joint_name_to_id[name]]]) for name in self.joint_names}
                self.publish_pose(target, float(self.get_parameter('ik_test_duration_sec').value))
                lines.append('IK solution sent to controller')

        response.success = ok
        response.message = '\n'.join(lines)
        self.get_logger().info(response.message)
        return response

    def update(self) -> None:
        if self.phase == StepPhase.IDLE:
            return

        elapsed = (self.get_clock().now() - self.phase_started_at).nanoseconds * 1e-9
        max_phase_time = self.active_phase_duration_sec + self.phase_timeout_margin_sec

        if self.pose_reached(self.active_target_pose):
            self.start_next_phase()
            return

        if elapsed > max_phase_time:
            self.get_logger().error(
                f'Phase {self.phase.name} timed out after {elapsed:.2f}s (max {max_phase_time:.2f}s).'
            )
            self.phase = StepPhase.IDLE
            self.phase_queue = []

    def have_all_joints(self) -> bool:
        return all(joint_name in self.current_joint_positions for joint_name in self.joint_names)

    def pose_reached(self, target_pose: Dict[str, float]) -> bool:
        for joint_name, target in target_pose.items():
            current = self.current_joint_positions.get(joint_name)
            if current is None:
                return False
            if math.fabs(current - target) > self.pose_tolerance:
                return False
        return True

    def start_next_phase(self) -> None:
        if not self.phase_queue:
            self.phase = StepPhase.IDLE
            self.get_logger().info('Step sequence complete')
            return

        phase, target_pose, duration_sec = self.phase_queue.pop(0)
        self.phase = phase
        self.active_target_pose = target_pose
        self.active_phase_duration_sec = duration_sec
        self.phase_started_at = self.get_clock().now()

        self.publish_pose(target_pose, duration_sec)
        self.get_logger().info(f'Started phase {phase.name} ({duration_sec:.2f}s)')

    def publish_pose(self, pose: Dict[str, float], duration_sec: float) -> None:
        msg = JointTrajectory()
        msg.joint_names = list(self.joint_names)

        point = JointTrajectoryPoint()
        point.positions = [pose[name] for name in self.joint_names]
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        msg.points = [point]

        self.traj_pub.publish(msg)

    def build_phase_queue(self) -> List[tuple[StepPhase, Dict[str, float], float]]:
        shift_left_pose = dict(self.neutral_pose)
        shift_left_pose['left_hip_roll'] = self.left_support_hip_roll
        shift_left_pose['right_hip_roll'] = self.left_support_hip_roll
        shift_left_pose['left_ankle_roll'] = self.left_support_ankle_roll
        shift_left_pose['right_ankle_roll'] = self.left_support_ankle_roll

        swing_pose = dict(shift_left_pose)
        swing_pose['right_hip_pitch'] = swing_pose['right_hip_pitch'] + self.swing_hip_pitch_delta
        swing_pose['right_knee'] = swing_pose['right_knee'] + self.swing_knee_delta
        swing_pose['right_ankle_pitch'] = swing_pose['right_ankle_pitch'] + self.swing_ankle_pitch_delta

        place_pose = dict(shift_left_pose)
        place_pose['right_hip_pitch'] = self.neutral_pose['right_hip_pitch'] + self.place_hip_pitch_delta
        place_pose['right_knee'] = self.neutral_pose['right_knee'] + self.place_knee_delta
        place_pose['right_ankle_pitch'] = self.neutral_pose['right_ankle_pitch'] + self.place_ankle_pitch_delta
        place_pose['left_ankle_roll'] = place_pose['left_ankle_roll'] + 0.1

        # Settle by re-centering lateral roll while keeping the placed-foot geometry.
        settle_pose = place_pose
        settle_pose['left_ankle_roll'] = settle_pose['left_ankle_roll'] + 0.05
        settle_pose['left_hip_pitch'] = settle_pose['left_hip_pitch'] + 0.05
        settle_pose['right_hip_pitch'] = settle_pose['right_hip_pitch'] + 0.05
        settle_pose['left_hip_roll'] = self.left_support_hip_roll * 0.5
        settle_pose['right_hip_roll'] = self.left_support_hip_roll * 0.5
        settle_pose['left_ankle_roll'] = self.left_support_ankle_roll * 0.5
        settle_pose['right_ankle_roll'] = self.left_support_ankle_roll * 0.5

        return [
            (StepPhase.SHIFT_TO_LEFT_SUPPORT, shift_left_pose, self.shift_duration_sec),
            (StepPhase.SWING_RIGHT_FOOT, swing_pose, self.swing_duration_sec),
            (StepPhase.PLACE_RIGHT_FOOT, place_pose, self.place_duration_sec),
            (StepPhase.SETTLE_DOUBLE_SUPPORT, settle_pose, self.settle_duration_sec),
        ]

    @staticmethod
    def mirror_pose(pose: Dict[str, float]) -> Dict[str, float]:
        """Mirror a pose about the sagittal plane: swap legs and negate roll/yaw joints."""
        mirrored = {}
        for name, value in pose.items():
            side, joint = name.split('_', 1)
            other = 'right' if side == 'left' else 'left'
            mirrored[f'{other}_{joint}'] = -value if joint in ('hip_roll', 'hip_yaw', 'ankle_roll') else value
        return mirrored

    def mirror_phase_queue(
        self, queue: List[tuple[StepPhase, Dict[str, float], float]]
    ) -> List[tuple[StepPhase, Dict[str, float], float]]:
        """Turn a right-foot step sequence into the equivalent left-foot step sequence."""
        mirrored_phase = {
            StepPhase.SHIFT_TO_LEFT_SUPPORT: StepPhase.SHIFT_TO_RIGHT_SUPPORT,
            StepPhase.SWING_RIGHT_FOOT: StepPhase.SWING_LEFT_FOOT,
            StepPhase.PLACE_RIGHT_FOOT: StepPhase.PLACE_LEFT_FOOT,
            StepPhase.SETTLE_DOUBLE_SUPPORT: StepPhase.SETTLE_DOUBLE_SUPPORT,
        }
        return [(mirrored_phase[phase], self.mirror_pose(pose), duration) for phase, pose, duration in queue]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HumanoidWalking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
