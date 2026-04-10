import math
from enum import Enum
from typing import Dict, List

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


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
        self.timer = self.create_timer(1.0 / max(self.update_rate_hz, 1.0), self.update)

        self.get_logger().info(
            'Humanoid walking node ready: call /humanoid_walking/step_once for test shift-swing-place-settle behavior'
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
            self.get_logger().info('Single-step sequence complete')
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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HumanoidWalking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
