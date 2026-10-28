"""Gait monitor: measured balance and tracking of the robot in Gazebo while it walks.

Combines the Gazebo ground-truth base pose (`/gazebo/model_states`) with `/joint_states` to get
the measured whole-body CoM and the foot soles (box collisions from the URDF) in the world frame.
Logs at `log_rate_hz`:

  - walk phase and step index (from the `humanoid_walk` action feedback),
  - support: which feet are on the ground and the height of each sole above it,
  - CoM margin: distance of the CoM ground projection to the edge of the support polygon
    (> 0 inside). Only meaningful for quasi-static motion,
  - DCM margin: same for the capture point xi = c + c'/omega, the dynamic stability indicator,
  - ref-meas: reference CoM (action feedback) minus measured CoM,
  - base roll/pitch, worst joint tracking error and the joint closest to its effort limit.

Warns when the DCM leaves the support polygon, a joint saturates, or the robot has fallen.
"""

import os

import numpy as np
import pinocchio as pin
import rclpy
from ament_index_python.packages import get_package_share_directory
from control_msgs.msg import JointTrajectoryControllerState
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from humanoid_control.gait.gait_params import GRAVITY
from humanoid_control.gait.support_polygon import convex_hull, signed_margin
from humanoid_control.humanoid_kinematics import HumanoidKinematics
from humanoid_interfaces.action import Walk


class GaitMonitor(Node):
    def __init__(self) -> None:
        super().__init__('gait_monitor')
        self.declare_parameter('model_name', 'humanoid')
        self.declare_parameter('model_states_topic', '/gazebo/model_states')
        self.declare_parameter('controller_name', 'effort_position_controller')
        self.declare_parameter('walk_action', '/humanoid_walk/walk')
        self.declare_parameter('log_rate_hz', 10.0)
        self.declare_parameter('ground_height', 0.0)
        self.declare_parameter('contact_tolerance', 0.005)  # sole within this height counts as grounded
        self.declare_parameter('fall_tilt_deg', 45.0)
        self.declare_parameter('saturation_ratio', 0.95)  # |effort| / limit that counts as saturated
        self.declare_parameter('com_velocity_filter', 0.3)  # low-pass weight of a new CoM velocity sample

        p = self.get_parameter
        self.model_name = str(p('model_name').value)
        self.ground = float(p('ground_height').value)
        self.contact_tol = float(p('contact_tolerance').value)
        self.fall_tilt = float(p('fall_tilt_deg').value)
        self.saturation = float(p('saturation_ratio').value)
        self.alpha = float(p('com_velocity_filter').value)

        share = get_package_share_directory('humanoid_description')
        self.kin = HumanoidKinematics.from_xacro(os.path.join(share, 'urdf', 'humanoid.xacro'))
        if any(c is None for c in self.kin.sole_corners.values()):
            raise RuntimeError('foot links need a box collision to define the soles')
        self.effort_limits = {
            name: float(self.kin.model.effortLimit[self.kin.model.idx_vs[jid]])
            for name, jid in self.kin.joint_name_to_id.items()
            if jid > 0 and self.kin.model.joints[jid].nv == 1
        }

        self.base = None
        self.have_joints = False
        self.prev = None  # (time, CoM) for the velocity estimate
        self.com_vel = np.zeros(3)
        self.controller = None
        self.feedback = None
        self.goal_id = None
        self.ref_frame = None  # base pose when the current walk started (feedback CoM is in that frame)
        self.fallen = False

        self.create_subscription(JointState, '/joint_states', self._joints_cb, 10)
        self.create_subscription(
            ModelStates, str(p('model_states_topic').value), self._model_states_cb, qos_profile_sensor_data)
        self.create_subscription(
            JointTrajectoryControllerState, f'/{p("controller_name").value}/controller_state',
            self._controller_cb, 10)
        self.create_subscription(
            Walk.Impl.FeedbackMessage, f'{p("walk_action").value}/_action/feedback', self._feedback_cb, 10)
        self.create_timer(1.0 / max(float(p('log_rate_hz').value), 0.1), self._report)
        self.get_logger().info(f'Gait monitor running for model "{self.model_name}"')

    # ------------------------------------------------------------------ inputs
    def _joints_cb(self, msg: JointState) -> None:
        self.kin.set_joint_positions(msg.name, msg.position)
        self.have_joints = True

    def _model_states_cb(self, msg: ModelStates) -> None:
        if self.model_name not in msg.name:
            return
        pose = msg.pose[msg.name.index(self.model_name)]
        self.base = pin.XYZQUATToSE3(np.array([
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]))

    def _controller_cb(self, msg: JointTrajectoryControllerState) -> None:
        self.controller = msg

    def _feedback_cb(self, msg) -> None:
        goal_id = bytes(msg.goal_id.uuid)
        if goal_id != self.goal_id and self.base is not None:
            # feedback starts within 0.1 s of the walk start, before the robot has moved
            self.goal_id = goal_id
            self.ref_frame = self.base.copy()
            self.fallen = False
        self.feedback = msg.feedback

    # ------------------------------------------------------------------ evaluation
    def _measure(self):
        kin = self.kin
        kin.q[:7] = pin.SE3ToXYZQUAT(self.base)
        pin.forwardKinematics(kin.model, kin.data, kin.q)
        pin.updateFramePlacements(kin.model, kin.data)
        com = np.array(pin.centerOfMass(kin.model, kin.data, kin.q))
        soles = {
            side: np.array([kin.data.oMf[fid].act(c) for c in kin.sole_corners[side]])
            for side, fid in kin.foot_frame_ids.items()
        }
        return com, soles

    def _update_com_velocity(self, com) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.prev is not None and now > self.prev[0]:
            vel = (com - self.prev[1]) / (now - self.prev[0])
            self.com_vel = (1.0 - self.alpha) * self.com_vel + self.alpha * vel
        self.prev = (now, com)

    def _tracking(self):
        """(worst tracking error [deg] with joint, joint closest to its effort limit with ratio)."""
        msg = self.controller
        if msg is None or not msg.error.positions:
            return None, None
        err = np.abs(np.degrees(np.array(msg.error.positions)))
        worst_err = (msg.joint_names[int(np.argmax(err))], float(err.max()))
        worst_effort = None
        for name, effort in zip(msg.joint_names, msg.output.effort):
            limit = self.effort_limits.get(name, 0.0)
            if limit > 0.0 and (worst_effort is None or abs(effort) / limit > worst_effort[1]):
                worst_effort = (name, abs(effort) / limit, effort)
        return worst_err, worst_effort

    def _report(self) -> None:
        if self.base is None or not self.have_joints:
            return
        com, soles = self._measure()
        self._update_com_velocity(com)
        height = com[2] - self.ground
        omega = np.sqrt(GRAVITY / height) if height > 0.05 else np.inf
        dcm = com + self.com_vel / omega

        lift = {side: float(corners[:, 2].min() - self.ground) for side, corners in soles.items()}
        grounded = [side for side in soles if lift[side] < self.contact_tol]
        if grounded:
            hull = convex_hull(np.vstack([soles[side][:, :2] for side in grounded]))
            com_margin, dcm_margin = signed_margin(com, hull), signed_margin(dcm, hull)
        else:
            com_margin = dcm_margin = float('nan')
        roll, pitch, _ = np.degrees(pin.rpy.matrixToRpy(self.base.rotation))

        parts = []
        if self.feedback is not None:
            parts.append(f'{self.feedback.phase:5} k={self.feedback.step_index}')
        support = {2: 'double', 1: f'{grounded[0] if grounded else ""} only', 0: 'none'}[len(grounded)]
        parts.append(f'support={support:10} lift L={lift["left"]:+.3f} R={lift["right"]:+.3f}')
        parts.append(f'com_m={com_margin:+.3f} dcm_m={dcm_margin:+.3f}')
        if self.feedback is not None and self.ref_frame is not None:
            c = self.feedback.com
            ref = self.ref_frame.act(np.array([c.x, c.y, c.z]))
            parts.append(f'ref-meas=({ref[0] - com[0]:+.3f},{ref[1] - com[1]:+.3f})')
        parts.append(f'rp=({roll:+.1f},{pitch:+.1f})')
        worst_err, worst_effort = self._tracking()
        if worst_err:
            parts.append(f'err {worst_err[0]}={worst_err[1]:.1f}deg')
        if worst_effort:
            parts.append(f'tau {worst_effort[0]}={worst_effort[2]:+.0f}Nm ({100 * worst_effort[1]:.0f}%)')
        line = ' '.join(parts)

        log = self.get_logger()
        tilt = np.degrees(np.arccos(np.clip(self.base.rotation[2, 2], -1.0, 1.0)))
        if tilt > self.fall_tilt:
            if not self.fallen:
                log.error(f'FALLEN (tilt {tilt:.0f} deg): {line}')
                self.fallen = True
            return
        if worst_effort and worst_effort[1] >= self.saturation:
            log.warn(f'SATURATED {worst_effort[0]}: {line}')
        elif grounded and dcm_margin < 0.0:
            log.warn(f'DCM outside support: {line}')
        else:
            log.info(line)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GaitMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
