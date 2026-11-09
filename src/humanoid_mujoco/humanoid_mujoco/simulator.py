"""MuJoCo simulator node that exposes the same ROS interface as the Gazebo backend.

Interface (identical names, types and frames to Gazebo):

* publishes `/clock`, `/joint_states`, `/odom` (`world` -> `base`, twist in the world
  frame like `gazebo_ros_p3d`) and the ankle wrenches `/left_foot_ft` and `/right_foot_ft`
  (force and torque on the foot from the ankle_pitch joint, in the foot frame, like
  `gazebo_ros_ft_sensor`);
* subscribes to `/<controller_name>/joint_trajectory` and tracks it with the same PD law
  and gains as the Gazebo controller (applied by MuJoCo position actuators);
* provides `/pause_physics`, `/unpause_physics` and `/controller_manager/list_controllers`
  (which reports the joint-state broadcaster and trajectory controller as active), so the
  startup node works unchanged.
"""

import threading
import time

from builtin_interfaces.msg import Time
from control_msgs.msg import JointTrajectoryControllerState
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import WrenchStamped
import mujoco
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectory

from humanoid_mujoco.model import (
    actuated_joints, BASE_BODY, build_model, FT_SENSORS, set_start_camera)
from humanoid_mujoco.trajectory import TrajectoryTracker

# Wall-clock period of the optional viewer refresh [s].
VIEWER_PERIOD = 1.0 / 60.0


def _stamp(sim_time: float) -> Time:
    """Convert a simulation time to a ROS time message.

    Args:
        sim_time: Simulation time [s].

    Returns:
        The equivalent time message.
    """
    nanoseconds = int(round(sim_time * 1e9))
    return Time(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


class MujocoSimulator(Node):
    """Step a MuJoCo model in real time and bridge it to ROS."""

    def __init__(self):
        """Build the model, declare parameters and create the ROS interface."""
        super().__init__('mujoco_sim')
        self.declare_parameter('robot_description', '')
        self.declare_parameter('world', '')
        self.declare_parameter('pause', True)
        self.declare_parameter('spawn_z', -1.0)
        self.declare_parameter('real_time_factor', 1.0)
        self.declare_parameter('joint_state_rate', 1000.0)
        self.declare_parameter('odom_rate', 100.0)
        self.declare_parameter('ft_rate', 500.0)
        self.declare_parameter('viewer', False)
        self.declare_parameter('viewer_distance', 3.0)
        self.declare_parameter('controller_name', 'effort_position_controller')
        self.declare_parameter('joint_state_broadcaster', 'joint_state_broadcaster')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('base_frame', 'base')

        urdf = str(self.get_parameter('robot_description').value)
        world = str(self.get_parameter('world').value)
        if not urdf or not world:
            raise RuntimeError('Parameters robot_description and world are required')
        spawn_z = float(self.get_parameter('spawn_z').value)
        gains = {
            name: (float(self.declare_parameter(f'gains.{name}.p', 0.0).value),
                   float(self.declare_parameter(f'gains.{name}.d', 0.0).value))
            for name in actuated_joints(urdf)
        }
        self.model = build_model(urdf, world, gains, None if spawn_z < 0.0 else spawn_z)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        if self.model.opt.integrator not in (mujoco.mjtIntegrator.mjINT_IMPLICIT,
                                             mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
            self.get_logger().warn(
                'The world does not use an implicit integrator; the stiff joint damping '
                'may be unstable')

        self._init_indices()
        self._init_controller(gains)
        self._init_rates()
        self.world_frame = str(self.get_parameter('world_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)

        self.lock = threading.Lock()
        self.add_on_set_parameters_callback(self._validate_gains)
        self.paused = bool(self.get_parameter('pause').value)
        self.running = True

        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.controller_state_pub = self.create_publisher(
            JointTrajectoryControllerState, f'/{self.controller_name}/controller_state', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.ft_pubs = {name: self.create_publisher(WrenchStamped, f'/{name}', 10)
                        for name in FT_SENSORS}
        self.create_subscription(
            JointTrajectory, f'/{self.controller_name}/joint_trajectory',
            self._trajectory_cb, 10)
        self.create_service(Empty, '/pause_physics', self._pause_cb)
        self.create_service(Empty, '/unpause_physics', self._unpause_cb)
        self.create_service(
            ListControllers, '/controller_manager/list_controllers', self._list_controllers_cb)

        self.viewer = None
        if bool(self.get_parameter('viewer').value):
            self._open_viewer()

        self.clock_pub.publish(Clock(clock=_stamp(self.data.time)))
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.get_logger().info(
            f'MuJoCo {mujoco.__version__} simulation ready '
            f'(nq={self.model.nq}, nu={self.model.nu}, dt={self.model.opt.timestep:g} s, '
            f'{"paused" if self.paused else "running"})')

    def _init_indices(self) -> None:
        """Look up the actuated joints, the base body and the sensor addresses."""
        model = self.model
        self.joint_names = [model.actuator(i).name for i in range(model.nu)]
        joint_ids = [int(model.actuator_trnid[i, 0]) for i in range(model.nu)]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
        self.ft_sensors = {
            name: (model.sensor(f'{name}_force').adr[0], model.sensor(f'{name}_torque').adr[0])
            for name in FT_SENSORS
        }

    def _init_controller(self, gains) -> None:
        """Create the trajectory tracker, holding the initial joint positions.

        Args:
            gains: `(p, d)` gains per actuated joint.
        """
        self.controller_name = str(self.get_parameter('controller_name').value)
        self.jsb_name = str(self.get_parameter('joint_state_broadcaster').value)
        self.tracker = TrajectoryTracker(
            self.joint_names, [gains[name][0] for name in self.joint_names],
            [gains[name][1] for name in self.joint_names])
        self.tracker.hold(self.data.time, self.data.qpos[self.qpos_adr])

    def _init_rates(self) -> None:
        """Convert the publish rates to step decimations."""
        dt = self.model.opt.timestep

        def every(rate):
            return max(1, int(round(1.0 / (float(rate) * dt))))

        self.joint_state_every = every(self.get_parameter('joint_state_rate').value)
        self.odom_every = every(self.get_parameter('odom_rate').value)
        self.ft_every = every(self.get_parameter('ft_rate').value)
        self.real_time_factor = float(self.get_parameter('real_time_factor').value)
        self.steps = 0

    def _open_viewer(self) -> None:
        """Open the passive MuJoCo viewer; continue headless if that fails."""
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as exc:  # The viewer needs a display and OpenGL.
            self.get_logger().warn(f'MuJoCo viewer unavailable, running headless: {exc}')
            return
        with self.viewer.lock():
            set_start_camera(self.model, self.data, self.viewer.cam,
                             float(self.get_parameter('viewer_distance').value))

    # ============================================================
    # Simulation loop
    # ============================================================

    def _run(self) -> None:
        """Step the physics in real time (scaled by the real-time factor) until shutdown."""
        period = self.model.opt.timestep / max(self.real_time_factor, 1e-6)
        next_step = time.perf_counter()
        next_view = next_step
        while self.running and rclpy.ok():
            now = time.perf_counter()
            if self.viewer is not None and now >= next_view:
                next_view = now + VIEWER_PERIOD
                if not self.viewer.is_running():
                    self.viewer = None
                else:
                    with self.lock:
                        self.viewer.sync()
            if self.paused:
                time.sleep(0.001)
                next_step = time.perf_counter()
                continue
            if now < next_step:
                time.sleep(next_step - now)
                continue
            with self.lock:
                self._step()
            next_step += period
            # Do not try to catch up after long stalls (e.g. a slow viewer frame).
            next_step = max(next_step, time.perf_counter() - 0.1)

    def _step(self) -> None:
        """Apply the actuator setpoints, advance one physics step and publish the due messages."""
        data = self.data
        if self.steps % self.joint_state_every == 0:
            self._publish_controller_state(_stamp(data.time))
        data.ctrl[:] = self.tracker.setpoint(data.time)
        mujoco.mj_step(self.model, data)
        self.steps += 1
        stamp = _stamp(data.time)
        self.clock_pub.publish(Clock(clock=stamp))
        if self.steps % self.joint_state_every == 0:
            self._publish_joint_states(stamp)
        if self.steps % self.odom_every == 0:
            self._publish_odom(stamp)
        if self.steps % self.ft_every == 0:
            self._publish_wrenches(stamp)

    def _publish_joint_states(self, stamp: Time) -> None:
        """Publish the actuated joint positions, velocities and applied efforts.

        Args:
            stamp: Simulation time of the state.
        """
        data = self.data
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = self.joint_names
        msg.position = data.qpos[self.qpos_adr].tolist()
        msg.velocity = data.qvel[self.dof_adr].tolist()
        msg.effort = data.qfrc_actuator[self.dof_adr].tolist()
        self.joint_state_pub.publish(msg)

    def _publish_controller_state(self, stamp: Time) -> None:
        """Publish the state and reference used for the next physics step.

        Args:
            stamp: Pre-step simulation timestamp.
        """
        msg = JointTrajectoryControllerState()
        msg.header.stamp = stamp
        msg.joint_names = self.joint_names
        positions, velocities = self.tracker.sample(self.data.time)
        msg.reference.positions = positions.tolist()
        msg.reference.velocities = velocities.tolist()
        msg.feedback.positions = self.data.qpos[self.qpos_adr].tolist()
        msg.feedback.velocities = self.data.qvel[self.dof_adr].tolist()
        msg.error.positions = (positions - self.data.qpos[self.qpos_adr]).tolist()
        msg.error.velocities = (velocities - self.data.qvel[self.dof_adr]).tolist()
        self.controller_state_pub.publish(msg)

    def _validate_gains(self, parameters):
        """Reject invalid gain batches without mutating the simulation.

        Args:
            parameters: Proposed ROS parameters.

        Returns:
            Validation result for the entire batch.
        """
        for parameter in parameters:
            if parameter.name.startswith('gains.'):
                if (parameter.type_ != rclpy.Parameter.Type.DOUBLE
                        or not np.isfinite(parameter.value)):
                    return SetParametersResult(
                        successful=False, reason='Gains must be finite floating-point values')
                if (parameter.name.endswith('.p') and parameter.value <= 0.0
                        or parameter.name.endswith('.d') and parameter.value < 0.0):
                    return SetParametersResult(
                        successful=False, reason='Gains require P > 0 and D >= 0')
        return SetParametersResult(successful=True)

    def set_parameters_atomically(self, parameters):
        """Commit effective gains only after every parameter callback accepts the batch.

        Args:
            parameters: Proposed ROS parameters.

        Returns:
            Atomic parameter update result.
        """
        if not hasattr(self, 'tracker'):
            return super().set_parameters_atomically(parameters)
        with self.lock:
            result = super().set_parameters_atomically(parameters)
            if result.successful and any(p.name.startswith('gains.') for p in parameters):
                p_gains = np.array([
                    self.get_parameter(f'gains.{name}.p').value for name in self.joint_names])
                d_gains = np.array([
                    self.get_parameter(f'gains.{name}.d').value for name in self.joint_names])
                self.model.actuator_gainprm[:, 0] = p_gains
                self.model.actuator_biasprm[:, 1] = -p_gains
                self.model.actuator_biasprm[:, 2] = -d_gains
                self.tracker.p_gains = p_gains
                self.tracker.d_gains = d_gains
            return result

    def _publish_odom(self, stamp: Time) -> None:
        """Publish the base pose and its world-frame twist.

        Args:
            stamp: Simulation time of the state.
        """
        data = self.data
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 0)
        position = data.xpos[self.base_id]
        w, x, y, z = data.xquat[self.base_id]
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.world_frame
        msg.child_frame_id = self.base_frame
        pose = msg.pose.pose
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in position)
        pose.orientation.x, pose.orientation.y = float(x), float(y)
        pose.orientation.z, pose.orientation.w = float(z), float(w)
        twist = msg.twist.twist
        twist.angular.x, twist.angular.y, twist.angular.z = (float(v) for v in velocity[:3])
        twist.linear.x, twist.linear.y, twist.linear.z = (float(v) for v in velocity[3:])
        self.odom_pub.publish(msg)

    def _publish_wrenches(self, stamp: Time) -> None:
        """Publish the ankle wrenches in the foot frames.

        Args:
            stamp: Simulation time of the state.
        """
        sensordata = self.data.sensordata
        for name, (force_adr, torque_adr) in self.ft_sensors.items():
            msg = WrenchStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = FT_SENSORS[name]
            force = sensordata[force_adr:force_adr + 3]
            torque = sensordata[torque_adr:torque_adr + 3]
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = (
                float(v) for v in force)
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = (
                float(v) for v in torque)
            self.ft_pubs[name].publish(msg)

    # ============================================================
    # ROS callbacks
    # ============================================================

    def _trajectory_cb(self, msg: JointTrajectory) -> None:
        """Replace the tracked trajectory; a zero header stamp starts it immediately.

        Args:
            msg: Joint trajectory command.
        """
        points = [
            (point.time_from_start.sec + point.time_from_start.nanosec * 1e-9,
             list(point.positions), list(point.velocities) if point.velocities else None)
            for point in msg.points
        ]
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            now = self.data.time
            try:
                self.tracker.set_trajectory(
                    now, stamp if stamp > 0.0 else now, msg.joint_names, points)
            except ValueError as exc:
                self.get_logger().error(f'Rejected joint trajectory: {exc}')

    def _pause_cb(self, request, response):
        """Pause the physics.

        Args:
            request: Empty request.
            response: Empty response.

        Returns:
            The empty response.
        """
        self.paused = True
        return response

    def _unpause_cb(self, request, response):
        """Resume the physics.

        Args:
            request: Empty request.
            response: Empty response.

        Returns:
            The empty response.
        """
        self.paused = False
        return response

    def _list_controllers_cb(self, request, response):
        """Report the built-in broadcaster and trajectory controller as active.

        Args:
            request: List request.
            response: Response to fill.

        Returns:
            The controller list.
        """
        response.controller = [
            ControllerState(name=self.jsb_name, state='active',
                            type='joint_state_broadcaster/JointStateBroadcaster'),
            ControllerState(name=self.controller_name, state='active',
                            type='joint_trajectory_controller/JointTrajectoryController'),
        ]
        return response

    def destroy_node(self):
        """Stop the simulation thread and close the viewer before destroying the node."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.viewer is not None:
            self.viewer.close()
        super().destroy_node()


def main(args=None):
    """Run the MuJoCo simulator node.

    Args:
        args: Command-line arguments passed to `rclpy.init`.
    """
    rclpy.init(args=args)
    node = MujocoSimulator()
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
