# Copyright 2026 Bjarne
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ROS experiment worker, isolated from the Qt event thread."""

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from queue import Empty, SimpleQueue
import threading
import time
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from control_msgs.msg import JointTrajectoryControllerState
from controller_manager_msgs.srv import ConfigureController, ListControllers, SwitchController
from humanoid_interfaces.msg import JointTuningState
from rcl_interfaces.srv import GetParameters, SetParametersAtomically
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty as EmptyService
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import xacro
import yaml

from humanoid_tuning.core import (
    export_config, native_gains, parse_limits, Settling, validate_gains, validate_position)


CONTROLLER = 'effort_position_controller'
COMMAND_TOPIC = f'/{CONTROLLER}/joint_trajectory'


@dataclass(frozen=True)
class Settings:
    """Editable experiment timings and settling criteria, in seconds and SI units."""

    prepare: float = 2.0
    window: float = 0.5
    timeout: float = 15.0
    observe: float = 5.0
    position_tolerance: float = 0.02
    velocity_tolerance: float = 0.02

    def validate(self):
        """Reject nonpositive/nonfinite timings or tolerances.

        Raises:
            ValueError: If a field is invalid or preparation cannot finish before timeout.
        """
        if not all(math.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError('Timings and settling tolerances must be finite and positive')
        if self.timeout <= self.prepare + self.window:
            raise ValueError('Preparation timeout must exceed motion plus settling window')


def seconds(stamp):
    """Convert a ROS timestamp to seconds.

    Args:
        stamp: ROS Time message.

    Returns:
        Seconds as a float.
    """
    return stamp.sec + stamp.nanosec * 1e-9


class Session(threading.Thread):
    """Own a ROS executor and serialize commands without blocking the GUI."""

    def __init__(self, simulator):
        """Load immutable backend metadata before starting the ROS worker.

        Args:
            simulator: Gazebo or MuJoCo backend name.
        """
        super().__init__(name='humanoid-tuning', daemon=True)
        package = 'humanoid_gazebo' if simulator == 'gazebo' else 'humanoid_mujoco'
        filename = 'ros2_control_controllers.yaml' if simulator == 'gazebo' else 'mujoco_sim.yaml'
        self.baseline = yaml.safe_load(
            (Path(get_package_share_directory(package)) / 'config' / filename).read_text())
        description = Path(get_package_share_directory('humanoid_description')) / 'urdf'
        self.urdf = xacro.process_file(
            str(description / 'humanoid.xacro'), mappings={'floating': 'true'}).toxml()
        self.limits = parse_limits(self.urdf)
        self.names = list(native_gains(self.baseline, simulator))
        if set(self.names) != set(self.limits):
            raise ValueError('Backend gain joints do not match the robot description')
        self.simulator = simulator
        self.commands = SimpleQueue()
        self.events = SimpleQueue()
        self.quit = threading.Event()
        self.interrupt = threading.Event()
        self.state = 'disconnected'
        self.applied = {}
        self.selected = {self.names[0]}
        self.references = deque(maxlen=200)
        self.sample = None
        self.sample_wall = 0.0
        self.reference_wall = 0.0
        self.aligned_wall = 0.0
        self.last_stamp = None
        self.ready = False
        self.pose = {name: 0.0 for name in self.names}
        self.settings = Settings()
        self.settling = None
        self.operation_start = 0.0
        self.telemetry = {}
        self.warned_limits = set()
        self.controllers_seen = False
        self.step_targets = None
        self.step_confirmed = False
        self.step_wall = 0.0

    def submit(self, command, *args):
        """Enqueue a UI operation; Stop interrupts a pending service wait.

        Args:
            command: Operation name.
            *args: Operation arguments, copied by the caller.
        """
        if command == 'stop':
            self.interrupt.set()
        self.commands.put((command, args))

    def report(self, state, message):
        """Notify the UI and invalidate readiness on errors.

        Args:
            state: New experiment state.
            message: User-visible status.
        """
        self.state = state
        if state == 'error':
            self.ready = False
            self.node.get_logger().error(message)
        self.events.put(('status', (state, message)))

    def run(self):
        """Create and spin a private ROS context until the plugin closes."""
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node('humanoid_tuning_session', context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.publisher = self.node.create_publisher(JointTrajectory, COMMAND_TOPIC, 10)
        self.node.create_subscription(
            JointState, '/joint_states', self._joint_state, qos_profile_sensor_data)
        self.node.create_subscription(
            JointTrajectoryControllerState, f'/{CONTROLLER}/controller_state',
            self._controller_state, qos_profile_sensor_data)
        self.clients = {}
        self.submit('connect')
        try:
            while not self.quit.is_set():
                self.executor.spin_once(timeout_sec=0.005)
                try:
                    command, args = self.commands.get_nowait()
                except Empty:
                    command = None
                if command:
                    try:
                        self._dispatch(command, args)
                    except (RuntimeError, ValueError, OSError, KeyError) as exc:
                        self.report('error', str(exc))
                self._tick()
        finally:
            self.executor.remove_node(self.node)
            self.node.destroy_node()
            self.executor.shutdown()
            self.context.shutdown()

    def _call(self, service_type, name, request):
        """Call a service with a bounded wall-time wait while continuing subscriptions.

        Args:
            service_type: ROS service type.
            name: Absolute service name.
            request: Service request.

        Returns:
            Service response.

        Raises:
            RuntimeError: If interrupted, unavailable, timed out or failed.
        """
        key = (service_type, name)
        if key not in self.clients:
            self.clients[key] = self.node.create_client(service_type, name)
        client = self.clients[key]
        deadline = time.monotonic() + 10.0
        future = None
        while time.monotonic() < deadline:
            if self.quit.is_set() or self.interrupt.is_set():
                raise RuntimeError(f'Interrupted while waiting for {name}')
            if future is None and client.service_is_ready():
                future = client.call_async(request)
            if future is not None and future.done():
                if future.exception() is not None:
                    raise RuntimeError(f'{name}: {future.exception()}')
                return future.result()
            self.executor.spin_once(timeout_sec=0.01)
        if future is not None:
            future.cancel()
        raise RuntimeError(f'Service unavailable or timed out: {name}')

    def _parameters(self, values):
        """Atomically set backend parameters and require explicit acceptance.

        Args:
            values: Parameter name to value mapping.

        Raises:
            RuntimeError: If the backend rejects the batch.
        """
        remote = CONTROLLER if self.simulator == 'gazebo' else 'mujoco_sim'
        request = SetParametersAtomically.Request()
        request.parameters = [
            Parameter(name, value=value).to_parameter_msg() for name, value in values.items()]
        response = self._call(
            SetParametersAtomically, f'/{remote}/set_parameters_atomically', request)
        if not response.result.successful:
            raise RuntimeError(f'Parameter update rejected: {response.result.reason}')

    def _read_gains(self):
        """Read confirmed native P/D parameters from the selected backend.

        Returns:
            Joint to P/D mapping.

        Raises:
            RuntimeError: If the backend returns missing or invalid values.
        """
        remote = CONTROLLER if self.simulator == 'gazebo' else 'mujoco_sim'
        names = [f'gains.{name}.{term}' for name in self.names for term in ('p', 'd')]
        response = self._call(
            GetParameters, f'/{remote}/get_parameters', GetParameters.Request(names=names))
        if len(response.values) != len(names) or any(v.type != 3 for v in response.values):
            raise RuntimeError('Backend did not return every P/D gain as a double')
        values = [v.double_value for v in response.values]
        result = dict(zip(self.names, zip(values[::2], values[1::2])))
        for gains in result.values():
            validate_gains(*gains)
        return result

    def _switch(self, activate):
        """Switch the trajectory controller while physics is running.

        Args:
            activate: Whether to activate rather than deactivate the controller.

        Raises:
            RuntimeError: If switching fails.
        """
        request = SwitchController.Request(strictness=SwitchController.Request.STRICT)
        request.timeout.sec = 5
        if activate:
            request.activate_controllers = [CONTROLLER]
        else:
            request.deactivate_controllers = [CONTROLLER]
        response = self._call(
            SwitchController, '/controller_manager/switch_controller', request)
        if not response.ok:
            raise RuntimeError('Controller switch failed')

    def _reconfigure(self, values):
        """Apply Gazebo parameters through its inactive/configure/activate lifecycle.

        Args:
            values: Parameters to apply.

        Raises:
            RuntimeError: If any transition fails; readiness remains disabled.
        """
        self._call(EmptyService, '/unpause_physics', EmptyService.Request())
        self._switch(False)
        self._call(EmptyService, '/pause_physics', EmptyService.Request())
        try:
            self._parameters(values)
        finally:
            # Configure also waits for the controller manager's update loop on Humble.
            self._call(EmptyService, '/unpause_physics', EmptyService.Request())
        response = self._call(
            ConfigureController, '/controller_manager/configure_controller',
            ConfigureController.Request(name=CONTROLLER))
        if not response.ok:
            raise RuntimeError('Controller reconfiguration failed')
        self._switch(True)
        self.references.clear()
        self.aligned_wall = 0.0
        self.ready = False

    def _check_exclusive(self):
        """Reject competing command publishers and duplicate tuning workers.

        Raises:
            RuntimeError: If another command source is present.
        """
        publishers = self.node.get_publishers_info_by_topic(COMMAND_TOPIC)
        if len(publishers) != 1 or publishers[0].node_name != self.node.get_name():
            raise RuntimeError('Conflicting trajectory publishers; use the isolated tuning launch')
        subscriptions = self.node.get_subscriptions_info_by_topic(COMMAND_TOPIC)
        if len(subscriptions) != 1:
            raise RuntimeError('Expected exactly one simulator trajectory subscriber')

    def _connect(self):
        """Verify floating description/controllers and read native applied gains.

        Raises:
            RuntimeError: If the session is not a floating, isolated backend.
        """
        self.report('connecting', 'Waiting for floating simulator and controllers')
        response = self._call(
            GetParameters, '/robot_state_publisher/get_parameters',
            GetParameters.Request(names=['robot_description']))
        if len(response.values) != 1 or response.values[0].type != 4:
            raise RuntimeError('robot_state_publisher did not return a robot description')
        try:
            robot = ET.fromstring(response.values[0].string_value)
        except ET.ParseError as exc:
            raise RuntimeError(f'Invalid published robot description: {exc}') from exc
        if not any(j.get('type') == 'fixed' and j.find('parent') is not None
                   and j.find('parent').get('link') == 'world'
                   for j in robot.findall('joint')):
            raise RuntimeError('Tuning requires floating:=true (base fixed to world)')
        self._call(EmptyService, '/unpause_physics', EmptyService.Request())
        deadline = time.monotonic() + 20
        while True:
            response = self._call(
                ListControllers, '/controller_manager/list_controllers', ListControllers.Request())
            active = {c.name for c in response.controller if c.state == 'active'}
            if {CONTROLLER, 'joint_state_broadcaster'} <= active:
                break
            if self.controllers_seen and any(
                    c.name == CONTROLLER and c.state == 'inactive' for c in response.controller):
                self._switch(True)
            if time.monotonic() >= deadline:
                raise RuntimeError('Joint-state broadcaster and trajectory controller not active')
            self.executor.spin_once(timeout_sec=0.05)
        self.controllers_seen = True
        self._check_exclusive()
        gains = self._read_gains()
        if self.simulator == 'gazebo':
            self._reconfigure({'state_publish_rate': 1000.0})
        self.applied = gains
        self.events.put(('gains', dict(self.applied)))
        self.report('idle', 'Connected. Prepare the holding pose before Start.')

    def _dispatch(self, command, args):
        """Execute one serialized UI command.

        Args:
            command: Operation name.
            args: Command arguments.

        Raises:
            ValueError: If the command or its inputs are invalid.
            RuntimeError: If an operation is not allowed or a backend operation fails.
        """
        if command == 'stop':
            self.interrupt.clear()
            self.ready = False
            self._call(EmptyService, '/pause_physics', EmptyService.Request())
            self.report('paused', 'Physics paused. Prepare before another test.')
            return
        if command == 'select':
            names, = args
            if not set(names) <= set(self.names):
                raise ValueError('Unknown plot joint')
            self.selected = set(names)
            return
        if command == 'export':
            if set(self.applied) != set(self.names):
                raise RuntimeError('Connect and confirm all applied gains before export')
            if self._read_gains() != self.applied:
                raise RuntimeError('Backend parameters changed outside this session; reconnect')
            export_config(args[0], self.baseline, self.simulator, self.applied)
            self.events.put(('notice', f'Exported applied gains to {args[0]}'))
            return
        if self.state in ('preparing', 'running'):
            raise RuntimeError('Finish or Stop the current operation first')
        if command == 'connect':
            self._connect()
        elif command == 'apply':
            name, p_gain, d_gain = args
            if name not in self.names:
                raise ValueError('Unknown test joint')
            validate_gains(p_gain, d_gain)
            self.ready = False
            self.report('applying', 'Applying gains; controller readiness will be checked')
            self._check_exclusive()
            self.applied = {}
            self.events.put(('gains', {}))
            values = {f'gains.{name}.p': float(p_gain), f'gains.{name}.d': float(d_gain)}
            if self.simulator == 'gazebo':
                self._reconfigure(values)
            else:
                self._parameters(values)
            readback = self._read_gains()
            if readback[name] != (p_gain, d_gain):
                raise RuntimeError('Gain readback does not match the requested values')
            self.applied = readback
            self.events.put(('gains', dict(readback)))
            self.report('idle', 'Gains applied. Prepare again before Start.')
        elif command == 'prepare':
            pose, settings = args
            settings.validate()
            if set(pose) != set(self.names):
                raise ValueError('Holding pose must specify every joint')
            for name, value in pose.items():
                validate_position(value, self.limits[name])
            self._check_exclusive()
            self.ready = False
            self._call(EmptyService, '/unpause_physics', EmptyService.Request())
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                self.executor.spin_once(timeout_sec=0.01)
                now = time.monotonic()
                if (self.sample and self.references and now - self.sample_wall < 0.1
                        and now - self.reference_wall < 0.1 and now - self.aligned_wall < 0.1):
                    break
            self._require_fresh()
            self.pose, self.settings = dict(pose), settings
            self.operation_start = self.sample[0]
            self.settling = Settling(
                settings.window, settings.position_tolerance, settings.velocity_tolerance)
            self._trajectory(self.pose, settings.prepare)
            self.report('preparing', 'Moving to holding pose and waiting for all joints to settle')
        elif command == 'start':
            name, target, settings = args
            settings.validate()
            if not self.ready or self.state != 'ready':
                raise RuntimeError('Prepare and settle before Start')
            validate_position(target, self.limits[name])
            self._check_exclusive()
            self._require_fresh()
            self.settings = settings
            targets = dict(self.pose)
            targets[name] = target
            self.step_targets = targets
            self.step_confirmed = False
            self.step_wall = time.monotonic()
            self._trajectory(targets, 0.0)
            self.ready = False
            self.operation_start = self.sample[0]
            self.warned_limits.clear()
            self.report('running', 'Observing step response; controller holds the target')
        else:
            raise ValueError(f'Unknown operation: {command}')

    def _trajectory(self, targets, duration):
        """Command every joint; a zero-time point is an immediate reference step.

        Args:
            targets: Absolute joint targets in radians.
            duration: Transition duration, or zero for a true step.
        """
        point = JointTrajectoryPoint()
        point.positions = [targets[name] for name in self.names]
        point.velocities = [0.0] * len(self.names)
        nanoseconds = round(duration * 1e9)
        point.time_from_start.sec = nanoseconds // 1_000_000_000
        point.time_from_start.nanosec = nanoseconds % 1_000_000_000
        msg = JointTrajectory(joint_names=self.names, points=[point])
        self.publisher.publish(msg)

    def _require_fresh(self):
        """Require complete state and controller feedback, including while paused.

        Raises:
            RuntimeError: If either stream has been absent for more than one wall second.
        """
        now = time.monotonic()
        if (self.sample is None or not self.references or now - self.sample_wall > 1.0
                or now - self.reference_wall > 1.0 or now - self.aligned_wall > 1.0):
            raise RuntimeError('Missing/stale joint or controller-reference telemetry')

    def _controller_state(self, msg):
        """Cache measured controller references without synthesizing commanded values.

        Args:
            msg: Actual controller feedback.
        """
        if (len(msg.joint_names) != len(msg.reference.positions)
                or not set(self.names) <= set(msg.joint_names)
                or not all(math.isfinite(q) for q in msg.reference.positions)):
            self.report('error', 'Controller feedback lacks reference positions')
            return
        stamp = seconds(msg.header.stamp)
        if self.references and stamp < self.references[-1][0]:
            self.references.clear()
            self.report('error', 'Simulation clock reset; Prepare again')
        self.references.append((stamp, msg.header.stamp, dict(
            zip(msg.joint_names, msg.reference.positions))))
        self.reference_wall = time.monotonic()
        if (self.state == 'running' and not self.step_confirmed
                and len(msg.reference.velocities) == len(msg.joint_names)):
            positions = dict(zip(msg.joint_names, msg.reference.positions))
            velocities = dict(zip(msg.joint_names, msg.reference.velocities))
            if all(abs(positions[name] - target) < 1e-10
                   and abs(velocities[name]) < 1e-10
                   for name, target in self.step_targets.items()):
                self.step_confirmed = True
                self.operation_start = stamp

    def _joint_state(self, msg):
        """Validate all measured state arrays and publish selected named telemetry.

        Args:
            msg: Measured joint state.
        """
        complete = all(len(values) == len(msg.name)
                       for values in (msg.position, msg.velocity, msg.effort))
        if (not set(self.names) <= set(msg.name) or not complete
                or len(set(msg.name)) != len(msg.name)):
            self.report('error', 'Joint states must include position, velocity and effort')
            return
        positions, velocities, efforts = (
            dict(zip(msg.name, values)) for values in (msg.position, msg.velocity, msg.effort))
        if not all(math.isfinite(values[name]) for values in (positions, velocities, efforts)
                   for name in self.names):
            self.report('error', 'Nonfinite joint-state sample')
            return
        stamp = seconds(msg.header.stamp)
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.references.clear()
            self.report('error', 'Simulation clock reset; Prepare again')
        self.last_stamp = stamp
        self.sample = (stamp, positions, velocities, efforts)
        self.sample_wall = time.monotonic()
        reference = next((r for r in reversed(self.references) if r[0] <= stamp + 1e-9), None)
        if reference is None or stamp - reference[0] > 0.01:
            return
        self.aligned_wall = time.monotonic()
        for name in self.selected:
            if name not in reference[2]:
                self.report('error', f'Reference feedback lacks {name}')
                return
            if name not in self.telemetry:
                self.telemetry[name] = self.node.create_publisher(
                    JointTuningState, f'/humanoid_tuning/{name}/state', 10)
            limit = self.limits[name]
            state = JointTuningState()
            state.header.stamp = msg.header.stamp
            state.reference_stamp = reference[1]
            state.joint_name = name
            state.position_deg = math.degrees(positions[name])
            state.reference_position_deg = math.degrees(reference[2][name])
            state.velocity_deg_s = math.degrees(velocities[name])
            state.velocity_upper_deg_s = math.degrees(limit.velocity)
            state.velocity_lower_deg_s = -state.velocity_upper_deg_s
            state.effort_nm = efforts[name]
            state.effort_upper_nm, state.effort_lower_nm = limit.effort, -limit.effort
            self.telemetry[name].publish(state)
            if (self.state == 'running' and name not in self.warned_limits
                    and (abs(velocities[name]) > limit.velocity
                         or abs(efforts[name]) > limit.effort * (1 + 1e-6))):
                self.warned_limits.add(name)
                self.events.put(('notice', f'URDF limit exceeded by {name}; physics continues'))

    def _tick(self):
        """Advance preparation/trial states on simulation time and detect stale streams."""
        if self.state not in ('preparing', 'ready', 'running'):
            return
        try:
            self._require_fresh()
        except RuntimeError as exc:
            self.report('error', str(exc))
            return
        stamp, positions, velocities, _ = self.sample
        if self.state == 'running' and not self.step_confirmed:
            if time.monotonic() - self.step_wall > 1.0:
                self.report('error', 'Controller did not confirm the requested step reference')
            return
        elapsed = stamp - self.operation_start
        if self.state == 'preparing':
            if elapsed > self.settings.timeout:
                self.report('error', 'Preparation timed out; physics remains running')
            elif elapsed >= self.settings.prepare and self.settling.update(
                    stamp, [positions[n] for n in self.names],
                    [velocities[n] for n in self.names], [self.pose[n] for n in self.names]):
                self.ready = True
                self.report('ready', 'Holding pose settled. Ready for a step.')
        elif self.state == 'running' and elapsed >= self.settings.observe:
            self.report('complete', 'Observation complete; target held. Prepare to repeat.')
