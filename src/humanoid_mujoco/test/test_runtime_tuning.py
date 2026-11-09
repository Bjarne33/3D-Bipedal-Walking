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
"""Verify committed runtime PD gains and actual pre-step reference feedback."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import mujoco
import numpy as np
import pytest
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.parameter import Parameter
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import xacro

from humanoid_mujoco.simulator import MujocoSimulator, _stamp


@pytest.fixture
def simulator():
    """Create a paused, pinned simulator with ROS parameter services.

    Yields:
        Simulator node; its worker cannot advance paused physics.
    """
    path = Path(get_package_share_directory('humanoid_description')) / 'urdf' / 'humanoid.xacro'
    urdf = xacro.process_file(str(path), mappings={'floating': 'true'}).toxml()
    root = Path(__file__).resolve().parents[1]
    rclpy.init(args=[
        '--ros-args', '--params-file', str(root / 'config' / 'mujoco_sim.yaml'),
        '-p', f'robot_description:={urdf}', '-p', f'world:={root / "worlds" / "world.xml"}'])
    node = MujocoSimulator()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_atomic_gain_updates_change_effective_actuators_and_physical_response(simulator):
    """Verify a committed P change alters actuator coefficients and measured motion."""
    node = simulator
    index = node.joint_names.index('left_hip_yaw')
    positions = np.zeros(len(node.joint_names))
    positions[index] = .1
    node.tracker.hold(0, positions)
    displacements = []
    for p_gain in (20.0, 600.0):
        result = node.set_parameters_atomically([
            Parameter('gains.left_hip_yaw.p', value=p_gain),
            Parameter('gains.left_hip_yaw.d', value=2.0)])
        assert result.successful
        assert node.model.actuator_gainprm[index, 0] == p_gain
        assert node.model.actuator_biasprm[index, 1:3] == pytest.approx([-p_gain, -2])
        assert node.tracker.p_gains[index] == p_gain
        assert node.tracker.d_gains[index] == 2
        mujoco.mj_resetData(node.model, node.data)
        for _ in range(50):
            node.data.ctrl[:] = node.tracker.setpoint(node.data.time)
            mujoco.mj_step(node.model, node.data)
        displacements.append(node.data.qpos[node.qpos_adr[index]])
    assert displacements[1] > displacements[0] * 2


@pytest.mark.parametrize('bad', [0.0, -1.0, float('nan'), float('inf')])
def test_invalid_batch_cannot_change_any_effective_gain(simulator, bad):
    """Reject invalid batches without changing parameters or actuators."""
    node = simulator
    before = node.model.actuator_gainprm.copy()
    p_before = node.get_parameter('gains.left_hip_yaw.p').value
    result = node.set_parameters_atomically([
        Parameter('gains.left_hip_yaw.d', value=1.0),
        Parameter('gains.left_hip_yaw.p', value=bad)])
    assert not result.successful
    assert np.array_equal(node.model.actuator_gainprm, before)
    assert node.get_parameter('gains.left_hip_yaw.p').value == p_before


def test_later_callback_rejection_does_not_mutate_effective_gains(simulator):
    """Leave actuators unchanged if any callback rejects a parameter update."""
    node = simulator
    before = node.model.actuator_gainprm.copy()
    node.add_on_set_parameters_callback(
        lambda parameters: SetParametersResult(successful=False, reason='test rejection'))
    result = node.set_parameters_atomically([
        Parameter('gains.left_hip_yaw.p', value=123.0)])
    assert not result.successful
    assert np.array_equal(node.model.actuator_gainprm, before)


def test_zero_time_point_is_true_step_with_zero_reference_velocity(simulator):
    """Apply a zero-time target immediately rather than interpolating a short ramp."""
    node = simulator
    positions = [0.0] * len(node.joint_names)
    index = node.joint_names.index('left_hip_yaw')
    positions[index] = .1
    node._trajectory_cb(JointTrajectory(
        joint_names=node.joint_names, points=[JointTrajectoryPoint(
            positions=positions, velocities=[0.0] * len(positions))]))
    q, velocity = node.tracker.sample(node.data.time)
    assert q == pytest.approx(positions)
    assert velocity == pytest.approx(np.zeros(len(positions)))
    assert node.tracker.setpoint(node.data.time) == pytest.approx(positions)


def test_feedback_reports_pre_step_reference_and_matching_measured_state(simulator):
    """Publish the actual pre-step reference with a matching measured-state timestamp."""
    node = simulator
    captured = []

    class Publisher:
        """Capture a controller-state message without a ROS subscriber."""

        def publish(self, message):
            """Record the published message.

            Args:
                message: Controller-state message.
            """
            captured.append(message)

    original = node.controller_state_pub
    node.controller_state_pub = Publisher()
    try:
        node._publish_controller_state(_stamp(node.data.time))
    finally:
        node.controller_state_pub = original
    msg, = captured
    assert msg.header.stamp == _stamp(node.data.time)
    assert msg.joint_names == node.joint_names
    assert msg.reference.positions == pytest.approx(node.tracker.sample(node.data.time)[0])
    assert msg.feedback.positions == pytest.approx(node.data.qpos[node.qpos_adr])
