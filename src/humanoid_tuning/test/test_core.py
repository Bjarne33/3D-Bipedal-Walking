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
"""Verify tuning validation, repeatability and non-destructive native export."""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from unittest.mock import Mock

from builtin_interfaces.msg import Time
from control_msgs.msg import JointTrajectoryControllerState
import pytest
from sensor_msgs.msg import JointState
import yaml

from humanoid_tuning.core import (
    export_config, JointLimits, native_gains, parse_limits, Settling,
    validate_gains, validate_position)
from humanoid_tuning.plots import SIGNALS, write_layout
from humanoid_tuning.session import Session, Settings


@pytest.mark.parametrize('p_gain,d_gain', [
    (0, 1), (-1, 0), (1, -1), (math.nan, 0), (1, math.inf)])
def test_invalid_gains(p_gain, d_gain):
    """Reject zero/negative P, negative D and nonfinite gains."""
    with pytest.raises(ValueError):
        validate_gains(p_gain, d_gain)


def test_gain_and_target_boundaries():
    """Accept inclusive URDF bounds and D zero, but not out-of-range knee targets."""
    validate_gains(0.01, 0)
    limit = JointLimits(-2.4, 0.0, 15, 180)
    validate_position(0, limit)
    validate_position(-2.4, limit)
    for angle in (0.001, -2.401, math.nan):
        with pytest.raises(ValueError):
            validate_position(angle, limit)


def test_urdf_limits_and_zero_pose():
    """Read the actual model limits and validate the zero holding pose."""
    session = Session('gazebo')
    assert len(session.limits) == 12
    for limit in session.limits.values():
        validate_position(0, limit)
    assert session.limits['left_knee'].upper == 0
    assert session.limits['left_hip_yaw'].velocity == 15
    with pytest.raises(ValueError):
        parse_limits('<robot/>')
    with pytest.raises(ValueError):
        parse_limits('<robot><joint name="bad" type="revolute"/></robot>')


def test_settling_requires_continuous_all_joint_position_and_speed():
    """Reset the settled window on motion, position error and clock discontinuities."""
    settling = Settling()
    assert not settling.update(1, [0, 0], [0, 0], [0, 0])
    assert not settling.update(1.3, [0, 0], [0, 0.03], [0, 0])
    assert not settling.update(2, [0.03, 0], [0, 0], [0, 0])
    assert not settling.update(3, [0.01, 0], [0, 0.01], [0, 0])
    assert settling.update(3.5, [0, 0], [0, 0], [0, 0])
    assert not settling.update(0, [0, 0], [0, 0], [0, 0])
    assert not settling.update(1, [math.nan, 0], [0, 0], [0, 0])


@pytest.mark.parametrize('backend', ['gazebo', 'mujoco'])
def test_export_preserves_entire_backend_and_never_overwrites(tmp_path, backend):
    """Change only applied native PD entries without mutating sources or existing exports."""
    session = Session(backend)
    original = deepcopy(session.baseline)
    applied = {'left_knee': (12.345, 0.0)}
    target = tmp_path / 'tuned.yaml'
    export_config(target, session.baseline, backend, applied)
    exported = yaml.safe_load(target.read_text())
    expected = deepcopy(original)
    native_gains(expected, backend)['left_knee'].update(p=12.345, d=0.0)
    assert exported == expected
    assert session.baseline == original
    with pytest.raises(FileExistsError):
        export_config(target, session.baseline, backend, applied)
    assert yaml.safe_load(target.read_text()) == expected
    with pytest.raises(ValueError):
        export_config(tmp_path / 'invalid.yaml', original, backend, {'bad': (1, 0)})
    assert not (tmp_path / 'invalid.yaml').exists()


def test_export_rejects_source_directory(tmp_path):
    """Keep new YAML exports out of source and generated directories."""
    session = Session('mujoco')
    folder = tmp_path / 'src'
    folder.mkdir()
    with pytest.raises(ValueError, match='outside'):
        export_config(folder / 'new.yaml', session.baseline, 'mujoco', {})


def test_settings_are_explicit_and_validated():
    """Validate the confirmed timing and tolerance defaults."""
    settings = Settings()
    timings = (settings.prepare, settings.window, settings.timeout, settings.observe)
    assert timings == (2, .5, 15, 5)
    assert settings.position_tolerance == settings.velocity_tolerance == .02
    settings.validate()
    for invalid in (Settings(prepare=0), Settings(observe=math.nan), Settings(timeout=1)):
        with pytest.raises(ValueError):
            invalid.validate()


def test_layout_has_only_selected_joints_separate_units_and_limit_lines(tmp_path):
    """Generate only selected joint curves in three separate unit-labelled plots."""
    target = tmp_path / 'plots.xml'
    write_layout(target, ['right_knee', 'left_hip_yaw'])
    root = ET.parse(target).getroot()
    assert root.attrib['pj4_version'] == '4'
    assert root.attrib['binding'] == 'generic'
    tabs = root.findall('./tabbed_widget/Tab')
    assert [tab.attrib['tab_name'] for tab in tabs] == ['right_knee', 'left_hip_yaw']
    for tab in tabs:
        plots = tab.findall('.//plot')
        assert len(plots) == 3
        assert [p.attrib['title'] for p in plots] == [title for title, _ in SIGNALS]
        assert all(p.attrib['style'] == 'Steps' for p in plots)
        assert [c.attrib['field'] for c in plots[0]] == [
            'position_deg', 'reference_position_deg']
        assert [c.attrib['field'] for c in plots[1]] == [
            'velocity_deg_s', 'velocity_upper_deg_s', 'velocity_lower_deg_s']
    assert not root.findall('.//curve[@dataset_id]')
    assert not list(Path(tmp_path).glob('*.yaml'))


def test_telemetry_uses_actual_named_feedback_degree_units_and_urdf_limits():
    """Map reordered measured arrays and real reference feedback to selected-joint telemetry."""
    session = Session('mujoco')
    session.node = Mock()
    session.node.create_publisher.side_effect = [Mock(), Mock()]
    names = list(reversed(session.names))
    state = JointState()
    state.name = names
    state.header.stamp = Time(sec=3, nanosec=1_000_000)
    state.position = [.01 * i for i in range(len(names))]
    state.velocity = [.02 * i for i in range(len(names))]
    state.effort = [float(i) for i in range(len(names))]
    reference = JointTrajectoryControllerState()
    reference.header.stamp = Time(sec=3)
    reference.joint_names = names
    reference.reference.positions = [.03 * i for i in range(len(names))]
    session.selected = {'left_knee'}
    session._controller_state(reference)
    session._joint_state(state)
    assert set(session.telemetry) == {'left_knee'}
    publisher = session.telemetry['left_knee']
    message = publisher.publish.call_args.args[0]
    index = names.index('left_knee')
    assert message.header.stamp == state.header.stamp
    assert message.reference_stamp == reference.header.stamp
    assert message.position_deg == pytest.approx(math.degrees(state.position[index]))
    assert message.reference_position_deg == pytest.approx(
        math.degrees(reference.reference.positions[index]))
    assert message.velocity_deg_s == pytest.approx(math.degrees(state.velocity[index]))
    assert message.velocity_upper_deg_s == pytest.approx(math.degrees(15))
    assert message.velocity_lower_deg_s == -message.velocity_upper_deg_s
    assert message.effort_nm == state.effort[index]
    assert (message.effort_lower_nm, message.effort_upper_nm) == (-180, 180)
    session.selected = {'right_knee'}
    session._joint_state(state)
    assert publisher.publish.call_count == 1
    assert session.telemetry['right_knee'].publish.call_count == 1


def test_stale_reference_is_not_replaced_by_a_command_or_zero_default():
    """Do not publish made-up reference values if timestamp-aligned feedback is missing."""
    session = Session('mujoco')
    session.node = Mock()
    reference = JointTrajectoryControllerState()
    reference.header.stamp = Time(sec=1)
    reference.joint_names = session.names
    reference.reference.positions = [0.0] * len(session.names)
    session._controller_state(reference)
    state = JointState(
        name=session.names, position=[0.0] * 12, velocity=[0.0] * 12, effort=[0.0] * 12)
    state.header.stamp = Time(sec=2)
    session._joint_state(state)
    assert not session.telemetry
    assert session.aligned_wall == 0
    session.state = 'ready'
    session._tick()
    assert session.state == 'error'
    assert not session.ready


def test_trial_clock_starts_only_after_actual_zero_velocity_reference_confirmation():
    """Do not declare a successful step from a published target or nonzero-velocity ramp."""
    session = Session('mujoco')
    session.node = Mock()
    session.state = 'running'
    session.step_targets = dict(session.pose)
    session.step_targets['left_hip_yaw'] = .1
    feedback = JointTrajectoryControllerState()
    feedback.header.stamp = Time(sec=5)
    feedback.joint_names = session.names
    feedback.reference.positions = [0.0] * len(session.names)
    feedback.reference.velocities = [0.0] * len(session.names)
    session._controller_state(feedback)
    assert not session.step_confirmed
    index = session.names.index('left_hip_yaw')
    feedback.reference.positions[index] = .1
    feedback.reference.velocities[index] = .2
    session._controller_state(feedback)
    assert not session.step_confirmed
    feedback.reference.velocities[index] = 0.0
    session._controller_state(feedback)
    assert session.step_confirmed
    assert session.operation_start == 5.0


def test_tuning_launch_is_isolated_floating_and_uses_standalone_arguments():
    """Force the floating backend and avoid competing bringup/startup command nodes."""
    from launch.actions import IncludeLaunchDescription
    from launch import LaunchContext
    from launch.utilities import perform_substitutions
    from launch_ros.actions import Node
    path = Path(__file__).resolve().parents[1] / 'launch' / 'tuning.launch.py'
    spec = importlib.util.spec_from_file_location('humanoid_tuning_launch_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    backends = [e for e in description.entities if isinstance(e, IncludeLaunchDescription)]
    assert len(backends) == 2
    for backend in backends:
        assert dict(backend.launch_arguments)['floating'] == 'true'
        assert dict(backend.launch_arguments)['pause'] == 'false'
    nodes = [e for e in description.entities if isinstance(e, Node)]
    assert len(nodes) == 1
    assert nodes[0].node_package == 'humanoid_tuning'
    assert nodes[0].node_executable == 'tuning_panel'
    context = LaunchContext()
    context.launch_configurations['simulator'] = 'mujoco'
    command = [perform_substitutions(context, part) for part in nodes[0].cmd]
    assert '--args' not in command
    assert command[1:3] == ['--simulator', 'mujoco']
