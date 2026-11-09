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
"""Opt-in real-backend lifecycle, gain-effect and reference-step verification."""

import os
import math
from pathlib import Path
from queue import Empty
import signal
import subprocess
import time

import pytest
import yaml
from humanoid_interfaces.msg import JointTuningState

from humanoid_tuning.session import seconds, Session, Settings


BACKEND = os.environ.get('HUMANOID_TUNING_TEST_BACKEND')
pytestmark = pytest.mark.skipif(
    BACKEND not in ('gazebo', 'mujoco'), reason='Set HUMANOID_TUNING_TEST_BACKEND')


class RecordingSession(Session):
    """Capture real controller and joint samples during the existing worker execution."""

    def __init__(self, backend):
        """Initialize sample captures.

        Args:
            backend: Selected real simulator.
        """
        super().__init__(backend)
        self.trial_references = []
        self.trial_samples = []
        self.trial_telemetry = []

    def _controller_state(self, msg):
        """Record actual controller references during a trial.

        Args:
            msg: Backend feedback message.
        """
        super()._controller_state(msg)
        if self.state == 'running':
            self.trial_references.append(msg)

    def _joint_state(self, msg):
        """Record measured named joint state during a trial.

        Args:
            msg: Backend measured state.
        """
        if not hasattr(self, 'telemetry_subscription'):
            self.telemetry_subscription = self.node.create_subscription(
                JointTuningState, '/humanoid_tuning/left_hip_yaw/state',
                self._tuning_state, 10)
        super()._joint_state(msg)
        if self.state == 'running':
            self.trial_samples.append(msg)

    def _tuning_state(self, msg):
        """Record the serialized ROS telemetry received by a real subscriber.

        Args:
            msg: Selected-joint plot telemetry.
        """
        if self.state == 'running':
            self.trial_telemetry.append(msg)


def wait_state(session, expected, timeout=40):
    """Wait for a worker state and fail immediately on a visible backend error.

    Args:
        session: Running experiment worker.
        expected: Desired state.
        timeout: Wall-clock deadline.

    Raises:
        AssertionError: If an error occurs or the operation times out.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            kind, value = session.events.get(timeout=.1)
        except Empty:
            continue
        if kind == 'status':
            assert value[0] != 'error', value[1]
            if value[0] == expected:
                return
    raise AssertionError(f'Expected {expected}, got {session.state}')


@pytest.fixture
def real_session(tmp_path, monkeypatch):
    """Launch an owned isolated simulator and clean up its worker and processes.

    Args:
        tmp_path: Test log directory.
        monkeypatch: Temporary environment configuration.

    Yields:
        Real recording session, ready for preparation.
    """
    monkeypatch.setenv('ROS_DOMAIN_ID', '79')
    monkeypatch.setenv('GAZEBO_MASTER_URI', 'http://127.0.0.1:11379')
    command = [
        'ros2', 'launch', f'humanoid_{BACKEND}', 'simulation.launch.py',
        'floating:=true', 'pause:=false']
    if BACKEND == 'mujoco':
        command.append('gui:=false')
    with (tmp_path / 'simulation.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        session = RecordingSession(BACKEND)
        try:
            session.start()
            wait_state(session, 'idle', 50)
            assert process.poll() is None
            yield session
        finally:
            session.quit.set()
            session.interrupt.set()
            session.join(timeout=5)
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)


def test_real_gain_effect_reference_step_and_repeatable_lifecycle(real_session, tmp_path):
    """Verify physical gain effects, real steps, held joints, export and Stop/re-Prepare."""
    session = real_session
    settings = Settings(observe=1.0)
    test_joint = 'left_hip_yaw'
    session.submit('select', [test_joint])
    responses = []
    for p_gain in (20.0, 600.0):
        session.submit('apply', test_joint, p_gain, 2.0)
        wait_state(session, 'idle')
        assert session.applied[test_joint] == (p_gain, 2.0)
        session.submit('prepare', dict(session.pose), settings)
        wait_state(session, 'ready')
        session.trial_samples.clear()
        session.trial_references.clear()
        session.trial_telemetry.clear()
        session.submit('start', test_joint, .1, settings)
        wait_state(session, 'complete')
        references = session.trial_references
        assert len(references) >= 100
        stepped = []
        for msg in references:
            index = msg.joint_names.index(test_joint)
            q_ref = msg.reference.positions[index]
            assert q_ref == pytest.approx(0.0, abs=1e-10) or q_ref == pytest.approx(.1)
            if abs(q_ref - .1) < 1e-10:
                stepped.append(msg)
                assert msg.reference.velocities[index] == pytest.approx(0.0)
            for name, position in zip(msg.joint_names, msg.reference.positions):
                if name != test_joint:
                    assert position == pytest.approx(0.0)
        assert len(stepped) >= 100
        assert len(session.trial_telemetry) >= 100
        for msg in session.trial_telemetry:
            assert msg.joint_name == test_joint
            assert 0 <= seconds(msg.header.stamp) - seconds(msg.reference_stamp) <= .010001
            assert msg.velocity_upper_deg_s == pytest.approx(math.degrees(15))
            assert (msg.effort_lower_nm, msg.effort_upper_nm) == (-200, 200)
        start = seconds(stepped[0].header.stamp)
        sample = next(msg for msg in session.trial_samples
                      if seconds(msg.header.stamp) >= start + .05)
        responses.append(sample.position[sample.name.index(test_joint)])
    assert responses[1] > responses[0] * 2
    session.submit('stop')
    wait_state(session, 'paused')
    time.sleep(1.2)
    session.submit('prepare', dict(session.pose), settings)
    wait_state(session, 'ready')
    destination = tmp_path / 'exported.yaml'
    session.submit('export', str(destination))
    deadline = time.monotonic() + 5
    while not destination.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert destination.exists()
    node = 'effort_position_controller' if BACKEND == 'gazebo' else 'mujoco_sim'
    config = yaml.safe_load(Path(destination).read_text())
    assert config[node]['ros__parameters']['gains'][test_joint]['p'] == 600.0
