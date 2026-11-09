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
"""Verify GUI units, state gating and separation from ROS service execution."""

import math
from unittest.mock import patch

from python_qt_binding.QtCore import QObject
from python_qt_binding.QtWidgets import QApplication
import pytest

from humanoid_tuning.panel import TuningPanel
from humanoid_tuning.core import validate_position
from humanoid_tuning.session import Session


class PluginContext(QObject):
    """Supply the minimal rqt context for offscreen widget tests."""

    def argv(self):
        """Select MuJoCo for the test panel.

        Returns:
            Plugin command-line arguments.
        """
        return ['--simulator', 'mujoco']

    def add_widget(self, widget):
        """Accept the widget normally docked by rqt.

        Args:
            widget: Plugin widget.
        """
        self.widget = widget


@pytest.fixture
def panel(monkeypatch):
    """Construct the real panel without launching a simulator or plotting process.

    Args:
        monkeypatch: Temporary offscreen platform setting.

    Yields:
        Constructed real plugin.
    """
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    app = QApplication.instance() or QApplication([])
    with patch.object(Session, 'start'), patch.object(Session, 'join'):
        context = PluginContext()
        plugin = TuningPanel(context)
        yield plugin
        plugin.shutdown_plugin()
        plugin.widget.deleteLater()
        app.processEvents()


def test_zero_holding_pose_and_degree_based_targets(panel):
    """Convert degree fields into ROS radians while retaining native gain units."""
    assert all(box.value() == 0 for box in panel.holding.values())
    panel.joint.setCurrentText('left_hip_yaw')
    panel.target.setValue(30)
    panel._start()
    commands = []
    while not panel.session.commands.empty():
        commands.append(panel.session.commands.get_nowait())
    command, args = commands[-1]
    assert command == 'start'
    assert args[0] == 'left_hip_yaw'
    assert args[1] == pytest.approx(math.pi / 6)
    assert args[2].position_tolerance == pytest.approx(.02, abs=1e-8)
    assert not panel.buttons['start'].isEnabled()
    assert not panel.buttons['apply'].isEnabled()
    assert panel.buttons['stop'].isEnabled()


def test_applied_versus_edited_gains_and_hold_changes_invalidate_start(panel):
    """Distinguish unapplied edits and require preparation after holding-pose edits."""
    name = panel.joint.currentText()
    panel.session.events.put(('gains', {name: (123.0, 4.0)}))
    panel.auto_plots = False
    panel.session.events.put(('status', ('ready', 'Ready')))
    panel._drain()
    assert panel.buttons['start'].isEnabled()
    panel.p_gain.setValue(456)
    assert 'NOT applied' in panel.applied_label.text()
    panel.holding[name].setValue(1)
    assert not panel.buttons['start'].isEnabled()
    assert panel.applied[name] == (123.0, 4.0)


def test_knee_target_is_bounded_and_test_joint_always_selected(panel):
    """Do not permit positive knee angles or deselecting the current test joint."""
    panel.joint.setCurrentText('left_knee')
    assert panel.target.maximum() == 0
    assert panel.checks['left_knee'].isChecked()
    assert not panel.checks['left_knee'].isEnabled()
    for name in panel.session.names:
        panel.joint.setCurrentText(name)
        limit = panel.session.limits[name]
        for value in (panel.target.minimum(), panel.target.maximum(),
                      panel.holding[name].minimum(), panel.holding[name].maximum()):
            validate_position(math.radians(value), limit)
