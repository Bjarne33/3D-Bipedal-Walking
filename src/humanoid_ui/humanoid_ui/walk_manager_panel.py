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
"""rqt panel with Stop, Walk and RL Walk buttons for the humanoid walk manager."""

import sys

from python_qt_binding.QtCore import QObject, QTimer, Signal
from python_qt_binding.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from rclpy.qos import DurabilityPolicy, QoSProfile
from rqt_gui_py.plugin import Plugin
from std_srvs.srv import Trigger

from humanoid_interfaces.msg import WalkManagerState

STATE_TOPIC = '/humanoid_walk_manager/state'
SERVICES = {
    'walk': '/humanoid_walk_manager/walk',
    'rl_walk': '/humanoid_walk_manager/rl_walk',
    'stop': '/humanoid_walk_manager/stop',
}


class _Bridge(QObject):
    """Move ROS callbacks from the rqt executor thread to the Qt thread."""

    state = Signal(object)
    response = Signal(str, str)


class WalkManagerPanel(Plugin):
    """Show the walk manager state and offer only the currently possible actions."""

    def __init__(self, context):
        """Build the widget and connect it to the walk manager topic and services.

        Args:
            context: rqt plugin context providing the ROS node.
        """
        super().__init__(context)
        self.setObjectName('HumanoidWalkManager')
        self.node = context.node
        self.latest = None
        self.bridge = _Bridge()
        self.bridge.state.connect(self._apply_state)
        self.bridge.response.connect(self._show_response)

        self.widget = QWidget()
        self.widget.setObjectName('HumanoidWalkManagerUi')
        self.widget.setWindowTitle('Humanoid walk manager')
        layout = QVBoxLayout(self.widget)
        self.state_label = QLabel()
        self.state_label.setStyleSheet('font-weight: bold; font-size: 14pt;')
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.response_label = QLabel()
        self.response_label.setWordWrap(True)
        buttons = QHBoxLayout()
        self.buttons = {
            'stop': QPushButton('Stop'),
            'walk': QPushButton('Walk'),
            'rl_walk': QPushButton('RL Walk'),
        }
        for name, button in self.buttons.items():
            button.setMinimumHeight(40)
            button.clicked.connect(lambda _checked=False, n=name: self._call(n))
            buttons.addWidget(button)
        layout.addWidget(self.state_label)
        layout.addWidget(self.message_label)
        layout.addLayout(buttons)
        layout.addWidget(self.response_label)
        layout.addStretch()
        context.add_widget(self.widget)

        self.clients = {
            name: self.node.create_client(Trigger, service) for name, service in SERVICES.items()
        }
        self.subscription = self.node.create_subscription(
            WalkManagerState, STATE_TOPIC, self._state_received,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._refresh()
        # Detects a stopped manager without relying on (possibly paused) simulation time.
        self.timer = QTimer(self.widget)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(500)

    def _manager_running(self) -> bool:
        """Check whether the walk manager publishes its state.

        Returns:
            True if a publisher of the state topic exists.
        """
        return self.node.count_publishers(STATE_TOPIC) > 0

    def _state_received(self, msg: WalkManagerState) -> None:
        """Forward a state message from the ROS executor thread to the Qt thread.

        Args:
            msg: Walk manager state.
        """
        self.bridge.state.emit(msg)

    def _apply_state(self, msg: WalkManagerState) -> None:
        """Store a received manager state and update the widget.

        Args:
            msg: Walk manager state.
        """
        self.latest = msg
        self._refresh()

    def _refresh(self) -> None:
        """Update labels and enable only the buttons the manager currently accepts."""
        msg = self.latest if self._manager_running() else None
        if msg is None:
            self.state_label.setText('Walk manager not running')
            self.message_label.setText(f'Waiting for {STATE_TOPIC}')
            enabled = {name: False for name in self.buttons}
        else:
            self.state_label.setText(msg.state)
            self.message_label.setText(msg.message)
            enabled = {'stop': msg.stop_enabled, 'walk': msg.walk_enabled,
                       'rl_walk': msg.rl_walk_enabled}
        for name, button in self.buttons.items():
            button.setEnabled(bool(enabled[name]) and self.clients[name].service_is_ready())

    def _call(self, name: str) -> None:
        """Send a request to a walk manager service.

        Args:
            name: Service key ('stop', 'walk' or 'rl_walk').
        """
        client = self.clients[name]
        if not client.service_is_ready():
            self._show_response(name, 'service not available')
            return
        self.buttons[name].setEnabled(False)
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda f: self.bridge.response.emit(name, self._result(f)))

    @staticmethod
    def _result(future) -> str:
        """Describe the outcome of a service call.

        Args:
            future: Completed service future.

        Returns:
            Human-readable outcome.
        """
        if future.exception() is not None:
            return f'failed: {future.exception()}'
        result = future.result()
        return ('' if result.success else 'refused: ') + result.message

    def _show_response(self, name: str, text: str) -> None:
        """Show a service response.

        Args:
            name: Service key.
            text: Outcome description.
        """
        self.response_label.setText(f'{self.buttons[name].text()}: {text}')
        self._refresh()

    def shutdown_plugin(self):
        """Release the ROS entities created on the shared rqt node."""
        self.timer.stop()
        self.node.destroy_subscription(self.subscription)
        for client in self.clients.values():
            self.node.destroy_client(client)


def main():
    """Start the panel as a standalone rqt application.

    Returns:
        rqt application exit status.
    """
    from rqt_gui.main import Main
    return Main().main(argv=sys.argv, standalone='humanoid_ui.walk_manager_panel.WalkManagerPanel')
