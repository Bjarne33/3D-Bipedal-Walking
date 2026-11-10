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
"""Verify that the panel grays out unavailable actions and calls the manager services."""

import os
import time

from python_qt_binding.QtCore import QObject
from python_qt_binding.QtWidgets import QApplication
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_srvs.srv import Trigger

from humanoid_interfaces.msg import WalkManagerState
from humanoid_ui.walk_manager_panel import SERVICES, STATE_TOPIC, WalkManagerPanel

# Keeps the Qt application alive for the widgets created by the tests.
APP = None


class PluginContext(QObject):
    """Supply the minimal rqt context for offscreen widget tests."""

    def __init__(self, node):
        """Store the ROS node handed to the plugin.

        Args:
            node: ROS node used by the panel.
        """
        super().__init__()
        self.node = node

    def add_widget(self, widget):
        """Accept the widget normally docked by rqt.

        Args:
            widget: Panel widget.
        """
        self.widget = widget


@pytest.fixture
def ros():
    """Provide an initialized ROS context, a panel node and an executor.

    Yields:
        Tuple (node, executor).
    """
    global APP
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    APP = QApplication.instance() or QApplication([])
    rclpy.init()
    node = rclpy.create_node('walk_manager_panel_test')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    yield node, executor
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


def spin_until(executor, condition, timeout=5.0):
    """Process ROS and Qt events until the condition holds.

    Args:
        executor: Executor to spin.
        condition: Callable returning True when done.
        timeout: Maximum time [s].

    Returns:
        The final condition value.
    """
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        executor.spin_once(timeout_sec=0.02)
        QApplication.processEvents()
        if condition():
            return True
    return condition()


def enabled(panel):
    """Read the button enable states.

    Args:
        panel: Panel under test.

    Returns:
        Dict of button key to enabled flag.
    """
    return {name: button.isEnabled() for name, button in panel.buttons.items()}


def test_buttons_follow_manager_state(ros):
    """Enable only the buttons the manager allows and call the selected service."""
    node, executor = ros
    # The context is the plugin's Qt parent and must outlive it.
    context = PluginContext(node)
    panel = WalkManagerPanel(context)
    assert enabled(panel) == {'stop': False, 'walk': False, 'rl_walk': False}

    manager = rclpy.create_node('fake_walk_manager')
    executor.add_node(manager)
    calls = []

    def handler(name):
        """Create a fake manager service callback."""
        def callback(_request, response):
            """Record the call and accept it."""
            calls.append(name)
            response.success = True
            response.message = f'{name} ok'
            return response
        return callback

    for name, service in SERVICES.items():
        manager.create_service(Trigger, service, handler(name))
    publisher = manager.create_publisher(
        WalkManagerState, STATE_TOPIC,
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    publisher.publish(WalkManagerState(
        state='IDLE', walk_enabled=True, rl_walk_enabled=False, stop_enabled=False,
        message='ready'))
    assert spin_until(executor, lambda: enabled(panel)['walk'])
    assert enabled(panel) == {'stop': False, 'walk': True, 'rl_walk': False}
    assert panel.state_label.text() == 'IDLE'

    panel.buttons['walk'].click()
    assert spin_until(executor, lambda: 'walk ok' in panel.response_label.text())
    assert calls == ['walk']

    publisher.publish(WalkManagerState(
        state='WALKING', walk_enabled=False, rl_walk_enabled=False, stop_enabled=True,
        message='walk running'))
    assert spin_until(executor, lambda: enabled(panel)['stop'])
    assert enabled(panel) == {'stop': True, 'walk': False, 'rl_walk': False}

    executor.remove_node(manager)
    manager.destroy_node()
    assert spin_until(executor, lambda: not any(enabled(panel).values())
                      and 'not running' in panel.state_label.text())
    panel.shutdown_plugin()
