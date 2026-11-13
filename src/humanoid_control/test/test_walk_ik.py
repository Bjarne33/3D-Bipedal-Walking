"""Test for walking-node IK parameter validation"""

import pytest
import rclpy

from humanoid_control.nodes.humanoid_walk import HumanoidWalk


@pytest.mark.parametrize('iterations', [0, -1])
def test_invalid_ik_iterations_rejected_at_startup(iterations):
    rclpy.init(args=['--ros-args', '-p', f'ik_iterations:={iterations}'])
    try:
        with pytest.raises(ValueError, match='ik_iterations must be at least 1'):
            HumanoidWalk()
    finally:
        rclpy.shutdown()


