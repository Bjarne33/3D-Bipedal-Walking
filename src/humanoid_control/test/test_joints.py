import os

import yaml

from humanoid_control.tools.joints import JOINT_NAMES, STANDING_POSE

CONTROLLERS_YAML = os.path.join(
    os.path.dirname(__file__), '..', '..', 'humanoid_description', 'config',
    'ros2_control_controllers.yaml')


def test_standing_pose_covers_all_joints():
    assert len(JOINT_NAMES) == 12
    assert set(STANDING_POSE) == set(JOINT_NAMES)


def test_joint_order_matches_controller_config():
    with open(CONTROLLERS_YAML) as f:
        config = yaml.safe_load(f)
    assert config['effort_position_controller']['ros__parameters']['joints'] == JOINT_NAMES
