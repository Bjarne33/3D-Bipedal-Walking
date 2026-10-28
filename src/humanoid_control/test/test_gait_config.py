import dataclasses
import os

import yaml

from humanoid_control.tools.gait_params import GaitParams

GAIT_YAML = os.path.join(os.path.dirname(__file__), '..', 'config', 'gait.yaml')

# Node-level parameters of humanoid_walk that are not GaitParams fields.
NODE_PARAMETERS = {
    'controller_name', 'update_rate_hz', 'ik_iterations', 'trajectory_lead_sec', 'first_swing',
    'forward_axis', 'ankle_to_sole', 'odom_topic', 'left_ft_topic', 'right_ft_topic',
    'ground_height', 'contact_tolerance',
}


def test_gait_yaml_matches_declared_parameters():
    with open(GAIT_YAML) as f:
        keys = set(yaml.safe_load(f)['humanoid_walk']['ros__parameters'])
    declared = {field.name for field in dataclasses.fields(GaitParams)} | NODE_PARAMETERS
    assert keys == declared
