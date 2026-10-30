import ast
import dataclasses
import os

import yaml

from humanoid_control.tools.gait_params import GaitParams

GAIT_YAML = os.path.join(os.path.dirname(__file__), '..', 'config', 'gait.yaml')

# Node-level parameters of humanoid_walk that are not GaitParams fields.
NODE_PARAMETERS = {
    'controller_name', 'update_rate_hz', 'ik_iterations', 'trajectory_lead_sec', 'first_swing',
    'forward_axis', 'ankle_to_sole', 'state_timeout_sec',
}


def test_gait_yaml_matches_declared_parameters():
    with open(GAIT_YAML) as f:
        keys = set(yaml.safe_load(f)['humanoid_walk']['ros__parameters'])
    declared = {field.name for field in dataclasses.fields(GaitParams)} | NODE_PARAMETERS
    assert keys == declared


def test_estimator_yaml_matches_node_defaults_and_shared_geometry():
    root = os.path.join(os.path.dirname(__file__), '..')
    with open(os.path.join(root, 'config', 'estimator.yaml')) as f:
        params = yaml.safe_load(f)['humanoid_estimator']['ros__parameters']
    with open(os.path.join(root, 'humanoid_control', 'nodes', 'humanoid_estimator.py')) as f:
        tree = ast.parse(f.read())
    defaults = next(
        ast.literal_eval(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'defaults'
                for target in node.targets)
    )
    assert params == defaults
    with open(GAIT_YAML) as f:
        gait = yaml.safe_load(f)['humanoid_walk']['ros__parameters']
    assert params['ankle_to_sole'] == gait['ankle_to_sole']
    assert params['update_rate_hz'] == gait['update_rate_hz']
