"""Tests of the NumPy RL policy deployment (`humanoid_control.tools.rl_policy`)."""

import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import numpy as np
import pytest
import xacro

from humanoid_control.tools.joints import JOINT_NAMES, STANDING_POSE
from humanoid_control.tools.rl_policy import (
    advance_phase, build_observation, clamp_speed, clock_obs, joint_limits_from_urdf,
    OBS_SIZE, quaternion_to_matrix, RlPolicy)

PACKAGE = Path(__file__).resolve().parents[1]
POLICY = PACKAGE / 'policies' / 'policy.npz'
RL_MJCF = PACKAGE.parents[1] / 'rl' / 'humanoid_rl' / 'assets' / 'humanoid_rl.xml'


@pytest.fixture(scope='module')
def policy():
    """Load the default policy once."""
    return RlPolicy(str(POLICY))


def _urdf() -> str:
    """Expand the shared description with a free base.

    Returns:
        The URDF XML.
    """
    path = Path(get_package_share_directory('humanoid_description')) / 'urdf' / 'humanoid.xacro'
    return xacro.process_file(str(path), mappings={'floating': 'false'}).toxml()


def _rl_model():
    """Load the RL training MJCF, skipping if MuJoCo or the asset is unavailable.

    Returns:
        Tuple of the `mujoco` module and the compiled model.
    """
    mujoco = pytest.importorskip('mujoco')
    if not RL_MJCF.exists():
        pytest.skip(f'{RL_MJCF} not found')
    return mujoco, mujoco.MjModel.from_xml_path(str(RL_MJCF))


def test_policy_metadata_matches_control_joints(policy):
    assert policy.joint_names == list(JOINT_NAMES)
    assert policy.default_pose == pytest.approx([STANDING_POSE[n] for n in JOINT_NAMES])
    assert policy.action_size == len(JOINT_NAMES)
    assert policy.ctrl_dt > 0.0 and policy.gait_period > 0.0 and policy.action_scale > 0.0


def test_policy_forward_pass_matches_manual_mlp(policy):
    data = np.load(POLICY)
    obs = np.random.default_rng(0).normal(size=OBS_SIZE).astype(np.float32)
    x = (obs - data['obs_mean']) / data['obs_std']
    n = int(data['num_layers'])
    for i in range(n):
        x = x @ data[f'w{i}'] + data[f'b{i}']
        if i < n - 1:
            with np.errstate(over='ignore'):
                x = x * (1.0 / (1.0 + np.exp(-x)))
    expected = np.tanh(x[:policy.action_size])
    action = policy(obs)
    assert action.shape == (policy.action_size,)
    assert np.allclose(action, expected, atol=1e-5)
    assert np.all(np.abs(action) <= 1.0)


def test_policy_rejects_incomplete_export(tmp_path):
    data = dict(np.load(POLICY))
    del data['ctrl_dt']
    path = tmp_path / 'policy.npz'
    np.savez(path, **data)
    with pytest.raises(ValueError, match='ctrl_dt'):
        RlPolicy(str(path))


def test_joint_targets_scale_and_clip(policy):
    lower, upper = joint_limits_from_urdf(_urdf(), JOINT_NAMES)
    zero = policy.joint_targets(np.zeros(12), lower, upper)
    assert zero == pytest.approx(policy.default_pose)
    full = policy.joint_targets(np.ones(12), lower, upper)
    assert np.all(full <= upper)
    assert full == pytest.approx(np.minimum(policy.default_pose + policy.action_scale, upper))


def test_joint_limits_match_training_model():
    mujoco, model = _rl_model()
    lower, upper = joint_limits_from_urdf(_urdf(), JOINT_NAMES)
    for i, name in enumerate(JOINT_NAMES):
        assert model.jnt_range[model.joint(name).id] == pytest.approx([lower[i], upper[i]])


def test_joint_limits_require_every_joint():
    with pytest.raises(ValueError, match='no position limits'):
        joint_limits_from_urdf('<robot name="r"/>', ['left_knee'])


def test_clamp_speed():
    assert clamp_speed(0.1, 0.05, 0.2) == (0.1, False)
    assert clamp_speed(0.0, 0.05, 0.2) == (0.05, True)
    assert clamp_speed(1.0, 0.05, 0.2) == (0.2, True)


def test_gait_clock():
    assert advance_phase(0.0, 0.02, 0.8) == pytest.approx(0.025)
    assert advance_phase(0.99, 0.02, 0.8) == pytest.approx(0.015)
    assert clock_obs(0.25) == pytest.approx([1.0, 0.0], abs=1e-12)


def test_quaternion_to_matrix():
    half = math.sqrt(0.5)
    rotation = quaternion_to_matrix((0.0, 0.0, half, half))
    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        quaternion_to_matrix((0.0, 0.0, 0.0, 0.0))


def test_observation_matches_training_sensors(policy):
    """Build the observation from world-frame base data like the training env's sensors."""
    mujoco, model = _rl_model()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(1)
    data.qpos[:] = model.keyframe('home').qpos
    data.qpos[3:7] = rng.normal(size=4)
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    data.qpos[7:] += rng.uniform(-0.1, 0.1, size=12)
    data.qvel[:] = rng.normal(size=model.nv)
    mujoco.mj_forward(model, data)

    base = model.body('base').id
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 0)
    w, x, y, z = data.xquat[base]
    last = rng.uniform(-1.0, 1.0, size=12)
    obs = build_observation(quaternion_to_matrix((x, y, z, w)), velocity[:3], 0.1,
                            data.qpos[7:], data.qvel[6:], policy.default_pose, last, 0.3)

    imu = data.site_xmat[model.site('imu').id].reshape(3, 3)
    expected = np.concatenate([
        data.sensor('gyro').data, imu.T @ [0.0, 0.0, -1.0], [0.1],
        data.qpos[7:] - policy.default_pose, data.qvel[6:], last, clock_obs(0.3)])
    assert obs.shape == (OBS_SIZE,)
    assert obs == pytest.approx(expected.astype(np.float32), abs=1e-5)


def test_policy_walks_forward_in_training_model(policy):
    """Close the loop in the training MJCF: the robot walks along +Y without falling."""
    mujoco, model = _rl_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe('home').id)
    mujoco.mj_forward(model, data)
    base = model.body('base').id
    lower, upper = model.jnt_range[1:].T
    substeps = int(round(policy.ctrl_dt / model.opt.timestep))
    start = data.xpos[base].copy()
    phase, last, speed, duration = 0.0, np.zeros(12), 0.15, 6.0
    while data.time < duration:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 0)
        obs = build_observation(data.xmat[base].reshape(3, 3), velocity[:3], speed,
                                data.qpos[7:], data.qvel[6:], policy.default_pose, last, phase)
        last = policy(obs)
        data.ctrl[:] = policy.joint_targets(last, lower, upper)
        phase = advance_phase(phase, policy.ctrl_dt, policy.gait_period)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        assert data.xmat[base][8] > math.cos(math.radians(30.0))
    forward = (data.xpos[base] - start)[1]
    assert forward > 0.5 * speed * duration
