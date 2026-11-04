"""Environment and domain randomization tests (JIT-compiles MJX on CPU)."""

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np
import pytest

from humanoid_rl import benchmark
from humanoid_rl import config as cfg
from humanoid_rl import constants as c
from humanoid_rl.env import HumanoidRLEnv, OBS_SIZE
from humanoid_rl.randomize import FIELDS, make_randomization_fn

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def env():
    return HumanoidRLEnv(cfg.env_config(2))


@pytest.fixture(scope='module')
def fns(env):
    return jax.jit(env.reset), jax.jit(env.step)


def test_reset_observation_and_info(env, fns):
    state = fns[0](jax.random.PRNGKey(0))
    assert state.obs.shape == (OBS_SIZE,) and env.observation_size == OBS_SIZE
    assert float(state.info['command']) == 0.0
    assert 0 <= int(state.info['delay']) <= env._config.latency.max_delay_steps
    assert set(cfg.REWARD_TERMS) <= {k[7:] for k in state.metrics if k.startswith('reward/')}
    feet_z = state.data.site_xpos[env._foot_sites][:, 2]
    assert float(jp.min(feet_z)) > -0.005


def test_zero_action_stands_with_both_feet(env, fns):
    reset, step = fns
    state = reset(jax.random.PRNGKey(1))
    for _ in range(50):
        state = step(state, jp.zeros(c.NUM_JOINTS))
        assert float(state.done) == 0.0
        assert np.isfinite(float(state.reward)) and float(state.reward) >= 0.0
    assert float(state.metrics['double_contact']) == 1.0
    assert float(state.metrics['tilt']) < 0.2


def test_action_latency_and_scaling(env, fns):
    reset, step = fns
    state = reset(jax.random.PRNGKey(2))
    action = jp.full(c.NUM_JOINTS, 0.1)
    delayed = step(state.replace(info={**state.info, 'delay': jp.array(1, jp.int32)}), action)
    np.testing.assert_allclose(delayed.info['motor_targets'], env._default_pose, atol=1e-6)
    direct = step(state.replace(info={**state.info, 'delay': jp.array(0, jp.int32)}), action)
    np.testing.assert_allclose(direct.info['motor_targets'],
                               env._default_pose + env._config.action_scale * 0.1, atol=1e-6)


def test_termination_resamples_episode_info(env, fns):
    reset, step = fns
    state = reset(jax.random.PRNGKey(3))
    qpos = state.data.qpos.at[3:7].set(jp.array([np.cos(0.6), np.sin(0.6), 0.0, 0.0]))
    info = {**state.info, 'last_act': jp.ones(c.NUM_JOINTS),
            'command_steps': jp.array(7, jp.int32)}
    state = state.replace(data=state.data.replace(qpos=qpos), info=info)
    state = step(state, jp.ones(c.NUM_JOINTS))
    assert float(state.done) == 1.0
    np.testing.assert_array_equal(state.info['last_act'], 0.0)
    assert int(state.info['command_steps']) == 0


def test_commands_per_phase():
    keys = jax.random.split(jax.random.PRNGKey(4), 64)
    for phase in (1, 2):
        env = HumanoidRLEnv(cfg.env_config(phase))
        assert float(jp.max(jp.abs(jax.vmap(env._sample_command)(keys)))) == 0.0
    env = HumanoidRLEnv(cfg.env_config(3))
    commands = jax.vmap(env._sample_command)(keys)
    low, high = env._config.command.forward_speed
    assert float(commands.min()) >= low and float(commands.max()) <= high


def test_foot_collision_option():
    with_feet = HumanoidRLEnv(cfg.env_config(1)).mj_model
    config = cfg.env_config(1)
    config.foot_collision = False
    without = HumanoidRLEnv(config).mj_model

    def collides(model, a, b):
        g1, g2 = model.geom(a), model.geom(b)
        return bool((g1.contype & g2.conaffinity) or (g2.contype & g1.conaffinity))

    assert collides(with_feet, *c.FOOT_GEOMS) and not collides(without, *c.FOOT_GEOMS)
    for model in (with_feet, without):
        for foot in c.FOOT_GEOMS:
            assert collides(model, foot, c.FLOOR_GEOM)


def test_randomization(env):
    dr = cfg.domain_randomization_config()
    fn = make_randomization_fn(env.mj_model, dr)
    model, in_axes = fn(env.mjx_model, jax.random.split(jax.random.PRNGKey(5), 6))
    assert isinstance(model, mjx.Model)
    for field in FIELDS:
        assert getattr(in_axes, field) == 0
        assert getattr(model, field).shape[0] == 6
    friction = model.geom_friction[:, env._foot_geoms, 0]
    floor = model.geom_friction[:, env.mj_model.geom(c.FLOOR_GEOM).id, 0]
    np.testing.assert_allclose(friction, np.repeat(floor[:, None], 2, axis=1))
    assert float(floor.min()) >= dr.friction[0] and float(floor.max()) <= dr.friction[1]
    kp = model.actuator_gainprm[:, :, 0]
    np.testing.assert_allclose(model.actuator_biasprm[:, :, 1], -kp)
    dr.enabled = False
    assert make_randomization_fn(env.mj_model, dr) is None


def test_benchmark_sweep():
    results = benchmark.sweep((2,), num_steps=3, env_overrides={'foot_collision': False})
    assert len(results) == 1
    assert results[0]['num_envs'] == 2 and results[0]['steps_per_s'] > 0
