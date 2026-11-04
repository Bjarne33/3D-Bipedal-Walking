"""Parameter helpers and the NumPy policy export versus brax inference."""

import functools

from brax.training import acme
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp
import numpy as np
import pytest

from humanoid_rl import config as cfg
from humanoid_rl import export
from humanoid_rl.env import OBS_SIZE

ACTIONS = 12


@pytest.fixture(scope='module')
def network_and_params():
    factory = functools.partial(ppo_networks.make_ppo_networks,
                                policy_hidden_layer_sizes=(32, 16),
                                value_hidden_layer_sizes=(16,))
    network = factory(OBS_SIZE, ACTIONS, preprocess_observations_fn=running_statistics.normalize)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    normalizer = running_statistics.init_state(acme.specs.Array((OBS_SIZE,), jp.float32))
    normalizer = normalizer.replace(
        mean=jax.random.normal(k1, (OBS_SIZE,)),
        std=jp.exp(0.3 * jax.random.normal(k2, (OBS_SIZE,))))
    params = (normalizer, network.policy_network.init(k3), network.value_network.init(k4))
    return network, params


def test_numpy_policy_matches_brax(network_and_params, tmp_path):
    network, params = network_and_params
    path = tmp_path / 'policy.npz'
    export.export_npz(str(path), params, cfg.env_config(3))
    policy = export.NumpyPolicy(str(path))
    assert int(policy.meta['phase']) == 3
    assert policy.meta['action_scale'] == pytest.approx(0.3)

    brax_policy = ppo_networks.make_inference_fn(network)(params, deterministic=True)
    obs = jax.random.normal(jax.random.PRNGKey(1), (16, OBS_SIZE)) * 2
    expected = np.asarray(brax_policy(obs, jax.random.PRNGKey(2))[0])
    np.testing.assert_allclose(policy(np.asarray(obs)), expected, atol=1e-5)
    np.testing.assert_allclose(policy(np.asarray(obs[0])), expected[0], atol=1e-5)


def test_reset_policy_std(network_and_params):
    network, params = network_and_params
    reset = export.reset_policy_std(params, 0.5)
    obs = jax.random.normal(jax.random.PRNGKey(3), (8, OBS_SIZE))
    logits = network.policy_network.apply(reset[0], reset[1], obs)
    dist = network.parametric_action_distribution.create_dist(logits)
    np.testing.assert_allclose(dist.scale, 0.5, rtol=1e-5)
    # The mean is unchanged by the reset.
    original = network.policy_network.apply(params[0], params[1], obs)
    np.testing.assert_allclose(logits[:, :ACTIONS], original[:, :ACTIONS], atol=1e-6)


def test_save_and_load_params(network_and_params, tmp_path):
    _, params = network_and_params
    path = str(tmp_path / 'params.pkl')
    export.save_params(path, params)
    loaded = export.load_params(path)
    for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(loaded)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
