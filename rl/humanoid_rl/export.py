"""Policy parameter utilities and the NumPy policy export.

Brax PPO params are a tuple `(normalizer_params, policy_params, value_params)`. The policy is an
MLP (`hidden_0` ... `hidden_n`, swish activations) producing `2 * action_size` values: the first
half is the Gaussian mean `loc`, the second half the raw scale (std = softplus(raw) + min_std).
The deterministic action is `tanh(loc)`.

The exported `.npz` contains:
    obs_mean, obs_std        observation normalizer (obs_norm = (obs - mean) / std)
    w0, b0, ..., wN, bN      dense layers (y = x @ w + b), swish between layers
    action_scale, default_pose, joint_names, obs_size, action_size, ctrl_dt, phase
The joint target is `default_pose + action_scale * tanh(loc)`.
"""

import pickle
from typing import Any, Dict, Optional

import jax
import numpy as np

from humanoid_rl import constants as c

MIN_STD = 0.001  # brax NormalTanhDistribution default.


def _layer_names(policy_params: Dict[str, Any]) -> list:
    layers = policy_params['params']
    names = sorted((k for k in layers if k.startswith('hidden_')), key=lambda k: int(k[7:]))
    if not names or len(names) != len(layers):
        raise ValueError(f'Unexpected policy parameter layout: {sorted(layers)}')
    return names


def reset_policy_std(params: tuple, std: float) -> tuple:
    """Return params whose policy std is `std` for every input (exploration reset).

    Zeroes the scale half of the output layer kernel and sets its bias to softplus^-1(std).
    """
    normalizer, policy, value = params
    policy = jax.tree_util.tree_map(np.array, jax.device_get(policy))
    last = policy['params'][_layer_names(policy)[-1]]
    action_size = last['bias'].shape[-1] // 2
    raw = np.log(np.expm1(std - MIN_STD))
    last['kernel'][..., action_size:] = 0.0
    last['bias'][..., action_size:] = raw
    return normalizer, policy, value


def save_params(path: str, params: tuple) -> None:
    with open(path, 'wb') as f:
        pickle.dump(jax.device_get(params), f)


def load_params(path: str) -> tuple:
    with open(path, 'rb') as f:
        return pickle.load(f)


def export_npz(path: str, params: tuple, env_config: Optional[Any] = None) -> None:
    """Write the deterministic policy (normalizer + actor MLP) as a NumPy `.npz`."""
    normalizer, policy, _ = jax.device_get(params)
    arrays = {
        'obs_mean': np.asarray(normalizer.mean, np.float32),
        'obs_std': np.asarray(normalizer.std, np.float32),
    }
    for i, name in enumerate(_layer_names(policy)):
        arrays[f'w{i}'] = np.asarray(policy['params'][name]['kernel'], np.float32)
        arrays[f'b{i}'] = np.asarray(policy['params'][name]['bias'], np.float32)
    action_size = arrays[f'b{i}'].shape[-1] // 2
    arrays.update(
        num_layers=np.int32(i + 1),
        obs_size=np.int32(arrays['obs_mean'].shape[-1]),
        action_size=np.int32(action_size),
        default_pose=np.asarray(c.STANDING_POSE, np.float32),
        joint_names=np.asarray(c.JOINT_NAMES),
    )
    if env_config is not None:
        arrays.update(action_scale=np.float32(env_config.action_scale),
                      ctrl_dt=np.float32(env_config.ctrl_dt),
                      gait_period=np.float32(env_config.gait.period),
                      phase=np.int32(env_config.phase))
    np.savez(path, **arrays)


class NumpyPolicy:
    """Dependency-free (NumPy only) deterministic policy loaded from an exported `.npz`."""

    def __init__(self, path: str) -> None:
        data = np.load(path)
        self.obs_mean = data['obs_mean']
        self.obs_std = data['obs_std']
        n = int(data['num_layers'])
        self.layers = [(data[f'w{i}'], data[f'b{i}']) for i in range(n)]
        self.action_size = int(data['action_size'])
        self.meta = {k: data[k] for k in data.files
                     if not k.startswith(('w', 'b', 'obs_')) or k == 'obs_size'}

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Action in [-1, 1] for an observation (or a batch of observations)."""
        x = (np.asarray(obs, np.float32) - self.obs_mean) / self.obs_std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = x / (1.0 + np.exp(-x))  # swish
        return np.tanh(x[..., :self.action_size])
