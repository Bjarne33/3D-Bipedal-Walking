"""Deterministic policy evaluation, curriculum thresholds and video rendering.

Evaluation uses the nominal model (no domain randomization), observation noise as in training,
and no command resampling inside an episode. Environments that terminate are masked out from
then on; `survival` is the fraction of environments that never terminated.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from humanoid_rl import config as cfg
from humanoid_rl.env import HumanoidRLEnv

Policy = Callable[[jax.Array, jax.Array], Tuple[jax.Array, dict]]
# Steps at the start of each episode excluded from the averages (initial transient).
SETTLE_STEPS = 50


def make_eval_env(phase: int, **overrides) -> HumanoidRLEnv:
    config = cfg.env_config(phase)
    config.command.resample_time = 1e9
    config.update(overrides)
    return HumanoidRLEnv(config)


def make_evaluator(env: HumanoidRLEnv, make_policy: Callable[[Any], Policy], num_envs: int,
                   num_steps: int) -> Callable[[Any, jax.Array], Dict[str, float]]:
    """Return `evaluate(params, rng) -> metrics`, compiled once for all params.

    Runs `num_envs` deterministic episodes of `num_steps` control steps; `make_policy(params)`
    must return a deterministic policy `(obs, rng) -> (action, extras)`.
    """

    @jax.jit
    def run(params, rng):
        policy = make_policy(params)
        reset_rng, policy_rng = jax.random.split(rng)
        state = jax.vmap(env.reset)(jax.random.split(reset_rng, num_envs))

        def body(carry, step_key):
            state, alive, step_index, sums, weight_sum = carry
            keys = jax.random.split(step_key, num_envs)
            action = jax.vmap(lambda o, k: policy(o, k)[0])(state.obs, keys)
            state = jax.vmap(env.step)(state, action)
            alive = alive * (1.0 - state.done)
            weight = alive * (step_index >= SETTLE_STEPS)
            sums = {k: sums[k] + jp.sum(jp.where(weight > 0, v, 0.0) * weight)
                    for k, v in state.metrics.items()}
            return (state, alive, step_index + 1, sums, weight_sum + jp.sum(weight)), None

        sums = {k: jp.zeros(()) for k in state.metrics}
        init = (state, jp.ones(num_envs), jp.zeros((), jp.int32), sums, jp.zeros(()))
        (state, alive, _, sums, weight_sum), _ = jax.lax.scan(
            body, init, jax.random.split(policy_rng, num_steps))
        final_drift = jp.sum(jp.where(alive > 0, state.metrics['drift'], 0.0))
        return sums, weight_sum, alive, final_drift

    def evaluate(params, rng: jax.Array) -> Dict[str, float]:
        sums, weight_sum, alive, final_drift = jax.device_get(run(params, rng))
        total = max(float(weight_sum), 1.0)

        def mean(name):
            return float(sums[name]) / total

        count = float(sums['swing_mid_count'])
        survivors = max(float(np.sum(alive)), 1.0)
        result = {
            'survival': float(np.mean(alive)),
            'tilt': mean('tilt'),
            'height_error': mean('height_error'),
            'double_contact': mean('double_contact'),
            'contact_match': mean('contact_match'),
            'swing_height_ratio': float(sums['swing_mid_height']) / count if count else 0.0,
            'drift': float(final_drift) / survivors,
            'vel_error': mean('vel_error'),
            'lateral_vel': mean('lateral_vel'),
            'forward_vel': mean('forward_vel'),
        }
        result.update({k: mean(k) for k in sums if k.startswith('reward/')})
        return result

    return evaluate


def rollout_metrics(env: HumanoidRLEnv, policy: Policy, rng: jax.Array, num_envs: int,
                    num_steps: int) -> Dict[str, float]:
    """One-off evaluation of a fixed policy (see `make_evaluator`)."""
    return make_evaluator(env, lambda _: policy, num_envs, num_steps)(None, rng)


def check_thresholds(phase: int, metrics: Dict[str, float],
                     thresholds: Optional[Dict[str, float]] = None
                     ) -> Tuple[bool, Dict[str, bool]]:
    """Compare metrics with `thresholds` (default `config.PHASE_THRESHOLDS[phase]`).

    Returns (passed, per-check results).
    """
    thresholds = thresholds if thresholds is not None else cfg.PHASE_THRESHOLDS[phase]
    checks = {}
    for name, value in thresholds.items():
        kind, metric = name.split('_', 1)
        if kind == 'min':
            checks[name] = metrics[metric] >= value
        elif kind == 'max':
            checks[name] = metrics[metric] <= value
        else:
            raise ValueError(f'Threshold names must start with min_ or max_: {name!r}')
    return all(checks.values()), checks


def rollout_qpos(env: HumanoidRLEnv, policy: Policy, rng: jax.Array,
                 num_steps: int, command: Optional[float] = None) -> np.ndarray:
    """Single deterministic episode; returns qpos per control step (stops at termination)."""
    reset_rng, policy_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    if command is not None:
        state.info['command'] = jp.asarray(command, dtype=jp.float32)
        state = state.replace(obs=env._get_obs(state.data, state.info, reset_rng))

    def body(state, key):
        action = policy(state.obs, key)[0]
        state = env.step(state, action)
        return state, (state.data.qpos, state.done)

    _, (qpos, done) = jax.jit(lambda s, k: jax.lax.scan(body, s, k))(
        state, jax.random.split(policy_rng, num_steps))
    qpos, done = np.asarray(qpos), np.asarray(done)
    end = int(np.argmax(done)) + 1 if done.any() else num_steps
    return np.concatenate([np.asarray(state.data.qpos)[None], qpos[:end]])


def render_qpos(mj_model: mujoco.MjModel, qpos: Sequence[np.ndarray], height: int = 360,
                width: int = 640, camera: str = 'track') -> list:
    """Render frames for a sequence of qpos with the MJCF tracking camera."""
    data = mujoco.MjData(mj_model)
    frames = []
    with mujoco.Renderer(mj_model, height=height, width=width) as renderer:
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(mj_model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
    return frames


def save_video(path: str, frames: list, fps: float) -> None:
    import mediapy
    mediapy.write_video(path, frames, fps=fps)
