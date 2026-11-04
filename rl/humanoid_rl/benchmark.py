"""Throughput benchmark of the batched training environment (run on the target GPU).

Measures environment steps per second of the wrapped, domain-randomized environment with
random actions for several batch sizes, which is the upper bound of PPO throughput and the
basis for choosing `ppo_config().num_envs`.
"""

import time
from typing import Dict, Iterable, Optional

import jax
import jax.numpy as jp
from mujoco_playground import wrapper

from humanoid_rl import config as cfg
from humanoid_rl.env import HumanoidRLEnv
from humanoid_rl.randomize import make_randomization_fn


def env_throughput(num_envs: int, num_steps: int = 100, phase: int = 3,
                   env_overrides: Optional[dict] = None, seed: int = 0) -> Dict[str, float]:
    """Return compile time [s] and env steps per second for `num_envs` parallel envs."""
    config = cfg.env_config(phase)
    config.update(env_overrides or {})
    env = HumanoidRLEnv(config)
    randomize = make_randomization_fn(env.mj_model)
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    wrapped = wrapper.wrap_for_brax_training(
        env, episode_length=config.episode_length, action_repeat=config.action_repeat,
        randomization_fn=None if randomize is None else (lambda m: randomize(m, keys)))

    # One jitted batched step called from Python: compiles like a single step (an XLA scan
    # over steps needs far more compile memory) and the dispatch overhead is negligible
    # against a batched physics step.
    @jax.jit
    def step(state, key):
        action = jax.random.uniform(key, (num_envs, env.action_size), minval=-1, maxval=1)
        return wrapped.step(state, action)

    def rollout(state, rng):
        for key in jax.random.split(rng, num_steps):
            state = step(state, key)
        return jax.block_until_ready(state)

    state = jax.jit(wrapped.reset)(keys)
    start = time.time()
    state = rollout(state, jax.random.PRNGKey(seed + 1))
    compile_time = time.time() - start
    start = time.time()
    state = rollout(state, jax.random.PRNGKey(seed + 2))
    elapsed = time.time() - start
    return {'num_envs': num_envs, 'compile_s': compile_time,
            'steps_per_s': num_envs * num_steps / elapsed,
            'done_fraction': float(jp.mean(state.done))}


def sweep(batch_sizes: Iterable[int] = (512, 1024, 2048, 4096, 8192),
          num_steps: int = 100, **kwargs) -> list:
    """Benchmark several batch sizes; stops at the first out-of-memory error."""
    results = []
    for num_envs in batch_sizes:
        try:
            result = env_throughput(num_envs, num_steps, **kwargs)
        except Exception as error:  # XlaRuntimeError RESOURCE_EXHAUSTED and similar.
            print(f'num_envs={num_envs}: failed ({type(error).__name__}: {error})'[:300])
            break
        print(f"num_envs={num_envs:6d}: {result['steps_per_s']:12,.0f} steps/s "
              f"(compile {result['compile_s']:.0f} s)")
        results.append(result)
    return results
