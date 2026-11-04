"""Chunked, resumable curriculum training with brax PPO.

Each curriculum phase is a separate PPO run on an environment with that phase's rewards. A phase
run is one `ppo.train` call (compiled once) split into iterations ("chunks") of `chunk_steps`;
after every chunk the params are saved, the deterministic policy is evaluated and the phase
thresholds are checked:

    passed and phase < 3   -> next phase, warm-started from the current params with the policy
                              std reset to `reset_std` and a fresh value function
    passed and phase == 3  -> finished
    phase budget exceeded  -> stop (status `budget_exceeded`) for manual inspection
    otherwise              -> next chunk of the same phase run

All progress is stored in `<run_dir>/curriculum_state.json`, so a run interrupted at any point
(e.g. a Colab disconnect) resumes from its last completed chunk (the PPO optimizer state then
restarts).
"""

import dataclasses
import functools
import json
import os
import time
from typing import Any, Callable, Dict, Optional

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax
from ml_collections import config_dict
from mujoco_playground import wrapper

from humanoid_rl import config as cfg
from humanoid_rl import evaluate
from humanoid_rl import export
from humanoid_rl.env import HumanoidRLEnv
from humanoid_rl.randomize import make_randomization_fn

STATE_FILE = 'curriculum_state.json'


@dataclasses.dataclass
class CurriculumState:
    phase: int = 1
    phase_steps: int = 0
    # Steps of the current phase counted against `max_phase_steps` (reset by `extend_budget`).
    budget_steps: int = 0
    total_steps: int = 0
    chunk: int = 0
    checkpoint: Optional[str] = None
    # 'training', 'finished' or 'budget_exceeded'.
    status: str = 'training'
    history: list = dataclasses.field(default_factory=list)

    @classmethod
    def load(cls, run_dir: str) -> Optional['CurriculumState']:
        path = os.path.join(run_dir, STATE_FILE)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, run_dir: str) -> None:
        path = os.path.join(run_dir, STATE_FILE)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(dataclasses.asdict(self), f, indent=2)
        os.replace(tmp, path)


@dataclasses.dataclass
class TrainingSetup:
    """Configs of one curriculum run; `smoke()` gives a tiny CPU-sized variant for tests."""

    ppo: config_dict.ConfigDict = dataclasses.field(default_factory=cfg.ppo_config)
    curriculum: config_dict.ConfigDict = dataclasses.field(
        default_factory=cfg.curriculum_config)
    randomization: config_dict.ConfigDict = dataclasses.field(
        default_factory=cfg.domain_randomization_config)
    env_overrides: Dict[str, Any] = dataclasses.field(default_factory=dict)
    thresholds: Optional[Dict[int, Dict[str, float]]] = None

    @classmethod
    def smoke(cls) -> 'TrainingSetup':
        setup = cls()
        setup.ppo.update(num_envs=8, unroll_length=4, batch_size=8,
                         num_minibatches=2, num_updates_per_batch=1,
                         training_metrics_steps=64)
        setup.ppo.network.update(policy_hidden_layer_sizes=(32, 32),
                                 value_hidden_layer_sizes=(32, 32))
        setup.curriculum.update(chunk_steps=64, max_phase_steps=(192, 192, 192),
                                eval_envs=4, eval_steps=60)
        setup.env_overrides = {'episode_length': 50, 'foot_collision': False}
        return setup


class Logger:
    """TensorBoard + stdout logging with a global env-step axis."""

    def __init__(self, run_dir: str, echo: Callable[[str], None] = print) -> None:
        from tensorboardX import SummaryWriter
        self._writer = SummaryWriter(os.path.join(run_dir, 'tensorboard'))
        self._echo = echo

    def scalars(self, prefix: str, values: Dict[str, Any], step: int) -> None:
        for key, value in values.items():
            try:
                self._writer.add_scalar(f'{prefix}/{key}', float(value), step)
            except (TypeError, ValueError):
                continue
        self._writer.flush()

    def echo(self, message: str) -> None:
        self._echo(message)

    def close(self) -> None:
        self._writer.close()


def _env_config(phase: int, setup: TrainingSetup) -> config_dict.ConfigDict:
    config = cfg.env_config(phase)
    config.update(setup.env_overrides)
    return config


def _thresholds(phase: int, setup: TrainingSetup) -> Dict[str, float]:
    if setup.thresholds is not None and phase in setup.thresholds:
        return setup.thresholds[phase]
    return cfg.PHASE_THRESHOLDS[phase]


def _network_factory(setup: TrainingSetup):
    net = setup.ppo.network
    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(net.policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(net.value_hidden_layer_sizes),
    )


def train_phase(env: HumanoidRLEnv, setup: TrainingSetup, num_steps: int, num_iterations: int,
                seed: int, restore_params: Optional[tuple], restore_value_fn: bool,
                progress_fn: Callable[[int, dict], None],
                policy_params_fn: Callable[[int, Any, tuple], None]):
    """One brax PPO run of `num_iterations` iterations (`policy_params_fn` after each)."""
    p = setup.ppo
    randomize = make_randomization_fn(env.mj_model, setup.randomization)
    return ppo.train(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomize,
        num_timesteps=num_steps,
        episode_length=env._config.episode_length,
        action_repeat=env._config.action_repeat,
        num_envs=p.num_envs,
        unroll_length=p.unroll_length,
        batch_size=p.batch_size,
        num_minibatches=p.num_minibatches,
        num_updates_per_batch=p.num_updates_per_batch,
        discounting=p.discounting,
        learning_rate=p.learning_rate,
        entropy_cost=p.entropy_cost,
        clipping_epsilon=p.clipping_epsilon,
        max_grad_norm=p.max_grad_norm,
        reward_scaling=p.reward_scaling,
        normalize_observations=p.normalize_observations,
        # num_evals - 1 training iterations; brax's own evaluation is replaced by ours.
        num_evals=num_iterations + 1,
        run_evals=False,
        log_training_metrics=True,
        training_metrics_steps=p.training_metrics_steps,
        network_factory=_network_factory(setup),
        seed=seed,
        restore_params=restore_params,
        restore_value_fn=restore_value_fn,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )


class _StopPhase(Exception):
    """Raised from the PPO callback to end the current phase run early."""


def make_policy_fn(setup: TrainingSetup, env: HumanoidRLEnv):
    """Build brax's inference function factory for given params (no training)."""
    network = _network_factory(setup)(
        env.observation_size, env.action_size,
        preprocess_observations_fn=_normalizer(setup))
    return ppo_networks.make_inference_fn(network)


def _normalizer(setup: TrainingSetup):
    from brax.training.acme import running_statistics
    if setup.ppo.normalize_observations:
        return running_statistics.normalize
    return lambda x, y: x


def run(run_dir: str, setup: Optional[TrainingSetup] = None, resume: bool = True,
        start_phase: Optional[int] = None, max_chunks: Optional[int] = None,
        extend_budget: bool = False,
        echo: Callable[[str], None] = print) -> CurriculumState:
    """Train the curriculum in `run_dir` until finished, over budget or `max_chunks` chunks.

    `start_phase` jumps a resumed run to another phase (warm start from its last checkpoint);
    `extend_budget` gives the current phase a fresh step budget (e.g. after reward tuning).
    """
    setup = setup if setup is not None else TrainingSetup()
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    state = CurriculumState.load(run_dir) if resume else None
    if state is None:
        state = CurriculumState(phase=start_phase or 1)
    elif start_phase is not None and start_phase != state.phase:
        echo(f'Switching resumed run from phase {state.phase} to phase {start_phase}')
        state.phase, state.phase_steps, state.budget_steps = start_phase, 0, 0
        state.status = 'training'
    if extend_budget and state.status == 'budget_exceeded':
        state.budget_steps, state.status = 0, 'training'
    state.save(run_dir)

    logger = Logger(run_dir, echo)
    cur = setup.curriculum
    chunks = 0
    try:
        while state.status == 'training' and (max_chunks is None or chunks < max_chunks):
            phase = state.phase
            env = HumanoidRLEnv(_env_config(phase, setup))
            eval_cfg = _env_config(phase, setup)
            eval_cfg.command.resample_time = 1e9
            eval_env = HumanoidRLEnv(eval_cfg)

            restore = None
            new_phase = state.phase_steps == 0
            if state.checkpoint is not None:
                restore = export.load_params(state.checkpoint)
                if new_phase:
                    restore = export.reset_policy_std(restore, cur.reset_std)
            remaining = cur.max_phase_steps[phase - 1] - state.budget_steps
            iterations = max(-(-remaining // cur.chunk_steps), 1)
            if max_chunks is not None:
                iterations = min(iterations, max_chunks - chunks)
            offset = state.total_steps
            tracker = {'step': 0, 'time': time.time(), 'evaluator': None, 'first': True}

            def progress(step, metrics, offset=offset, phase=phase):
                logger.scalars('train', metrics, offset + int(step))
                logger.scalars('curriculum', {'phase': phase}, offset + int(step))

            def after_iteration(current_step, make_policy, params, phase=phase,
                                eval_env=eval_env, tracker=tracker):
                nonlocal chunks
                steps = int(current_step) - tracker['step']
                if steps <= 0:  # Initial call before any training.
                    return
                elapsed = time.time() - tracker['time']
                tracker['step'] = int(current_step)
                state.phase_steps += steps
                state.budget_steps += steps
                state.total_steps += steps

                ckpt = os.path.join(run_dir, 'checkpoints',
                                    f'phase{phase}_{state.total_steps:012d}.pkl')
                export.save_params(ckpt, params)
                state.checkpoint = ckpt

                if tracker['evaluator'] is None:
                    tracker['evaluator'] = evaluate.make_evaluator(
                        eval_env, lambda p: make_policy(p, deterministic=True),
                        cur.eval_envs, cur.eval_steps)
                metrics = tracker['evaluator'](
                    params, jax.random.PRNGKey(cur.seed + 1000 + state.chunk))
                passed, checks = evaluate.check_thresholds(phase, metrics,
                                                           _thresholds(phase, setup))
                rate = steps / max(elapsed, 1e-6)
                logger.scalars('eval', metrics, state.total_steps)
                logger.scalars('perf', {'steps_per_second': rate}, state.total_steps)
                note = ' (incl. compile)' if tracker['first'] else ''
                tracker['first'] = False
                echo(f'[phase {phase}] chunk {state.chunk}: {state.phase_steps:,} phase steps, '
                     f'{rate:,.0f} steps/s{note}; eval '
                     + ', '.join(f'{k}={metrics[k]:.3f}' for k in sorted(metrics)
                                 if not k.startswith('reward/'))
                     + f'; thresholds {checks}')
                state.history.append(dict(chunk=state.chunk, phase=phase,
                                          total_steps=state.total_steps, checkpoint=ckpt,
                                          passed=passed, metrics=metrics))
                state.chunk += 1
                chunks += 1

                stop = True
                if passed and phase == cfg.PHASES[-1]:
                    state.status = 'finished'
                    echo('Curriculum finished.')
                elif passed:
                    state.phase, state.phase_steps, state.budget_steps = phase + 1, 0, 0
                    echo(f'Phase {phase} passed -> phase {phase + 1}')
                elif state.budget_steps >= cur.max_phase_steps[phase - 1]:
                    state.status = 'budget_exceeded'
                    echo(f'Phase {phase} exceeded its budget of '
                         f'{cur.max_phase_steps[phase - 1]} steps without passing; inspect the '
                         'run and adjust before resuming (extend_budget=True).')
                else:
                    stop = max_chunks is not None and chunks >= max_chunks
                state.save(run_dir)
                tracker['time'] = time.time()
                if stop:
                    raise _StopPhase()

            echo(f'[phase {phase}] training up to {iterations} chunk(s) of {cur.chunk_steps:,} '
                 f'steps (phase steps {state.phase_steps:,}, total {state.total_steps:,})')
            try:
                train_phase(env, setup, iterations * cur.chunk_steps, iterations,
                            cur.seed + state.chunk, restore, restore_value_fn=not new_phase,
                            progress_fn=progress, policy_params_fn=after_iteration)
            except _StopPhase:
                pass
    finally:
        logger.close()
    return state
