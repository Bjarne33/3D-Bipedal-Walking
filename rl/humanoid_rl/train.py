"""Command line entry point: curriculum training, evaluation, video and policy export.

    humanoid-rl-train train  --run_dir runs/walk [--no-resume] [--phase N] [--extend-budget]
    humanoid-rl-train eval   --run_dir runs/walk [--phase N] [--checkpoint PATH]
    humanoid-rl-train video  --run_dir runs/walk [--phase N] [--command 0.15] [--seconds 10]
    humanoid-rl-train export --run_dir runs/walk [--checkpoint PATH] [--output policy.npz]

`--smoke` switches to tiny CPU-sized settings (pipeline checks, not real training).
"""

import argparse
import json
import os
import sys
from typing import Optional

import jax

from humanoid_rl import curriculum
from humanoid_rl import evaluate
from humanoid_rl import export


def _setup(args) -> curriculum.TrainingSetup:
    return curriculum.TrainingSetup.smoke() if args.smoke else curriculum.TrainingSetup()


def _resolve(args):
    """(phase, checkpoint) from the arguments or the run's curriculum state."""
    state = curriculum.CurriculumState.load(args.run_dir)
    checkpoint = args.checkpoint or (state.checkpoint if state else None)
    if checkpoint is None:
        sys.exit(f'No checkpoint given and none found in {args.run_dir}')
    phase = args.phase
    if phase is None:
        name = os.path.basename(checkpoint)
        phase = int(name[5]) if name.startswith('phase') else (state.phase if state else 1)
    return phase, checkpoint


def _policy(setup: curriculum.TrainingSetup, phase: int, checkpoint: str):
    env = evaluate.make_eval_env(phase, **setup.env_overrides)
    params = export.load_params(checkpoint)
    make_policy = curriculum.make_policy_fn(setup, env)
    return env, params, make_policy(params, deterministic=True)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('train', 'eval', 'video', 'export'))
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--phase', type=int, choices=(1, 2, 3))
    parser.add_argument('--checkpoint')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--extend-budget', action='store_true')
    parser.add_argument('--max-chunks', type=int)
    parser.add_argument('--command', dest='speed', type=float,
                        help='forward speed command for videos (phase 3)')
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    setup = _setup(args)

    if args.command == 'train':
        state = curriculum.run(args.run_dir, setup, resume=args.resume, start_phase=args.phase,
                               max_chunks=args.max_chunks, extend_budget=args.extend_budget)
        print(f'status={state.status} phase={state.phase} total_steps={state.total_steps}')
        return

    phase, checkpoint = _resolve(args)
    env, params, policy = _policy(setup, phase, checkpoint)
    rng = jax.random.PRNGKey(args.seed)

    if args.command == 'eval':
        cur = setup.curriculum
        metrics = evaluate.rollout_metrics(env, policy, rng, cur.eval_envs, cur.eval_steps)
        passed, checks = evaluate.check_thresholds(phase, metrics,
                                                   curriculum._thresholds(phase, setup))
        print(json.dumps({'phase': phase, 'checkpoint': checkpoint, 'passed': passed,
                          'checks': checks, 'metrics': metrics}, indent=2))
    elif args.command == 'video':
        steps = int(args.seconds / env.dt)
        # Commands are only part of the observation distribution in phase 3.
        command = args.speed if phase == 3 else None
        qpos = evaluate.rollout_qpos(env, policy, rng, steps, command=command)
        frames = evaluate.render_qpos(env.mj_model, qpos)
        output = args.output or os.path.join(args.run_dir, f'phase{phase}.mp4')
        evaluate.save_video(output, frames, fps=1.0 / env.dt)
        print(f'{len(frames)} frames -> {output}')
    elif args.command == 'export':
        output = args.output or os.path.join(args.run_dir, f'policy_phase{phase}.npz')
        export.export_npz(output, params, env._config)
        print(f'exported -> {output}')


if __name__ == '__main__':
    main()
