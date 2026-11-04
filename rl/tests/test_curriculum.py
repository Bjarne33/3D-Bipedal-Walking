"""End-to-end curriculum pipeline with tiny CPU settings (no learning is expected)."""

import json
import os
import shutil

import jax
import numpy as np
import pytest

from humanoid_rl import curriculum
from humanoid_rl import evaluate
from humanoid_rl import export
from humanoid_rl import train

pytestmark = pytest.mark.slow

ALWAYS = {phase: {'min_survival': 0.0} for phase in (1, 2, 3)}
NEVER = {phase: {'min_survival': 2.0} for phase in (1, 2, 3)}


def _setup(thresholds):
    setup = curriculum.TrainingSetup.smoke()
    setup.thresholds = thresholds
    return setup


def _policy_std(params):
    last = params[1]['params']
    last = last[max(last, key=lambda k: int(k[7:]))]
    return np.asarray(last['bias'])[12:], np.asarray(last['kernel'])[:, 12:]


def test_curriculum_advances_resumes_and_exports(tmp_path):
    run_dir = str(tmp_path / 'run')

    # Phase 1 passes after one chunk -> phase 2 (interrupted after a single chunk).
    state = curriculum.run(run_dir, _setup(ALWAYS), max_chunks=1, echo=lambda _: None)
    assert (state.phase, state.status, state.chunk, state.phase_steps) == (2, 'training', 1, 0)
    assert os.path.exists(state.checkpoint) and 'phase1_' in state.checkpoint
    phase1_ckpt = state.checkpoint

    # Resume from the state file: phase 2 is warm-started from the phase 1 checkpoint with the
    # exploration std reset, then phase 3 passes and the curriculum finishes.
    state = curriculum.run(run_dir, _setup(ALWAYS), echo=lambda _: None)
    assert state.status == 'finished' and state.phase == 3
    assert [h['phase'] for h in state.history] == [1, 2, 3]
    assert state.total_steps == sum(64 for _ in state.history)
    with open(os.path.join(run_dir, curriculum.STATE_FILE)) as f:
        assert json.load(f)['status'] == 'finished'
    assert os.listdir(os.path.join(run_dir, 'tensorboard'))

    # The phase 2 run started from phase 1's params: its first checkpoint keeps the hidden
    # layers close to phase 1 (only a few updates), not a fresh initialization.
    p1 = export.load_params(phase1_ckpt)
    p2 = export.load_params(state.history[1]['checkpoint'])
    w1, w2 = (np.asarray(p[1]['params']['hidden_0']['kernel']) for p in (p1, p2))
    assert np.abs(w2 - w1).max() < 0.1 * np.abs(w1).max()

    # CLI: evaluate and export the final checkpoint, then reload the exported policy.
    train.main(['eval', '--run_dir', run_dir, '--smoke'])
    output = str(tmp_path / 'policy.npz')
    train.main(['export', '--run_dir', run_dir, '--smoke', '--output', output])
    policy = export.NumpyPolicy(output)
    assert int(policy.meta['phase']) == 3
    assert policy(np.zeros(45)).shape == (12,)


def test_budget_stop_and_extension(tmp_path, monkeypatch):
    run_dir = str(tmp_path / 'run')
    setup = _setup(NEVER)
    setup.curriculum.max_phase_steps = (128, 128, 128)
    calls = []
    original = curriculum.train_phase
    monkeypatch.setattr(curriculum, 'train_phase',
                        lambda *a, **k: calls.append(a[3]) or original(*a, **k))
    state = curriculum.run(run_dir, setup, echo=lambda _: None)
    assert (state.status, state.phase, state.chunk) == ('budget_exceeded', 1, 2)
    # Both chunks of the phase ran in one PPO call (compiled once).
    assert calls == [2]
    # Resuming without extension does nothing; extending the budget trains further.
    state = curriculum.run(run_dir, setup, echo=lambda _: None)
    assert state.chunk == 2
    state = curriculum.run(run_dir, setup, max_chunks=1, extend_budget=True,
                           echo=lambda _: None)
    assert (state.status, state.chunk, state.phase_steps) == ('training', 3, 192)


def test_reset_std_applied_on_phase_start(tmp_path, monkeypatch):
    run_dir = str(tmp_path / 'run')
    curriculum.run(run_dir, _setup(ALWAYS), max_chunks=1, echo=lambda _: None)
    restored = {}

    def fake_train_phase(env, setup, num_steps, num_iterations, seed, restore_params,
                         restore_value_fn, progress_fn, policy_params_fn):
        restored.update(params=restore_params, value=restore_value_fn)
        raise KeyboardInterrupt

    monkeypatch.setattr(curriculum, 'train_phase', fake_train_phase)
    with pytest.raises(KeyboardInterrupt):
        curriculum.run(run_dir, _setup(ALWAYS), echo=lambda _: None)
    bias, kernel = _policy_std(restored['params'])
    std = jax.nn.softplus(bias) + export.MIN_STD
    np.testing.assert_allclose(std, 0.5, rtol=1e-5)
    np.testing.assert_array_equal(kernel, 0.0)
    assert restored['value'] is False


def test_video(tmp_path):
    run_dir = str(tmp_path / 'run')
    curriculum.run(run_dir, _setup(ALWAYS), max_chunks=1, echo=lambda _: None)
    try:
        import mujoco
        mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>'), 16, 16).close()
    except Exception as error:  # No OpenGL context (e.g. headless without EGL/OSMesa).
        pytest.skip(f'rendering unavailable: {error}')
    env, _, policy = train._policy(curriculum.TrainingSetup.smoke(), 2,
                                   curriculum.CurriculumState.load(run_dir).checkpoint)
    qpos = evaluate.rollout_qpos(env, policy, jax.random.PRNGKey(0), 10)
    frames = evaluate.render_qpos(env.mj_model, qpos, height=48, width=64)
    assert len(frames) == len(qpos) and frames[0].shape == (48, 64, 3)
    assert np.ptp(frames[0]) > 0
    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg not installed: video file writing not tested')
    output = str(tmp_path / 'video.mp4')
    train.main(['video', '--run_dir', run_dir, '--smoke', '--seconds', '0.5', '--output', output])
    assert os.path.getsize(output) > 0
