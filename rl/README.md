# humanoid_rl: curriculum RL walking controller

A learning-based walking controller for the humanoid biped, independent of the model-based gait in
`humanoid_control`. A PPO policy (Brax) is trained on MuJoCo MJX through
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) on a single GPU (free
Google Colab tier), with a three-phase reward curriculum:

| Phase | Name | Goal | Command |
|---|---|---|---|
| 1 | Posture and balance | stand upright on both feet, CoM over the support polygon | 0 |
| 2 | Weight shifting and foot clearance | step in place following the gait clock | 0 |
| 3 | Low-speed locomotion | walk forward following a speed command | U(0.05, 0.2) m/s |

The package is ROS-free (`COLCON_IGNORE`) and is not built by `colcon`. Policies are evaluated in
MuJoCo and exported as NumPy `.npz`; there is no ROS deployment node yet.

**Status:** the pipeline is implemented and tested locally on CPU (environment, rewards, domain
randomization, curriculum driver, evaluation, export). No policy has been trained yet; reward
weights, thresholds and PPO settings are literature-based starting points, see the
[tuning checklist](#tuning-checklist).

## Layout

```text
rl/
  COLCON_IGNORE
  pyproject.toml              Package humanoid_rl, extras [test] and [cuda]
  scripts/export_mjcf.py      Xacro -> training MJCF (needs ROS and the built workspace)
  humanoid_rl/
    assets/humanoid_rl.xml    Generated training model (committed)
    constants.py              Joint order, standing pose, RL PD gains, model names
    config.py                 Env, reward weights per phase, thresholds, DR, PPO, curriculum
    gait_clock.py             Gait phase, swing/stance indicators, swing height reference
    rewards.py                Reward terms (pure JAX functions)
    env.py                    HumanoidRLEnv (Playground MjxEnv)
    randomize.py              Per-environment domain randomization
    curriculum.py             Chunked, resumable PPO curriculum driver
    evaluate.py               Deterministic evaluation metrics, thresholds, video rendering
    export.py                 Checkpoints, exploration-std reset, NumPy policy export
    benchmark.py              GPU throughput benchmark of the batched environment
    train.py                  CLI: humanoid-rl-train {train,eval,video,export}
  notebooks/train_humanoid_colab.ipynb
  tests/                      Unit, environment and end-to-end pipeline tests
```

## Prerequisites

- Training: an NVIDIA GPU with CUDA 12 jax (Colab T4 or better), Python >= 3.10. Colab uses
  Python 3.12 and gets the current `playground` release.
- Local development and tests: Python >= 3.10, CPU is enough (tests take a few minutes and up to
  ~3 GB RAM for the XLA compilation of MJX).
- Regenerating the MJCF: ROS 2 Humble and the built workspace (`xacro`, `humanoid_mujoco`).

## Local setup

Use a virtual environment separate from ROS; unset `PYTHONPATH` so the ROS Python packages do not
leak in:

```bash
cd rl
python3 -m venv .venv
env -u PYTHONPATH .venv/bin/pip install -e ".[test]"
env -u PYTHONPATH JAX_PLATFORMS=cpu .venv/bin/python -m pytest              # all tests
env -u PYTHONPATH JAX_PLATFORMS=cpu .venv/bin/python -m pytest -m "not slow"  # unit tests only
env -u PYTHONPATH .venv/bin/flake8 humanoid_rl tests scripts
```

If `python3 -m venv` fails because `ensurepip` is missing, create it with `--without-pip` and
install pip with `get-pip.py`.

Smoke-run the whole pipeline on CPU (tiny settings, no learning):

```bash
env -u PYTHONPATH JAX_PLATFORMS=cpu .venv/bin/humanoid-rl-train train --run_dir /tmp/smoke --smoke --max-chunks 1
```

## Training model (MJCF)

`humanoid_rl/assets/humanoid_rl.xml` is generated from the description Xacro with the same URDF
to MJCF conversion as the MuJoCo simulator (`humanoid_mujoco.model.urdf_to_mjcf`) and then
adapted for MJX: free joint `root`, 4 ms step, Newton solver with few iterations, `implicitfast`,
contacts only between the floor and the two foot boxes (bitmasks), position actuators with the RL
gains, armature, IMU and sole sites, sensors and the `home` keyframe. Regenerate it after changing
the robot description:

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 rl/scripts/export_mjcf.py
```

A test compares the committed file with a fresh export.

### RL gains

The policy outputs joint position offsets that are tracked by PD control at the 250 Hz simulation
rate. The gains in `constants.py` (kp 150-350, kd 4-10) are deliberately softer than the Gazebo
controller gains (2000-2500): stiff PD loops need a smaller time step in MJX and make policies
brittle and jerky, whereas compliant joints are the common choice for learned locomotion
(legged_gym, Playground humanoids). The robot stands stably with these gains and zero action,
including at the domain randomization extremes. A policy trained with them expects the same gains
on deployment.

## Design

### Observation (45, actor = critic)

| Signal | Size | Noise (uniform +-) |
|---|---|---|
| IMU gyro (torso frame) | 3 | 0.2 rad/s |
| Projected gravity (torso frame) | 3 | 0.05 |
| Forward speed command | 1 | - |
| Joint positions - standing pose | 12 | 0.03 rad |
| Joint velocities | 12 | 1.5 rad/s |
| Last action | 12 | - |
| Gait clock sin/cos (period 0.8 s) | 2 | - |

Base linear velocity is not observed (not measurable on hardware without an estimator), as in most
recent humanoid locomotion work. The critic uses the same observation (symmetric actor-critic).

### Action (12)

`a` in [-1, 1] (tanh-squashed Gaussian), joint target = standing pose + 0.3 * `a`, clipped to the
joint limits, at 50 Hz. A per-episode latency of 0-1 control steps is simulated.

### Gait clock

One cycle (0.8 s) contains one left and one right swing of 40 % of the cycle each, with two
double-support intervals. Phase 2 and 3 rewards follow the periodic reward composition of
Siekmann et al. (2021): no foot force during swing, no foot velocity during stance, plus a smooth
swing-height reference (6 cm apex).

### Rewards

The total reward is `sum(weight * term) * dt`, clipped at zero. Positive weights multiply bounded
kernels `exp(-e^2 / sigma)`, negative weights multiply costs. Weights per phase (`config.py`):

| Term | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| alive | 1.0 | 1.0 | 0.5 |
| tracking_lin_vel (forward speed) | | | 2.0 |
| stand_still (base xy velocity) | 1.0 | 0.5 | |
| com_over_support | 1.0 | | |
| both_feet_contact | 0.5 | | |
| periodic_force / periodic_vel | | 1.0 / 1.0 | 1.0 / 1.0 |
| feet_height (swing reference) | | 2.0 | 1.0 |
| weight_shift (CoM over stance foot) | | 1.0 | |
| orientation | -5 | -5 | -5 |
| base_height | -50 | -20 | -10 |
| pose / hip_deviation | -0.5 / -1 | - / -1 | - / -0.5 |
| heading (yaw drift) | -0.5 | -0.5 | -0.5 |
| drift (xy from start) | | -1 | |
| lateral_vel / yaw_rate | | | -2 / -0.5 |
| feet_distance (< 0.22 m) | | -10 | -10 |
| regularization (all phases) | torques, energy, action_rate, dof_acc, joint limits, feet slip, foot flatness, vertical and roll/pitch base velocity |||

### Curriculum

Each phase is a separate PPO run on that phase's environment, warm-started from the previous phase
(normalizer and policy; the policy std is reset to 0.5 and the value function is re-initialized).
Within a phase, one PPO call runs (compiled once, optimizer state kept), split into chunks of
20 M steps (`chunk_steps` = one Brax iteration). After each chunk, a checkpoint is saved and 128
deterministic 20 s episodes on the nominal model (no domain randomization, no command change
within an episode) are evaluated; the phase ends as soon as its thresholds are met. On resume,
the phase continues from the latest checkpoint in a new PPO call (fresh optimizer state):

| Phase | Advance when |
|---|---|
| 1 | survival >= 0.95, mean tilt <= 0.05 rad, torso height error <= 2 cm, double contact >= 95 % |
| 2 | survival >= 0.9, contact pattern match >= 80 %, mid-swing height >= 80 % of the reference, drift <= 0.2 m |
| 3 | survival >= 0.9, forward speed error <= 0.05 m/s, lateral speed <= 0.05 m/s |

A phase that does not pass within its budget (100 M / 200 M / 300 M steps) stops with status
`budget_exceeded` for manual inspection; continue with `extend_budget=True` (`--extend-budget`).

### Domain randomization (per environment)

Floor/foot friction 0.4-1.2, link masses x0.9-1.1, torso mass +-1 kg and CoM offset +-2 cm,
kp x0.9-1.1, kd x0.8-1.2, joint damping/friction loss x0.8-1.2, armature x0.9-1.1, plus observation
noise and action latency. No pushes and no rough terrain.

## Training on Colab

Open `notebooks/train_humanoid_colab.ipynb` in Colab with a GPU runtime. It mounts Google Drive,
clones the repository (set `REPO_URL`/`BRANCH`; a private repository needs a token URL), installs
`rl[cuda]`, runs the throughput benchmark, starts TensorBoard and trains. Everything is written to
`DRIVE_ROOT/RUN_NAME`; after a disconnect, re-run the setup cells and the training cell to resume.
Free Colab sessions are limited to a few hours, so a full curriculum will take several sessions.

CLI equivalent:

```bash
humanoid-rl-train train  --run_dir RUN [--phase N] [--extend-budget] [--no-resume]
humanoid-rl-train eval   --run_dir RUN [--checkpoint PATH]
humanoid-rl-train video  --run_dir RUN [--command 0.15] [--seconds 10]
humanoid-rl-train export --run_dir RUN [--output policy.npz]
```

## Exported policy

`export.NumpyPolicy(path)(obs)` returns the deterministic action from the `.npz` with NumPy only:
`tanh(MLP((obs - mean) / std))[:12]`, swish activations. The file also stores the joint order,
standing pose, action scale, control period and gait period needed to rebuild the observation and
joint targets.

## Tuning checklist

Run on the GPU first, before long trainings:

1. Benchmark and set `ppo_config().num_envs` (and `chunk_steps` to a chunk of roughly 10-20 min).
2. Phase 1: check in TensorBoard (`train/episode/reward/*`) that no single cost dominates; look at
   a video. Adjust `base_height`/`orientation` weights if the robot crouches or leans.
3. Phase 2: verify actual stepping in the video (not shuffling). If the policy keeps both feet on
   the ground, raise `feet_height`/`periodic_force` or lower `stand_still`.
4. Phase 3: check the speed tracking per command; adjust `tracking_sigma` and `lateral_vel`.
5. Adjust thresholds if a phase is visibly solved but does not advance (or vice versa).
6. If training is unstable: lower `learning_rate`, raise `entropy_cost`, or disable latency and
   domain randomization (`domain_randomization_config().enabled = False`) to isolate the cause.

## References

- Siekmann et al., "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward
  Composition", ICRA 2021.
- Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL" (legged_gym),
  CoRL 2021.
- Zakka et al., "MuJoCo Playground", 2025.
