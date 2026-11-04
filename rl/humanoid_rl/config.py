"""Configuration of the RL environment, curriculum phases and PPO.

All values are literature-based starting points (legged_gym, MuJoCo Playground humanoid
joystick tasks, Siekmann et al. 2021). They have not been tuned by training yet; see the
README tuning checklist.
"""

from ml_collections import config_dict

PHASES = (1, 2, 3)
PHASE_NAMES = {
    1: 'posture_balance',
    2: 'weight_shift_clearance',
    3: 'low_speed_locomotion',
}

# Every reward term computed by the environment. Weights of 0 disable a term in a phase.
# Positive terms are bounded kernels exp(-e^2 / sigma) in [0, 1]; negative weights scale costs.
REWARD_TERMS = (
    'alive',
    'tracking_lin_vel',
    'lateral_vel',
    'yaw_rate',
    'heading',
    'lin_vel_z',
    'ang_vel_xy',
    'orientation',
    'base_height',
    'stand_still',
    'pose',
    'hip_deviation',
    'both_feet_contact',
    'com_over_support',
    'periodic_force',
    'periodic_vel',
    'feet_height',
    'weight_shift',
    'drift',
    'feet_distance',
    'feet_slip',
    'foot_flat',
    'torques',
    'energy',
    'action_rate',
    'dof_acc',
    'dof_pos_limits',
    'termination',
)

_REGULARIZATION = dict(
    torques=-2.0e-5,
    energy=-1.0e-4,
    action_rate=-0.01,
    dof_acc=-2.5e-7,
    dof_pos_limits=-1.0,
    feet_slip=-0.25,
    foot_flat=-2.0,
    lin_vel_z=-2.0,
    ang_vel_xy=-0.15,
)

PHASE_REWARD_SCALES = {
    # Posture and balance: stand still on both feet, CoM centred over the support polygon.
    1: dict(
        _REGULARIZATION,
        alive=1.0,
        orientation=-5.0,
        base_height=-50.0,
        stand_still=1.0,
        pose=-0.5,
        hip_deviation=-1.0,
        both_feet_contact=0.5,
        com_over_support=1.0,
        heading=-0.5,
    ),
    # Weight shifting and foot clearance: step in place following the gait clock.
    2: dict(
        _REGULARIZATION,
        alive=1.0,
        orientation=-5.0,
        base_height=-20.0,
        stand_still=0.5,
        hip_deviation=-1.0,
        periodic_force=1.0,
        periodic_vel=1.0,
        feet_height=2.0,
        weight_shift=1.0,
        drift=-1.0,
        heading=-0.5,
        feet_distance=-10.0,
    ),
    # Low-speed locomotion: track a positive forward speed with the periodic gait.
    3: dict(
        _REGULARIZATION,
        alive=0.5,
        tracking_lin_vel=2.0,
        lateral_vel=-2.0,
        yaw_rate=-0.5,
        heading=-0.5,
        orientation=-5.0,
        base_height=-10.0,
        hip_deviation=-0.5,
        periodic_force=1.0,
        periodic_vel=1.0,
        feet_height=1.0,
        feet_distance=-10.0,
    ),
}

# Thresholds for advancing the curriculum, evaluated on deterministic policy rollouts
# (see humanoid_rl.evaluate.compute_metrics). `min_*` must be reached, `max_*` not exceeded.
PHASE_THRESHOLDS = {
    1: dict(min_survival=0.95, max_tilt=0.05, max_height_error=0.02, min_double_contact=0.95),
    2: dict(min_survival=0.90, min_contact_match=0.80, min_swing_height_ratio=0.8,
            max_drift=0.2),
    3: dict(min_survival=0.90, max_vel_error=0.05, max_lateral_vel=0.05),
}


def _reward_scales(phase: int) -> config_dict.ConfigDict:
    scales = {term: 0.0 for term in REWARD_TERMS}
    unknown = set(PHASE_REWARD_SCALES[phase]) - set(scales)
    if unknown:
        raise ValueError(f'Unknown reward terms for phase {phase}: {sorted(unknown)}')
    scales.update(PHASE_REWARD_SCALES[phase])
    return config_dict.create(**scales)


def env_config(phase: int = 1) -> config_dict.ConfigDict:
    """Environment configuration for a curriculum phase."""
    if phase not in PHASES:
        raise ValueError(f'phase must be one of {PHASES}, got {phase}')
    return config_dict.create(
        phase=phase,
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        action_repeat=1,
        # Joint target = standing pose + action_scale * action, action in [-1, 1].
        action_scale=0.3,
        impl='jax',
        # Sole-sole (box-box) collision. Only disabled by the CPU smoke tests: box-box contact
        # roughly doubles the XLA compile memory of the environment step.
        foot_collision=True,
        # Gains applied to the actuators at load time; None keeps the MJCF values
        # (humanoid_rl.constants.KP/KD). Dicts map leg joint names to gains.
        kp=None,
        kd=None,
        gait=config_dict.create(
            # Duration of one full gait cycle (one left and one right step) [s].
            period=0.8,
            # Fraction of the cycle each foot is in swing (stance = 1 - swing_ratio).
            swing_ratio=0.4,
            # Width of the smooth stance/swing transition (fraction of the cycle).
            smoothing=0.02,
            # Peak height of the swing-foot reference above the floor [m].
            swing_height=0.06,
        ),
        command=config_dict.create(
            # Forward speed range [m/s] (phase 3 only; phases 1-2 always command 0).
            forward_speed=(0.05, 0.2),
            # A new command is sampled after this time [s] and on every episode start.
            resample_time=5.0,
        ),
        init=config_dict.create(
            joint_noise=0.05,
            base_tilt=0.05,
            base_lin_vel=0.1,
            base_ang_vel=0.1,
        ),
        noise=config_dict.create(
            level=1.0,
            joint_pos=0.03,
            joint_vel=1.5,
            gravity=0.05,
            gyro=0.2,
        ),
        latency=config_dict.create(
            # Policy actions are delayed by 0..max_delay_steps control steps (per episode).
            max_delay_steps=1,
        ),
        termination=config_dict.create(
            min_torso_height=0.75,
            max_tilt=0.8,
        ),
        reward=config_dict.create(
            scales=_reward_scales(phase),
            # Clip the summed reward at zero (legged_gym "only positive rewards").
            only_positive=True,
            tracking_sigma=0.01,
            stand_still_sigma=0.01 if phase == 1 else 0.04,
            com_sigma=0.005,
            weight_shift_sigma=0.002,
            feet_height_sigma=0.0025,
            # Swing-phase foot force kernel width, as a fraction of the robot weight.
            periodic_force_sigma=0.1,
            periodic_vel_sigma=0.01,
            base_height_target=None,  # None: torso height of the home keyframe.
            min_feet_distance=0.22,
            soft_joint_limit_factor=0.95,
            contact_force_threshold=10.0,
        ),
    )


def domain_randomization_config() -> config_dict.ConfigDict:
    """Ranges of the per-environment model randomization (see humanoid_rl.randomize)."""
    return config_dict.create(
        enabled=True,
        friction=(0.4, 1.2),
        body_mass_scale=(0.9, 1.1),
        torso_mass_add=(-1.0, 1.0),
        torso_com_offset=0.02,
        kp_scale=(0.9, 1.1),
        kd_scale=(0.8, 1.2),
        damping_scale=(0.8, 1.2),
        frictionloss_scale=(0.8, 1.2),
        armature_scale=(0.9, 1.1),
    )


def ppo_config() -> config_dict.ConfigDict:
    """Brax PPO hyperparameters (sized for a single Colab T4; adjust after benchmarking)."""
    return config_dict.create(
        num_envs=2048,
        unroll_length=20,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0.005,
        clipping_epsilon=0.2,
        max_grad_norm=1.0,
        reward_scaling=1.0,
        normalize_observations=True,
        # Training episode metrics are sent to TensorBoard every this many env steps.
        training_metrics_steps=1_000_000,
        network=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
        ),
    )


def curriculum_config() -> config_dict.ConfigDict:
    """Chunked training driver settings."""
    return config_dict.create(
        # Environment steps per training chunk; a checkpoint and evaluation follow each chunk.
        chunk_steps=20_000_000,
        # Safety budget per phase (phases 1, 2, 3); training stops without advancing when a
        # phase exceeds it.
        max_phase_steps=(100_000_000, 200_000_000, 300_000_000),
        # Policy std at the start of each phase (exploration reset for warm starts).
        reset_std=0.5,
        eval_envs=128,
        eval_steps=1000,
        seed=0,
    )
