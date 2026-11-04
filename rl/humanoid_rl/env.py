"""MJX environment for the curriculum RL walking controller.

Observation (45, identical for actor and critic), all in the torso/IMU frame:
    gyro (3), projected gravity (3), forward speed command (1), joint positions relative to the
    standing pose (12), joint velocities (12), last policy action (12), gait clock sin/cos (2).
Action (12): joint position offsets in [-1, 1]; target = standing pose + action_scale * action,
    tracked by the MJCF position actuators (PD) at the simulation rate.

The curriculum phase is a constructor argument (`config.phase`), so reward weights are compile
time constants of each training run. Playground's auto-reset restores only `data` and `obs` on
episode end, so the episode-level `info` (command, clock phase, latency) is re-sampled in
`step` when an episode terminates.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import MjxEnv, State
import numpy as np

from humanoid_rl import config as cfg
from humanoid_rl import constants as c
from humanoid_rl import gait_clock
from humanoid_rl import rewards as rw

OBS_SIZE = 45
EVAL_METRICS = (
    'tilt', 'height_error', 'double_contact', 'contact_match', 'swing_mid_height',
    'swing_mid_count', 'forward_vel', 'command', 'vel_error', 'lateral_vel', 'drift',
)
GRAVITY = 9.81


def _quat_from_rpy(roll: jax.Array, pitch: jax.Array, yaw: jax.Array) -> jax.Array:
    """Quaternion (w, x, y, z) of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = jp.cos(roll / 2), jp.sin(roll / 2)
    cp, sp = jp.cos(pitch / 2), jp.sin(pitch / 2)
    cy, sy = jp.cos(yaw / 2), jp.sin(yaw / 2)
    return jp.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def apply_gains(model: mujoco.MjModel, kp: Optional[Dict[str, float]],
                kd: Optional[Dict[str, float]]) -> None:
    """Override position actuator gains per leg joint name (in place)."""
    for name in c.JOINT_NAMES:
        leg_joint = name.split('_', 1)[1]
        act = model.actuator(name).id
        if kp is not None and leg_joint in kp:
            model.actuator_gainprm[act, 0] = kp[leg_joint]
            model.actuator_biasprm[act, 1] = -kp[leg_joint]
        if kd is not None and leg_joint in kd:
            model.actuator_biasprm[act, 2] = -kd[leg_joint]


def disable_foot_collision(model: mujoco.MjModel) -> None:
    """Keep sole-floor contacts but drop sole-sole contacts (bitmask filtering, in place)."""
    for name in c.FOOT_GEOMS:
        model.geom(name).contype = 1
        model.geom(name).conaffinity = 2
    model.geom(c.FLOOR_GEOM).contype = 2
    model.geom(c.FLOOR_GEOM).conaffinity = 1


class HumanoidRLEnv(MjxEnv):
    """Biped standing, stepping and walking environment with phase-dependent rewards."""

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config if config is not None else cfg.env_config(1), config_overrides)
        self._phase = int(self._config.phase)
        self._mj_model = mujoco.MjModel.from_xml_path(c.XML_PATH)
        self._mj_model.opt.timestep = self.sim_dt
        apply_gains(self._mj_model, self._config.kp, self._config.kd)
        if not self._config.foot_collision:
            disable_foot_collision(self._mj_model)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = c.XML_PATH
        self._post_init()

    def _post_init(self) -> None:
        model = self._mj_model
        if model.nu != c.NUM_JOINTS:
            raise ValueError(f'Expected {c.NUM_JOINTS} actuators, model has {model.nu}')
        for index, name in enumerate(c.JOINT_NAMES):
            joint = model.joint(name).id
            if model.actuator_trnid[index, 0] != joint or model.jnt_qposadr[joint] != 7 + index:
                raise ValueError(f'Joint/actuator order differs from JOINT_NAMES at {name!r}')

        key = model.keyframe(c.HOME_KEYFRAME)
        self._init_q = jp.array(key.qpos)
        self._default_pose = jp.array(key.qpos[7:])
        self._lowers, self._uppers = (jp.array(x) for x in model.jnt_range[1:].T)
        center = (self._lowers + self._uppers) / 2
        half = (self._uppers - self._lowers) / 2
        factor = self._config.reward.soft_joint_limit_factor
        self._soft_lowers = center - half * factor
        self._soft_uppers = center + half * factor
        self._hip_ids = jp.array([c.JOINT_NAMES.index(f'{side}_{joint}')
                                  for side in c.SIDES for joint in ('hip_roll', 'hip_yaw')])

        self._base_body = model.body(c.BASE_BODY).id
        self._torso_body = model.body(c.TORSO_BODY).id
        self._imu_site = model.site(c.IMU_SITE).id
        self._foot_sites = np.array([model.site(name).id for name in c.FOOT_SITES])
        self._foot_geoms = np.array([model.geom(name).id for name in c.FOOT_GEOMS])
        self._foot_half = jp.array(model.geom_size[self._foot_geoms])

        def adr(name):
            sensor = model.sensor(name).id
            start = model.sensor_adr[sensor]
            return np.arange(start, start + model.sensor_dim[sensor])

        self._gyro_adr = adr(c.GYRO_SENSOR)
        self._global_linvel_adr = adr(c.GLOBAL_LINVEL_SENSOR)
        self._foot_linvel_adr = np.stack([adr(name) for name in c.FOOT_LINVEL_SENSORS])
        self._touch_adr = np.concatenate([adr(name) for name in c.FOOT_TOUCH_SENSORS])

        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, key.id)
        mujoco.mj_forward(model, data)
        self._home_torso_height = float(data.xpos[self._torso_body][2])
        target = self._config.reward.base_height_target
        self._height_target = self._home_torso_height if target is None else float(target)
        self._weight = float(model.body_subtreemass[self._base_body]) * GRAVITY
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        self._corner_signs = jp.array(corners, dtype=jp.float32)

        scales = self._config.reward.scales
        self._reward_scales = {term: float(scales[term]) for term in cfg.REWARD_TERMS}

    # ---------------- Helpers ----------------

    def _lowest_foot_corner(self, data: mjx.Data) -> jax.Array:
        pos = data.geom_xpos[self._foot_geoms]
        rot = data.geom_xmat[self._foot_geoms]
        local = self._corner_signs[None] * self._foot_half[:, None]
        corners = pos[:, None] + jp.einsum('fij,fcj->fci', rot, local)
        return jp.min(corners[..., 2])

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        if self._phase < 3:
            return jp.zeros(())
        low, high = self._config.command.forward_speed
        return jax.random.uniform(rng, (), minval=low, maxval=high)

    def _sample_episode_info(self, rng: jax.Array) -> Dict[str, jax.Array]:
        phase_rng, cmd_rng, delay_rng = jax.random.split(rng, 3)
        max_delay = self._config.latency.max_delay_steps
        return {
            'phase': jax.random.uniform(phase_rng, ()),
            'command': self._sample_command(cmd_rng),
            'command_steps': jp.zeros((), dtype=jp.int32),
            'delay': jax.random.randint(delay_rng, (), 0, max_delay + 1),
            'action_buffer': jp.zeros((max_delay + 1, self.action_size)),
            'last_act': jp.zeros(self.action_size),
        }

    # ---------------- MjxEnv API ----------------

    def reset(self, rng: jax.Array) -> State:
        init = self._config.init
        rng, q_rng, tilt_rng, yaw_rng, v_rng, w_rng, info_rng = jax.random.split(rng, 7)
        qpos = self._init_q
        qpos = qpos.at[7:].add(jax.random.uniform(
            q_rng, (c.NUM_JOINTS,), minval=-init.joint_noise, maxval=init.joint_noise))
        qpos = qpos.at[7:].set(jp.clip(qpos[7:], self._lowers, self._uppers))
        roll, pitch = jax.random.uniform(
            tilt_rng, (2,), minval=-init.base_tilt, maxval=init.base_tilt)
        yaw = jax.random.uniform(yaw_rng, (), minval=-jp.pi, maxval=jp.pi)
        qpos = qpos.at[3:7].set(_quat_from_rpy(roll, pitch, yaw))
        qvel = jp.zeros(self.mjx_model.nv)
        qvel = qvel.at[0:3].set(jax.random.uniform(
            v_rng, (3,), minval=-init.base_lin_vel, maxval=init.base_lin_vel))
        qvel = qvel.at[3:6].set(jax.random.uniform(
            w_rng, (3,), minval=-init.base_ang_vel, maxval=init.base_ang_vel))

        data = mjx.make_data(self.mjx_model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self._default_pose)
        data = mjx.kinematics(self.mjx_model, data)
        # Rest the lowest sole corner 1 mm above the floor after the random tilt.
        qpos = qpos.at[2].add(0.001 - self._lowest_foot_corner(data))
        data = mjx.forward(self.mjx_model, data.replace(qpos=qpos))

        start_xy = data.xpos[self._base_body, :2]
        start_yaw = rw.heading_yaw(data.site_xmat[self._imu_site])
        info = {
            'rng': rng,
            'start_xy': start_xy,
            'start_yaw': start_yaw,
            'init_xy': start_xy,
            'init_yaw': start_yaw,
            'motor_targets': self._default_pose,
            **self._sample_episode_info(info_rng),
        }
        metrics = {f'reward/{term}': jp.zeros(()) for term in cfg.REWARD_TERMS}
        metrics.update({name: jp.zeros(()) for name in EVAL_METRICS})
        obs = self._get_obs(data, info, rng)
        return State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        info = dict(state.info)
        rng, noise_rng, episode_rng, cmd_rng = jax.random.split(info['rng'], 4)

        buffer = jp.roll(info['action_buffer'], 1, axis=0).at[0].set(action)
        applied = buffer[info['delay']]
        targets = jp.clip(self._default_pose + self._config.action_scale * applied,
                          self._lowers, self._uppers)

        def substep(data, _):
            return mjx.step(self.mjx_model, data.replace(ctrl=targets)), None

        data = jax.lax.scan(substep, state.data, (), self.n_substeps)[0]
        phase = gait_clock.advance(info['phase'], self.dt, self._config.gait.period)
        info.update(action_buffer=buffer, motor_targets=targets, phase=phase)

        quantities = self._quantities(data, info)
        terms = self._reward_terms(data, info, action, quantities)
        scaled = {k: v * self._reward_scales[k] for k, v in terms.items()}
        reward = sum(scaled.values()) * self.dt
        if self._config.reward.only_positive:
            reward = jp.maximum(reward, 0.0)

        done = self._terminated(data, quantities)
        reward = jp.where(jp.isfinite(reward), reward, 0.0)

        metrics = dict(state.metrics)
        metrics.update({f'reward/{k}': v for k, v in scaled.items()})
        metrics.update(self._eval_metrics(info, quantities))

        info['last_act'] = action
        info['command_steps'] = info['command_steps'] + 1
        resample = info['command_steps'] * self.dt >= self._config.command.resample_time
        info['command'] = jp.where(resample, self._sample_command(cmd_rng), info['command'])
        info['command_steps'] = jp.where(resample, 0, info['command_steps'])

        # New episode after termination: restart the episode-level info (data is restored by
        # the auto-reset wrapper to the cached initial state).
        fresh = self._sample_episode_info(episode_rng)
        fresh.update(start_xy=info['init_xy'], start_yaw=info['init_yaw'])
        for key, value in fresh.items():
            info[key] = jp.where(done > 0.5, value, info[key])
        info['rng'] = rng

        obs = self._get_obs(data, info, noise_rng)
        return state.replace(data=data, obs=obs, reward=reward, done=done, metrics=metrics,
                             info=info)

    # ---------------- Observation ----------------

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any], rng: jax.Array) -> jax.Array:
        noise = self._config.noise
        rot = data.site_xmat[self._imu_site]
        gravity = rot.T @ jp.array([0.0, 0.0, -1.0])
        parts = [
            (data.sensordata[self._gyro_adr], noise.gyro),
            (gravity, noise.gravity),
            (jp.reshape(info['command'], (1,)), 0.0),
            (data.qpos[7:] - self._default_pose, noise.joint_pos),
            (data.qvel[6:], noise.joint_vel),
            (info['last_act'], 0.0),
            (gait_clock.clock_obs(info['phase']), 0.0),
        ]
        keys = jax.random.split(rng, len(parts))
        obs = [value + noise.level * scale * jax.random.uniform(
                   key, value.shape, minval=-1.0, maxval=1.0)
               for (value, scale), key in zip(parts, keys)]
        return jp.concatenate(obs)

    # ---------------- Rewards and termination ----------------

    def _quantities(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, jax.Array]:
        gait = self._config.gait
        rot = data.site_xmat[self._imu_site]
        yaw = rw.heading_yaw(rot)
        base_vel = data.sensordata[self._global_linvel_adr]
        base_vel_h = rw.to_heading_frame(base_vel[:2], yaw)
        feet_pos = data.site_xpos[self._foot_sites]
        feet_h = rw.to_heading_frame(feet_pos[:, :2], yaw)
        com = data.subtree_com[self._base_body]
        com_h = rw.to_heading_frame(com[:2], yaw)
        force = data.sensordata[self._touch_adr]
        return {
            'rot': rot,
            'gravity': rot.T @ jp.array([0.0, 0.0, -1.0]),
            'ang_vel': data.sensordata[self._gyro_adr],
            'yaw': yaw,
            'yaw_rate': data.qvel[5],
            'base_vel': base_vel,
            'forward_vel': base_vel_h[1],
            'lateral_vel': base_vel_h[0],
            'torso_height': data.xpos[self._torso_body, 2],
            'base_xy': data.xpos[self._base_body, :2],
            'feet_pos': feet_pos,
            'feet_lateral': feet_h[:, 0],
            'feet_vel_xy': data.sensordata[self._foot_linvel_adr][:, :2],
            'feet_z_axes': data.site_xmat[self._foot_sites][:, :, 2],
            'force': force,
            'contact': force > self._config.reward.contact_force_threshold,
            'com': com,
            'com_lateral': com_h[0],
            'swing': gait_clock.swing_indicator(info['phase'], gait.swing_ratio,
                                                gait.smoothing),
            'desired_contact': gait_clock.desired_contact(info['phase'], gait.swing_ratio),
            'ref_z': gait_clock.swing_height(info['phase'], gait.swing_ratio,
                                             gait.swing_height),
        }

    def _reward_terms(self, data: mjx.Data, info: Dict[str, Any], action: jax.Array,
                      q: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
        r = self._config.reward
        qpos = data.qpos[7:]
        torque = data.actuator_force
        contact = q['contact'].astype(jp.float32)
        stance = 1.0 - q['swing']
        terms = {
            'alive': jp.ones(()),
            'tracking_lin_vel': rw.tracking_lin_vel(q['forward_vel'], info['command'],
                                                    r.tracking_sigma),
            'lateral_vel': rw.lateral_vel(q['lateral_vel']),
            'yaw_rate': rw.yaw_rate(q['yaw_rate']),
            'heading': rw.heading(q['yaw'], info['start_yaw']),
            'lin_vel_z': rw.lin_vel_z(q['base_vel'][2]),
            'ang_vel_xy': rw.ang_vel_xy(q['ang_vel']),
            'orientation': rw.orientation(q['gravity']),
            'base_height': rw.base_height(q['torso_height'], self._height_target),
            'stand_still': rw.stand_still(q['base_vel'][:2], r.stand_still_sigma),
            'pose': rw.pose(qpos, self._default_pose),
            'hip_deviation': rw.hip_deviation(qpos, self._default_pose, self._hip_ids),
            'both_feet_contact': rw.both_feet_contact(q['contact']),
            'com_over_support': rw.com_over_support(q['com'][:2], q['feet_pos'][:, :2],
                                                    r.com_sigma),
            'periodic_force': rw.periodic_force(q['swing'], q['force'],
                                                r.periodic_force_sigma * self._weight),
            'periodic_vel': rw.periodic_vel(q['swing'], q['feet_vel_xy'], r.periodic_vel_sigma),
            'feet_height': rw.feet_height(q['feet_pos'][:, 2], q['ref_z'], r.feet_height_sigma),
            'weight_shift': rw.weight_shift(q['com_lateral'], q['feet_lateral'], stance,
                                            r.weight_shift_sigma),
            'drift': rw.drift(q['base_xy'], info['start_xy']),
            'feet_distance': rw.feet_distance(q['feet_lateral'], r.min_feet_distance),
            'feet_slip': rw.feet_slip(contact, q['feet_vel_xy']),
            'foot_flat': rw.foot_flat(q['feet_z_axes']),
            'torques': rw.torques(torque),
            'energy': rw.energy(torque, data.qvel[6:]),
            'action_rate': rw.action_rate(action, info['last_act']),
            'dof_acc': rw.dof_acc(data.qacc[6:]),
            'dof_pos_limits': rw.dof_pos_limits(qpos, self._soft_lowers, self._soft_uppers),
            'termination': self._terminated(data, q),
        }
        return {k: jp.where(jp.isfinite(v), v, 0.0) for k, v in terms.items()}

    def _terminated(self, data: mjx.Data, q: Dict[str, jax.Array]) -> jax.Array:
        t = self._config.termination
        fallen = q['torso_height'] < t.min_torso_height
        tilted = q['gravity'][2] > -jp.cos(t.max_tilt)
        invalid = ~jp.all(jp.isfinite(data.qpos)) | ~jp.all(jp.isfinite(data.qvel))
        return (fallen | tilted | invalid).astype(jp.float32)

    def _eval_metrics(self, info: Dict[str, Any],
                      q: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
        gait = self._config.gait
        local = gait_clock.foot_phases(info['phase'])
        mid_swing = jp.abs(local - 0.5 * gait.swing_ratio) < 0.1 * gait.swing_ratio
        tilt = jp.arccos(jp.clip(-q['gravity'][2], -1.0, 1.0))
        return {
            'tilt': tilt,
            'height_error': jp.abs(q['torso_height'] - self._height_target),
            'double_contact': jp.all(q['contact']).astype(jp.float32),
            'contact_match': jp.mean((q['contact'] == q['desired_contact']).astype(jp.float32)),
            'swing_mid_height': jp.sum(mid_swing * q['feet_pos'][:, 2]) / gait.swing_height,
            'swing_mid_count': jp.sum(mid_swing).astype(jp.float32),
            'forward_vel': q['forward_vel'],
            'command': info['command'],
            'vel_error': jp.abs(q['forward_vel'] - info['command']),
            'lateral_vel': jp.abs(q['lateral_vel']),
            'drift': jp.linalg.norm(q['base_xy'] - info['start_xy']),
        }

    # ---------------- Accessors ----------------

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def home_torso_height(self) -> float:
        return self._home_torso_height

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return c.NUM_JOINTS

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
