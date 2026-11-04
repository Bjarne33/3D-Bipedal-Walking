"""Reward terms as pure functions of arrays.

Kernel terms return values in [0, 1] (higher is better) and are used with positive weights.
Cost terms return non-negative values and are used with negative weights. Geometry follows the
robot convention: forward is +Y, lateral is X. "Heading frame" quantities are expressed in a
frame rotated by the base yaw around the world Z axis.
"""

import jax
import jax.numpy as jp


def kernel(error_sq: jax.Array, sigma: float) -> jax.Array:
    """Gaussian kernel exp(-error^2 / sigma)."""
    return jp.exp(-error_sq / sigma)


def heading_yaw(rot: jax.Array) -> jax.Array:
    """Yaw of the robot heading (+Y axis of the body) relative to world +Y."""
    forward = rot[:, 1]
    return jp.arctan2(-forward[0], forward[1])


def to_heading_frame(vec_xy: jax.Array, yaw: jax.Array) -> jax.Array:
    """Rotate a world XY vector into the heading frame (x = lateral, y = forward)."""
    c, s = jp.cos(yaw), jp.sin(yaw)
    return jp.stack([c * vec_xy[..., 0] + s * vec_xy[..., 1],
                     -s * vec_xy[..., 0] + c * vec_xy[..., 1]], axis=-1)


def wrap_angle(angle: jax.Array) -> jax.Array:
    return jp.arctan2(jp.sin(angle), jp.cos(angle))


# ---------------- Task terms ----------------

def tracking_lin_vel(forward_vel: jax.Array, command: jax.Array, sigma: float) -> jax.Array:
    return kernel(jp.square(forward_vel - command), sigma)


def lateral_vel(lateral_vel_: jax.Array) -> jax.Array:
    return jp.square(lateral_vel_)


def yaw_rate(yaw_rate_: jax.Array) -> jax.Array:
    return jp.square(yaw_rate_)


def heading(yaw: jax.Array, start_yaw: jax.Array) -> jax.Array:
    return jp.square(wrap_angle(yaw - start_yaw))


def stand_still(base_vel_xy: jax.Array, sigma: float) -> jax.Array:
    return kernel(jp.sum(jp.square(base_vel_xy)), sigma)


def drift(base_xy: jax.Array, start_xy: jax.Array) -> jax.Array:
    return jp.sum(jp.square(base_xy - start_xy))


# ---------------- Base terms ----------------

def lin_vel_z(vel_z: jax.Array) -> jax.Array:
    return jp.square(vel_z)


def ang_vel_xy(ang_vel_local: jax.Array) -> jax.Array:
    return jp.sum(jp.square(ang_vel_local[:2]))


def orientation(gravity_local: jax.Array) -> jax.Array:
    """Squared horizontal components of the projected gravity (0 when upright)."""
    return jp.sum(jp.square(gravity_local[:2]))


def base_height(height: jax.Array, target: float) -> jax.Array:
    return jp.square(height - target)


# ---------------- Posture terms ----------------

def pose(qpos: jax.Array, default: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - default))


def hip_deviation(qpos: jax.Array, default: jax.Array, hip_ids: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos[hip_ids] - default[hip_ids]))


def both_feet_contact(contact: jax.Array) -> jax.Array:
    return jp.all(contact).astype(jp.float32)


def com_over_support(com_xy: jax.Array, feet_xy: jax.Array, sigma: float) -> jax.Array:
    """Horizontal CoM close to the midpoint of the two soles."""
    return kernel(jp.sum(jp.square(com_xy - jp.mean(feet_xy, axis=0))), sigma)


def weight_shift(com_lateral: jax.Array, feet_lateral: jax.Array, stance: jax.Array,
                 sigma: float) -> jax.Array:
    """Lateral CoM over the stance-weighted foot position.

    In single support the target is the stance foot; in double support it is the midpoint.
    """
    weights = stance / jp.maximum(jp.sum(stance), 1e-6)
    target = jp.sum(weights * feet_lateral)
    return kernel(jp.square(com_lateral - target), sigma)


# ---------------- Periodic gait terms (Siekmann et al. 2021) ----------------

def periodic_force(swing: jax.Array, foot_force: jax.Array, sigma: float) -> jax.Array:
    """Reward small sole contact force while a foot should swing (mean over feet)."""
    return jp.mean(swing * kernel(jp.square(foot_force), sigma ** 2))


def periodic_vel(swing: jax.Array, foot_vel_xy: jax.Array, sigma: float) -> jax.Array:
    """Reward a stationary sole while a foot should be in stance (mean over feet)."""
    speed_sq = jp.sum(jp.square(foot_vel_xy), axis=-1)
    return jp.mean((1.0 - swing) * kernel(speed_sq, sigma))


def feet_height(foot_z: jax.Array, ref_z: jax.Array, sigma: float) -> jax.Array:
    """Track the swing height reference with both soles."""
    return kernel(jp.sum(jp.square(foot_z - ref_z)), sigma)


def feet_distance(feet_lateral: jax.Array, min_distance: float) -> jax.Array:
    """Penalize lateral sole separation below the minimum (heading frame)."""
    separation = feet_lateral[0] - feet_lateral[1]
    return jp.clip(min_distance - separation, 0.0, 0.1)


def feet_slip(contact: jax.Array, foot_vel_xy: jax.Array) -> jax.Array:
    return jp.sum(contact * jp.sum(jp.square(foot_vel_xy), axis=-1))


def foot_flat(foot_z_axes: jax.Array) -> jax.Array:
    """Squared horizontal components of each sole's normal (0 when flat)."""
    return jp.sum(jp.square(foot_z_axes[:, :2]))


# ---------------- Regularization ----------------

def torques(torque: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torque))


def energy(torque: jax.Array, qvel: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torque * qvel))


def action_rate(action: jax.Array, last_action: jax.Array) -> jax.Array:
    return jp.sum(jp.square(action - last_action))


def dof_acc(qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))


def dof_pos_limits(qpos: jax.Array, lower: jax.Array, upper: jax.Array) -> jax.Array:
    out = jp.clip(lower - qpos, 0.0, None) + jp.clip(qpos - upper, 0.0, None)
    return jp.sum(out)
