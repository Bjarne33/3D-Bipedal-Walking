"""Periodic gait clock: phase variable, smooth stance/swing indicators and swing height reference.

The clock phase `phi` in [0, 1) advances by `dt / period` per control step. The left foot swings
for `phi` in [0, swing_ratio), the right foot half a cycle later. With `swing_ratio` < 0.5 there
are two double-support intervals per cycle. The indicators follow Siekmann et al. 2021
("Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition") with a
logistic smoothing of the phase boundaries instead of von Mises distributions.
"""

import jax
import jax.numpy as jp

# Phase offset of each foot (left, right) as a fraction of the cycle.
FOOT_OFFSETS = (0.0, 0.5)


def advance(phi: jax.Array, dt: float, period: float) -> jax.Array:
    """Advance the clock phase by one control step."""
    return jp.mod(phi + dt / period, 1.0)


def clock_obs(phi: jax.Array) -> jax.Array:
    """Clock observation [sin(2 pi phi), cos(2 pi phi)]."""
    angle = 2.0 * jp.pi * phi
    return jp.stack([jp.sin(angle), jp.cos(angle)], axis=-1)


def foot_phases(phi: jax.Array) -> jax.Array:
    """Local phase of each foot (left, right) in [0, 1); swing starts at 0."""
    return jp.mod(phi[..., None] - jp.array(FOOT_OFFSETS), 1.0)


def swing_indicator(phi: jax.Array, swing_ratio: float, smoothing: float) -> jax.Array:
    """Smooth swing indicator per foot in [0, 1] (1 = foot should be in the air)."""
    local = foot_phases(phi)
    # Circular distance of the local phase to the centre of the swing interval.
    dist = jp.abs(local - 0.5 * swing_ratio)
    dist = jp.minimum(dist, 1.0 - dist)
    return jax.nn.sigmoid((0.5 * swing_ratio - dist) / smoothing)


def desired_contact(phi: jax.Array, swing_ratio: float) -> jax.Array:
    """Hard desired contact per foot (True = stance)."""
    return foot_phases(phi) >= swing_ratio


def _bezier(t: jax.Array) -> jax.Array:
    return t ** 3 + 3.0 * t ** 2 * (1.0 - t)


def swing_height(phi: jax.Array, swing_ratio: float, height: float) -> jax.Array:
    """Reference sole height per foot: 0 in stance, a smooth rise and fall in swing.

    The cubic Bezier profile has zero vertical velocity at lift-off, apex and touchdown.
    """
    local = foot_phases(phi)
    x = jp.clip(local / swing_ratio, 0.0, 1.0)
    up = height * _bezier(2.0 * x)
    down = height * (1.0 - _bezier(2.0 * x - 1.0))
    z = jp.where(x <= 0.5, up, down)
    return jp.where(local < swing_ratio, z, 0.0)
