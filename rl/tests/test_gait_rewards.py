"""Gait clock and reward term unit tests."""

import jax.numpy as jp
import numpy as np
import pytest

from humanoid_rl import config as cfg
from humanoid_rl import gait_clock
from humanoid_rl import rewards as rw

SWING = 0.4
HEIGHT = 0.06


def test_advance_wraps():
    assert float(gait_clock.advance(jp.array(0.99), 0.02, 0.8)) == pytest.approx(0.015, abs=1e-6)


def test_clock_obs_unit_circle():
    obs = gait_clock.clock_obs(jp.linspace(0, 1, 7))
    np.testing.assert_allclose(np.linalg.norm(obs, axis=-1), 1.0, rtol=1e-6)


def test_feet_alternate_with_double_support():
    phases = jp.linspace(0, 1, 101)[:-1]
    stance = np.asarray(gait_clock.desired_contact(phases, SWING))
    assert not np.any(~stance[:, 0] & ~stance[:, 1]), 'both feet in swing'
    assert np.all(stance.sum(axis=1) >= 1)
    np.testing.assert_allclose((~stance).mean(axis=0), SWING, atol=0.011)
    np.testing.assert_array_equal(stance[:50, 0], stance[50:, 1])


def test_swing_indicator_matches_hard_contact():
    phases = jp.linspace(0, 1, 401)[:-1]
    swing = np.asarray(gait_clock.swing_indicator(phases, SWING, 0.02))
    stance = np.asarray(gait_clock.desired_contact(phases, SWING))
    at_boundary = np.abs(swing - 0.5) < 1e-3
    assert np.all(((swing > 0.5) == ~stance) | at_boundary)
    assert swing.min() >= 0 and swing.max() <= 1


def test_swing_height_profile():
    local = jp.array([0.0, SWING / 4, SWING / 2, SWING, 0.7])
    z = np.asarray(gait_clock.swing_height(local, SWING, HEIGHT))[:, 0]
    assert z[0] == pytest.approx(0.0, abs=1e-7)
    assert z[1] == pytest.approx(HEIGHT / 2, rel=1e-5)
    assert z[2] == pytest.approx(HEIGHT, rel=1e-6)
    assert z[3] == pytest.approx(0.0, abs=1e-7) and z[4] == 0.0


def test_reward_terms_complete_and_scales_valid():
    assert len(set(cfg.REWARD_TERMS)) == len(cfg.REWARD_TERMS)
    for phase in cfg.PHASES:
        scales = cfg.env_config(phase).reward.scales
        assert set(scales.keys()) == set(cfg.REWARD_TERMS)
    assert cfg.env_config(1).reward.scales.tracking_lin_vel == 0
    assert cfg.env_config(3).reward.scales.tracking_lin_vel > 0
    with pytest.raises(ValueError):
        cfg.env_config(4)


def test_heading_frame():
    yaw = rw.heading_yaw(jp.eye(3))
    assert float(yaw) == pytest.approx(0.0)
    rot90 = jp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90 deg about Z
    yaw = rw.heading_yaw(rot90)
    assert float(yaw) == pytest.approx(np.pi / 2)
    # The robot now faces world -X: a world velocity along -X is forward in the heading frame.
    vel = rw.to_heading_frame(jp.array([-1.0, 0.0]), yaw)
    np.testing.assert_allclose(vel, [0.0, 1.0], atol=1e-6)


def test_kernels_and_costs():
    assert float(rw.tracking_lin_vel(jp.array(0.1), jp.array(0.1), 0.01)) == pytest.approx(1)
    assert float(rw.tracking_lin_vel(jp.array(0.0), jp.array(0.1), 0.01)) < 0.4
    assert float(rw.orientation(jp.array([0.0, 0.0, -1.0]))) == 0
    assert float(rw.heading(jp.array(3.1), jp.array(-3.1))) == pytest.approx(
        (2 * np.pi - 6.2) ** 2, rel=1e-4)
    assert float(rw.dof_pos_limits(jp.array([0.0, 2.0]), jp.array([-1.0, -1.0]),
                                   jp.array([1.0, 1.0]))) == pytest.approx(1.0)
    assert float(rw.feet_distance(jp.array([0.15, -0.15]), 0.22)) == 0
    assert float(rw.feet_distance(jp.array([0.05, -0.05]), 0.22)) == pytest.approx(0.1)


def test_periodic_terms():
    swing = jp.array([1.0, 0.0])  # left swings, right stands
    good = rw.periodic_force(swing, jp.array([0.0, 300.0]), 30.0)
    bad = rw.periodic_force(swing, jp.array([150.0, 150.0]), 30.0)
    assert float(good) == pytest.approx(0.5) and float(bad) < 1e-3
    still = rw.periodic_vel(swing, jp.array([[1.0, 0.0], [0.0, 0.0]]), 0.01)
    sliding = rw.periodic_vel(swing, jp.array([[0.0, 0.0], [0.5, 0.0]]), 0.01)
    assert float(still) == pytest.approx(0.5) and float(sliding) < 1e-3


def test_weight_shift_targets_stance_foot():
    feet = jp.array([0.125, -0.125])
    left_stance = jp.array([1.0, 0.0])
    assert float(rw.weight_shift(jp.array(0.125), feet, left_stance, 0.002)) == pytest.approx(1)
    assert float(rw.weight_shift(jp.array(0.0), feet, jp.ones(2), 0.002)) == pytest.approx(1)
    assert float(rw.weight_shift(jp.array(-0.1), feet, left_stance, 0.002)) < 1e-6


def test_thresholds_reference_existing_metrics():
    from humanoid_rl.evaluate import check_thresholds
    metrics = dict(survival=1.0, tilt=0.0, height_error=0.0, double_contact=1.0,
                   contact_match=1.0, swing_height_ratio=1.0, drift=0.0, vel_error=0.0,
                   lateral_vel=0.0)
    for phase in cfg.PHASES:
        passed, checks = check_thresholds(phase, metrics)
        assert passed and checks
    passed, checks = check_thresholds(1, dict(metrics, tilt=1.0))
    assert not passed and not checks['max_tilt']
