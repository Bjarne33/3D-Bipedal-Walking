import numpy as np
import pytest

from humanoid_control.tools import lipm
from humanoid_control.tools.footstep_planner import FixedFootstepPlanner
from humanoid_control.tools.gait_params import GaitParams
from humanoid_control.tools.swing_trajectory import SwingFootTrajectory
from humanoid_control.tools.walking_pattern import WalkingPattern

LEFT = np.array([0.0, 0.125, 0.075])
RIGHT = np.array([0.0, -0.125, 0.075])


def make_pattern(num_steps, **kw):
    kw.setdefault('zmp_lateral', 0.0)  # tests below assume a foot-centred single-support ZMP
    params = GaitParams(**kw)
    planner = FixedFootstepPlanner(params, LEFT[:2], RIGHT[:2], 'right', num_steps)
    return params, planner, WalkingPattern(params, planner, LEFT, RIGHT)


def test_quasi_static_lead_in_holds_com_over_first_stance_foot():
    params, planner, pat = make_pattern(2, shift_time=2.0, hold_time=1.0, zmp_offset_forward=0.05)
    stance = planner.step(0).stance + np.array([0.05, 0.0])
    ref = pat.sample(2.5)
    assert ref.phase == 'hold'
    assert np.allclose(ref.com[:2], stance) and np.allclose(ref.com_vel, 0.0)
    assert pat.sample(0.5).phase == 'shift'
    assert pat.step_start_time(0) == pytest.approx(3.0 + params.init_time)
    assert pat.sample(pat.step_start_time(0) + 0.01).phase == 'step'


def test_planner_alternates_and_keeps_width():
    params = GaitParams(step_length=0.1, step_width=0.25)
    planner = FixedFootstepPlanner(params, LEFT[:2], RIGHT[:2], 'right', 4)
    assert planner.total_steps == 5
    assert planner.step(5) is None
    sides = [planner.step(i).swing_side for i in range(5)]
    assert sides == ['right', 'left', 'right', 'left', 'right']
    advances = [planner.step(i).swing_target[0] - planner.step(i).stance[0] for i in range(5)]
    np.testing.assert_allclose(advances, [0.05, 0.1, 0.1, 0.1, 0.0])
    for i in range(5):
        s = planner.step(i)
        assert abs(abs(s.swing_target[1] - s.stance[1]) - 0.25) < 1e-12
    # each step starts where the previous stance foot was
    np.testing.assert_allclose(planner.step(2).swing_start, planner.step(1).stance)


def test_planner_endless():
    planner = FixedFootstepPlanner(GaitParams(), LEFT[:2], RIGHT[:2])
    assert planner.total_steps is None
    assert planner.step(100) is not None


def test_swing_endpoints_and_height():
    traj = SwingFootTrajectory([0, 0, 0.07], [0.1, 0, 0.07], 0.05, 0.64)
    p0, v0 = traj.sample(0.0)
    p1, v1 = traj.sample(0.64)
    pm, _ = traj.sample(0.32)
    np.testing.assert_allclose(p0, [0, 0, 0.07], atol=1e-12)
    np.testing.assert_allclose(p1, [0.1, 0, 0.07], atol=1e-12)
    np.testing.assert_allclose(v0, 0, atol=1e-12)
    np.testing.assert_allclose(v1, 0, atol=1e-12)
    assert pm[2] == pytest.approx(0.12)
    ts = np.linspace(0, 0.64, 50)
    xs = [traj.sample(t)[0][0] for t in ts]
    assert np.all(np.diff(xs) >= 0)


def test_lipm_linear_zmp_matches_numerical_integration():
    w = lipm.omega(0.75)
    p0, slope = np.array([0.0, 0.1]), np.array([0.1, -0.5])
    c0, v0 = np.array([0.02, 0.05]), np.array([0.1, 0.0])
    dt, c, v = 1e-5, c0.copy(), v0.copy()
    for i in range(int(0.3 / dt)):
        p = p0 + slope * (i * dt)
        v += w * w * (c - p) * dt
        c += v * dt
    pos, vel, _, zmp = lipm.linear_zmp_state(c0, v0, p0, slope, w, 0.3)
    np.testing.assert_allclose(pos, c, atol=1e-4)
    np.testing.assert_allclose(vel, v, atol=1e-3)
    np.testing.assert_allclose(zmp, p0 + slope * 0.3)


def test_zmp_moves_between_feet_during_dsp():
    params, planner, pat = make_pattern(4)
    k, td = 2, params.dsp_ratio * params.step_time
    t0 = pat.step_start_time(k)
    start = pat.sample(t0 + 1e-9).zmp
    half = pat.sample(t0 + 0.5 * td).zmp
    end = pat.sample(t0 + td + 1e-9).zmp
    np.testing.assert_allclose(start, planner.step(k - 1).stance, atol=1e-6)
    np.testing.assert_allclose(
        half, 0.5 * (planner.step(k - 1).stance + planner.step(k).stance), atol=1e-6)
    np.testing.assert_allclose(end, planner.step(k).stance, atol=1e-6)
    # the swing foot only leaves the ground once the ZMP has reached the stance foot
    side = planner.step(k).swing_side
    assert pat.sample(t0 + td).feet[side][2] == pytest.approx(0.075)


def test_com_sways_between_the_feet_and_zmp_is_always_between_them():
    params, planner, pat = make_pattern(6, step_time=1.0, dsp_ratio=0.3)
    for t in np.arange(0.0, pat.duration, 0.02):
        ref = pat.sample(t)
        assert abs(ref.com[1]) < 0.125 + 1e-6
        assert abs(ref.zmp[1]) < 0.13  # the final settling phase may overshoot by a few mm


def test_com_and_feet_continuous():
    params, _, pat = make_pattern(4)
    ts = np.arange(0.0, pat.duration, 0.002)
    refs = [pat.sample(t) for t in ts]
    com = np.array([r.com for r in refs])
    assert np.max(np.abs(np.diff(com, axis=0))) < 0.002
    for side in ('left', 'right'):
        pos = np.array([r.feet[side] for r in refs])
        assert np.max(np.abs(np.diff(pos, axis=0))) < 0.002
    # CoM velocity consistent with finite differences of the position
    vel = np.array([r.com_vel for r in refs])[1:-1]
    fd = (com[2:] - com[:-2]) / (2 * 0.002)
    assert np.max(np.abs(vel[:, :2] - fd[:, :2])) < 5e-3


def test_pattern_ends_at_rest_between_feet():
    params, planner, pat = make_pattern(3)
    end = pat.sample(pat.duration + 0.1)
    assert end.finished
    np.testing.assert_allclose(end.com_vel, 0, atol=1e-9)
    mid = 0.5 * (end.feet['left'][:2] + end.feet['right'][:2])
    np.testing.assert_allclose(end.com[:2], mid, atol=1e-9)
    assert end.feet['left'][0] == pytest.approx(end.feet['right'][0])
    assert end.feet['left'][0] == pytest.approx(0.05 + 2 * 0.1)
    assert end.feet['left'][1] - end.feet['right'][1] == pytest.approx(0.25)


def test_zmp_inside_stance_foot_during_steps():
    params, planner, pat = make_pattern(4)
    for k in range(planner.total_steps):
        ref = pat.sample(pat.step_start_time(k) + 0.4)
        np.testing.assert_allclose(ref.zmp, planner.step(k).stance)
        assert ref.feet[ref.stance_side][2] == pytest.approx(0.075)


def test_swing_foot_stays_down_in_dsp_and_lifts_after():
    params, planner, pat = make_pattern(3)
    k = 1
    t0 = pat.step_start_time(k)
    start = pat.sample(t0 + 0.5 * params.dsp_ratio * params.step_time)
    mid = pat.sample(t0 + 0.5 * params.step_time + 0.5 * params.dsp_ratio * params.step_time)
    side = planner.step(k).swing_side
    assert start.feet[side][2] == pytest.approx(0.075)
    assert mid.feet[side][2] == pytest.approx(0.075 + params.swing_height, abs=1e-6)
    assert not start.single_support and mid.single_support
    assert not pat.sample(0.0).single_support


def test_com_stays_bounded_in_long_continuous_walk():
    params = GaitParams()
    planner = FixedFootstepPlanner(params, LEFT[:2], RIGHT[:2])
    pat = WalkingPattern(params, planner, LEFT, RIGHT)
    for k in (0, 10, 100, 400):
        ref = pat.sample(pat.step_start_time(k) + 0.3)
        assert abs(ref.com[1] - planner.step(k).stance[1]) < 0.15  # lateral
        assert abs(ref.com[0] - planner.step(k).stance[0]) < 0.15  # forward


def test_com_state_continuous_at_step_boundaries():
    _, _, pat = make_pattern(5)
    for k in range(1, 5):
        tb = pat.step_start_time(k)
        a = pat.sample(tb - 1e-9)
        b = pat.sample(tb + 1e-9)
        np.testing.assert_allclose(a.com, b.com, atol=1e-6)
        np.testing.assert_allclose(a.com_vel, b.com_vel, atol=1e-6)


def test_stop_pattern_hands_over_smoothly_and_closes_feet():
    params = GaitParams()
    planner = FixedFootstepPlanner(params, LEFT[:2], RIGHT[:2])
    pat = WalkingPattern(params, planner, LEFT, RIGHT)
    k = 6
    stop = pat.stop_pattern(k)
    tb = pat.step_start_time(k)
    a = pat.sample(tb)
    b = stop.sample(0.0)
    np.testing.assert_allclose(a.com, b.com, atol=1e-9)
    np.testing.assert_allclose(a.com_vel, b.com_vel, atol=1e-9)
    for s in ('left', 'right'):
        np.testing.assert_allclose(a.feet[s], b.feet[s], atol=1e-9)
    end = stop.sample(stop.duration + 0.1)
    assert end.finished
    np.testing.assert_allclose(end.com_vel, 0, atol=1e-9)
    assert end.feet['left'][0] == pytest.approx(end.feet['right'][0])
