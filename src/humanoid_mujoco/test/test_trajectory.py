"""Check the trajectory tracker against the trajectory controller semantics."""

import numpy as np
import pytest

from humanoid_mujoco.trajectory import TrajectoryTracker

NAMES = ['a', 'b']


def _tracker():
    """Create a two-joint tracker holding zero.

    Returns:
        The tracker.
    """
    tracker = TrajectoryTracker(NAMES, [100.0, 200.0], [10.0, 4.0])
    tracker.hold(1.0, [0.0, 0.0])
    return tracker


def test_hold():
    """Hold the activation positions with zero velocity."""
    tracker = _tracker()
    positions, velocities = tracker.sample(5.0)
    assert positions == pytest.approx([0.0, 0.0])
    assert velocities == pytest.approx([0.0, 0.0])


def test_linear_segment_and_hold_after_end():
    """Interpolate positions linearly without velocities and hold the last point."""
    tracker = _tracker()
    tracker.set_trajectory(1.0, 1.0, NAMES, [(2.0, [1.0, -2.0], None)])
    positions, velocities = tracker.sample(2.0)
    assert positions == pytest.approx([0.5, -1.0])
    assert velocities == pytest.approx([0.5, -1.0])
    positions, velocities = tracker.sample(10.0)
    assert positions == pytest.approx([1.0, -2.0])
    assert velocities == pytest.approx([0.0, 0.0])


def test_cubic_segment_starts_from_desired_state():
    """Use a cubic segment from the current desired state when velocities are given."""
    tracker = _tracker()
    tracker.set_trajectory(1.0, 1.0, NAMES, [(1.0, [1.0, 1.0], [0.0, 0.0])])
    tracker.set_trajectory(1.5, 1.5, NAMES, [(1.0, [2.0, 0.0], [1.0, 0.0])])
    start = tracker.knots[0]
    assert start.positions == pytest.approx([0.5, 0.5])
    assert start.velocities == pytest.approx([1.5, 1.5])
    positions, velocities = tracker.sample(2.5 - 1e-9)
    assert positions == pytest.approx([2.0, 0.0], abs=1e-6)
    assert velocities == pytest.approx([1.0, 0.0], abs=1e-6)
    positions, velocities = tracker.sample(3.0)
    assert positions == pytest.approx([2.0, 0.0])
    assert velocities == pytest.approx([1.0, 0.0])


def test_partial_joints_keep_desired_position():
    """Keep joints that are missing from the message at their desired position."""
    tracker = _tracker()
    tracker.set_trajectory(1.0, 1.0, NAMES, [(1.0, [0.0, 3.0], None)])
    tracker.set_trajectory(2.0, 2.0, ['a'], [(1.0, [1.0], None)])
    positions, _ = tracker.sample(3.0)
    assert positions == pytest.approx([1.0, 3.0])


def test_past_and_empty_trajectories():
    """Jump to the last point when all points are past; hold on an empty trajectory."""
    tracker = _tracker()
    tracker.set_trajectory(5.0, 1.0, NAMES, [(1.0, [1.0, 1.0], None), (2.0, [2.0, 2.0], None)])
    assert tracker.sample(5.0)[0] == pytest.approx([2.0, 2.0])
    tracker.set_trajectory(6.0, 6.0, NAMES, [])
    positions, velocities = tracker.sample(7.0)
    assert positions == pytest.approx([2.0, 2.0])
    assert velocities == pytest.approx([0.0, 0.0])


def test_invalid_input():
    """Reject unknown joints, mismatched arrays and non-positive p gains."""
    tracker = _tracker()
    with pytest.raises(ValueError):
        tracker.set_trajectory(1.0, 1.0, ['c'], [(1.0, [0.0], None)])
    with pytest.raises(ValueError):
        tracker.set_trajectory(1.0, 1.0, NAMES, [(1.0, [0.0], None)])
    with pytest.raises(ValueError):
        TrajectoryTracker(NAMES, [1.0, 0.0], [1.0, 1.0])


def test_setpoint_reproduces_pd_law():
    """Make the position actuator force equal p * position error + d * velocity error."""
    tracker = _tracker()
    tracker.set_trajectory(1.0, 1.0, NAMES, [(1.0, [1.0, -1.0], [0.5, 2.0])])
    desired_pos, desired_vel = tracker.sample(1.4)
    q, qdot = np.array([0.1, -0.2]), np.array([0.3, -0.4])
    actuator = tracker.p_gains * (tracker.setpoint(1.4) - q) - tracker.d_gains * qdot
    pd_law = tracker.p_gains * (desired_pos - q) + tracker.d_gains * (desired_vel - qdot)
    assert actuator == pytest.approx(pd_law)
