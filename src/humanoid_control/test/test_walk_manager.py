"""Tests for the ROS-free walk manager state machine."""

import math

import pytest

from humanoid_control.tools.walk_manager import (
    ManagerParams,
    ManagerState,
    RobotObservation,
    tilt_from_quaternion,
    WalkKind,
    WalkManager,
)

REST = RobotObservation(com_height=0.9, com_speed=0.0, tilt=0.0)
MOVING = RobotObservation(com_height=0.9, com_speed=0.2, tilt=0.0)


def idle_manager():
    manager = WalkManager(ManagerParams(rest_hold=0.5))
    manager.update(0.0, True, REST)
    manager.update(0.1, True, REST)
    manager.update(0.7, True, REST)
    assert manager.state == ManagerState.IDLE
    return manager


def test_waits_for_startup_then_rest():
    manager = WalkManager(ManagerParams(rest_hold=0.5))
    manager.update(0.0, False, REST)
    assert manager.state == ManagerState.STARTUP
    assert manager.allowed(True, True) == (False, False, False)
    manager.update(1.0, True, REST)
    assert manager.state == ManagerState.SETTLING
    manager.update(1.3, True, MOVING)
    manager.update(1.4, True, REST)
    manager.update(1.8, True, REST)
    assert manager.state == ManagerState.SETTLING
    manager.update(1.9, True, REST)
    assert manager.state == ManagerState.IDLE
    assert manager.standing_com_height == pytest.approx(0.9)


def test_missing_state_blocks_idle():
    manager = idle_manager()
    manager.update(1.0, True, None)
    assert manager.state == ManagerState.SETTLING


def test_allowed_actions_follow_state_and_servers():
    manager = idle_manager()
    assert manager.allowed(True, True) == (True, True, False)
    assert manager.allowed(True, False) == (True, False, False)
    assert manager.request_walk(WalkKind.RL_WALK, False)[0] is False
    assert manager.state == ManagerState.IDLE
    assert manager.request_walk(WalkKind.WALK, True)[0] is True
    assert manager.state == ManagerState.WALKING
    assert manager.allowed(True, True) == (False, False, True)
    assert manager.request_walk(WalkKind.RL_WALK, True)[0] is False


def test_stop_then_goal_end_returns_to_idle_after_rest():
    manager = idle_manager()
    manager.request_walk(WalkKind.RL_WALK, True)
    manager.goal_accepted(WalkKind.RL_WALK)
    manager.update(1.0, True, MOVING)
    assert manager.state == ManagerState.RL_WALKING
    assert manager.request_stop()[0] is True
    assert manager.state == ManagerState.STOPPING
    assert manager.allowed(True, True) == (False, False, False)
    assert manager.request_stop()[0] is False
    manager.goal_ended(WalkKind.RL_WALK, 'canceled')
    assert manager.state == ManagerState.SETTLING
    manager.update(2.0, True, REST)
    manager.update(2.6, True, REST)
    assert manager.state == ManagerState.IDLE


def test_rejected_goal_returns_to_settling():
    manager = idle_manager()
    manager.request_walk(WalkKind.WALK, True)
    manager.goal_ended(WalkKind.WALK, 'rejected')
    assert manager.state == ManagerState.SETTLING


def test_stop_refused_when_idle():
    manager = idle_manager()
    assert manager.request_stop()[0] is False


def test_fall_while_walking_cancels_and_disables():
    manager = idle_manager()
    manager.request_walk(WalkKind.WALK, True)
    low = RobotObservation(com_height=0.3, com_speed=0.5, tilt=0.2)
    assert manager.update(1.0, True, low) is True
    assert manager.state == ManagerState.FALLEN
    assert manager.allowed(True, True) == (False, False, False)
    manager.goal_ended(WalkKind.WALK, 'canceled')
    assert manager.state == ManagerState.FALLEN
    assert manager.update(2.0, True, REST) is False
    assert manager.state == ManagerState.SETTLING


def test_tilt_detects_fall_before_reference_height():
    manager = WalkManager()
    manager.update(0.0, True, REST)
    tilted = RobotObservation(com_height=0.9, com_speed=0.0, tilt=1.2)
    assert manager.update(0.1, True, tilted) is False
    assert manager.state == ManagerState.FALLEN


def test_tilt_from_quaternion():
    assert tilt_from_quaternion(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)
    s = math.sin(0.25)
    assert tilt_from_quaternion(s, 0.0, 0.0, math.cos(0.25)) == pytest.approx(0.5)
    assert tilt_from_quaternion(0.0, 0.0, s, math.cos(0.25)) == pytest.approx(0.0)
    assert tilt_from_quaternion(1.0, 0.0, 0.0, 0.0) == pytest.approx(math.pi)


@pytest.mark.parametrize('kwargs', [
    {'rest_speed': 0.0}, {'rest_hold': -1.0}, {'fall_tilt': 0.0}, {'fall_height_ratio': 1.0},
])
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        ManagerParams(**kwargs)
