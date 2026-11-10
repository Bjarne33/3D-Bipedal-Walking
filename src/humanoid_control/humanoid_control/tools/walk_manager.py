"""ROS-free state machine that decides when the walking controllers may be started or stopped."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple


class ManagerState(str, Enum):
    """States of the walk manager; values match `WalkManagerState` constants."""

    STARTUP = 'STARTUP'
    SETTLING = 'SETTLING'
    IDLE = 'IDLE'
    WALKING = 'WALKING'
    RL_WALKING = 'RL_WALKING'
    STOPPING = 'STOPPING'
    FALLEN = 'FALLEN'


class WalkKind(str, Enum):
    """Walking controllers managed by the state machine."""

    WALK = 'walk'
    RL_WALK = 'rl_walk'


WALK_STATES = {WalkKind.WALK: ManagerState.WALKING, WalkKind.RL_WALK: ManagerState.RL_WALKING}


@dataclass
class RobotObservation:
    """Measurements of the shared robot state used for rest and fall detection.

    Attributes:
        com_height: World-frame CoM height [m].
        com_speed: Norm of the world-frame CoM velocity [m/s].
        tilt: Angle between the base z axis and the world z axis [rad].
    """

    com_height: float
    com_speed: float
    tilt: float


@dataclass
class ManagerParams:
    """Thresholds of the walk manager.

    Attributes:
        rest_speed: CoM speed below which the robot counts as at rest [m/s].
        rest_hold: Time the robot must stay at rest before walking is allowed [s].
        fall_tilt: Base tilt above which the robot counts as fallen [rad].
        fall_height_ratio: Fraction of the standing CoM height below which the robot counts
            as fallen.
    """

    rest_speed: float = 0.02
    rest_hold: float = 1.0
    fall_tilt: float = 0.8
    fall_height_ratio: float = 0.6

    def __post_init__(self):
        """Validate the thresholds.

        Raises:
            ValueError: If a threshold is out of range.
        """
        if not self.rest_speed > 0.0:
            raise ValueError('rest_speed must be positive')
        if not self.rest_hold >= 0.0:
            raise ValueError('rest_hold must not be negative')
        if not 0.0 < self.fall_tilt < math.pi:
            raise ValueError('fall_tilt must be in (0, pi)')
        if not 0.0 < self.fall_height_ratio < 1.0:
            raise ValueError('fall_height_ratio must be in (0, 1)')


def tilt_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Compute the angle between the rotated z axis and the world z axis.

    Args:
        x: Quaternion x component.
        y: Quaternion y component.
        z: Quaternion z component.
        w: Quaternion w component.

    Returns:
        The tilt angle [rad]; 0 for an upright body.
    """
    norm2 = x * x + y * y + z * z + w * w
    if norm2 <= 0.0:
        return math.pi
    # z component of the rotated unit z axis.
    cos_tilt = 1.0 - 2.0 * (x * x + y * y) / norm2
    return math.acos(max(-1.0, min(1.0, cos_tilt)))


class WalkManager:
    """Track the walking mode and allowed user actions.

    The owner feeds startup readiness and robot observations through `update`, forwards user
    requests and reports action goal outcomes. Methods return what the owner has to do with
    the action servers; the class itself performs no I/O.
    """

    def __init__(self, params: Optional[ManagerParams] = None):
        """Start in the STARTUP state.

        Args:
            params: Thresholds; defaults if None.
        """
        self.params = params or ManagerParams()
        self.state = ManagerState.STARTUP
        self.message = 'waiting for the startup sequence'
        self.standing_com_height: Optional[float] = None
        self._rest_since: Optional[float] = None

    # ------------------------------------------------------------------ queries
    def allowed(self, walk_ready: bool, rl_walk_ready: bool) -> Tuple[bool, bool, bool]:
        """Return which user requests are currently accepted.

        Args:
            walk_ready: The model-based walk action server is available.
            rl_walk_ready: The RL walk action server is available.

        Returns:
            Tuple (walk, rl_walk, stop).
        """
        idle = self.state == ManagerState.IDLE
        walking = self.state in (ManagerState.WALKING, ManagerState.RL_WALKING)
        return idle and walk_ready, idle and rl_walk_ready, walking

    def is_fallen(self, obs: RobotObservation) -> bool:
        """Check the fall criteria.

        Args:
            obs: Current robot observation.

        Returns:
            True if the base is tilted too far or the CoM dropped too low.
        """
        if not obs.tilt <= self.params.fall_tilt:
            return True
        if self.standing_com_height is None:
            return False
        return not obs.com_height >= self.params.fall_height_ratio * self.standing_com_height

    # ------------------------------------------------------------------ transitions
    def _set(self, state: ManagerState, message: str) -> None:
        """Change state and reset the rest timer.

        Args:
            state: New state.
            message: Human-readable detail.
        """
        self.state = state
        self.message = message
        self._rest_since = None

    def update(self, now: float, startup_ready: bool,
               obs: Optional[RobotObservation]) -> bool:
        """Advance the time-driven transitions.

        Args:
            now: Current time [s].
            startup_ready: humanoid_startup reported the standing pose.
            obs: Fresh robot observation, or None if no valid state is available.

        Returns:
            True if the active walk goal must be canceled (the robot fell while walking).
        """
        if self.state == ManagerState.STARTUP:
            if startup_ready:
                self._set(ManagerState.SETTLING, 'waiting until the robot is at rest')
            else:
                return False

        if obs is None:
            if self.state in (ManagerState.SETTLING, ManagerState.IDLE):
                self._set(ManagerState.SETTLING, 'no valid robot state')
            return False

        fallen = self.is_fallen(obs)
        if self.state == ManagerState.FALLEN:
            if not fallen:
                self._set(ManagerState.SETTLING, 'robot upright again, waiting for rest')
            return False
        if fallen:
            cancel = self.state in (ManagerState.WALKING, ManagerState.RL_WALKING,
                                    ManagerState.STOPPING)
            self._set(ManagerState.FALLEN, 'robot fell; walking disabled')
            return cancel

        if self.state == ManagerState.SETTLING:
            if obs.com_speed < self.params.rest_speed:
                if self._rest_since is None:
                    self._rest_since = now
                elif now - self._rest_since >= self.params.rest_hold:
                    if self.standing_com_height is None:
                        self.standing_com_height = obs.com_height
                    self._set(ManagerState.IDLE, 'ready')
            else:
                self._rest_since = None
        elif self.state == ManagerState.IDLE and obs.com_speed >= 2.0 * self.params.rest_speed:
            self._set(ManagerState.SETTLING, 'robot moving, waiting for rest')
        return False

    def request_walk(self, kind: WalkKind, server_ready: bool) -> Tuple[bool, str]:
        """Handle a user request to start walking.

        Args:
            kind: Requested walking controller.
            server_ready: Its action server is available.

        Returns:
            Tuple (accepted, message); if accepted the owner must send the goal.
        """
        if self.state != ManagerState.IDLE:
            return False, f'cannot start {kind.value} in state {self.state.value}'
        if not server_ready:
            return False, f'{kind.value} action server not available'
        self._set(WALK_STATES[kind], f'{kind.value} goal sent')
        return True, f'{kind.value} started'

    def request_stop(self) -> Tuple[bool, str]:
        """Handle a user stop request.

        Returns:
            Tuple (accepted, message); if accepted the owner must cancel the active goal.
        """
        if self.state not in (ManagerState.WALKING, ManagerState.RL_WALKING):
            return False, f'nothing to stop in state {self.state.value}'
        self._set(ManagerState.STOPPING, 'stopping')
        return True, 'stopping'

    def goal_accepted(self, kind: WalkKind) -> None:
        """Report that the action server accepted the goal.

        Args:
            kind: Walking controller of the goal.
        """
        if self.state == WALK_STATES[kind]:
            self.message = f'{kind.value} running'

    def goal_ended(self, kind: WalkKind, detail: str) -> None:
        """Report that the goal was rejected or finished with any terminal status.

        Args:
            kind: Walking controller of the goal.
            detail: Outcome description for the message.
        """
        if self.state in (WALK_STATES[kind], ManagerState.STOPPING):
            self._set(ManagerState.SETTLING, f'{kind.value} ended ({detail}), waiting for rest')
