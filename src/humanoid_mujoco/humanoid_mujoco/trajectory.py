"""Joint trajectory tracking that mirrors the Gazebo effort trajectory controller.

The Gazebo backend runs `joint_trajectory_controller` on the effort interface. This module
reproduces the parts the control nodes rely on:

* on activation the controller holds the current joint positions;
* a new trajectory replaces the active one and starts from the current desired state;
* segments are interpolated linearly when a point has no velocities and with cubic
  Hermite splines when it has velocities (accelerations are ignored);
* after the last point the controller holds that point;
* the effort is `p * position_error + d * velocity_error`, with `i = 0` and no
  velocity feed-forward as in the Gazebo gains.

MuJoCo applies this PD law with position actuators (`force = p * (ctrl - q) - d * qdot`,
damping integrated implicitly), so `setpoint` returns `ctrl = q_des + (d / p) * qdot_des`.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Knot:
    """Desired joint state at an absolute time.

    Attributes:
        time: Absolute simulation time [s].
        positions: Joint positions [rad].
        velocities: Joint velocities [rad/s].
        has_velocities: Whether the velocities were commanded (selects cubic interpolation
            into this knot) or are zero/derived.
    """

    time: float
    positions: np.ndarray
    velocities: np.ndarray
    has_velocities: bool


class TrajectoryTracker:
    """Sample joint trajectories and convert them to position-actuator setpoints."""

    def __init__(self, joint_names: Sequence[str], p_gains: Sequence[float],
                 d_gains: Sequence[float]):
        """Store the joint order and gains; hold zero positions until `hold` is called.

        Args:
            joint_names: Joint order of all arrays.
            p_gains: Proportional gains [Nm/rad], one positive gain per joint.
            d_gains: Derivative gains [Nm s/rad], one per joint.

        Raises:
            ValueError: If the gain arrays do not match the joints or a p gain is not positive.
        """
        self.joint_names = list(joint_names)
        self.index = {name: i for i, name in enumerate(self.joint_names)}
        self.p_gains = np.asarray(p_gains, dtype=float)
        self.d_gains = np.asarray(d_gains, dtype=float)
        n = len(self.joint_names)
        if self.p_gains.shape != (n,) or self.d_gains.shape != (n,):
            raise ValueError('One p and one d gain are required per joint')
        if np.any(self.p_gains <= 0.0):
            raise ValueError('All p gains must be positive')
        self.knots: List[Knot] = [Knot(0.0, np.zeros(n), np.zeros(n), False)]

    def hold(self, now: float, positions: Sequence[float]) -> None:
        """Hold the given positions with zero velocity (controller activation).

        Args:
            now: Current simulation time [s].
            positions: Positions to hold, in joint order.
        """
        n = len(self.joint_names)
        self.knots = [Knot(now, np.asarray(positions, dtype=float).copy(), np.zeros(n), False)]

    def set_trajectory(self, now: float, start_time: float, joint_names: Sequence[str],
                       points: Sequence[Tuple[float, Sequence[float],
                                              Optional[Sequence[float]]]]) -> None:
        """Replace the active trajectory.

        Joints that are not in `joint_names` keep their current desired position. Points that
        lie in the past are skipped; if all of them do, the last point is applied at once.
        An empty point list holds the current desired state.

        Args:
            now: Current simulation time [s].
            start_time: Absolute time of `time_from_start = 0` [s].
            joint_names: Joint names of the point arrays.
            points: `(time_from_start, positions, velocities or None)` per point.

        Raises:
            ValueError: If a joint name is unknown or an array has the wrong length.
        """
        indices = []
        for name in joint_names:
            if name not in self.index:
                raise ValueError(f'Unknown joint {name!r}')
            indices.append(self.index[name])
        start_pos, start_vel = self.sample(now)
        knots = [Knot(now, start_pos, start_vel, False)]
        for time_from_start, positions, velocities in points:
            if len(positions) != len(indices) or (
                    velocities is not None and len(velocities) != len(indices)):
                raise ValueError('Point arrays must match the joint names')
            pos = start_pos.copy()
            vel = np.zeros_like(start_vel)
            pos[indices] = positions
            if velocities is not None:
                vel[indices] = velocities
            knots.append(Knot(start_time + float(time_from_start), pos, vel,
                              velocities is not None))
        future = [knot for knot in knots[1:] if knot.time > now]
        if future:
            self.knots = [knots[0]] + future
        elif len(knots) > 1:
            last = knots[-1]
            self.knots = [Knot(now, last.positions, last.velocities, last.has_velocities)]
        else:
            self.knots = [Knot(now, start_pos, np.zeros_like(start_vel), False)]

    def sample(self, now: float) -> Tuple[np.ndarray, np.ndarray]:
        """Desired positions and velocities at a time.

        Args:
            now: Simulation time [s].

        Returns:
            `(positions, velocities)` in joint order.
        """
        knots = self.knots
        if now <= knots[0].time:
            return knots[0].positions.copy(), knots[0].velocities.copy()
        for start, end in zip(knots, knots[1:]):
            if now < end.time:
                return _interpolate(start, end, now)
        last = knots[-1]
        return last.positions.copy(), last.velocities.copy()

    def setpoint(self, now: float) -> np.ndarray:
        """Position-actuator setpoints that reproduce the PD law on the desired state.

        Args:
            now: Simulation time [s].

        Returns:
            Setpoints `q_des + (d / p) * qdot_des` [rad], in joint order.
        """
        desired_pos, desired_vel = self.sample(now)
        return desired_pos + self.d_gains / self.p_gains * desired_vel


def _interpolate(start: Knot, end: Knot, now: float) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate between two knots.

    Args:
        start: Segment start.
        end: Segment end; its `has_velocities` selects cubic or linear interpolation.
        now: Time inside the segment [s].

    Returns:
        `(positions, velocities)` at `now`.
    """
    duration = end.time - start.time
    s = now - start.time
    if not end.has_velocities:
        slope = (end.positions - start.positions) / duration
        return start.positions + slope * s, slope
    p0, v0, p1, v1 = start.positions, start.velocities, end.positions, end.velocities
    a2 = (3.0 * (p1 - p0) - (2.0 * v0 + v1) * duration) / duration ** 2
    a3 = (2.0 * (p0 - p1) + (v0 + v1) * duration) / duration ** 3
    return p0 + v0 * s + a2 * s ** 2 + a3 * s ** 3, v0 + 2.0 * a2 * s + 3.0 * a3 * s ** 2
