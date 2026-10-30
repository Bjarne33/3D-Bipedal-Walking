"""Swing-foot trajectory: minimum-jerk motion in the plane and a smooth bump in z."""

from typing import Tuple

import numpy as np


def min_jerk(s: float) -> Tuple[float, float]:
    """Evaluate the minimum-jerk blend.

    Args:
        s: Normalised time in [0, 1].

    Returns:
        Tuple (position blend, derivative with respect to `s`).
    """
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4


class SwingFootTrajectory:
    """Foot path from `start` to `target` (3D, same ground height) lasting `duration`.

    Velocity and acceleration vanish at lift-off and touch-down in all axes.
    """

    def __init__(self, start, target, height: float, duration: float) -> None:
        """Create the trajectory.

        Args:
            start: Start position (3,).
            target: Target position (3,).
            height: Peak swing height above the straight line.
            duration: Swing duration in seconds.

        Raises:
            ValueError: If `duration` is not positive.
        """
        if duration <= 0.0:
            raise ValueError('duration must be positive')
        self.start = np.asarray(start, float).reshape(3)
        self.target = np.asarray(target, float).reshape(3)
        self.height = float(height)
        self.duration = float(duration)

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the swing foot path.

        Args:
            t: Time since lift-off in seconds (clamped to the duration).

        Returns:
            Tuple (position, velocity).
        """
        s = min(max(t / self.duration, 0.0), 1.0)
        blend, dblend = min_jerk(s)
        pos = self.start + (self.target - self.start) * blend
        vel = (self.target - self.start) * dblend / self.duration
        # 16 s^2 (1-s)^2 peaks at exactly `height` for s = 0.5
        pos[2] += self.height * 16 * s**2 * (1 - s) ** 2
        vel[2] += self.height * 16 * (2 * s * (1 - s) ** 2 - 2 * s**2 * (1 - s)) / self.duration
        return pos, vel
