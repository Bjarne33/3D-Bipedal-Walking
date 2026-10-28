"""Swing-foot trajectory: minimum-jerk motion in the plane and a smooth bump in z."""

from typing import Tuple

import numpy as np


def min_jerk(s: float) -> Tuple[float, float]:
    """Return (position, d/ds) of the minimum-jerk blend for s in [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4


class SwingFootTrajectory:
    """
    Foot path from `start` to `target` (3D, same ground height) lasting `duration`.

    Velocity and acceleration vanish at lift-off and touch-down in all axes.
    """

    def __init__(self, start, target, height: float, duration: float) -> None:
        if duration <= 0.0:
            raise ValueError('duration must be positive')
        self.start = np.asarray(start, float).reshape(3)
        self.target = np.asarray(target, float).reshape(3)
        self.height = float(height)
        self.duration = float(duration)

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, velocity) at time `t` since lift-off (clamped to the duration)."""
        s = min(max(t / self.duration, 0.0), 1.0)
        blend, dblend = min_jerk(s)
        pos = self.start + (self.target - self.start) * blend
        vel = (self.target - self.start) * dblend / self.duration
        # 16 s^2 (1-s)^2 peaks at exactly `height` for s = 0.5
        pos[2] += self.height * 16 * s**2 * (1 - s) ** 2
        vel[2] += self.height * 16 * (2 * s * (1 - s) ** 2 - 2 * s**2 * (1 - s)) / self.duration
        return pos, vel
