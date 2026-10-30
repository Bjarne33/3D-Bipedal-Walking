"""Footstep planning with fixed step length and width (plane frame: x forward, y left)."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from humanoid_control.tools.gait_params import GaitParams


@dataclass
class Footstep:
    """One step: the swing foot moves from `swing_start` to `swing_target`.

    The `stance` foot carries the robot meanwhile.
    """

    index: int
    swing_side: str  # 'left' or 'right'
    stance: np.ndarray  # (2,) stance foot position
    swing_start: np.ndarray  # (2,)
    swing_target: np.ndarray  # (2,)

    @property
    def stance_side(self) -> str:
        """Side of the stance foot ('left' or 'right')."""
        return 'left' if self.swing_side == 'right' else 'right'


class FixedFootstepPlanner:
    """Alternating steps of constant length and width.

    The first step is a half step, every later step has `step_length`. For a finite plan
    (`num_steps` is not None) one closing step with zero length is appended so that the
    feet end side by side. `num_steps=None` plans forever.
    """

    def __init__(
        self,
        params: GaitParams,
        left_xy,
        right_xy,
        first_swing: str = 'right',
        num_steps: Optional[int] = None,
        first_step_length: Optional[float] = None,
    ) -> None:
        """Plan the footsteps.

        Args:
            params: Gait parameters.
            left_xy: Initial left foot position (2,) in the plane frame.
            right_xy: Initial right foot position (2,) in the plane frame.
            first_swing: Side of the first swing foot, 'left' or 'right'.
            num_steps: Number of steps before the closing step, or None to plan forever.
            first_step_length: Length of the first step; defaults to half a step.

        Raises:
            ValueError: If `first_swing` is invalid or `num_steps` is negative.
        """
        if first_swing not in ('left', 'right'):
            raise ValueError("first_swing must be 'left' or 'right'")
        if num_steps is not None and num_steps < 0:
            raise ValueError('num_steps must be >= 0')
        self.params = params
        self.num_steps = num_steps
        self.first_swing = first_swing
        self.first_step_length = (
            0.5 * params.step_length if first_step_length is None else first_step_length
        )
        feet = {'left': np.asarray(left_xy, float)[:2], 'right': np.asarray(right_xy, float)[:2]}
        self.y_center = 0.5 * (feet['left'][1] + feet['right'][1])
        self._feet = feet
        self._steps: List[Footstep] = []

    @property
    def total_steps(self) -> Optional[int]:
        """Return the number of steps including the closing step, or None for an endless plan."""
        return None if self.num_steps is None else self.num_steps + 1

    def _lateral(self, side: str) -> float:
        """Return the lateral (y) position of a foot.

        Args:
            side: 'left' or 'right'.

        Returns:
            The y coordinate, half a step width from the centre line.
        """
        half = 0.5 * self.params.step_width
        return self.y_center + (half if side == 'left' else -half)

    def _extend(self, index: int) -> None:
        """Plan steps lazily until step `index` exists.

        Args:
            index: Highest step index that must be available.
        """
        while len(self._steps) <= index:
            k = len(self._steps)
            if k == 0:
                swing_side = self.first_swing
                stance = self._feet['right' if swing_side == 'left' else 'left']
                swing_start = self._feet[swing_side]
            else:
                prev = self._steps[-1]
                swing_side = prev.stance_side
                stance = prev.swing_target
                swing_start = prev.stance
            if self.num_steps is not None and k >= self.num_steps:
                advance = 0.0  # closing step
            elif k == 0:
                advance = self.first_step_length
            else:
                advance = self.params.step_length
            target = np.array([stance[0] + advance, self._lateral(swing_side)])
            self._steps.append(Footstep(k, swing_side, stance.copy(), swing_start.copy(), target))

    def step(self, index: int) -> Optional[Footstep]:
        """Return step `index` (0-based), or None beyond the end of a finite plan.

        Args:
            index: Step index.

        Returns:
            The footstep, or None if `index` is outside the plan.
        """
        total = self.total_steps
        if index < 0 or (total is not None and index >= total):
            return None
        self._extend(index)
        return self._steps[index]
