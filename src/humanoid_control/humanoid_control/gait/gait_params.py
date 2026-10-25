"""Gait parameters shared by the footstep planner and the trajectory generators."""

from dataclasses import dataclass

GRAVITY = 9.81


@dataclass
class GaitParams:
    """All lengths in metres, times in seconds.

    The gait is planned in a 2D plane frame (x forward, y towards the left foot) with z up.
    """

    step_length: float = 0.10
    step_width: float = 0.25  # lateral distance between the feet (ankle frames)
    step_time: float = 0.8
    dsp_ratio: float = 0.2  # fraction of a step in which the swing foot stays on the ground
    swing_height: float = 0.05
    com_height: float = 0.75  # CoM height above the ground used for the LIPM
    zmp_offset_forward: float = 0.0  # ZMP / CoM reference offset from the ankle frame (foot centre)
    init_time: float = 1.5  # CoM shift from the standing state onto the first stance foot
    shift_time: float = 0.0  # quasi-static CoM shift above the first stance foot (0 disables the lead-in)
    hold_time: float = 0.0  # pause above the first stance foot before the dynamic init phase
    shift_lateral: float = 0.0  # max lateral CoM travel in the quasi-static shift (0 = full shift onto the foot)
    zmp_lateral: float = 0.0  # lateral ZMP distance from the body midline during single support (0 = stance foot centre)
    final_time: float = 1.5  # CoM settling to the centre of the feet after the last step
    stop_transition_time: float = 0.8  # blend time used when a running walk is stopped
    lookahead_steps: int = 6  # steps of preview used for the DCM end-of-step values

    def validate(self) -> None:
        if self.step_time <= 0.0 or self.com_height <= 0.0:
            raise ValueError('step_time and com_height must be positive')
        if not 0.0 <= self.dsp_ratio < 1.0:
            raise ValueError('dsp_ratio must be in [0, 1)')
        if self.step_width <= 0.0:
            raise ValueError('step_width must be positive')
        if self.swing_height < 0.0 or self.step_length < 0.0:
            raise ValueError('swing_height and step_length must be non-negative')
        if self.shift_time < 0.0 or self.hold_time < 0.0:
            raise ValueError('shift_time and hold_time must be non-negative')
        if min(self.init_time, self.final_time, self.stop_transition_time) <= 0.0:
            raise ValueError('init_time, final_time and stop_transition_time must be positive')
        if self.lookahead_steps < 2:
            raise ValueError('lookahead_steps must be at least 2')
