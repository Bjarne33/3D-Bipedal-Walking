"""Gait parameters shared by the footstep planner and the trajectory generators."""

from dataclasses import dataclass

GRAVITY = 9.81


@dataclass
class GaitParams:
    """
    All lengths in metres, times in seconds.

    The gait is planned in a 2D plane frame (x forward, y towards the left foot) with z up.
    """

    step_length: float = 0.10
    step_width: float = 0.25  # lateral distance between the feet (ankle frames)
    step_time: float = 0.8
    dsp_ratio: float = 0.2  # fraction of a step in which the swing foot stays on the ground
    swing_height: float = 0.05
    com_height: float = 0.75  # CoM height above the ground used for the LIPM
    # ZMP / CoM reference offset from the ankle frame (foot centre)
    zmp_offset_forward: float = 0.0
    init_time: float = 1.5  # CoM shift from the standing state onto the first stance foot
    # quasi-static CoM shift above the first stance foot (0 disables the lead-in)
    shift_time: float = 0.0
    hold_time: float = 0.0  # pause above the first stance foot before the dynamic init phase
    # max lateral CoM travel in the quasi-static shift (0 = full shift onto the foot)
    shift_lateral: float = 0.0
    # lateral ZMP distance from the body midline during single support (0 = stance foot centre)
    zmp_lateral: float = 0.08
    final_time: float = 1.5  # CoM settling to the centre of the feet after the last step
    stop_transition_time: float = 0.8  # blend time used when a running walk is stopped
    lookahead_steps: int = 6  # steps of preview used for the DCM end-of-step values
    # Closed-loop balance (tools/balance.py); omega is derived from com_height.
    balance_enabled: bool = True
    balance_state_alpha: float = 0.5  # low-pass weight of the measured CoM velocity (1 = raw)
    balance_zmp_margin: float = 0.02  # desired ZMP is kept this far inside the support polygon
    balance_k_xi_lateral: float = 2.0  # p_des = p_ref + k_xi (xi - xi_ref), i.e. 1 + k / omega
    balance_k_xi_sagittal: float = 2.0
    balance_k_zmp_lateral: float = 4.0  # CoM admittance on the ZMP error [1/s]
    balance_k_zmp_sagittal: float = 4.0
    balance_zmp_dsp_scale: float = 1.0  # k_zmp factor in double support
    balance_k_com_lateral: float = 3.0  # CoM admittance on the CoM error [1/s]
    balance_k_com_sagittal: float = 3.0
    balance_com_leak_time: float = 2.0  # CoM offset decay time constant [s]
    balance_com_offset_limit_lateral: float = 0.06
    balance_com_offset_limit_sagittal: float = 0.06
    balance_k_ankle_roll: float = 0.0  # ankle rotation rate per CoP error [rad/(m s)]
    balance_k_ankle_pitch: float = 0.0
    balance_ankle_leak_time: float = 1.0  # ankle offset decay time constant [s]
    balance_ankle_offset_limit: float = 0.10  # [rad]
    balance_k_torso_p: float = 0.6  # stance hip rotation per torso tilt [rad/rad]
    balance_k_torso_d: float = 0.06  # [rad/(rad/s)]
    balance_hip_offset_limit: float = 0.10  # [rad]
    step_adjust_enabled: bool = False
    step_adjust_gain_lateral: float = 1.0
    step_adjust_gain_sagittal: float = 1.0
    step_adjust_limit_lateral_out: float = 0.08
    step_adjust_limit_lateral_in: float = 0.04
    step_adjust_limit_sagittal: float = 0.08
    step_adjust_lock_time: float = 0.20
    step_adjust_min_norm: float = 0.01

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
        if not 0.0 <= self.balance_state_alpha <= 1.0:
            raise ValueError('balance_state_alpha must be in [0, 1]')
        if not 0.0 <= self.balance_zmp_dsp_scale <= 1.0:
            raise ValueError('balance_zmp_dsp_scale must be in [0, 1]')
        if min(self.balance_com_leak_time, self.balance_ankle_leak_time) <= 0.0:
            raise ValueError('balance_com_leak_time and balance_ankle_leak_time must be positive')
        if min(
            self.balance_com_offset_limit_lateral,
            self.balance_com_offset_limit_sagittal,
            self.balance_ankle_offset_limit,
            self.balance_hip_offset_limit,
            self.step_adjust_limit_lateral_out,
            self.step_adjust_limit_lateral_in,
            self.step_adjust_limit_sagittal,
        ) < 0.0:
            raise ValueError('balance and step-adjust limits must be non-negative')
        if min(self.step_adjust_lock_time, self.step_adjust_min_norm) < 0.0:
            raise ValueError('step_adjust_lock_time and step_adjust_min_norm must be non-negative')
