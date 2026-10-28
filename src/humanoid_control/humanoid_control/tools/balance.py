"""
Closed-loop balance stabiliser for humanoid_walk (pure Python, gait plane frame).

Given the walking reference and the measured state it computes
- the desired ZMP from DCM tracking: p_des = p_ref + K_xi (xi - xi_ref), clamped to the
  measured support polygon,
- a CoM offset for the IK target (CoM-ZMP admittance),
- a stance-ankle rotation (foot damping) that drives the measured CoP towards p_des,
- a stance-hip rotation that keeps the torso upright, and
- a footstep offset when the desired ZMP saturates (applied by the node).

Rotations are body-relative-to-support rotations in the plane frame, [about forward, about
left]; the node maps them onto joint offsets with the model's joint axes.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from humanoid_control.tools.support_polygon import clamp_to_hull

GRAVITY = 9.81
SIDES = ('left', 'right')


@dataclass
class BalanceEstimate:
    """Measured state in the plane frame (x forward, y left) of the walk."""

    com: np.ndarray
    com_vel: np.ndarray
    dcm: np.ndarray
    zmp: np.ndarray
    zmp_valid: bool
    support_hull: np.ndarray
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(2))
    tilt_rate: np.ndarray = field(default_factory=lambda: np.zeros(2))
    foot_tilt: Dict[str, np.ndarray] = field(default_factory=dict)
    foot_force: Dict[str, float] = field(default_factory=dict)


@dataclass
class BalanceOutput:
    zmp_des: np.ndarray = field(default_factory=lambda: np.zeros(2))
    dcm_error: np.ndarray = field(default_factory=lambda: np.zeros(2))
    saturated: bool = False
    com_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    ankle_rotation: Dict[str, np.ndarray] = field(default_factory=dict)
    torso_rotation: np.ndarray = field(default_factory=lambda: np.zeros(2))
    torso_side: Optional[str] = None
    step_adjust: np.ndarray = field(default_factory=lambda: np.zeros(2))


def cop_shift_to_rotation(dp: np.ndarray) -> np.ndarray:
    """
    Body rotation (about forward, about left) whose ankle torque moves the CoP by +dp.

    Tilting the body about +x (to the right) loads the left sole edge (CoP +y); tilting it
    about +y (forward) loads the heel (CoP -x).
    """
    return np.array([dp[1], -dp[0]])


class BalanceController:
    """DCM feedback + CoM admittance + foot damping + torso feedback."""

    def __init__(self, params) -> None:
        self.p = params
        self.omega = float(np.sqrt(GRAVITY / params.com_height))
        self.com_offset = np.zeros(2)
        self.ankle_rotation = {s: np.zeros(2) for s in SIDES}

    def reset(self) -> None:
        self.com_offset[:] = 0.0
        for s in SIDES:
            self.ankle_rotation[s][:] = 0.0

    def dcm_reference(self, ref) -> np.ndarray:
        return np.asarray(ref.com[:2], float) + np.asarray(ref.com_vel[:2], float) / self.omega

    def _leak(self, value: np.ndarray, tau: float, dt: float) -> np.ndarray:
        return value * max(0.0, 1.0 - dt / tau) if tau > 0.0 else value * 0.0

    def update(self, ref, est: BalanceEstimate, dt: float) -> BalanceOutput:
        p = self.p
        out = BalanceOutput(zmp_des=np.array(ref.zmp[:2], float))
        out.dcm_error = est.dcm - self.dcm_reference(ref)
        if not p.balance_enabled:
            return out

        k_xi = np.array([p.balance_k_xi_sagittal, p.balance_k_xi_lateral])
        zmp_des = np.asarray(ref.zmp[:2], float) + k_xi * out.dcm_error
        if len(est.support_hull) >= 3:
            zmp_des, out.saturated = clamp_to_hull(zmp_des, est.support_hull, p.balance_zmp_margin)
        out.zmp_des = zmp_des

        # CoM-ZMP admittance: moving the CoM target back moves the ZMP forward and vice versa.
        k_zmp = np.array([p.balance_k_zmp_sagittal, p.balance_k_zmp_lateral])
        k_com = np.array([p.balance_k_com_sagittal, p.balance_k_com_lateral])
        com_dot = -k_com * (est.com - np.asarray(ref.com[:2], float))
        # In double support the measured ZMP also reflects the internal force between the legs,
        # which the CoM shift excites (side-to-side load rocking), hence the separate scale.
        if est.zmp_valid:
            scale = 1.0 if ref.single_support else p.balance_zmp_dsp_scale
            com_dot -= scale * k_zmp * (zmp_des - est.zmp)
        self.com_offset = self._leak(self.com_offset + com_dot * dt, p.balance_com_leak_time, dt)
        limit = np.array([p.balance_com_offset_limit_sagittal, p.balance_com_offset_limit_lateral])
        self.com_offset = np.clip(self.com_offset, -limit, limit)
        out.com_offset = self.com_offset.copy()

        # Foot damping on the single stance foot; other feet relax back to the plan.
        stance = ref.stance_side if ref.single_support else None
        k_ankle = np.array([p.balance_k_ankle_roll, p.balance_k_ankle_pitch])
        for side in SIDES:
            rot = self.ankle_rotation[side]
            if side == stance and est.zmp_valid:
                rot = rot - k_ankle * cop_shift_to_rotation(est.zmp - zmp_des) * dt
            rot = self._leak(rot, p.balance_ankle_leak_time, dt)
            self.ankle_rotation[side] = np.clip(
                rot, -p.balance_ankle_offset_limit, p.balance_ankle_offset_limit)
            out.ankle_rotation[side] = self.ankle_rotation[side].copy()

        # Torso feedback through the stance hip (both hips in double support).
        out.torso_rotation = np.clip(
            -p.balance_k_torso_p * est.tilt - p.balance_k_torso_d * est.tilt_rate,
            -p.balance_hip_offset_limit, p.balance_hip_offset_limit)
        out.torso_side = stance

        if p.step_adjust_enabled and out.saturated and ref.swing_side:
            # The DCM error that the ZMP cannot absorb is caught by moving the next footstep.
            k = np.array([p.step_adjust_gain_sagittal, p.step_adjust_gain_lateral])
            adjust = k * out.dcm_error
            lim = p.step_adjust_limit_sagittal
            adjust[0] = np.clip(adjust[0], -lim, lim)
            sign = 1.0 if ref.swing_side == 'left' else -1.0  # +y is outwards for the left foot
            outward = np.clip(sign * adjust[1], -p.step_adjust_limit_lateral_in,
                              p.step_adjust_limit_lateral_out)
            adjust[1] = sign * outward
            if np.linalg.norm(adjust) >= p.step_adjust_min_norm:
                out.step_adjust = adjust
        return out
