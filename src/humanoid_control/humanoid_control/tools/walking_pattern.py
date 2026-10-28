"""
Walking pattern: combines the footstep plan, LIPM CoM and swing-foot trajectories.

Timeline of a pattern (all times relative to the start of the pattern):
  0. optional quasi-static lead-in (`shift_time` + `hold_time`): the CoM is moved slowly (quintic,
     at rest at both ends) above the first stance foot and held there before anything else happens.
  1. `pre` phase (`pre_time`): the CoM moves from its initial state to the state the plan needs at
     the start of the first step, driven by one constant ZMP (exact LIPM solution). Feet stay put.
  2. one segment of `step_time` per footstep. In the first `dsp_ratio` of the step (double
     support, swing foot on the ground) the ZMP moves linearly from the previous stance foot to the
     new stance foot; afterwards it stays at the stance foot while the swing foot moves.
  3. (finite plans only) `final` phase (`final_time`): quintic CoM settling to the feet centre.

Everything is expressed in the plane frame (x forward, y left, z up).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from humanoid_control.tools import lipm
from humanoid_control.tools.footstep_planner import FixedFootstepPlanner, Footstep
from humanoid_control.tools.gait_params import GaitParams
from humanoid_control.tools.swing_trajectory import SwingFootTrajectory


@dataclass
class WalkingReference:
    t: float
    com: np.ndarray  # (3,)
    com_vel: np.ndarray  # (3,)
    zmp: np.ndarray  # (2,)
    feet: Dict[str, np.ndarray]  # side -> (3,)
    feet_vel: Dict[str, np.ndarray]  # side -> (3,)
    phase: str  # 'shift', 'hold', 'init', 'step' or 'final'
    step_index: int  # -1 outside of the step phase
    stance_side: Optional[str]
    swing_side: Optional[str]
    finished: bool = False
    single_support: bool = False  # swing foot in the air (step phase after the double support)


@dataclass
class _Segment:
    c0: np.ndarray  # CoM at the start of the step
    v0: np.ndarray  # CoM velocity at the start of the step
    zmp_start: np.ndarray  # ZMP at the start of the step (previous stance foot)
    zmp_stance: np.ndarray  # ZMP during single support (stance foot)


class WalkingPattern:
    def __init__(
        self,
        params: GaitParams,
        planner: FixedFootstepPlanner,
        left_pos,
        right_pos,
        com_state: Optional[Tuple] = None,
        com_z: Optional[float] = None,
        pre_time: Optional[float] = None,
        lead_in: bool = True,
    ) -> None:
        """
        Set up the pattern.

        Inputs:
            left_pos, right_pos: current foot positions (3,) in the plane frame.
            com_state: (position(2), velocity(2), acceleration(2)) of the CoM at t = 0; defaults to
                a CoM at rest at the centre of the feet.
            com_z: constant CoM height reference (z coordinate); defaults to `params.com_height`.
            pre_time: duration of the initial transition; defaults to `params.init_time`.
            lead_in: apply the quasi-static shift/hold above the first stance foot.
        """
        params.validate()
        self.params = params
        self.planner = planner
        self.w = lipm.omega(params.com_height)
        self.feet0 = {
            'left': np.asarray(left_pos, float).reshape(3),
            'right': np.asarray(right_pos, float).reshape(3),
        }
        self.com_z = params.com_height if com_z is None else float(com_z)
        self.pre_time = params.init_time if pre_time is None else float(pre_time)
        self._offset = np.array([params.zmp_offset_forward, 0.0])
        self._mid_y = 0.5 * (self.feet0['left'][1] + self.feet0['right'][1])

        mid = 0.5 * (self.feet0['left'][:2] + self.feet0['right'][:2]) + self._offset
        zero = np.zeros(2)
        self.com_state = tuple(np.asarray(x, float)
                               for x in com_state) if com_state else (mid, zero, zero)

        self.shift_time = params.shift_time if lead_in else 0.0
        self.hold_time = params.hold_time if lead_in and params.shift_time > 0.0 else 0.0
        self.lead = self.shift_time + self.hold_time
        self._lead_start = self.com_state
        if self.shift_time > 0.0:
            self._hold_com = self._zmp(planner.step(0))
            if params.shift_lateral > 0.0:
                dy = self._hold_com[1] - self.com_state[0][1]
                self._hold_com = self._hold_com.copy()
                self._hold_com[1] = self.com_state[0][1] + \
                    float(np.clip(dy, -params.shift_lateral, params.shift_lateral))
            self.com_state = (self._hold_com, zero, zero)

        self.step_time = params.step_time
        self.dsp_time = params.dsp_ratio * params.step_time
        self.num_steps = planner.total_steps  # None: endless
        self._segments: List[_Segment] = []

        # Initial phase: constant ZMP p* chosen so that the CoM arrives with exactly the state the
        # plan needs at the start of step 0. That state depends on p* (the DSP of step 0 starts at
        # the ZMP of the initial phase), hence the short fixed-point iteration.
        c0, v0 = self.com_state[0], self.com_state[1]
        xi0 = c0 + v0 / self.w
        growth = math.exp(self.w * self.pre_time)
        self._pre_zmp = c0.copy()
        for _ in range(8):
            self._start_zmp = self._pre_zmp
            xi_target = self._xi_start(0)
            self._pre_zmp = (xi_target - xi0 * growth) / (1.0 - growth)
        self._start_zmp = self._pre_zmp
        c_start, v_start, _, _ = lipm.linear_zmp_state(
            c0, v0, self._pre_zmp, np.zeros(2), self.w, self.pre_time)
        self._segments.append(
            _Segment(
                c_start,
                v_start,
                self._start_zmp,
                self._zmp(
                    planner.step(0))))

        self._final_start: Optional[Tuple] = None
        self._final_target: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ planning helpers
    def _zmp(self, step: Footstep) -> np.ndarray:
        zmp = step.stance + self._offset
        if self.params.zmp_lateral > 0.0:
            zmp = zmp.copy()
            sign = 1.0 if step.stance_side == 'left' else -1.0
            zmp[1] = self._mid_y + sign * self.params.zmp_lateral
        return zmp

    def _zmp_start(self, k: int) -> np.ndarray:
        return self._start_zmp if k == 0 else self._zmp(self.planner.step(k - 1))

    def _final_com(self) -> np.ndarray:
        last = self.planner.step(self.num_steps - 1)
        return 0.5 * (last.stance + last.swing_target) + self._offset

    def _xi_start(self, k: int) -> np.ndarray:
        """DCM at the start of step k, planned backwards over a limited preview of later steps."""
        last = k + self.params.lookahead_steps
        finite_end = self.num_steps is not None and last >= self.num_steps - 1
        if finite_end:
            last = self.num_steps - 1
        xi = self._final_com() if finite_end else self._zmp(self.planner.step(last))
        for j in range(last, k - 1, -1):
            stance = self._zmp(self.planner.step(j))
            xi = lipm.dcm_back_const(xi, stance, self.step_time - self.dsp_time, self.w)
            xi = lipm.dcm_back_linear(xi, self._zmp_start(j), stance, self.dsp_time, self.w)
        return xi

    def _eval(self, seg: _Segment, tau: float) -> Tuple:
        """Evaluate the CoM position, velocity, acceleration and ZMP `tau` s into a step."""
        tau = min(max(tau, 0.0), self.step_time)
        td = self.dsp_time
        if td > 0.0:
            slope = (seg.zmp_stance - seg.zmp_start) / td
            c, v, a, zmp = lipm.linear_zmp_state(
                seg.c0, seg.v0, seg.zmp_start, slope, self.w, min(tau, td))
        else:
            c, v, a, zmp = seg.c0, seg.v0, np.zeros(2), seg.zmp_stance
        if tau > td:
            c, v, a, zmp = lipm.linear_zmp_state(
                c, v, seg.zmp_stance, np.zeros(2), self.w, tau - td)
        return c, v, a, zmp

    def _segment(self, k: int) -> _Segment:
        while len(self._segments) <= k:
            prev = self._segments[-1]
            c_end, _, _, _ = self._eval(prev, self.step_time)
            idx = len(self._segments)
            v_start = self.w * (self._xi_start(idx) - c_end)
            self._segments.append(
                _Segment(c_end, v_start, prev.zmp_stance, self._zmp(self.planner.step(idx)))
            )
        return self._segments[k]

    def _step_state(self, k: int, tau: float):
        return self._eval(self._segment(k), tau)

    # ------------------------------------------------------------------ timeline
    @property
    def duration(self) -> Optional[float]:
        if self.num_steps is None:
            return None
        return self.lead + self.pre_time + self.num_steps * self.step_time + self.params.final_time

    def step_start_time(self, k: int) -> float:
        return self.lead + self.pre_time + k * self.step_time

    def step_index_at(self, t: float) -> int:
        """Index of the active step at `t` (-1 initial phase, `num_steps` after the steps)."""
        t -= self.lead
        if t < self.pre_time:
            return -1
        k = int((t - self.pre_time) // self.step_time)
        return k if self.num_steps is None else min(k, self.num_steps)

    def _foot_positions(self, step: Footstep, tau: float):
        z = {s: self.feet0[s][2] for s in ('left', 'right')}
        swing_duration = (1.0 - self.params.dsp_ratio) * self.step_time
        traj = SwingFootTrajectory(
            np.append(step.swing_start, z[step.swing_side]),
            np.append(step.swing_target, z[step.swing_side]),
            self.params.swing_height,
            swing_duration,
        )
        swing_pos, swing_vel = traj.sample(tau - self.params.dsp_ratio * self.step_time)
        stance_pos = np.append(step.stance, z[step.stance_side])
        feet = {step.stance_side: stance_pos, step.swing_side: swing_pos}
        vels = {step.stance_side: np.zeros(3), step.swing_side: swing_vel}
        return feet, vels

    def _resting_feet(self, left_xy, right_xy):
        feet = {
            'left': np.append(left_xy, self.feet0['left'][2]),
            'right': np.append(right_xy, self.feet0['right'][2]),
        }
        return feet, {'left': np.zeros(3), 'right': np.zeros(3)}

    def sample(self, t: float) -> WalkingReference:
        t = max(t, 0.0)
        t_abs = t
        p = self.params
        finished = False
        step_index = -1
        stance_side = swing_side = None
        single_support = False

        if t < self.lead:
            phase = 'shift' if t < self.shift_time else 'hold'
            c, v, a = lipm.quintic(
                *self._lead_start, self._hold_com, np.zeros(2), np.zeros(2),
                self.shift_time, min(t, self.shift_time),
            )
            zmp = c - a / self.w**2
            first = self.planner.step(0)
            xy = {first.stance_side: first.stance, first.swing_side: first.swing_start}
            feet, feet_vel = self._resting_feet(xy['left'], xy['right'])
            return WalkingReference(
                t=t_abs, com=np.append(c, self.com_z), com_vel=np.append(v, 0.0),
                zmp=np.asarray(zmp),
                feet=feet, feet_vel=feet_vel, phase=phase, step_index=-1,
                stance_side=None, swing_side=None,
            )
        t -= self.lead

        if t < self.pre_time:
            phase = 'init'
            c, v, a, zmp = lipm.linear_zmp_state(
                self.com_state[0], self.com_state[1], self._pre_zmp, np.zeros(2), self.w, t)
            first = self.planner.step(0)
            xy = {first.stance_side: first.stance, first.swing_side: first.swing_start}
            feet, feet_vel = self._resting_feet(xy['left'], xy['right'])
        else:
            k = int((t - self.pre_time) // self.step_time)
            if self.num_steps is not None and k >= self.num_steps:
                phase = 'final'
                if self._final_start is None:
                    self._final_start = self._step_state(self.num_steps - 1, self.step_time)[:3]
                    self._final_target = self._final_com()
                tf = t - self.pre_time - self.num_steps * self.step_time
                c, v, a = lipm.quintic(
                    *self._final_start, self._final_target, np.zeros(2), np.zeros(2),
                    p.final_time, tf,
                )
                last = self.planner.step(self.num_steps - 1)
                xy = {last.stance_side: last.stance, last.swing_side: last.swing_target}
                feet, feet_vel = self._resting_feet(xy['left'], xy['right'])
                zmp = c - a / self.w**2
                finished = tf >= p.final_time
            else:
                phase = 'step'
                step = self.planner.step(k)
                tau = t - k * self.step_time - self.pre_time
                c, v, a, zmp = self._step_state(k, tau)
                feet, feet_vel = self._foot_positions(step, tau)
                step_index, stance_side, swing_side = k, step.stance_side, step.swing_side
                single_support = tau >= self.dsp_time

        return WalkingReference(
            t=t_abs,
            com=np.append(c, self.com_z),
            com_vel=np.append(v, 0.0),
            zmp=np.asarray(zmp),
            feet=feet,
            feet_vel=feet_vel,
            phase=phase,
            step_index=step_index,
            stance_side=stance_side,
            swing_side=swing_side,
            finished=finished,
            single_support=single_support,
        )

    # ------------------------------------------------------------------ stopping an endless walk
    def stop_pattern(self, k: int) -> 'WalkingPattern':
        """
        Create a new finite pattern that starts at the beginning of step `k` and closes the feet.

        The new pattern's time zero corresponds to `step_start_time(k)` of this pattern, and its
        CoM starts from the exact CoM state there (smooth hand-over).
        """
        step = self.planner.step(k)
        seg = self._segment(k)
        v0, a0 = seg.v0, self.w**2 * (seg.c0 - seg.zmp_start)
        xy = {step.stance_side: step.stance, step.swing_side: step.swing_start}
        planner = FixedFootstepPlanner(
            self.params, xy['left'], xy['right'], first_swing=step.swing_side, num_steps=0
        )
        feet = {s: np.append(xy[s], self.feet0[s][2]) for s in ('left', 'right')}
        return WalkingPattern(
            self.params, planner, feet['left'], feet['right'],
            com_state=(seg.c0, v0, a0), com_z=self.com_z,
            pre_time=self.params.stop_transition_time, lead_in=False,
        )
