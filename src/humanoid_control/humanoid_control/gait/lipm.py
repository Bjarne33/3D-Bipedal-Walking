"""Analytic linear inverted pendulum (LIPM) helpers using the divergent component of motion (DCM).

With omega = sqrt(g / z_c) and ZMP p the LIPM is  c'' = omega^2 (c - p).  The DCM
xi = c + c'/omega obeys xi' = omega (xi - p). Planning the DCM backwards from a terminal condition gives the initial
velocity c' = omega (xi - c) of every step; the CoM is then evaluated with the exact closed-form
solution for a ZMP that is constant or moves linearly in time.
All functions accept scalars or numpy arrays (e.g. the 2D plane position).
"""

import math
from typing import Sequence, Tuple

import numpy as np

from humanoid_control.gait.gait_params import GRAVITY


def omega(com_height: float) -> float:
    return math.sqrt(GRAVITY / com_height)


def dcm_back_const(xi_end, stance, duration: float, w: float):
    """DCM at the start of a segment with constant ZMP `stance`, given the DCM at its end."""
    return stance + (xi_end - stance) * math.exp(-w * duration)


def dcm_back_linear(xi_end, zmp_start, zmp_end, duration: float, w: float):
    """DCM at the start of a segment whose ZMP moves linearly from `zmp_start` to `zmp_end`."""
    if duration <= 0.0:
        return xi_end
    slope = (zmp_end - zmp_start) / duration
    return zmp_start + slope / w + (xi_end - zmp_end - slope / w) * math.exp(-w * duration)


def linear_zmp_state(c0, v0, zmp_start, slope, w: float, t: float) -> Tuple:
    """CoM state after `t` s with ZMP p(t) = zmp_start + slope * t (exact LIPM solution).

    Returns (position, velocity, acceleration, zmp).
    """
    d0 = c0 - zmp_start
    dv0 = v0 - slope
    ch, sh = math.cosh(w * t), math.sinh(w * t)
    zmp = zmp_start + slope * t
    pos = zmp + d0 * ch + dv0 / w * sh
    vel = slope + d0 * w * sh + dv0 * ch
    return pos, vel, w * w * (pos - zmp), zmp


def quintic(p0, v0, a0, p1, v1, a1, duration: float, t: float) -> Tuple:
    """Quintic polynomial through (p0, v0, a0) -> (p1, v1, a1); returns (pos, vel, acc) at `t`."""
    T = duration
    t = min(max(t, 0.0), T)
    p0, v0, a0, p1, v1, a1 = (np.asarray(x, float) for x in (p0, v0, a0, p1, v1, a1))
    c0, c1, c2 = p0, v0, 0.5 * a0
    # remaining coefficients from the end conditions
    A = np.array([[T**3, T**4, T**5], [3 * T**2, 4 * T**3, 5 * T**4], [6 * T, 12 * T**2, 20 * T**3]])
    rhs = np.stack([p1 - (c0 + c1 * T + c2 * T**2), v1 - (c1 + 2 * c2 * T), a1 - 2 * c2])
    c3, c4, c5 = np.linalg.solve(A, rhs)
    pos = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    vel = c1 + 2 * c2 * t + 3 * c3 * t**2 + 4 * c4 * t**3 + 5 * c5 * t**4
    acc = 2 * c2 + 6 * c3 * t + 12 * c4 * t**2 + 20 * c5 * t**3
    return pos, vel, acc
