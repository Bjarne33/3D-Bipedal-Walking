"""ROS-independent helpers that turn kinematics and parameters into a gait plan."""

import dataclasses
from typing import Dict

import numpy as np

from humanoid_control.tools.footstep_planner import FixedFootstepPlanner
from humanoid_control.tools.gait_params import GaitParams
from humanoid_control.tools.state_estimator import axial_to_plane
from humanoid_control.tools.walking_pattern import WalkingPattern


class PlaneFrame:
    """Maps between the base frame and the gait plane frame (x forward, y left, z up)."""

    def __init__(self, forward, left) -> None:
        self.forward = np.asarray(forward, float) / np.linalg.norm(forward)
        self.left = np.asarray(left, float) / np.linalg.norm(left)
        self.up = np.array([0.0, 0.0, 1.0])

    def to_plane(self, p) -> np.ndarray:
        return np.array([p @ self.forward, p @ self.left, p @ self.up])

    def to_base(self, p) -> np.ndarray:
        return p[0] * self.forward + p[1] * self.left + p[2] * self.up


def plan_walk(standing, config, num_steps: int):
    """
    Build the gait plan from the standing kinematics and the node parameters.

    Inputs:
        standing: `KinematicsResult` of the standing pose (base frame).
        config: mapping with `forward_axis`, `first_swing`, `ankle_to_sole`, `step_width`,
            `com_height` (<= 0: derived from the standing pose) and the `GaitParams` fields.
        num_steps: number of steps, 0 walks until stopped.

    Result:
        (frame, params, planner, pattern)
    """
    left, right = standing.feet['left'].position, standing.feet['right'].position
    lateral = (left - right).copy()
    lateral[2] = 0.0
    frame = PlaneFrame(np.array(config['forward_axis'], float), lateral)

    values = {
        f.name: config[f.name]
        for f in dataclasses.fields(GaitParams)
        if f.name not in ('step_width', 'com_height') and f.name in config
    }
    width = float(config['step_width'])
    if width <= 0.0:
        width = abs((left - right) @ frame.left)
    height = float(config['com_height'])
    if height <= 0.0:
        height = standing.com[2] - (0.5 * (left[2] + right[2]) - float(config['ankle_to_sole']))
    params = GaitParams(step_width=width, com_height=height, **values)
    params.validate()

    feet0 = {s: frame.to_plane(standing.feet[s].position) for s in ('left', 'right')}
    planner = FixedFootstepPlanner(
        params, feet0['left'][:2], feet0['right'][:2],
        first_swing=str(config['first_swing']),
        num_steps=num_steps if num_steps > 0 else None,
    )
    com0 = frame.to_plane(standing.com)
    zero = np.zeros(2)
    pattern = WalkingPattern(
        params, planner, feet0['left'], feet0['right'],
        com_state=(com0[:2], zero, zero), com_z=com0[2],
    )
    return frame, params, planner, pattern


def rotation_joint_maps(kin, frame) -> Dict[str, Dict[str, tuple]]:
    """
    Per side and joint group: (joint names, M) with joint offsets dq = M @ r.

    r is a rotation [about forward, about left] of the body relative to the distal link
    (stance foot for the ankle group, stance thigh for the hip group). With the distal link
    held, a joint offset dq rotates the body by -dq about the joint axis, so M = -pinv(A)
    with the plane components of the joint axes as the columns of A. Uses `kin.q`.
    """
    maps = {}
    for side in ('left', 'right'):
        maps[side] = {}
        for group, names in (('ankle', ('ankle_roll', 'ankle_pitch')),
                             ('hip', ('hip_roll', 'hip_pitch'))):
            joints = [f'{side}_{n}' for n in names]
            axes = np.array([axial_to_plane(frame, a) for a in kin.joint_axes(joints)])
            maps[side][group] = (joints, -np.linalg.pinv(axes.T))
    return maps
