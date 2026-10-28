"""Offline check of the walking pipeline: footstep plan -> LIPM pattern -> Pink IK.

Builds the pattern exactly like `humanoid_walk` (same parameters file, standing pose from
`humanoid_startup`) and runs the IK at the node rate, without ROS or Gazebo. Reports per phase
how far the reference ZMP and CoM stay inside the support polygon of the grounded soles (sole
boxes from the URDF), how well the IK reaches the CoM target and how much the torso tilts.

    ros2 run humanoid_control gait_check                    # installed gait.yaml, 2 steps
    ros2 run humanoid_control gait_check --steps 4 --set step_time=1.0 --set zmp_lateral=0.06 -v

Exits with status 1 if the ZMP leaves the support polygon or the IK CoM error exceeds `--max-ik-error`.
"""

import argparse
import os
import sys

import numpy as np
import pinocchio as pin
import yaml
from ament_index_python.packages import get_package_share_directory

from humanoid_control.gait.support_polygon import convex_hull, signed_margin
from humanoid_control.humanoid_kinematics import HumanoidKinematics
from humanoid_control.humanoid_startup import STANDING_POSE
from humanoid_control.humanoid_walk import JOINT_NAMES, plan_walk

GROUND_TOLERANCE = 1e-4  # a foot counts as grounded when within this height of its standing height


def _parse_args(argv):
    share = get_package_share_directory('humanoid_control')
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', default=os.path.join(share, 'config', 'gait.yaml'))
    parser.add_argument('--steps', type=int, default=2, help='number of steps (> 0)')
    parser.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                        help='override a parameter, e.g. --set shift_lateral=0.06')
    parser.add_argument('--max-ik-error', type=float, default=0.005, help='[m]')
    parser.add_argument('-v', '--verbose', action='store_true', help='print a row every 0.2 s')
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error('--steps must be positive')
    return args


def _load_config(path, overrides):
    with open(path) as f:
        config = yaml.safe_load(f)['humanoid_walk']['ros__parameters']
    for item in overrides:
        name, _, value = item.partition('=')
        if name not in config:
            raise SystemExit(f"unknown parameter '{name}'")
        config[name] = yaml.safe_load(value)
    return config


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config = _load_config(args.config, args.set)

    kin = HumanoidKinematics.from_xacro(
        os.path.join(get_package_share_directory('humanoid_description'), 'urdf', 'humanoid.xacro'))
    kin.set_joint_positions(JOINT_NAMES, [STANDING_POSE[n] for n in JOINT_NAMES])
    kin.reset_ik()
    standing = kin.compute()
    frame, params, planner, pattern = plan_walk(standing, config, args.steps)

    sides = ('left', 'right')
    rotations = {s: standing.feet[s].rotation for s in sides}
    feet_z0 = {s: frame.to_plane(standing.feet[s].position)[2] for s in sides}
    # sole corners relative to the ankle, in the plane frame (feet keep their standing orientation)
    soles = {s: np.array([frame.to_plane(rotations[s] @ c) for c in kin.sole_corners[s]]) for s in sides}
    torso_id = kin.model.getFrameId('torso_link')
    torso0 = kin.data.oMf[torso_id].rotation.copy()  # standing torso orientation (after compute())

    print(f'step_width={params.step_width:.3f} com_height={params.com_height:.3f} '
          f'step_time={params.step_time} dsp_ratio={params.dsp_ratio}')
    for i in range(planner.total_steps):
        s = planner.step(i)
        print(f'  step {i}: swing {s.swing_side:5} stance={np.round(s.stance, 3)} '
              f'{np.round(s.swing_start, 3)} -> {np.round(s.swing_target, 3)}')

    dt = 1.0 / float(config['update_rate_hz'])
    iterations = int(config['ik_iterations'])
    stats = {}
    if args.verbose:
        print(f'{"t":>6} {"phase":6} {"k":>2} {"zmp":>16} {"com_ref":>16} {"zmp_m":>7} {"com_m":>7} '
              f'{"ik_err":>7} {"tilt":>5} support')
    for i, t in enumerate(np.arange(0.0, pattern.duration + dt, dt)):
        ref = pattern.sample(t)
        poses = {s: pin.SE3(rotations[s], frame.to_base(ref.feet[s])) for s in sides}
        for _ in range(iterations):
            kin.compute_ik_step(kin.foot_frames['left'], poses['left'], kin.foot_frames['right'],
                                poses['right'], frame.to_base(ref.com), dt)
        com_ik = frame.to_plane(kin.compute().com)
        torso = kin.data.oMf[torso_id].rotation
        tilt = np.degrees(np.linalg.norm(pin.log3(torso0.T @ torso)))

        grounded = [s for s in sides if abs(ref.feet[s][2] - feet_z0[s]) < GROUND_TOLERANCE]
        hull = convex_hull(np.vstack([ref.feet[s][:2] + soles[s][:, :2] for s in grounded]))
        zmp_m = signed_margin(ref.zmp, hull)
        com_m = signed_margin(ref.com[:2], hull)
        ik_err = float(np.linalg.norm(com_ik[:2] - ref.com[:2]))

        st = stats.setdefault(ref.phase, [np.inf, np.inf, 0.0, 0.0])
        st[0], st[1] = min(st[0], zmp_m), min(st[1], com_m)
        st[2], st[3] = max(st[2], ik_err), max(st[3], tilt)
        if (args.verbose and i % max(1, round(0.2 / dt)) == 0) or zmp_m < 0.0:
            support = 'double' if len(grounded) == 2 else f'{grounded[0]} only'
            print(f'{t:6.2f} {ref.phase:6} {ref.step_index:2d} {np.round(ref.zmp, 3)!s:>16} '
                  f'{np.round(ref.com[:2], 3)!s:>16} {zmp_m:7.3f} {com_m:7.3f} {ik_err:7.4f} '
                  f'{tilt:5.1f} {support}' + ('  <-- ZMP outside support' if zmp_m < 0.0 else ''))

    print('\nper phase (margins > 0: inside the support polygon) [m, deg]')
    print(f'  {"phase":6} {"min ZMP margin":>14} {"min CoM margin":>14} {"max IK error":>12} {"max tilt":>8}')
    for phase, (zmp_m, com_m, ik_err, tilt) in stats.items():
        print(f'  {phase:6} {zmp_m:14.3f} {com_m:14.3f} {ik_err:12.4f} {tilt:8.1f}')
    print('  (CoM margin < 0 is fine while walking dynamically; the ZMP margin must stay > 0)')

    zmp_ok = min(s[0] for s in stats.values()) >= 0.0
    ik_ok = max(s[2] for s in stats.values()) <= args.max_ik_error
    print(f'\nZMP inside support: {"OK" if zmp_ok else "FAIL"}   '
          f'IK CoM error <= {args.max_ik_error} m: {"OK" if ik_ok else "FAIL"}')
    return 0 if zmp_ok and ik_ok else 1


if __name__ == '__main__':
    sys.exit(main())
