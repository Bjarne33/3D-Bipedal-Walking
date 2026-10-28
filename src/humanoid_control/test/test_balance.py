import numpy as np
import pinocchio as pin
import pytest

from humanoid_control.gait.balance import (
    BalanceController,
    BalanceEstimate,
    cop_shift_to_rotation,
)
from humanoid_control.gait.gait_params import GaitParams
from humanoid_control.gait.state_estimator import (
    axial_to_plane,
    BalanceStateEstimator,
    cop_from_ankle_wrench,
)

HULL = np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])


class Ref:
    def __init__(self, stance='left', swing='right'):
        self.com = np.array([0.0, 0.0, 0.7])
        self.com_vel = np.zeros(3)
        self.zmp = np.zeros(2)
        self.stance_side = stance
        self.swing_side = swing
        self.single_support = swing is not None


def _estimate(**kw):
    values = dict(com=np.zeros(2), com_vel=np.zeros(2), dcm=np.zeros(2), zmp=np.zeros(2),
                  zmp_valid=True, support_hull=HULL)
    values.update(kw)
    return BalanceEstimate(**values)


def _params(**kw):
    return GaitParams(com_height=0.7, balance_enabled=True, **kw)


def test_disabled_controller_reports_error_but_does_not_act():
    out = BalanceController(GaitParams(com_height=0.7, balance_enabled=False)).update(
        Ref(), _estimate(dcm=np.array([0.0, 0.05])), 0.01)
    assert out.dcm_error[1] == pytest.approx(0.05)
    np.testing.assert_allclose(out.zmp_des, 0.0)
    np.testing.assert_allclose(out.com_offset, 0.0)
    assert not out.ankle_rotation


def test_zero_error_keeps_zero_output():
    out = BalanceController(_params()).update(Ref(), _estimate(), 0.01)
    np.testing.assert_allclose(out.com_offset, 0.0)
    np.testing.assert_allclose(out.zmp_des, 0.0)
    np.testing.assert_allclose(out.ankle_rotation['left'], 0.0)
    np.testing.assert_allclose(out.torso_rotation, 0.0)
    assert not out.saturated


def test_omega_and_dcm_reference_follow_com_height():
    ctrl = BalanceController(_params())
    assert ctrl.omega == pytest.approx(np.sqrt(9.81 / 0.7))
    ref = Ref()
    ref.com_vel = np.array([0.2, -0.1, 0.0])
    np.testing.assert_allclose(ctrl.dcm_reference(ref), ref.com_vel[:2] / ctrl.omega)


def test_dcm_error_moves_desired_zmp_and_clamps():
    params = _params(balance_k_xi_lateral=2.0, balance_k_xi_sagittal=1.5, balance_zmp_margin=0.0)
    out = BalanceController(params).update(Ref(), _estimate(dcm=np.array([0.02, 0.01])), 0.01)
    np.testing.assert_allclose(out.zmp_des, [0.03, 0.02])
    assert not out.saturated
    out = BalanceController(params).update(Ref(), _estimate(dcm=np.array([0.0, 0.2])), 0.01)
    assert out.saturated
    assert out.zmp_des[1] == pytest.approx(0.1)


def test_admittance_moves_com_against_zmp_and_com_errors():
    # Desired ZMP ahead of the measured ZMP -> accelerate the CoM back (offset < 0).
    params = _params(balance_k_xi_sagittal=1.0, balance_k_com_lateral=0.0,
                     balance_k_com_sagittal=0.0, balance_k_zmp_sagittal=2.0)
    ref = Ref()
    ref.zmp = np.array([0.05, 0.0])
    out = BalanceController(params).update(ref, _estimate(), 0.1)
    assert out.com_offset[0] < 0.0
    # Double support: the ZMP term is scaled by balance_zmp_dsp_scale.
    dsp = Ref('left', None)
    dsp.zmp = np.array([0.05, 0.0])
    single = out.com_offset[0]
    params.balance_zmp_dsp_scale = 0.5
    out = BalanceController(params).update(dsp, _estimate(), 0.1)
    assert out.com_offset[0] == pytest.approx(0.5 * single)
    params.balance_zmp_dsp_scale = 0.0
    out = BalanceController(params).update(dsp, _estimate(), 0.1)
    np.testing.assert_allclose(out.com_offset, 0.0)
    # Measured CoM left of the reference -> shift the CoM target right.
    params = _params(balance_k_zmp_lateral=0.0, balance_k_zmp_sagittal=0.0,
                     balance_k_com_lateral=2.0, balance_k_com_sagittal=2.0,
                     balance_k_xi_lateral=1.0, balance_k_xi_sagittal=1.0)
    out = BalanceController(params).update(Ref(), _estimate(com=np.array([0.0, 0.01])), 0.1)
    assert out.com_offset[1] < 0.0 and out.com_offset[0] == pytest.approx(0.0)


def test_com_offset_is_limited_and_leaks():
    params = _params(balance_k_com_lateral=100.0, balance_com_offset_limit_lateral=0.01,
                     balance_com_leak_time=0.5)
    ctrl = BalanceController(params)
    out = ctrl.update(Ref(), _estimate(com=np.array([0.0, 1.0]), zmp_valid=False), 0.1)
    assert out.com_offset[1] == pytest.approx(-0.01)
    params.balance_k_com_lateral = 0.0
    out = ctrl.update(Ref(), _estimate(zmp_valid=False), 0.1)
    assert -0.01 < out.com_offset[1] < 0.0


def test_cop_shift_to_rotation_signs():
    # Tilting the body about +x (right) loads the left edge; about +y (forward) the heel.
    np.testing.assert_allclose(cop_shift_to_rotation(np.array([0.0, 1.0])), [1.0, 0.0])
    np.testing.assert_allclose(cop_shift_to_rotation(np.array([1.0, 0.0])), [0.0, -1.0])


def test_foot_damping_moves_cop_towards_desired_on_stance_foot_only():
    params = _params(balance_k_ankle_roll=5.0, balance_k_ankle_pitch=5.0,
                     balance_k_com_lateral=0.0, balance_k_com_sagittal=0.0,
                     balance_k_zmp_lateral=0.0, balance_k_zmp_sagittal=0.0,
                     balance_k_xi_lateral=1.0, balance_k_xi_sagittal=1.0)
    # Measured CoP too far left and forward of the desired ZMP.
    est = _estimate(zmp=np.array([0.02, 0.04]))
    out = BalanceController(params).update(Ref(), est, 0.1)
    rot = out.ankle_rotation['left']
    shift = np.array([-rot[1], rot[0]])  # inverse of cop_shift_to_rotation
    assert shift[0] < 0.0 and shift[1] < 0.0
    np.testing.assert_allclose(out.ankle_rotation['right'], 0.0)
    # Double support: no foot damping.
    out = BalanceController(params).update(Ref('left', None), est, 0.1)
    np.testing.assert_allclose(out.ankle_rotation['left'], 0.0)


def test_torso_feedback_opposes_tilt_and_is_limited():
    params = _params(balance_k_torso_p=10.0, balance_k_torso_d=10.0, balance_hip_offset_limit=0.1)
    out = BalanceController(params).update(Ref(), _estimate(tilt=np.array([0.05, -0.05])), 0.01)
    np.testing.assert_allclose(out.torso_rotation, [-0.1, 0.1])
    assert out.torso_side == 'left'


def test_step_adjustment_only_when_enabled_and_saturated():
    params = _params(balance_k_xi_lateral=4.0, step_adjust_gain_lateral=1.0,
                     step_adjust_limit_lateral_out=0.08, step_adjust_limit_lateral_in=0.02)
    est = _estimate(dcm=np.array([0.0, -0.2]))
    out = BalanceController(params).update(Ref(), est, 0.01)
    assert out.saturated and not np.any(out.step_adjust)
    params.step_adjust_enabled = True
    out = BalanceController(params).update(Ref(), est, 0.01)
    assert out.step_adjust[1] == pytest.approx(-0.08)  # right swing foot moves outwards (-y)
    out = BalanceController(params).update(Ref('right', 'left'), est, 0.01)
    assert out.step_adjust[1] == pytest.approx(-0.02)  # left swing foot moves inwards


@pytest.mark.parametrize('sign', [1.0, -1.0])
def test_cop_from_ankle_wrench_is_sign_independent(sign):
    force = np.array([10.0, -5.0, 300.0])
    cop = np.array([0.02, 0.03, -0.05])
    torque = np.cross(cop, force)
    np.testing.assert_allclose(cop_from_ankle_wrench(sign * force, sign * torque, 0.05), cop[:2])
    assert cop_from_ankle_wrench(np.array([0.0, 0.0, 5.0]), np.zeros(3), 0.05) is None


def test_estimator_averages_wrench_samples_between_estimates():
    est = BalanceStateEstimator()
    est.set_wrench('left', [0.0, 0.0, 100.0], [1.0, 0.0, 0.0])
    est.set_wrench('left', [0.0, 0.0, 300.0], [3.0, 0.0, 0.0])
    est._consume_wrenches()
    np.testing.assert_allclose(est.wrenches['left'].force, [0.0, 0.0, 200.0])
    np.testing.assert_allclose(est.wrenches['left'].torque, [2.0, 0.0, 0.0])
    est._consume_wrenches()  # no new samples: keep the last mean
    np.testing.assert_allclose(est.wrenches['left'].force, [0.0, 0.0, 200.0])
    est.set_wrench('left', [0.0, 0.0, 50.0], np.zeros(3))
    est._consume_wrenches()
    np.testing.assert_allclose(est.wrenches['left'].force, [0.0, 0.0, 50.0])


# ---------------------------------------------------------------- with the robot model
@pytest.fixture(scope='module')
def model():
    pytest.importorskip('pink')
    try:
        from ament_index_python.packages import get_package_share_directory
        import os
        share = get_package_share_directory('humanoid_description')
    except Exception:  # noqa: BLE001 - package not built / sourced
        pytest.skip('humanoid_description not installed')
    from humanoid_control.humanoid_kinematics import HumanoidKinematics
    from humanoid_control.humanoid_startup import STANDING_POSE
    kin = HumanoidKinematics.from_xacro(os.path.join(share, 'urdf', 'humanoid.xacro'))
    kin.set_joint_positions(STANDING_POSE.keys(), STANDING_POSE.values())
    standing = kin.compute()
    return kin, standing, STANDING_POSE


class Frame:
    def __init__(self, standing):
        lateral = standing.feet['left'].position - standing.feet['right'].position
        lateral[2] = 0.0
        self.forward = np.array([0.0, 1.0, 0.0])
        self.left = lateral / np.linalg.norm(lateral)
        self.up = np.array([0.0, 0.0, 1.0])

    def to_plane(self, p):
        return np.array([p @ self.forward, p @ self.left, p @ self.up])


def test_estimator_does_not_touch_ik_state(model):
    kin, standing, pose = model
    frame = Frame(standing)
    q_before = kin.q.copy()
    est = BalanceStateEstimator(alpha=1.0)
    # Tilt the body to its right (towards the right foot) by 0.1 rad about the forward axis.
    fwd = frame.forward
    rot = pin.exp3(0.1 * fwd * np.sign(np.cross(frame.forward, frame.left) @ frame.up))
    up = rot @ np.array([0.0, 0.0, 1.0])
    assert frame.to_plane(up)[1] < 0.0
    est.set_base(pin.SE3.Identity(), 0.0)
    est.set_walk_reference_frame()
    est.set_base(pin.SE3(rot, np.zeros(3)), 0.0, [0.0, 0.1, 0.0], [0.0, 0.0, 0.0])
    est.set_joint_state(list(pose), list(pose.values()), [0.0] * len(pose))
    omega = np.sqrt(9.81 / 0.7)
    out = est.estimate(kin, frame, -10.0, 0.005, omega, 0.05)
    np.testing.assert_array_equal(kin.q, q_before)
    np.testing.assert_allclose(out.tilt, [0.1, 0.0], atol=1e-6)
    com_world = rot @ standing.com
    np.testing.assert_allclose(out.com, frame.to_plane(com_world)[:2], atol=1e-9)
    np.testing.assert_allclose(out.com_vel, [0.1, 0.0], atol=1e-9)
    np.testing.assert_allclose(out.dcm, out.com + out.com_vel / omega)
    assert not out.zmp_valid  # no FT data and the feet are far above the ground


def test_estimator_support_and_zmp_come_from_loaded_feet(model):
    kin, standing, pose = model
    frame = Frame(standing)
    est = BalanceStateEstimator()
    est.set_base(pin.SE3.Identity(), 0.0)
    est.set_walk_reference_frame()
    est.set_joint_state(list(pose), list(pose.values()))
    feet = standing.feet
    ground = min((feet[s].position + feet[s].rotation @ c)[2]
                 for s in ('left', 'right') for c in kin.sole_corners[s])
    # Left foot carries the weight at its ankle; the right foot is (nearly) unloaded.
    est.set_wrench('left', [0.0, 0.0, 300.0], np.zeros(3))
    est.set_wrench('right', [0.0, 0.0, 5.0], np.zeros(3))
    out = est.estimate(kin, frame, ground, 0.005, 3.7, 0.05)
    left = frame.to_plane(standing.feet['left'].position)[:2]
    right = frame.to_plane(standing.feet['right'].position)[:2]
    assert out.zmp_valid
    # CoP only from the loaded foot (its sole point below the ankle; the foot is not quite level)
    np.testing.assert_allclose(out.zmp, left, atol=2e-3)
    assert out.support_hull[:, 1].min() > right[1]  # the unloaded foot is not support
    est.set_wrench('right', [0.0, 0.0, 300.0], np.zeros(3))
    out = est.estimate(kin, frame, ground, 0.005, 3.7, 0.05)
    assert out.support_hull[:, 1].min() < right[1] < left[1] < out.support_hull[:, 1].max()
    np.testing.assert_allclose(out.zmp, 0.5 * (left + right), atol=2e-3)
    # Without FT data, the feet touching the ground are support.
    est = BalanceStateEstimator()
    est.set_base(pin.SE3.Identity(), 0.0)
    est.set_walk_reference_frame()
    est.set_joint_state(list(pose), list(pose.values()))
    out = est.estimate(kin, frame, ground, 0.005, 3.7, 0.05)
    assert not out.zmp_valid and len(out.support_hull) >= 4


@pytest.mark.parametrize('group', ['ankle', 'hip'])
def test_rotation_joint_maps_rotate_body_as_requested(model, group):
    pytest.importorskip('humanoid_interfaces')
    from humanoid_control.humanoid_walk import rotation_joint_maps
    kin, standing, pose = model
    frame = Frame(standing)
    maps = rotation_joint_maps(kin, frame)
    # Distal link that is held fixed: the foot for the ankle group, the thigh for the hip.
    distal = {'ankle': lambda s: kin.model.getFrameId(f'{s}_foot_link'),
              'hip': lambda s: kin.model.getFrameId(f'{s}_knee')}[group]
    data = kin.model.createData()
    for side in ('left', 'right'):
        for r in (np.array([0.05, 0.0]), np.array([0.0, 0.05])):
            joints, m = maps[side][group]
            q = kin.q.copy()
            pin.framesForwardKinematics(kin.model, data, q)
            r0 = data.oMf[distal(side)].rotation.copy()
            for name, dq in zip(joints, m @ r):
                q[kin.model.idx_qs[kin.joint_name_to_id[name]]] += dq
            pin.framesForwardKinematics(kin.model, data, q)
            r1 = data.oMf[distal(side)].rotation
            body = pin.log3(r0 @ r1.T)  # base rotation when the distal link is held still
            np.testing.assert_allclose(axial_to_plane(frame, body), r, atol=2e-3)


def test_axial_to_plane_is_right_handed_for_left_handed_axes():
    class LeftHanded:  # this robot: forward = base +y, left = base +x
        forward = np.array([0.0, 1.0, 0.0])
        left = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])

        def to_plane(self, p):
            return np.array([p @ self.forward, p @ self.left, p @ self.up])

    frame = LeftHanded()
    for axis, moved in ((frame.forward, 1), (frame.left, 0)):
        for rotvec in (0.1 * axis, -0.1 * axis):
            up = frame.to_plane(pin.exp3(rotvec) @ frame.up)
            plane = axial_to_plane(frame, rotvec)
            if moved == 1:  # +rotation about forward tilts the body right (-y)
                assert np.sign(plane[0]) == -np.sign(up[1])
            else:  # +rotation about left tilts the body forward (+x)
                assert np.sign(plane[1]) == np.sign(up[0])
