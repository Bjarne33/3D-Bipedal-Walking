"""Generated MJCF, robot constants and consistency with the ROS packages."""

import importlib.util
import os
import subprocess

import mujoco
from mujoco import mjx
import numpy as np
import pytest

from humanoid_rl import constants as c

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_DIR = os.path.dirname(RL_DIR)
JOINTS_PY = os.path.join(WS_DIR, 'src', 'humanoid_control', 'humanoid_control', 'tools',
                         'joints.py')
ROS_SETUP = '/opt/ros/humble/setup.bash'
WS_SETUP = os.path.join(WS_DIR, 'install', 'setup.bash')


@pytest.fixture(scope='module')
def model():
    return mujoco.MjModel.from_xml_path(c.XML_PATH)


def test_joint_order_and_pose_match_humanoid_control():
    spec = importlib.util.spec_from_file_location('ros_joints', JOINTS_PY)
    joints = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(joints)
    assert tuple(joints.JOINT_NAMES) == c.JOINT_NAMES
    assert tuple(joints.STANDING_POSE[name] for name in c.JOINT_NAMES) == c.STANDING_POSE


def test_model_structure(model):
    assert (model.nq, model.nv, model.nu) == (7 + c.NUM_JOINTS, 6 + c.NUM_JOINTS, c.NUM_JOINTS)
    assert model.joint(0).name == c.ROOT_JOINT
    for index, name in enumerate(c.JOINT_NAMES):
        assert model.actuator(index).name == name
        assert model.actuator_trnid[index, 0] == model.joint(name).id
        assert model.jnt_qposadr[model.joint(name).id] == 7 + index
    for name in (c.BASE_BODY, c.TORSO_BODY, *c.FOOT_BODIES):
        model.body(name)
    for name in (c.FLOOR_GEOM, *c.FOOT_GEOMS):
        model.geom(name)
    for name in (c.IMU_SITE, *c.FOOT_SITES):
        model.site(name)
    for name in (c.GYRO_SENSOR, c.GLOBAL_LINVEL_SENSOR, *c.FOOT_LINVEL_SENSORS,
                 *c.FOOT_TOUCH_SENSORS):
        model.sensor(name)
    model.camera('track')


def test_actuator_gains_match_constants(model):
    for name in c.JOINT_NAMES:
        joint = name.split('_', 1)[1]
        act = model.actuator(name)
        assert act.gainprm[0] == pytest.approx(c.KP[joint])
        assert act.biasprm[1] == pytest.approx(-c.KP[joint])
        assert act.biasprm[2] == pytest.approx(-c.KD[joint])
        assert model.dof_armature[model.joint(name).dofadr[0]] == pytest.approx(c.ARMATURE)


def test_home_keyframe_is_standing_pose_above_floor(model):
    key = model.keyframe(c.HOME_KEYFRAME)
    np.testing.assert_allclose(key.qpos[7:], c.STANDING_POSE)
    np.testing.assert_allclose(key.ctrl, c.STANDING_POSE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key.id)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    soles = data.site_xpos[[model.site(s).id for s in c.FOOT_SITES]]
    assert np.all(soles[:, 2] > 0) and np.all(soles[:, 2] < 0.01)


def test_standing_with_home_ctrl_is_stable(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe(c.HOME_KEYFRAME).id)
    torso = model.body(c.TORSO_BODY).id
    for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert data.xpos[torso][2] > 1.0
    assert data.xmat[torso].reshape(3, 3)[2, 2] > np.cos(0.1)


def test_mjx_put_model(model):
    mx = mjx.put_model(model, impl='jax')
    assert mx.nu == c.NUM_JOINTS


@pytest.mark.skipif(not (os.path.exists(ROS_SETUP) and os.path.exists(WS_SETUP)),
                    reason='ROS 2 Humble and the built workspace are required')
def test_committed_mjcf_matches_export(tmp_path):
    output = tmp_path / 'humanoid_rl.xml'
    script = os.path.join(RL_DIR, 'scripts', 'export_mjcf.py')
    command = f'source {ROS_SETUP} && source {WS_SETUP} && python3 {script} --output {output}'
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'VIRTUAL_ENV')}
    env['PATH'] = '/usr/bin:/bin'
    subprocess.run(['bash', '-c', command], check=True, env=env, capture_output=True)
    with open(c.XML_PATH) as committed:
        assert output.read_text() == committed.read(), \
            'Regenerate the MJCF with rl/scripts/export_mjcf.py'
