"""Check the MuJoCo scene built from the shared description."""

import importlib.util
import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import mujoco
import numpy as np
import pytest
import xacro
import yaml

from humanoid_mujoco.model import actuated_joints, build_model, build_scene_xml, FT_SENSORS
from humanoid_mujoco.model import ARMATURE, FREE_JOINT_NAME, set_start_camera
from humanoid_mujoco.trajectory import TrajectoryTracker

WORLD = str(Path(__file__).resolve().parents[1] / 'worlds' / 'world.xml')
CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'mujoco_sim.yaml'
RL_CONSTANTS = Path(__file__).resolve().parents[3] / 'rl' / 'humanoid_rl' / 'constants.py'
JOINT_NAMES = [
    f'{side}_{joint}' for side in ('left', 'right')
    for joint in ('hip_roll', 'hip_yaw', 'hip_pitch', 'knee', 'ankle_roll', 'ankle_pitch')
]
EFFORT_LIMITS = {
    'hip_roll': 200.0, 'hip_yaw': 200.0, 'hip_pitch': 220.0, 'knee': 180.0,
    'ankle_roll': 120.0, 'ankle_pitch': 120.0,
}
GAINS = {name: (2000.0 + i, 10.0 + i) for i, name in enumerate(JOINT_NAMES)}


def _urdf(floating: bool) -> str:
    """Expand the shared description.

    Args:
        floating: Value of the `floating` xacro argument.

    Returns:
        The URDF XML.
    """
    path = Path(get_package_share_directory('humanoid_description')) / 'urdf' / 'humanoid.xacro'
    document = xacro.process_file(str(path), mappings={'floating': str(floating).lower()})
    return document.toxml()


@pytest.fixture(scope='module')
def free_model():
    """Compile the free-base model once."""
    return build_model(_urdf(False), WORLD, GAINS)


def test_free_base(free_model):
    """Add a free joint at the Gazebo spawn height and actuate the twelve leg joints."""
    model = free_model
    assert actuated_joints(_urdf(False)) == JOINT_NAMES
    assert (model.nq, model.nv, model.nu) == (19, 18, 12)
    assert model.joint(0).name == FREE_JOINT_NAME
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.body(model.jnt_bodyid[0]).name == 'base'
    assert np.allclose(model.body('base').pos, [0.0, 0.0, 0.0])
    assert [model.actuator(i).name for i in range(model.nu)] == JOINT_NAMES
    assert math.isclose(model.body_subtreemass[model.body('base').id], 30.6)


def test_floating_base():
    """Weld the base 0.15 m above the world origin like the Gazebo floating spawn."""
    model = build_model(_urdf(True), WORLD, GAINS)
    assert (model.nq, model.nv, model.nu) == (12, 12, 12)
    base = model.body('base')
    assert base.parentid[0] == 0 and model.body_jntnum[base.id] == 0
    assert np.allclose(base.pos, [0.0, 0.0, 0.15])


def test_actuators_follow_gains_and_effort_limits(free_model):
    """Apply each joint's p and d gain and clamp the force at the URDF effort limit."""
    model = free_model
    for i, name in enumerate(JOINT_NAMES):
        p_gain, d_gain = GAINS[name]
        assert model.actuator_gainprm[i, 0] == pytest.approx(p_gain)
        assert model.actuator_biasprm[i, :3] == pytest.approx([0.0, -p_gain, -d_gain])
        joint = model.joint(name)
        limit = EFFORT_LIMITS[name.split('_', 1)[1]]
        assert model.jnt_actfrclimited[joint.id]
        assert model.jnt_actfrcrange[joint.id] == pytest.approx([-limit, limit])
        assert model.dof_armature[joint.dofadr[0]] == pytest.approx(ARMATURE)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.timestep == pytest.approx(0.001)


def test_missing_gain_is_rejected():
    """Require a positive p gain for every actuated joint."""
    gains = dict(GAINS)
    del gains['right_knee']
    with pytest.raises(ValueError, match='right_knee'):
        build_scene_xml(_urdf(False), WORLD, gains)


def test_geoms_and_sensors(free_model):
    """Name every geom, collide the robot only with the world and place the ankle sensors."""
    model = free_model
    names = [model.geom(i).name for i in range(model.ngeom)]
    assert all(names) and len(set(names)) == len(names)
    robot = [i for i in range(model.ngeom) if model.geom_bodyid[i] != 0]
    floor = model.geom('ground_plane').id
    for i in robot:
        assert not (model.geom_contype[i] & model.geom_conaffinity[robot[0]])
        assert model.geom_contype[floor] & model.geom_conaffinity[i]
    for name, body in FT_SENSORS.items():
        site = model.site(f'{name}_site')
        assert model.body(site.bodyid[0]).name == body
        assert np.allclose(site.pos, 0.0)
        assert model.sensor(f'{name}_force').type[0] == mujoco.mjtSensor.mjSENS_FORCE
        assert model.sensor(f'{name}_torque').type[0] == mujoco.mjtSensor.mjSENS_TORQUE


def test_start_camera_centres_robot(free_model):
    """Look at the robot centre of mass from the world's view direction."""
    data = mujoco.MjData(free_model)
    mujoco.mj_forward(free_model, data)
    camera = mujoco.MjvCamera()
    set_start_camera(free_model, data, camera, 3.0)
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert camera.distance == pytest.approx(3.0)
    assert camera.azimuth == pytest.approx(free_model.vis.global_.azimuth)
    assert camera.elevation == pytest.approx(free_model.vis.global_.elevation)
    assert np.allclose(camera.lookat, data.subtree_com[free_model.body('base').id])
    assert 0.5 < camera.lookat[2] < 0.9


def test_startup_pose_is_stable():
    """Come to rest near the standing pose with the configured gains, weight on both ankles.

    The soft RL gains leave a static gravity droop (mostly in the ankle pitch joints), so the
    tolerances are wider than the tracking accuracy of the stiff Gazebo gains.
    """
    configured = yaml.safe_load(CONFIG.read_text())['mujoco_sim']['ros__parameters']['gains']
    gains = {name: (gain['p'], gain['d']) for name, gain in configured.items()}
    model = build_model(_urdf(False), WORLD, gains)
    data = mujoco.MjData(model)
    qpos_adr = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    tracker = TrajectoryTracker(
        JOINT_NAMES, [gains[n][0] for n in JOINT_NAMES], [gains[n][1] for n in JOINT_NAMES])
    standing = {'hip_pitch': 0.275, 'knee': -0.5, 'ankle_pitch': 0.25}
    target = [standing.get(name.split('_', 1)[1], 0.0) for name in JOINT_NAMES]
    tracker.set_trajectory(0.0, 0.0, JOINT_NAMES, [(2.0, target, None)])
    while data.time < 6.0:
        data.ctrl[:] = tracker.setpoint(data.time)
        mujoco.mj_step(model, data)
    tilt = 2.0 * math.acos(min(1.0, abs(data.qpos[3])))
    assert tilt < math.radians(8.0)
    assert np.abs(data.qpos[qpos_adr] - target).max() < 0.1
    assert np.abs(data.qvel).max() < 0.01
    # Each ankle carries half the weight above it (1.1 kg of foot and ankle per leg).
    expected = -0.5 * (30.6 - 2.2) * 9.81
    for name in FT_SENSORS:
        assert data.sensor(f'{name}_force').data[2] == pytest.approx(expected, rel=0.1)


def test_gains_and_armature_match_rl_training():
    """Keep the hardcoded gains and armature equal to the RL training constants."""
    if not RL_CONSTANTS.exists():
        pytest.skip(f'{RL_CONSTANTS} not found')
    spec = importlib.util.spec_from_file_location('rl_constants', RL_CONSTANTS)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    gains = yaml.safe_load(CONFIG.read_text())['mujoco_sim']['ros__parameters']['gains']
    assert sorted(gains) == sorted(JOINT_NAMES)
    for name, gain in gains.items():
        joint = name.split('_', 1)[1]
        assert gain['p'] == pytest.approx(constants.KP[joint])
        assert gain['d'] == pytest.approx(constants.KD[joint])
    assert ARMATURE == pytest.approx(constants.ARMATURE)
