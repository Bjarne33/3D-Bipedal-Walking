"""Build a MuJoCo scene from the shared URDF description and an MJCF world file.

Every hinge joint gets a `<position kp kv>` actuator. With the `implicitfast` integrator
MuJoCo integrates its damping implicitly, which keeps the stiff joint PD stable when the
feet are in contact (the same gains applied as explicit motor torques make the standing
robot oscillate and fall). The trajectory tracker turns the desired state into the
equivalent position setpoint, so the force equals the trajectory controller's PD law.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import mujoco

# Name of the free joint added to the base when the robot is not pinned.
FREE_JOINT_NAME = 'base_free'
# Root body of the description; it carries the free joint and is reported as odometry.
BASE_BODY = 'base'
# Ankle force/torque sensors: sensor (and topic) name -> body whose parent joint is measured.
FT_SENSORS: Dict[str, str] = {
    'left_foot_ft': 'left_foot_link',
    'right_foot_ft': 'right_foot_link',
}
# Robot geoms only collide with the world (contype 2 & conaffinity 1), like Gazebo's
# default self_collide=false; world geoms keep the MuJoCo defaults (1/1).
ROBOT_CONTYPE = '2'
ROBOT_CONAFFINITY = '1'


def is_floating(urdf: str) -> bool:
    """Check whether the description pins the base to a `world` link (`floating:=true`).

    Args:
        urdf: Expanded URDF of the shared description.

    Returns:
        True if the URDF contains a link named `world`.
    """
    return ET.fromstring(urdf).find("link[@name='world']") is not None


def actuated_joints(urdf: str) -> List[str]:
    """List the revolute and continuous joints of a URDF in document order.

    Args:
        urdf: Expanded URDF of the shared description.

    Returns:
        Names of the joints that become hinge joints with actuators.
    """
    return [joint.get('name') for joint in ET.fromstring(urdf).findall('joint')
            if joint.get('type') in ('revolute', 'continuous')]


def urdf_to_mjcf(urdf: str) -> ET.Element:
    """Convert a URDF to MJCF with MuJoCo's URDF importer.

    Visual geometry is discarded (the collision geometry is identical) and static bodies are
    kept so `base` and the links remain addressable by name.

    Args:
        urdf: Expanded URDF of the shared description.

    Returns:
        Root `<mujoco>` element of the converted model.
    """
    robot = ET.fromstring(urdf)
    extension = ET.SubElement(robot, 'mujoco')
    ET.SubElement(extension, 'compiler', discardvisual='true', fusestatic='false')
    model = mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding='unicode'))
    fd, path = tempfile.mkstemp(suffix='.xml')
    os.close(fd)
    try:
        mujoco.mj_saveLastXML(path, model)
        return ET.parse(path).getroot()
    finally:
        os.remove(path)


def _add_robot_elements(robot: ET.Element, floating: bool, spawn_z: float,
                        gains: Dict[str, Tuple[float, float]]) -> None:
    """Add the base joint, geom names, sensor sites, actuators and sensors to the robot.

    Args:
        robot: Root `<mujoco>` element of the converted robot; modified in place.
        floating: Weld the base to the world instead of adding a free joint.
        spawn_z: Height of the base frame above the world origin.
        gains: `(p, d)` gains per actuated joint.

    Raises:
        ValueError: If a required body is missing or a hinge joint has no positive p gain.
    """
    base = robot.find(f"worldbody/body[@name='{BASE_BODY}']")
    if base is None:
        raise ValueError(f'Description has no root body named {BASE_BODY!r}')
    base.set('pos', f'0 0 {spawn_z:g}')
    if not floating:
        base.insert(0, ET.Element('freejoint', name=FREE_JOINT_NAME))

    for body in robot.iter('body'):
        geoms = body.findall('geom')
        for index, geom in enumerate(geoms):
            suffix = '' if len(geoms) == 1 else f'_{index}'
            geom.set('name', f"{body.get('name')}_collision{suffix}")
            geom.set('contype', ROBOT_CONTYPE)
            geom.set('conaffinity', ROBOT_CONAFFINITY)

    sensors = ET.SubElement(robot, 'sensor')
    for name, body_name in FT_SENSORS.items():
        body = robot.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(f'Description has no body named {body_name!r}')
        # The site sits at the body origin, i.e. on the ankle_pitch joint axis.
        ET.SubElement(body, 'site', name=f'{name}_site')
        ET.SubElement(sensors, 'force', name=f'{name}_force', site=f'{name}_site')
        ET.SubElement(sensors, 'torque', name=f'{name}_torque', site=f'{name}_site')

    actuators = ET.SubElement(robot, 'actuator')
    for joint in robot.iter('joint'):
        if joint.get('type', 'hinge') != 'hinge':
            continue
        name = joint.get('name')
        p_gain, d_gain = gains.get(name, (0.0, 0.0))
        if p_gain <= 0.0:
            raise ValueError(f'Joint {name!r} needs a positive p gain')
        # The importer converts the URDF effort limit to the joint's actuatorfrcrange,
        # which clamps the actuator force like Gazebo's effort limit.
        ET.SubElement(actuators, 'position', name=name, joint=name,
                      kp=f'{p_gain:g}', kv=f'{d_gain:g}')


def _merge(world: ET.Element, robot: ET.Element) -> None:
    """Append the robot's MJCF sections to the world model.

    Args:
        world: Root `<mujoco>` element of the world; modified in place.
        robot: Root `<mujoco>` element of the robot.
    """
    compiler = world.find('compiler')
    if compiler is None:
        compiler = ET.Element('compiler')
        world.insert(0, compiler)
    # The converted robot is expressed in radians.
    compiler.set('angle', 'radian')
    for section in robot:
        if section.tag == 'compiler':
            continue
        target = world.find(section.tag)
        if target is None:
            target = ET.SubElement(world, section.tag)
        target.extend(list(section))


def build_scene_xml(urdf: str, world_path: str, gains: Dict[str, Tuple[float, float]],
                    spawn_z: Optional[float] = None) -> str:
    """Build the simulation MJCF from the shared description and a world file.

    Args:
        urdf: Expanded URDF of the shared description; `floating:=true` pins the base.
        world_path: MJCF world file with options, floor, lights and defaults.
        gains: `(p, d)` gains per actuated joint.
        spawn_z: Base height; defaults to 0.15 m when floating and 0.0 m otherwise,
            matching the Gazebo spawn height.

    Returns:
        MJCF XML of the complete scene.
    """
    floating = is_floating(urdf)
    if spawn_z is None:
        spawn_z = 0.15 if floating else 0.0
    robot = urdf_to_mjcf(urdf)
    _add_robot_elements(robot, floating, spawn_z, gains)
    world = ET.parse(world_path).getroot()
    _merge(world, robot)
    return ET.tostring(world, encoding='unicode')


def build_model(urdf: str, world_path: str, gains: Dict[str, Tuple[float, float]],
                spawn_z: Optional[float] = None) -> mujoco.MjModel:
    """Compile the simulation model from the shared description and a world file.

    Args:
        urdf: Expanded URDF of the shared description; `floating:=true` pins the base.
        world_path: MJCF world file with options, floor, lights and defaults.
        gains: `(p, d)` gains per actuated joint.
        spawn_z: Base height; see `build_scene_xml`.

    Returns:
        Compiled MuJoCo model.
    """
    return mujoco.MjModel.from_xml_string(build_scene_xml(urdf, world_path, gains, spawn_z))


def set_start_camera(model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera,
                     distance: float) -> None:
    """Point a free camera at the robot's centre of mass.

    The azimuth and elevation come from the world's `<visual><global>` settings, which the
    passive viewer does not apply on its own.

    Args:
        model: Compiled simulation model.
        data: Simulation data after `mj_forward`.
        camera: Viewer camera to set.
        distance: Distance from the centre of mass [m].
    """
    mujoco.mjv_defaultFreeCamera(model, camera)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    camera.lookat[:] = data.subtree_com[base_id]
    camera.distance = distance
