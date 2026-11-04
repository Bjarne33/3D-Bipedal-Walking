"""Robot constants shared by the MJCF exporter, the environment and the policy export.

This module has no third-party dependencies so that `scripts/export_mjcf.py` can import it from
the ROS Python environment. Joint order and standing pose mirror
`humanoid_control/tools/joints.py`; a test checks that they stay identical.

Axis convention of the robot: forward is +Y, lateral is X (left leg at +X), up is +Z.
"""

import os

LEG_JOINTS = ('hip_roll', 'hip_yaw', 'hip_pitch', 'knee', 'ankle_roll', 'ankle_pitch')
SIDES = ('left', 'right')
JOINT_NAMES = tuple(f'{side}_{joint}' for side in SIDES for joint in LEG_JOINTS)
NUM_JOINTS = len(JOINT_NAMES)

# Standing pose per leg, in the order of LEG_JOINTS (same as humanoid_control).
STANDING_LEG_POSITIONS = (0.0, 0.0, 0.275, -0.5, 0.0, 0.25)
STANDING_POSE = tuple(STANDING_LEG_POSITIONS) * len(SIDES)

# Unit vectors of the robot frame.
FORWARD_AXIS = 1  # +Y
LATERAL_AXIS = 0  # X

# PD gains of the training model's position actuators (per leg joint). They are softer than the
# Gazebo controller gains (2000-2500); see README "RL gains".
KP = {
    'hip_roll': 300.0,
    'hip_yaw': 200.0,
    'hip_pitch': 300.0,
    'knee': 350.0,
    'ankle_roll': 150.0,
    'ankle_pitch': 150.0,
}
KD = {
    'hip_roll': 8.0,
    'hip_yaw': 5.0,
    'hip_pitch': 8.0,
    'knee': 10.0,
    'ankle_roll': 4.0,
    'ankle_pitch': 4.0,
}
# Reflected rotor inertia added to every actuated joint [kg m^2]. It is not in the URDF; it
# models the motor/gearbox and keeps the light hip/ankle dummy links stable at the 4 ms step.
ARMATURE = 0.02

# Names in the generated MJCF.
ROOT_JOINT = 'root'
BASE_BODY = 'base'
TORSO_BODY = 'torso_link'
FLOOR_GEOM = 'floor'
IMU_SITE = 'imu'
FOOT_BODIES = ('left_foot_link', 'right_foot_link')
FOOT_GEOMS = ('left_foot_link_collision', 'right_foot_link_collision')
# Sole centre sites (one per foot) used for foot position/velocity and swing height.
FOOT_SITES = ('left_foot', 'right_foot')
HOME_KEYFRAME = 'home'

GYRO_SENSOR = 'gyro'
ACCELEROMETER_SENSOR = 'accelerometer'
ORIENTATION_SENSOR = 'orientation'
LOCAL_LINVEL_SENSOR = 'local_linvel'
GLOBAL_LINVEL_SENSOR = 'global_linvel'
FOOT_POS_SENSORS = tuple(f'{site}_pos' for site in FOOT_SITES)
FOOT_LINVEL_SENSORS = tuple(f'{site}_global_linvel' for site in FOOT_SITES)
FOOT_TOUCH_SENSORS = tuple(f'{site}_touch' for site in FOOT_SITES)

# Foot geometry from humanoid_description/urdf/properties.xacro and leg_macro.xacro.
FOOT_SIZE = (0.20, 0.34, 0.05)
FOOT_OFFSET = (0.0, 0.05, -0.025)

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
XML_PATH = os.path.join(ASSET_DIR, 'humanoid_rl.xml')
