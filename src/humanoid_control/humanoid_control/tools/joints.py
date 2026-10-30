"""Actuated joint order and standing pose shared by all control nodes.

The order must match `config/ros2_control_controllers.yaml` and `urdf/ros2_control.xacro`
in `humanoid_description`.
"""

LEG_JOINTS = ('hip_roll', 'hip_yaw', 'hip_pitch', 'knee', 'ankle_roll', 'ankle_pitch')

JOINT_NAMES = [f'{side}_{joint}' for side in ('left', 'right') for joint in LEG_JOINTS]

# Standing pose per leg, in the order of LEG_JOINTS.
STANDING_LEG_POSITIONS = (0.0, 0.0, 0.275, -0.5, 0.0, 0.25)

STANDING_POSE = {
    f'{side}_{joint}': value
    for side in ('left', 'right')
    for joint, value in zip(LEG_JOINTS, STANDING_LEG_POSITIONS)
}
