"""Actuated joint order and standing pose shared by all control nodes.

The order must match the Gazebo controller YAML and `urdf/ros2_control.xacro` in
`humanoid_gazebo`, and the gains in `config/mujoco_sim.yaml` in `humanoid_mujoco`.
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
