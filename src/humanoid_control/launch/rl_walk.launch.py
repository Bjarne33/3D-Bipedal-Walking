"""Launch file that starts the RL policy walking node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create the RL walking launch description.

    Returns:
        The launch description.
    """
    config = os.path.join(
        get_package_share_directory('humanoid_control'), 'config', 'rl_walk.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    policy_path = LaunchConfiguration('policy_path')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'policy_path', default_value='',
            description='Exported policy (.npz); empty uses the installed policies/policy.npz'),
        Node(
            package='humanoid_control',
            executable='humanoid_rl_walk',
            name='humanoid_rl_walk',
            output='screen',
            parameters=[config, {'use_sim_time': use_sim_time, 'policy_path': policy_path}],
        ),
    ])
