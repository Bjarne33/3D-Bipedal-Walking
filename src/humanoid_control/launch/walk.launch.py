"""Launch file that starts the model-based walking node with the gait parameters."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create the walking launch description.

    Returns:
        The launch description.
    """
    config = os.path.join(get_package_share_directory('humanoid_control'), 'config', 'gait.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(
            package='humanoid_control',
            executable='humanoid_walk',
            name='humanoid_walk',
            output='screen',
            parameters=[config, {'use_sim_time': use_sim_time}],
        ),
    ])
