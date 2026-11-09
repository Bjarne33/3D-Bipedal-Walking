# Copyright 2026 Bjarne
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Launch one isolated floating simulator and the joint tuning panel."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Compose the selected backend without startup or walking command publishers.

    Returns:
        Tuning launch description with a pinned robot and standalone rqt panel.
    """
    simulator = LaunchConfiguration('simulator')
    backends = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare(f'humanoid_{name}'), 'launch', 'simulation.launch.py'])),
            condition=LaunchConfigurationEquals('simulator', name),
            launch_arguments={'floating': 'true', 'pause': 'false',
                              'use_sim_time': 'true'}.items())
        for name in ('gazebo', 'mujoco')
    ]
    return LaunchDescription([
        DeclareLaunchArgument('simulator', default_value='gazebo', choices=['gazebo', 'mujoco']),
        *backends,
        Node(package='humanoid_tuning', executable='tuning_panel', output='screen',
             arguments=['--simulator', simulator]),
    ])
