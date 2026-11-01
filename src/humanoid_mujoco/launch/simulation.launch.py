"""Start the MuJoCo simulator and publish the shared robot description."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    Command, EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Compose the MuJoCo backend with the same arguments as the Gazebo backend.

    Returns:
        The MuJoCo simulation launch description.
    """
    pause = LaunchConfiguration('pause')
    world = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time')
    floating = LaunchConfiguration('floating')
    gui = LaunchConfiguration('gui')
    robot_description = ParameterValue(Command([
        'xacro', ' ', PathJoinSubstitution([
            FindPackageShare('humanoid_description'), 'urdf', 'humanoid.xacro',
        ]), ' floating:=', floating,
    ]), value_type=str)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description},
                    {'use_sim_time': use_sim_time}],
    )
    # The simulator owns /clock and stamps with its own time, so it ignores use_sim_time.
    simulator = Node(
        package='humanoid_mujoco',
        executable='mujoco_sim',
        name='mujoco_sim',
        output='screen',
        # The Wayland GLFW 3.3 library bundled with pyGLFW delivers no mouse or keyboard
        # input to the viewer, so the X11 library (via XWayland) is the default. A value set
        # in the environment takes precedence.
        additional_env={'PYGLFW_LIBRARY_VARIANT': EnvironmentVariable(
            'PYGLFW_LIBRARY_VARIANT', default_value='x11')},
        parameters=[
            PathJoinSubstitution([FindPackageShare('humanoid_mujoco'), 'config',
                                  'mujoco_sim.yaml']),
            {
                'robot_description': robot_description,
                'world': world,
                'pause': ParameterValue(pause, value_type=bool),
                'viewer': ParameterValue(gui, value_type=bool),
            },
        ],
    )
    return LaunchDescription([
        DeclareLaunchArgument('pause', default_value='true',
                              description='Start the MuJoCo simulation paused'),
        DeclareLaunchArgument('world', default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_mujoco'), 'worlds', 'world.xml',
        ]), description='Absolute path to the MJCF world file'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulation clock for TF and visualization timestamps',
        ),
        DeclareLaunchArgument('floating', default_value='false',
                              description='Pin the robot base in the air (true/false)'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='Open the MuJoCo viewer'),
        robot_state_publisher,
        simulator,
    ])
