"""Start Gazebo Classic, publish the model and spawn the robot and controllers."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch.substitutions import PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Compose the Gazebo backend with the existing spawn and controller sequence.

    Returns:
        The Gazebo simulation launch description.
    """
    pause = LaunchConfiguration('pause')
    world = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time')
    floating = LaunchConfiguration('floating')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py',
        ])),
        launch_arguments={
            'pause': pause,
            'world': world,
            'params_file': PathJoinSubstitution([
                FindPackageShare('humanoid_gazebo'), 'config', 'gazebo_params.yaml',
            ]),
        }.items(),
    )
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command([
                'xacro', ' ', PathJoinSubstitution([
                    FindPackageShare('humanoid_gazebo'), 'urdf', 'humanoid.xacro',
                ]), ' floating:=', floating,
            ]),
        }, {'use_sim_time': use_sim_time}],
    )
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'humanoid', '-topic', 'robot_description',
            '-z', PythonExpression(["'0.15' if '", floating, "' == 'true' else '0.0'"]),
        ],
        output='screen',
    )
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )
    effort_position_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['effort_position_controller', '--controller-manager', '/controller_manager'],
    )
    return LaunchDescription([
        DeclareLaunchArgument('pause', default_value='true',
                              description='Start Gazebo simulation paused'),
        DeclareLaunchArgument('world', default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_gazebo'), 'worlds', 'world.sdf',
        ]), description='Absolute path to world file'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulation clock for TF and visualization timestamps',
        ),
        DeclareLaunchArgument('floating', default_value='false',
                              description='Pin the robot base in the air (true/false)'),
        gazebo,
        robot_state_publisher,
        spawn_entity,
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_entity, on_exit=[joint_state_broadcaster],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[effort_position_controller],
        )),
    ])
