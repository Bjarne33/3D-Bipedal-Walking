from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz_config_path = PathJoinSubstitution([FindPackageShare('humanoid_description'), 'rviz', 'urdf.rviz'])
    pause = LaunchConfiguration('pause')
    world = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time')

    pause_arg = DeclareLaunchArgument(
        'pause',
        default_value='true',
        description='Start Gazebo simulation paused',
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution(
            [FindPackageShare('humanoid_description'), 'worlds', 'world.sdf']
        ),
        description='Absolute path to world file',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock for TF and visualization timestamps',
    )

    humanoid_xacro = PathJoinSubstitution(
        [FindPackageShare('humanoid_description'), 'urdf', 'humanoid.xacro']
    )
    gazebo_launch_file = PathJoinSubstitution(
        [FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py']
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={'pause': pause, 'world': world}.items(),
    )

    robot_description = {
        'robot_description': Command(['xacro', ' ', humanoid_xacro])
    }

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'humanoid', '-topic', 'robot_description'],
        output='screen',
    )

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    effort_position_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["effort_position_controller", "--controller-manager", "/controller_manager"],
    )

    humanoid_startup = Node(
        package='humanoid_control',
        executable='humanoid_startup',
        name='humanoid_startup',
        output='screen',
    )

    com_visualizer = Node(
        package='humanoid_control',
        executable='com_visualizer',
        name='com_visualizer',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'world_frame': 'world',
            'base_frame': 'base',
            'model_name': 'humanoid',
            'publish_world_markers': True,
            'publish_world_to_base_tf': True,
        }],
    )

    humanoid_walking = Node(
        package='humanoid_control',
        executable='humanoid_walking',
        name='humanoid_walking',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
    )

    load_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[joint_state_broadcaster],
        )
    )

    load_effort_position_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[effort_position_controller],
        )
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': use_sim_time}],
    )
    
    return LaunchDescription([
        pause_arg,
        world_arg,
        use_sim_time_arg,
        gazebo,
        robot_state_publisher,
        spawn_entity,
        load_joint_state_broadcaster,
        load_effort_position_controller,
        rviz,
        humanoid_startup,
        com_visualizer,
        humanoid_walking,
    ])
