"""Launch a simulator backend, spawn the humanoid and start the existing control nodes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, PythonExpression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the simulation launch description.

    Returns:
        The launch description.
    """
    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare('humanoid_description'), 'rviz', 'urdf.rviz'])
    pause = LaunchConfiguration('pause')
    world = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time')
    floating = LaunchConfiguration('floating')
    visualization = LaunchConfiguration('visualization')
    simulator = LaunchConfiguration('simulator')

    simulator_arg = DeclareLaunchArgument(
        'simulator',
        default_value='gazebo',
        choices=['gazebo', 'mujoco'],
        description='Simulator backend',
    )

    visualization_arg = DeclareLaunchArgument(
        'visualization',
        default_value='false',
        description='Launch RViz and the CoM visualizer',
    )

    floating_arg = DeclareLaunchArgument(
        'floating',
        default_value='false',
        description='Pin the robot base in the air so it cannot fall (true/false)',
    )

    pause_arg = DeclareLaunchArgument(
        'pause',
        default_value='true',
        description='Start the simulation paused',
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=PythonExpression([
            "'", PathJoinSubstitution(
                [FindPackageShare('humanoid_gazebo'), 'worlds', 'world.sdf']),
            "' if '", simulator, "' == 'gazebo' else '", PathJoinSubstitution(
                [FindPackageShare('humanoid_mujoco'), 'worlds', 'world.xml']),
            "'",
        ]),
        description='Absolute path to the world file (SDF for Gazebo, MJCF for MuJoCo)',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock for TF and visualization timestamps',
    )

    backend_arguments = list({
        'pause': pause,
        'world': world,
        'use_sim_time': use_sim_time,
        'floating': floating,
    }.items())

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('humanoid_gazebo'), 'launch', 'simulation.launch.py'])),
        condition=LaunchConfigurationEquals('simulator', 'gazebo'),
        launch_arguments=backend_arguments,
    )

    mujoco = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('humanoid_mujoco'), 'launch', 'simulation.launch.py'])),
        condition=LaunchConfigurationEquals('simulator', 'mujoco'),
        launch_arguments=backend_arguments,
    )

    humanoid_startup = Node(
        package='humanoid_control',
        executable='humanoid_startup',
        name='humanoid_startup',
        output='screen',
    )

    humanoid_estimator = Node(
        package='humanoid_control',
        executable='humanoid_estimator',
        name='humanoid_estimator',
        output='screen',
        parameters=[PathJoinSubstitution([
            FindPackageShare('humanoid_control'), 'config', 'estimator.yaml',
        ]), {
            'use_sim_time': use_sim_time,
            # When floating, robot_state_publisher already provides world -> base.
            'publish_world_to_base_tf': PythonExpression(["'", floating, "' != 'true'"]),
        }],
    )

    com_visualizer = Node(
        package='humanoid_control',
        executable='com_visualizer',
        condition=IfCondition(visualization),
        name='com_visualizer',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'world_frame': 'world',
            'base_frame': 'base',
            'publish_world_markers': True,
        }],
    )

    humanoid_predefined_trajs = Node(
        package='humanoid_control',
        executable='humanoid_predefined_trajs',
        name='humanoid_predefined_trajs',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        condition=IfCondition(visualization),
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        simulator_arg,
        pause_arg,
        world_arg,
        use_sim_time_arg,
        floating_arg,
        visualization_arg,
        gazebo,
        mujoco,
        rviz,
        humanoid_startup,
        humanoid_estimator,
        com_visualizer,
        humanoid_predefined_trajs,
    ])
