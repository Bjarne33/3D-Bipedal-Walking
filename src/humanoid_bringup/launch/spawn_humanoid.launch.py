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
    ui = LaunchConfiguration('ui')

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

    ui_arg = DeclareLaunchArgument(
        'ui',
        default_value='true',
        description='Open the walk manager rqt panel (Stop / Walk / RL Walk)',
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
        parameters=[{
            # MuJoCo's soft RL-training PD gains leave ~0.07 rad ankle error under load.
            'pose_tolerance': PythonExpression(
                ["0.1 if '", simulator, "' == 'mujoco' else 0.02"]),
        }],
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

    control_config = PathJoinSubstitution([FindPackageShare('humanoid_control'), 'config'])

    humanoid_walk = Node(
        package='humanoid_control',
        executable='humanoid_walk',
        name='humanoid_walk',
        output='screen',
        parameters=[PathJoinSubstitution([control_config, 'gait.yaml']),
                    {'use_sim_time': use_sim_time}],
    )

    # humanoid_rl_walk and the manager time themselves from the robot-state stamps and stay
    # on wall time: a sim-time Python node spends ~1/3 core on the 1 kHz /clock topic, which
    # slows the simulation and the model-based walk loop.
    humanoid_rl_walk = Node(
        package='humanoid_control',
        executable='humanoid_rl_walk',
        name='humanoid_rl_walk',
        output='screen',
        parameters=[PathJoinSubstitution([control_config, 'rl_walk.yaml'])],
    )

    humanoid_walk_manager = Node(
        package='humanoid_control',
        executable='humanoid_walk_manager',
        name='humanoid_walk_manager',
        output='screen',
        parameters=[PathJoinSubstitution([control_config, 'walk_manager.yaml'])],
    )

    walk_manager_panel = Node(
        package='humanoid_ui',
        executable='walk_manager_panel',
        condition=IfCondition(ui),
        name='walk_manager_panel',
        output='screen',
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
        ui_arg,
        gazebo,
        mujoco,
        rviz,
        humanoid_startup,
        humanoid_estimator,
        com_visualizer,
        humanoid_predefined_trajs,
        humanoid_walk,
        humanoid_rl_walk,
        humanoid_walk_manager,
        walk_manager_panel,
    ])
