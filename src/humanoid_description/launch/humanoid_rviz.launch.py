from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    urdf_file = PathJoinSubstitution([
        FindPackageShare('humanoid_description'),
        'urdf',
        'humanoid.xacro'
    ])

    robot_description = Command([FindExecutable(name='xacro'), ' ', urdf_file])
    rviz_config_path = PathJoinSubstitution([FindPackageShare('humanoid_description'), 'rviz', 'urdf.rviz'])
    world_frame = LaunchConfiguration('world_frame')
    base_frame = LaunchConfiguration('base_frame')
    publish_world_to_base_tf = LaunchConfiguration('publish_world_to_base_tf')

    world_frame_arg = DeclareLaunchArgument(
        'world_frame',
        default_value='world',
        description='Global frame used for visualization',
    )

    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base',
        description='Robot base frame',
    )

    publish_world_to_base_tf_arg = DeclareLaunchArgument(
        'publish_world_to_base_tf',
        default_value='true',
        description='Publish a static world->base transform when no other source provides it',
    )
    
    joint_state_publisher = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
        parameters=[{"robot_description": robot_description}],
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{"robot_description": robot_description}],
    )

    world_to_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', world_frame, base_frame],
        condition=IfCondition(publish_world_to_base_tf),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path]
    )

    return LaunchDescription([
        world_frame_arg,
        base_frame_arg,
        publish_world_to_base_tf_arg,
        joint_state_publisher,
        world_to_base_tf,
        robot_state_publisher,
        rviz,
    ])