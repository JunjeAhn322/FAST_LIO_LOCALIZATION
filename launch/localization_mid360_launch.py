import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz'
    )
    map_arg = DeclareLaunchArgument(
        'map',
        default_value='',
        description='Map file for global mapping'
    )

    # Define paths for the parameter and RViz config files
    pkg_share = FindPackageShare(package='fast_lio_localization')
    param_file = PathJoinSubstitution([pkg_share, 'config', 'mid360.yaml'])
    rviz_config_file = PathJoinSubstitution([pkg_share, 'rviz', 'localization.rviz'])

    # Additional parameters (overrides loaded from the YAML file)
    # Note: Original ROS1 values are preserved here.
    additional_params = {
        'feature_extract_enable': False,  # 0 in ROS1 means false
        'point_filter_num': 1,            # Original value was 1
        'max_iteration': 3,
        'filter_size_surf': 0.5,
        'filter_size_map': 0.5,
        'cube_side_length': 1000.0,
        'runtime_pos_log_enable': False,  # 0 in ROS1 means false
        'pcd_save_enable': False,         # 0 in ROS1 means false
    }

    # Node for laserMapping
    laserMapping_node = Node(
        package='fast_lio_localization',
        executable='fastlio_mapping',
        name='laserMapping',
        output='screen',
        parameters=[param_file, additional_params]
    )

    # Node for global localization (Python executable)
    # In ROS2 it is common to install Python nodes as entry points, but here we keep the .py name.
    global_localization_node = Node(
        package='fast_lio_localization',
        executable='global_localization.py',
        name='global_localization',
        output='screen'
    )
    
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_base_link_static_tf',
        arguments=['0', '0', '0', '0', '0', '0', '1', 'map', 'base_link'],
    )

    # Node for transform fusion (Python executable)
    transform_fusion_node = Node(
        package='fast_lio_localization',
        executable='transform_fusion.py',
        name='transform_fusion',
        output='screen'
    )

    # Node for global map: using pcl_ros's pcd_to_pointcloud
    # The "map" file is passed as a parameter, and we set the frame_id explicitly.
    # Remapping is done for the cloud_pcd topic.
    map_publishe_node = Node(
        package='pcl_ros',
        executable='pcd_to_pointcloud',
        name='map_publishe',
        output='screen',
        parameters=[{
            'file_name': LaunchConfiguration('map'),
            '_frame_id': '/map'
        }],
        arguments=['5', '--ros-args', '--remap', 'cloud_pcd:=/map']
    )

    # RViz node (only launched if the "rviz" argument is true)
    # ROS2 uses rviz2, and we use a launch prefix "nice" as in ROS1.
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        output='screen',
        arguments=['-d', rviz_config_file],
        prefix='nice'
    )

    ld = LaunchDescription()
    ld.add_action(rviz_arg)
    ld.add_action(map_arg)
    ld.add_action(laserMapping_node)
    ld.add_action(global_localization_node)
    ld.add_action(transform_fusion_node)
    ld.add_action(map_publishe_node)
    ld.add_action(
        GroupAction(
            actions=[rviz_node],
            condition=IfCondition(LaunchConfiguration('rviz'))
        )
    )
    ld.add_action(static_tf_node)

    return ld