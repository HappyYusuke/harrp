import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    # MPC config file
    mpc_config = os.path.join(
        get_package_share_directory('harrp'),
        'config',
        'mpc_params.yaml'
    )

    # rviz2 config file
    rviz_config = os.path.join(
        get_package_share_directory('harrp'),
        'rviz',
        'harrp.rviz'
    )

    namespace = 'harrp'

    return LaunchDescription([
        Node(
            package='harrp',
            executable='mpc_controller.py',
            name='mpc_controller',
            namespace=namespace,
            output='screen',
            parameters=[mpc_config],
        ),

        Node(
            package='harrp',
            executable='reid3d_with_tracker.py',
            name='reid3d_with_tracker',
            namespace=namespace,
            output='screen',
        ),

        Node(
            package='harrp',
            executable='pub_pillar_bbox.py',
            name='pub_pillar_box',
            namespace=namespace,
            output='screen',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config, '--ros-args', '--log-level', 'fatal'],
        )
    ])
