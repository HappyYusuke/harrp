import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory('harrp'),
        'rviz',
        'reid3d.rviz'
    )

    namespace = 'harrp'

    return LaunchDescription([
        Node(
            package='harrp',
            executable='reid3d', 
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        )
    ])
