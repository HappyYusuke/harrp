import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    # HOME
    home = os.environ['HOME']

    # MPC config file
    mpc_config = os.path.join(
        get_package_share_directory('harrp'),
        'config',
        'mpc_params.yaml'
    )

    reid3d_tracker_config = os.path.join(
        get_package_share_directory('harrp'),
        'config',
        'reid3d_with_tracker_params.yaml'
    )

    # rviz2 config file
    rviz_config = os.path.join(
        get_package_share_directory('harrp'),
        'rviz',
        'harrp_kachaka.rviz'
    )

    namespace = 'harrp'

    # 購読するオドメトリのトピック名
    odom_topic_name = '/kachaka/odometry/odometry'

    return LaunchDescription([
        #Node(
        #    package='harrp',
        #    executable='livox_tf_broadcaster.py',
        #    name='odom_tf_broadcaster',
        #    output='screen',
        #    parameters=[{
        #        'parent_frame': 'base_link',
        #        'child_frame': 'livox_frame',
        #        # 3D LiDARの位置 (kachakaのurdf基準)
        #        'x': 0.156,
        #        'y': 0.0,
        #        'z': 0.1049 + 0.05,
        #    }],
        #    remappings=[
        #        # オドメトリ購読用トピック
        #        ('/odom', odom_topic_name),
        #    ],
        #),

        Node(
            package='harrp',
            executable='mpc_controller.py',
            name='mpc_controller',
            namespace=namespace,
            output='screen',
            parameters=[mpc_config],
            remappings=[
                # ロボット制御用トピック
                ('/cmd_vel', '/kachaka/manual_control/cmd_vel'),
                # オドメトリ購読用トピック
                ('/odom', odom_topic_name),
                # 3D LiDAR購読用トピック
                ('/livox/lidar', '/livox/lidar'),
                # tf用
                ('tf', '/tf'),
                ('tf_static', '/tf_static'),
            ],
        ),

        Node(
            package='harrp',
            executable='reid3d_with_tracker.py',
            name='reid3d_with_tracker',
            namespace=namespace,
            output='screen',
            parameters=[
                reid3d_tracker_config,
                {'weight_path': f'{home}/ReID3D/reidnet/log/ckpt_best.pth'}
            ]
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
