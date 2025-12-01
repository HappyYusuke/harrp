import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory('harrp'),
        'rviz',
        'pointpillars.rviz'
    )

    # 1. Composable Node の定義
    # ground_removal_node はライブラリとしてロードされます
    ground_removal_component = ComposableNode(
        package='harrp',
        plugin='harrp::GroundRemovalNode',  # C++で登録したクラス名
        name='ground_removal_node',
        extra_arguments=[{'use_intra_process_comms': True}] # プロセス内通信を有効化
    )

    # 2. Container の定義
    # このコンテナプロセスの中で ground_removal_component が動きます
    container = ComposableNodeContainer(
        name='harrp_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ground_removal_component
        ],
        output='screen',
    )

    return LaunchDescription([
        # 3. 通常のPythonノード (pub_pillar_bbox.py)
        # これはコンポーネントではないため、これまで通り Node で起動します
        Node(
            package='harrp',
            executable='pub_pillar_bbox.py', 
            output='screen'
        ),

        # 4. コンテナ (ground_removal_node を内包)
        container,

        # 5. RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        )
    ])
