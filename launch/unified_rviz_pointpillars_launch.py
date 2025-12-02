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

    # ---------------------------------------------------------
    # 1. Ground Removal コンポーネント (harrp)
    # ---------------------------------------------------------
    ground_removal_component = ComposableNode(
        package='harrp',
        plugin='harrp::GroundRemovalNode',
        name='ground_removalcomp_node',
        # プロセス内通信を有効化 (重要)
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # ---------------------------------------------------------
    # 2. PointPillars コンポーネント (pp_infer)
    # ---------------------------------------------------------
    pointpillars_component = ComposableNode(
        package='pp_infer',
        plugin='pp_infer::PointPillarsHarrpNode',
        name='pp_infer_harrp',
        parameters=[{
            'nms_iou_thresh': 0.01,
            'pre_nms_top_n': 4096,
            'class_names': ['Pedestrian'],
            # パスは環境に合わせてください
            'model_path': '/home/demulab-kohei/colcon_ws/src/ros2_tao_pointpillars/include/harrp_epoch400.onnx', 
            'engine_path': '/home/nvidia/Projects/PointPillars/trt.fp16.engine', # パス注意
            'data_type': 'fp32',
            'intensity_scale': 255.0,
        }],
        # トピックの接続: 
        # harrpの出力トピック名に合わせて変更してください。
        # ここでは harrp が "/livox/lidar/no_ground" に出すと仮定して、
        # pp_infer の入力 "/point_cloud" をそれに繋いでいます。
        remappings=[
            ('/point_cloud', '/livox/lidar/no_ground')
        ],
        # プロセス内通信を有効化 (重要)
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # ---------------------------------------------------------
    # 3. 統合コンテナ (Unified Container)
    # ---------------------------------------------------------
    # このコンテナの中に上記2つのノードが同居します。
    # これにより、メモリ空間が共有され、ゼロコピー通信が可能になります。
    container = ComposableNodeContainer(
        name='unified_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt', # マルチスレッド版なら component_container_mt も可
        composable_node_descriptions=[
            ground_removal_component,
            pointpillars_component
        ],
        output='screen',
    )

    return LaunchDescription([
        # 通常のPythonノード (BBox Publisher)
        Node(
            package='harrp',
            executable='pub_pillar_bbox.py', 
            output='screen'
        ),

        # 統合コンテナ (ここで2つのC++ノードが高速に連携)
        container,

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        )
    ])
