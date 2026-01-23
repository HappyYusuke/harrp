import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 2. DWAパラメータファイルのパス (前回作成したもの)
    config_path = os.path.join(
        get_package_share_directory('harrp'), # 自分のパッケージ名
        'config', # ディレクトリ構成に合わせて調整
        'mpc_params.yaml'
    )

    # 3. DWAコントローラ
    dwa_node = Node(
        package='harrp',
        executable='mpc_controller.py',
        output='screen',
        parameters=[config_path],
    )

    # 4. 仮想ターゲット (Virtual Person)
    target_node = Node(
        package='harrp',
        executable='virtual_target.py',
        output='screen',
    )

    return LaunchDescription([
        dwa_node,
        target_node
    ])
