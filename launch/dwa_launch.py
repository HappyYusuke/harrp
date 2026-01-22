import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 注意: ここではYAMLファイルのパスを直接指定する例ですが、
    # 通常はパッケージのshareディレクトリなどから取得します。
    # 例: config = os.path.join(get_package_share_directory('my_dwa_pkg'), 'config', 'dwa_params.yaml')
    
    # 実行場所と同じディレクトリに yaml ファイルがある場合の簡易パス
    config = os.path.join(
        os.getcwd(), 
        'dwa_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='my_dwa_pkg', # あなたのパッケージ名に変更してください
            executable='dwa_controller', # setup.pyのエントリーポイント名
            name='dwa_controller',
            output='screen',
            parameters=[config] # ここでYAMLファイルをロード
        )
    ])
