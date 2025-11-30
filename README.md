# HARRP
3D-LiDARを用いた再識別可能な人追従のROS2パッケージ。 <br>
ReID3DとPointPillarsを使用したHuman Following systemです。

# Installation

ros2_tao_pointpillarsのファイルを置換します。
```
# 元のファイルを削除
rm ~/ros2_ws/src/ros2_tao_pointpillars/launch/pp_infer_launch.py
rm ~/ros2_ws/src/ros2_tao_pointpillars/package.xml

# 本リポジトリのファイルをコピー
cp ~/ros2_ws/src/follow_me_by_3d_lidar/external_files/pp_infer_launch.py ~/ros2_ws/src/ros2_tao_pointpillars/launch/
cp ~/ros2_ws/src/follow_me_by_3d_lidar/external_files/point_cloud2_iterator.hpp ~/ros2_ws/src/ros2_tao_pointpillars/include/pp_infer/
cp ~/ros2_ws/src/follow_me_by_3d_lidar/external_files/package.xml ~/ros2_ws/src/ros2_tao_pointpillars/
```
