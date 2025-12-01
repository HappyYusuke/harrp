cmake_minimum_required(VERSION 3.5)
project(harrp)

# デフォルト設定
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 依存関係
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
# find_package(pcl_conversions REQUIRED)

# ---------------------------------------------------------
# C++ ノードのビルド
# ---------------------------------------------------------
# 実行ファイルとして定義
add_executable(ground_removal_node src/ground_removal_node.cpp)

# 依存関係のリンク
ament_target_dependencies(ground_removal_node
  rclcpp
  rclcpp_components
  sensor_msgs
  # pcl_conversions
)

# 【削除】以下の行がエラーの原因でした。実行ファイルには使用できないため削除済みです。
# rclcpp_components_register_nodes(ground_removal_node "harrp::GroundRemovalNode")

# インストール設定
install(TARGETS ground_removal_node
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------------------------------------------------
# Python スクリプトのインストール
# ---------------------------------------------------------
install(PROGRAMS
  scripts/pub_pillar_bbox.py
  scripts/reid3d.py
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------------------------------------------------
# Launch / Config のインストール
# ---------------------------------------------------------
install(DIRECTORY
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
