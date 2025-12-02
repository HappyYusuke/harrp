cmake_minimum_required(VERSION 3.5)
project(harrp)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED) # 必須
find_package(sensor_msgs REQUIRED)
# find_package(pcl_conversions REQUIRED)

# ---------------------------------------------------------
# C++ノードのビルド
# ---------------------------------------------------------
# 実行ファイルとして定義
add_executable(ground_removal_node src/ground_removal_node.cpp)

ament_target_dependencies(ground_removal_node
  rclcpp
  sensor_msgs
)

# ---------------------------------------------------------
# コンポーネント（ライブラリ）のビルド
# ---------------------------------------------------------
# SHARED ライブラリとして追加します
add_library(ground_removal_component SHARED
  src/ground_removal_node.cpp
)

target_compile_definitions(ground_removal_component
  PRIVATE "COMPOSITION_BUILDING_DLL"
)

ament_target_dependencies(ground_removal_component
  rclcpp
  rclcpp_components
  sensor_msgs
  # pcl_conversions
)

# コンポーネントの登録（これが重要です）
# ライブラリターゲット名と、C++のクラス名(namespace込み)を指定
rclcpp_components_register_nodes(ground_removal_component
  "harrp::GroundRemovalNode"
)

# ---------------------------------------------------------
# インストール設定
# ---------------------------------------------------------
install(TARGETS ground_removal_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Pythonスクリプト
install(PROGRAMS
  scripts/pub_pillar_bbox.py
  scripts/reid3d.py
  DESTINATION lib/${PROJECT_NAME}
)

# Launch / Config
install(DIRECTORY
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
