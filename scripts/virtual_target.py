#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
import math
import time

class VirtualTarget(Node):
    def __init__(self):
        super().__init__('virtual_target')
        self.pose_publisher_ = self.create_publisher(PoseStamped, '/harrp/tracker/target_pose', 10)
        self.marker_publisher_ = self.create_publisher(Marker, 'target_marker', 10)
        
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.start_time = time.time()
        
        # --- 変更: 位置調整用のパラメータ ---
        self.center_x = -3.7219
        self.center_y = -5.8182  # ★追加: Y軸方向の中心位置 (ここを変更して調整)
        self.radius = 1.5    # 半径
        self.speed = 0.0     # 回転速度
        # --------------------------------

    def timer_callback(self):
        elapsed = time.time() - self.start_time
        now = self.get_clock().now().to_msg()
        
        # --- ターゲット位置の計算 ---
        # 円運動の中心 (center_x, center_y) の周りを回る
        tx = self.center_x + self.radius * math.cos(self.speed * elapsed)
        ty = self.center_y + self.radius * math.sin(self.speed * elapsed) # ★修正: center_y を加算
        
        # 1. PoseStamped (ロボット制御用)
        msg = PoseStamped()
        msg.header.stamp = now
        msg.header.frame_id = "odom"
        msg.pose.position.x = tx
        msg.pose.position.y = ty
        msg.pose.orientation.w = 1.0
        self.pose_publisher_.publish(msg)
        
        # 2. Marker (RViz可視化用)
        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = "odom"
        marker.ns = "target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = tx
        marker.pose.position.y = ty
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.marker_publisher_.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = VirtualTarget()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
