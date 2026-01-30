#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

class LidarStaticPublisher(Node):
    def __init__(self):
        super().__init__('livox_tf_broadcaster')
        
        # --- パラメータ設定 ---
        # 親フレーム: ロボットの取り付け基準点 (通常は base_link)
        self.declare_parameter('parent_frame', 'base_link')
        # 子フレーム: LiDARのフレーム名
        self.declare_parameter('child_frame', 'livox_frame')
        
        # --- 設置位置オフセット (m) ---
        self.declare_parameter('x', 0.2)  # 前
        self.declare_parameter('y', 0.0)  # 左
        self.declare_parameter('z', 0.0)  # 高さ
        
        self.parent_frame = self.get_parameter('parent_frame').value
        self.child_frame = self.get_parameter('child_frame').value
        self.x = self.get_parameter('x').value
        self.y = self.get_parameter('y').value
        self.z = self.get_parameter('z').value

        # Static Broadcasterの初期化
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # 配信実行
        self.publish_static_tf()
        
        self.get_logger().info(f'Static TF Published: {self.parent_frame} -> {self.child_frame}')

    def publish_static_tf(self):
        static_tf = TransformStamped()
        
        # Static TFは現在時刻でOK
        static_tf.header.stamp = self.get_clock().now().to_msg()
        static_tf.header.frame_id = self.parent_frame
        static_tf.child_frame_id = self.child_frame
        
        # 位置
        static_tf.transform.translation.x = float(self.x)
        static_tf.transform.translation.y = float(self.y)
        static_tf.transform.translation.z = float(self.z)
        
        # 回転 (0,0,0)
        static_tf.transform.rotation.x = 0.0
        static_tf.transform.rotation.y = 0.0
        static_tf.transform.rotation.z = 0.0
        static_tf.transform.rotation.w = 1.0
        
        self.static_broadcaster.sendTransform(static_tf)

def main(args=None):
    rclpy.init(args=args)
    node = LidarStaticPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
