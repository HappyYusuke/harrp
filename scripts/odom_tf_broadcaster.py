#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class OdomTFBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')
        
        # パラメータ: フレーム名の設定（必要に応じて変更可能）
        self.declare_parameter('parent_frame', 'odom')
        self.declare_parameter('child_frame', 'livox_frame')
        
        self.parent_frame = self.get_parameter('parent_frame').value
        self.child_frame = self.get_parameter('child_frame').value

        # TF Broadcasterの初期化
        self.tf_broadcaster = TransformBroadcaster(self)

        # QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # オドメトリの購読
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile)
        
        self.get_logger().info(f'Start broadcasting TF: {self.parent_frame} -> {self.child_frame}')

    def odom_callback(self, msg):
        t = TransformStamped()

        # タイムスタンプを現在の時刻ではなく、メッセージの時刻に合わせるのが一般的
        # （データの同期ズレを防ぐため）
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        # 位置（Translation）
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        # 回転（Rotation）
        t.transform.rotation = msg.pose.pose.orientation

        # TF配信
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = OdomTFBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
