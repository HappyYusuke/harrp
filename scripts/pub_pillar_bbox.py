#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

class PubPillarBbox(Node):
    def __init__(self):
        super().__init__('pub_pillar_bbox')
        
        # 3D検出結果トピックの購読
        # TODO: 実際のトピック名に合わせて変更してください
        self.subscription = self.create_subscription(
            Detection3DArray,
            '/bbox',  
            self.detection_callback,
            10)
        
        # MarkerArray配信用パブリッシャー
        self.marker_publisher = self.create_publisher(MarkerArray, '/detections/markers', 10)

    def detection_callback(self, msg: Detection3DArray):
    	# 処理速度計測用
        start_time = time.perf_counter()
    	
        marker_array = MarkerArray()
    	
    	# 検出された各オブジェクトに対してマーカーを作成
        for i, detection in enumerate(msg.detections):
            marker = Marker()
            marker.header = msg.header
            marker.ns = "detections"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # バウンディングボックスのz軸を下げる(TAO PointPillarsとROS2のギャップ)
            detection.bbox.center.position.z \
                    = detection.bbox.center.position.z #+ (detection.bbox.size.z/2)

            # バウンディングボックスの中心位置と姿勢を設定
            marker.pose = detection.bbox.center
            
            # バウンディングボックスの大きさを設定
            marker.scale = detection.bbox.size
            
            # マーカーの色を設定 (例: 半透明の赤色)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.4  # 透明度
            
            # マーカーの表示時間を設定 (例: 1秒)
            marker.lifetime = Duration(seconds=1.0).to_msg()
            
            marker_array.markers.append(marker)
            
        # MarkerArrayをパブリッシュ
        self.marker_publisher.publish(marker_array)
        
        # 処理速度計測用
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        #self.get_logger().info(f'Processing time: {elapsed_ms:.3f} ms')


def main(args=None):
    rclpy.init(args=args)
    node = PubPillarBbox()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
