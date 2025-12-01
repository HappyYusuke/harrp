import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import sys

class GroundRemovalNode(Node):
    def __init__(self):
        super().__init__('ground_removal_node')

        # パラメータの宣言 (デフォルト: -0.13m)
        self.declare_parameter('z_threshold', -0.13)
        self.z_threshold = self.get_parameter('z_threshold').get_parameter_value().double_value

        # dtypeのキャッシュ用
        self.cached_dtype = None
        self.last_fields = None

        # QoS設定: センサーデータはBestEffort (qos_profile_sensor_data) で受信する
        # これにより、遅延した古いパケットの再送待ちなどを防ぎ、最新データを処理する
        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.listener_callback,
            qos_profile_sensor_data
        )

        # パブリッシャーは後段のノード(pp_infer)がReliableで待っている可能性があるため
        # デフォルト(Reliable)のままで出力する
        self.publisher = self.create_publisher(
            PointCloud2,
            '/livox/lidar/no_ground',
            10
        )

        self.get_logger().info(f'Fast Ground Removal Node started using NumPy & Cached Dtype. Threshold Z > {self.z_threshold}')

        # PointFieldのデータタイプとNumPyの型のマッピング
        self.datatype_map = {
            PointField.INT8:    np.int8,
            PointField.UINT8:   np.uint8,
            PointField.INT16:   np.int16,
            PointField.UINT16:  np.uint16,
            PointField.INT32:   np.int32,
            PointField.UINT32:  np.uint32,
            PointField.FLOAT32: np.float32,
            PointField.FLOAT64: np.float64,
        }

    def listener_callback(self, msg):
        # メッセージが空ならスキップ
        if len(msg.data) == 0:
            return

        # 1. PointCloud2のフィールド情報からNumPyのdtypeを構築 (キャッシュ利用)
        # 毎回作成するとPythonのオーバーヘッドがかかるため、フィールド構成が変わった時のみ作成
        if self.cached_dtype is None or msg.fields != self.last_fields:
            names = []
            formats = []
            offsets = []
            
            for field in msg.fields:
                if field.datatype in self.datatype_map:
                    numpy_type = self.datatype_map[field.datatype]
                else:
                    self.get_logger().error(f"Unsupported datatype: {field.datatype}")
                    return
                
                names.append(field.name)
                formats.append(numpy_type)
                offsets.append(field.offset)
            
            try:
                # itemsize=point_stepを指定することでパディングも扱える
                self.cached_dtype = np.dtype({
                    'names': names,
                    'formats': formats,
                    'offsets': offsets,
                    'itemsize': msg.point_step
                })
                self.last_fields = msg.fields
            except Exception as e:
                self.get_logger().error(f"Failed to create dtype: {e}")
                return

        # 2. バイナリデータをNumPy配列として一括読み込み (コピーなしで高速)
        try:
            cloud_array = np.frombuffer(msg.data, dtype=self.cached_dtype)
        except ValueError as e:
            self.get_logger().error(f"Failed to convert buffer: {e}. Data len: {len(msg.data)}, step: {msg.point_step}")
            return

        # 3. NumPyのブールインデックスで高速フィルタリング
        # 'z'フィールドが一括で評価され、True/Falseのマスクが作られる
        mask = cloud_array['z'] > self.z_threshold
        filtered_array = cloud_array[mask]

        # フィルタリング結果が空の場合
        if len(filtered_array) == 0:
            return

        # 4. 新しいPointCloud2メッセージの作成
        output_msg = PointCloud2()
        output_msg.header = msg.header
        output_msg.height = 1
        output_msg.width = len(filtered_array)
        output_msg.fields = msg.fields
        output_msg.is_bigendian = msg.is_bigendian
        output_msg.point_step = msg.point_step
        output_msg.row_step = output_msg.point_step * output_msg.width
        output_msg.is_dense = msg.is_dense
        
        # NumPy配列をバイト列に戻す
        output_msg.data = filtered_array.tobytes()

        self.publisher.publish(output_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GroundRemovalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
