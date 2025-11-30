import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

class GroundRemovalNode(Node):
    def __init__(self):
        super().__init__('ground_removal_node')

        # パラメータの宣言 (デフォルト: -0.2m)
        self.declare_parameter('z_threshold', -0.13)
        self.z_threshold = self.get_parameter('z_threshold').get_parameter_value().double_value

        # サブスクライバー
        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.listener_callback,
            10
        )

        # パブリッシャー
        self.publisher = self.create_publisher(
            PointCloud2,
            '/livox/lidar/no_ground',
            10
        )

        self.get_logger().info(f'Ground Removal Node started using threshold Z > {self.z_threshold}')

    def listener_callback(self, msg):
        # 1. PointCloud2メッセージからデータを読み込む
        # 修正点: Livoxのデータ形式に合わせて全てのフィールドを指定します
        # トピック情報にある: x, y, z, intensity, tag, line, timestamp を全て取得
        points_generator = point_cloud2.read_points(
            msg, 
            field_names=("x", "y", "z", "intensity", "tag", "line", "timestamp"), 
            skip_nans=True
        )

        # 2. 床面除去フィルタリング
        filtered_points = []
        # point変数は上記のfield_namesの順序でタプルになっています
        # index 0: x, 1: y, 2: z ...
        for point in points_generator:
            if point[2] > self.z_threshold:
                filtered_points.append(point)

        # 点が一つも残らなかった場合は処理をスキップ
        if not filtered_points:
            return

        # 3. 新しいPointCloud2メッセージを作成
        header = msg.header
        
        # msg.fieldsには7つのフィールド定義が含まれています。
        # filtered_pointsの各要素も7つの値を持つようになったため、整合性が取れます。
        output_msg = point_cloud2.create_cloud(
            header,
            msg.fields, 
            filtered_points
        )

        # 4. パブリッシュ
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
