import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

import ros2_numpy as rnp
import numpy as np
import torch
import torch.nn.functional as F
import collections

# モデルとユーティリティ関数のインポート
HOME_DIR = os.environ['HOME']
sys.path.insert(0, f'{HOME_DIR}/ReID3D/reidnet/')

# ============================
# parser対策
# ============================
"""
from model import networkをするときに、parserがimportされros2 launchできないので、
ROSの引数をoriginal_argvに隠し、import後にもとに戻す。
"""
original_argv = sys.argv
sys.argv = [original_argv[0]]

from model import network

sys.argv = original_argv

# ============================

# normalize_point_cloud はクラスの状態に依存しないため、クラス外のヘルパー関数としておくのが適切
def normalize_point_cloud(points, num_points=256):
    # ... (この関数の内容は変更なし) ...
    if points.shape[0] == 0:
        return np.zeros((num_points, 3))
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    num_current_points = points_centered.shape[0]
    if num_current_points > num_points:
        indices = np.random.choice(num_current_points, num_points, replace=False)
        normalized_points = points_centered[indices, :]
    else:
        extra_indices = np.random.choice(num_current_points, num_points - num_current_points, replace=True)
        additional_points = points_centered[extra_indices, :]
        normalized_points = np.vstack((points_centered, additional_points))
    return normalized_points


class ReID3D_ROS2(Node):
    def __init__(self):
        super().__init__('reid3d_ros2')

        # --- パラメータ設定 ---
        self.ROI_X_RANGE = (0.5, 3.0)
        self.ROI_Y_RANGE = (-0.5, 0.5)
        self.ROI_Z_RANGE = (-0.2, 1.8)
        self.MIN_POINTS_THRESHOLD = 10
        self.SEQUENCE_LENGTH = 30
        self.SIMILARITY_THRESHOLD = 0.85

        # --- ROSインターフェース ---
        self.create_subscription(PointCloud2, '/livox/lidar', self.callback, 10)
        self.person_points_pub = self.create_publisher(PointCloud2, '/person_points', 10)
        self.label_marker_pub = self.create_publisher(Marker, '/person_label', 10)

        # --- Re-ID関連 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._load_model()
        self.point_cloud_queue = collections.deque(maxlen=self.SEQUENCE_LENGTH)
        self.gallery = {}
        self.next_person_id = 0
        
        self.get_logger().info('初期化完了。推論待機中です...')

    def _load_model(self):
        """モデルをロードして初期化する"""
        net = torch.nn.DataParallel(network.reid3d(1024, num_class=222, stride=1))
        weight = torch.load(f'{HOME_DIR}/ReID3D/reidnet/log/ckpt_best.pth')
        net.load_state_dict(weight)
        net.to(self.device)
        net.eval()
        self.get_logger().info('モデルの読み込みが完了しました。')
        return net

    def parse_point_cloud(self, msg):
        """ROSのPointCloud2メッセージをNumPy配列に変換する"""
        try:
            return rnp.point_cloud2.pointcloud2_to_xyz_array(msg)
        except Exception as e:
            self.get_logger().error(f"PointCloudのパースに失敗: {e}")
            return None

    def extract_person_points(self, pc_data: np.ndarray):
        """ROIに基づいて人物の点群を抽出する"""
        mask = (pc_data[:, 0] >= self.ROI_X_RANGE[0]) & (pc_data[:, 0] <= self.ROI_X_RANGE[1]) & \
               (pc_data[:, 1] >= self.ROI_Y_RANGE[0]) & (pc_data[:, 1] <= self.ROI_Y_RANGE[1]) & \
               (pc_data[:, 2] >= self.ROI_Z_RANGE[0]) & (pc_data[:, 2] <= self.ROI_Z_RANGE[1])
        return pc_data[mask]

    def perform_reid(self, query_feature: torch.Tensor):
        """クエリ特徴量とギャラリーを比較し、IDを判定または新規登録する"""
        if not self.gallery:
            new_id = f"person_{self.next_person_id}"
            self.gallery[new_id] = query_feature
            self.next_person_id += 1
            return f"NEW: {new_id}"
        
        max_similarity, best_match_id = -1.0, None
        for person_id, gallery_feature in self.gallery.items():
            similarity = F.cosine_similarity(query_feature.unsqueeze(0), gallery_feature.unsqueeze(0)).item()
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_id = person_id
        
        if max_similarity > self.SIMILARITY_THRESHOLD:
            return f"ID: {best_match_id}\nSim: {max_similarity:.2f}"
        else:
            new_id = f"person_{self.next_person_id}"
            self.gallery[new_id] = query_feature
            self.next_person_id += 1
            return f"NEW: {new_id}"

    def publish_visualizations(self, points: np.ndarray, text: str, header):
        """検出した点群とIDラベルをrviz2にパブリッシュする"""
        # 1. 色付き点群のパブリッシュ
        colored_points = np.zeros(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        colored_points['x'], colored_points['y'], colored_points['z'] = points[:, 0], points[:, 1], points[:, 2]
        colored_points['r'], colored_points['g'], colored_points['b'] = 0, 255, 0 # Green
        pc2_msg = rnp.point_cloud2.array_to_pointcloud2(colored_points, stamp=header.stamp, frame_id=header.frame_id)
        self.person_points_pub.publish(pc2_msg)

        # 2. テキストマーカーのパブリッシュ
        marker = Marker(header=header, ns="person_reid", id=0, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
        text_position = np.mean(points, axis=0)
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = float(text_position[0]), float(text_position[1]), float(text_position[2]) + 0.5
        marker.pose.orientation.w = 1.0
        marker.text, marker.scale.z = text, 0.2
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        marker.lifetime = rclpy.duration.Duration(seconds=2.0).to_msg()
        self.label_marker_pub.publish(marker)

    def callback(self, msg: PointCloud2):
        """メインのコールバック関数。各メソッドを呼び出して処理を調整する。"""
        pc_data = self.parse_point_cloud(msg)
        if pc_data is None:
            return

        person_points = self.extract_person_points(pc_data)
        if person_points.shape[0] < self.MIN_POINTS_THRESHOLD:
            self.point_cloud_queue.clear()
            return
            
        normalized_frame = normalize_point_cloud(person_points, num_points=256)
        self.point_cloud_queue.append(normalized_frame)

        if len(self.point_cloud_queue) < self.SEQUENCE_LENGTH:
            self.publish_visualizations(person_points, "Gathering data...", msg.header)
            return

        sequence_np = np.array(self.point_cloud_queue)
        tensor = torch.from_numpy(sequence_np).float()
        input_tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.net(input_tensor)
        
        query_feature = output['val_bn'][0]
        display_text = self.perform_reid(query_feature)
        
        self.publish_visualizations(person_points, display_text, msg.header)

# main関数 (変更なし)
def main():
    rclpy.init()
    node = ReID3D_ROS2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
