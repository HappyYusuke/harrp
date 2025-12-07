#!/usr/bin/env python3

import os
import sys
import collections
import time
import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import Detection3DArray

import ros2_numpy as rnp

# ============================
# ReIDモデルのインポート
# ============================
HOME_DIR = os.environ['HOME']
sys.path.insert(0, f'{HOME_DIR}/ReID3D/reidnet/')

original_argv = sys.argv
sys.argv = [original_argv[0]]
from model import network
sys.argv = original_argv
# ============================

def get_unique_rgb(id_val):
    """IDに基づいて一意のRGB値を生成"""
    np.random.seed(id_val)
    rgb = np.random.rand(3)
    rgb = rgb * 0.5 + 0.5 
    return (rgb[0], rgb[1], rgb[2])

def normalize_point_cloud(points, num_points=256):
    """点群の正規化"""
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

class PersonTracker:
    def __init__(self, track_id, initial_pos, size, orientation):
        self.id = track_id
        self.pos = initial_pos       
        self.size = size             
        self.orientation = orientation
        
        self.queue = collections.deque(maxlen=30) 
        self.last_seen = time.time()
        self.reid_result = "Gathering..."
        self.person_id_int = -1 
        
        # ★追加: 最後にReID推論を行った時間
        self.last_reid_time = 0.0

    def update(self, pos, points, size, orientation):
        self.pos = pos
        self.size = size
        self.orientation = orientation
        self.last_seen = time.time()
        
        norm_points = normalize_point_cloud(points, num_points=256)
        self.queue.append(norm_points)

class ReID3D_PointPillars(Node):
    def __init__(self):
        super().__init__('reid3d_pointpillars')

        # --- パラメータ ---
        self.SEQUENCE_LENGTH = 30
        self.SIMILARITY_THRESHOLD = 0.85
        self.TRACK_DISTANCE_THRESH = 1.0
        self.TRACK_TIMEOUT = 2.0
        
        # ★追加: ReID推論を行う間隔 (秒)
        # ここを長くすると負荷が下がり遅延が減ります
        self.REID_INTERVAL = 1.0 

        self.create_subscription(Detection3DArray, '/bbox', self.bbox_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/person_reid_markers', 10)

        self.trackers = []
        self.next_track_id = 0
        self.gallery = {}
        self.next_person_id = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._load_model()
        
        self.get_logger().info('ReID3D Node Started. Optimized for low latency.')

    def _load_model(self):
        try:
            net = torch.nn.DataParallel(network.reid3d(1024, num_class=222, stride=1))
            weight_path = f'{HOME_DIR}/ReID3D/reidnet/log/ckpt_best.pth'
            if not os.path.exists(weight_path):
                self.get_logger().error(f"モデルファイルが見つかりません: {weight_path}")
                sys.exit(1)
            weight = torch.load(weight_path)
            net.load_state_dict(weight)
            net.to(self.device)
            net.eval()
            return net
        except Exception as e:
            self.get_logger().error(f"モデル読み込みエラー: {e}")
            sys.exit(1)

    def bbox_callback(self, msg: Detection3DArray):
        current_time = time.time()
        matched_tracker_indices = []

        for detection in msg.detections:
            bbox = detection.bbox
            center = np.array([bbox.center.position.x, bbox.center.position.y, bbox.center.position.z])
            size = np.array([bbox.size.x, bbox.size.y, bbox.size.z])
            orientation = bbox.center.orientation

            # 点群取得
            if detection.source_cloud.width * detection.source_cloud.height == 0:
                continue
            try:
                cloud_arr = rnp.point_cloud2.pointcloud2_to_xyz_array(detection.source_cloud)
            except Exception:
                continue
            if cloud_arr.shape[0] < 10:
                continue

            # トラッキング (位置合わせ)
            best_idx = -1
            min_dist = self.TRACK_DISTANCE_THRESH

            for i, tracker in enumerate(self.trackers):
                if i in matched_tracker_indices:
                    continue
                dist = np.linalg.norm(tracker.pos - center)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            if best_idx != -1:
                # 既存トラッカー更新
                # ★ここが重要: 位置(pos)は毎フレーム更新されるため、描画はスムーズになります
                self.trackers[best_idx].update(center, cloud_arr, size, orientation)
                matched_tracker_indices.append(best_idx)
                target_tracker = self.trackers[best_idx]
            else:
                # 新規作成
                new_tracker = PersonTracker(self.next_track_id, center, size, orientation)
                new_tracker.update(center, cloud_arr, size, orientation)
                self.trackers.append(new_tracker)
                self.next_track_id += 1
                target_tracker = new_tracker

            # --- ReID推論判定 ---
            # 1. データが溜まっていること
            # 2. 前回の推論から一定時間(REID_INTERVAL)経過していること
            # 3. または、まだIDが確定していない(Gathering...)場合は優先的に実行してもよい
            is_ready = len(target_tracker.queue) == self.SEQUENCE_LENGTH
            time_elapsed = (current_time - target_tracker.last_reid_time) > self.REID_INTERVAL
            is_unknown = target_tracker.person_id_int == -1

            if is_ready and (time_elapsed or is_unknown):
                self.run_reid_inference(target_tracker, current_time)

        self.trackers = [t for t in self.trackers if (current_time - t.last_seen) < self.TRACK_TIMEOUT]
        self.publish_markers(msg.header)

    def run_reid_inference(self, tracker, current_time):
        sequence_np = np.array(tracker.queue)
        tensor = torch.from_numpy(sequence_np).float()
        input_tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.net(input_tensor)
        query_feature = output['val_bn'][0]

        # 最後に推論した時間を更新
        tracker.last_reid_time = current_time

        if not self.gallery:
            self._register_new_person(tracker, query_feature)
            return

        max_sim = -1.0
        best_id_str = None
        best_id_int = -1
        
        for pid_str, (feat, pid_int) in self.gallery.items():
            sim = F.cosine_similarity(query_feature.unsqueeze(0), feat.unsqueeze(0)).item()
            if sim > max_sim:
                max_sim = sim
                best_id_str = pid_str
                best_id_int = pid_int
        
        if max_sim > self.SIMILARITY_THRESHOLD:
            tracker.reid_result = f"{best_id_str}\n({max_sim:.2f})"
            tracker.person_id_int = best_id_int
        else:
            self._register_new_person(tracker, query_feature)

    def _register_new_person(self, tracker, feature):
        new_id_str = f"Person_{self.next_person_id}"
        self.gallery[new_id_str] = (feature, self.next_person_id)
        tracker.reid_result = f"{new_id_str}\n(New)"
        tracker.person_id_int = self.next_person_id
        self.next_person_id += 1

    def publish_markers(self, header):
        marker_array = MarkerArray()
        for tracker in self.trackers:
            if tracker.person_id_int == -1:
                r, g, b = 1.0, 1.0, 1.0
                box_alpha = 0.3
            else:
                r, g, b = get_unique_rgb(tracker.person_id_int)
                box_alpha = 0.5

            # Box
            box_marker = Marker()
            box_marker.header = header
            box_marker.ns = "reid_box"
            box_marker.id = tracker.id
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD
            box_marker.pose.position.x = tracker.pos[0]
            box_marker.pose.position.y = tracker.pos[1]
            box_marker.pose.position.z = tracker.pos[2]
            box_marker.pose.orientation = tracker.orientation
            box_marker.scale.x = tracker.size[0]
            box_marker.scale.y = tracker.size[1]
            box_marker.scale.z = tracker.size[2]
            box_marker.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=box_alpha)
            box_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            marker_array.markers.append(box_marker)

            # Text
            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "reid_text"
            text_marker.id = tracker.id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = tracker.pos[0]
            text_marker.pose.position.y = tracker.pos[1]
            text_marker.pose.position.z = tracker.pos[2] + (tracker.size[2] / 2.0) + 0.5
            text_marker.scale.z = 0.3
            text_marker.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0)
            text_marker.text = tracker.reid_result
            text_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            marker_array.markers.append(text_marker)
            
        self.marker_pub.publish(marker_array)

def main():
    rclpy.init()
    node = ReID3D_PointPillars()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
