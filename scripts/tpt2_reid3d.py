#!/usr/bin/env python3

import os
import sys
import collections
import time
import numpy as np
import torch
import torch.nn.functional as F
import math
import json
import cv2 
import glob
import bisect
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseStamped

import ros2_numpy as rnp

# ============================
# ReIDモデル設定
# ============================
HOME_DIR = os.environ['HOME']
sys.path.insert(0, f'{HOME_DIR}/ReID3D/reidnet/')

original_argv = sys.argv
sys.argv = [original_argv[0]]
try:
    from model import network
except ImportError:
    print("Error: Could not import 'model.network'.")
sys.argv = original_argv
# ============================

def normalize_point_cloud(points, num_points=256):
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

def get_yaw_from_orientation(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

# ==========================================
# UKF用の関数定義 (状態遷移関数と観測関数)
# ==========================================

def fx(x, dt):
    """
    状態遷移関数 (State Transition Function)
    等速直線運動モデル:
    位置 = 位置 + 速度 * dt
    速度 = 速度 (変化なし)
    x: [x, y, z, vx, vy, vz]
    """
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return np.dot(F, x)

def hx(x):
    """
    観測関数 (Measurement Function)
    センサー(LiDAR)は位置(x, y, z)のみを観測する
    """
    return x[:3]

# ==========================================
# KalmanBoxTracker (UKF版)
# ==========================================

class KalmanBoxTracker(object):
    def __init__(self, initial_pos):
        # シグマ点の生成設定
        # n=6 (状態変数の次元数), alpha, beta, kappaは一般的な推奨値
        points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=0.)

        # UKFの初期化
        # dim_x=6 (x,y,z,vx,vy,vz), dim_z=3 (x,y,z)
        self.ukf = UKF(dim_x=6, dim_z=3, dt=0.1, fx=fx, hx=hx, points=points)

        # 初期状態の設定
        self.ukf.x = np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0, 0, 0])

        # 共分散行列 P (初期の不確かさ)
        self.ukf.P *= 1.0

        # ★プロセスノイズ Q (モデルの不確かさ)
        # ここを小さくすると「等速直線運動」を強く信じるようになります
        # すれ違い対策には小さめ (0.01~0.05) が推奨
        self.ukf.Q = np.eye(6) * 0.05

        # ★観測ノイズ R (センサーの不確かさ)
        # ここを大きくすると「観測値（他人の位置）」に飛びつきにくくなります
        # すれ違い対策には大きめ (2.0~5.0) が推奨
        self.ukf.R = np.eye(3) * 5.0

        self.last_timestamp = None
        
        # 外部参照用 (互換性のため)
        self.x = self.ukf.x

    def predict(self, current_time):
        if self.last_timestamp is None:
            dt = 0.1 # 初回用ダミー
        else:
            dt = current_time - self.last_timestamp
        
        self.last_timestamp = current_time
        
        # UKFの予測ステップ
        self.ukf.predict(dt=dt)
        
        # 参照用変数の更新
        self.x = self.ukf.x
        
        # 予測位置 (x, y, z) を返す
        return self.ukf.x[:3]

    def update(self, measurement):
        # measurement: numpy array [x, y, z]
        
        # UKFの更新ステップ
        self.ukf.update(measurement)
        
        # 参照用変数の更新
        self.x = self.ukf.x

# ==========================================

class Candidate:
    def __init__(self, id, pos, size, orientation, initial_points):
        self.id = id
        self.size = size
        self.orientation = orientation
        self.queue = collections.deque(maxlen=30)
        self.queue.append(initial_points)
        self.last_sim = 0.0
        
        self.kf = KalmanBoxTracker(pos)
        self.pos = pos 
        self.pred_pos = pos 
        self.last_seen_time = time.time()

    def predict(self, current_time):
        self.pred_pos = self.kf.predict(current_time)
        return self.pred_pos

    def update_state(self, pos, size, orientation):
        self.kf.update(pos) 
        self.pos = self.kf.x[:3] 
        self.size = size
        self.orientation = orientation
        self.last_seen_time = time.time()

    def add_points(self, points):
        self.queue.append(points)

class PersonTrackerClickInitNode(Node):
    def __init__(self):
        super().__init__('person_tracker_click_init')

        # --- パラメータ ---
        self.declare_parameter('max_missing_time', 1.0)
        self.declare_parameter('reid_sim_thresh', 0.70)
        
        self.MAX_MISSING_TIME = self.get_parameter('max_missing_time').value
        self.REID_SIM_THRESH = self.get_parameter('reid_sim_thresh').value
        
        # --- モデルロード ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._load_model()
        self.get_logger().info(f'Model Loaded on {self.device}')

        # --- ★修正: 暗闇のみのGTをロード ---
        self.gt_timestamps = []
        # ファイル名を 0035_dark.json に変更
        gt_path = os.path.expanduser("~/tpt-bench/GTs/0035_dark.json") 
        
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
                # 検索用に整数化してソートしておく
                self.gt_timestamps = sorted([int(k) for k in gt_data.keys()])
            self.get_logger().info(f">>> Loaded {len(self.gt_timestamps)} DARK frames from {gt_path}")
        else:
            self.get_logger().error(f"GT file not found: {gt_path}")

        # --- 通信 ---
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_bbox = self.create_subscription(Detection3DArray, '/bbox', self.bbox_callback, qos_profile)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        self.pub_markers = self.create_publisher(MarkerArray, '/person_reid_markers', 10)
        self.pub_status = self.create_publisher(String, '/target_status', 10)

        # --- 状態 ---
        self.state = "WAIT_FOR_CLICK"
        self.target_candidate = None    
        self.registered_feature = None 
        self.feature_locked = False     
        self.captured_frames_count = 0 
        
        self.candidates = {} 
        self.next_candidate_id = 0
        
        self.json_results = {}
        
        self.get_logger().info(">>> Waiting for Click... Use '2D Goal Pose' in RViz.")

    def _load_model(self):
        try:
            net = torch.nn.DataParallel(network.reid3d(1024, num_class=222, stride=1))
            weight_path = f'{HOME_DIR}/ReID3D/reidnet/log/ckpt_best.pth'
            if not os.path.exists(weight_path):
                sys.exit(1)
            weight = torch.load(weight_path, map_location=self.device)
            net.load_state_dict(weight)
            net.to(self.device)
            net.eval()
            return net
        except Exception as e:
            sys.exit(1)

    def extract_feature_single(self, points_sequence):
        if len(points_sequence) < 30: return None
        seq_np = np.array(points_sequence)
        tensor = torch.from_numpy(seq_np).float()
        input_tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.net(input_tensor)
            feature = output['val_bn'][0]
            feature = F.normalize(feature.unsqueeze(0), p=2, dim=1).squeeze(0)
        return feature

    def goal_callback(self, msg: PoseStamped):
        click_x = msg.pose.position.x
        click_y = msg.pose.position.y
        self.get_logger().info(f">>> Click Received at ({click_x:.2f}, {click_y:.2f})")
        
        closest_cand = None
        min_dist = 3.0
        
        for cid, cand in self.candidates.items():
            dist = math.sqrt((cand.pos[0] - click_x)**2 + (cand.pos[1] - click_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_cand = cand
        
        if closest_cand:
            self.target_candidate = closest_cand
            self.captured_frames_count = len(closest_cand.queue)
            self.feature_locked = False
            self.registered_feature = None
            self.state = "INITIALIZING"
            self.get_logger().info(f">>> Target Selected: ID {closest_cand.id}. Gathering frames...")
        else:
            self.get_logger().warn(">>> No candidate found near click position!")

    def bbox_callback(self, msg: Detection3DArray):
        current_time = time.time()
        
        # 1. LiDARの生時刻を取得
        if hasattr(msg.header.stamp, 'sec'):
            raw_ts = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        else:
            raw_ts = msg.header.stamp.nanoseconds

        # 2. ★修正: 暗闇GTリストとのマッチングとフィルタリング
        target_timestamp_str = None
        
        if self.gt_timestamps:
            # 二分探索で最も近いGT時刻を探す
            idx = bisect.bisect_left(self.gt_timestamps, raw_ts)
            
            candidates = []
            if idx < len(self.gt_timestamps): candidates.append(self.gt_timestamps[idx])
            if idx > 0: candidates.append(self.gt_timestamps[idx - 1])
            
            if candidates:
                # 差が最小のGT時刻を選ぶ
                nearest_ts = min(candidates, key=lambda x: abs(x - raw_ts))
                diff = abs(nearest_ts - raw_ts)
                
                # ★フィルタリング条件: 
                # 誤差が100ms (100,000,000ns) 以内なら「暗闇フレーム」とみなして処理する
                # それ以外は「明るいフレーム」または「GTがないフレーム」として無視する
                if diff < 100000000:
                    target_timestamp_str = str(nearest_ts)
                else:
                    # 暗闇GTに含まれない時刻なので、このコールバックはJSON保存せずに終了しても良いが、
                    # 追跡(KalmanFilter)自体は継続させたほうが精度が良いので処理は通す。
                    # ただし、JSONキーとしては無効にする。
                    target_timestamp_str = None 
        else:
            # GTファイルがない場合のフォールバック（今回は使わない想定）
            target_timestamp_str = str(raw_ts)

        # --- 以下、通常の推論処理 ---

        detections = []
        for detection in msg.detections:
            if detection.source_cloud.width * detection.source_cloud.height == 0: continue
            try:
                raw_points = rnp.point_cloud2.pointcloud2_to_xyz_array(detection.source_cloud)
            except: continue
            if raw_points.shape[0] < 5: continue 

            norm_points = normalize_point_cloud(raw_points, num_points=256)
            bbox = detection.bbox
            pos = np.array([bbox.center.position.x, bbox.center.position.y, bbox.center.position.z])
            size = np.array([bbox.size.x, bbox.size.y, bbox.size.z])
            detections.append({'pos': pos, 'size': size, 'ori': bbox.center.orientation, 'points': norm_points})

        # 3. 候補更新 (常に実行してトラッキングを維持する)
        self.update_candidates(detections, current_time)

        # 4. JSONデータ生成準備
        current_json_data = [0, 0, 0, 0, -1]
        
        if self.state == "INITIALIZING":
            self.process_initialization()
            if self.target_candidate:
                current_json_data = self.get_2d_target_info(self.target_candidate)
        elif self.state == "TRACKING" or self.state == "LOST":
            # 自律追跡ロジック
            # ※ここで渡すタイムスタンプはログ用なので raw_ts でも target_timestamp_str でもOK
            self.process_autonomous_tracking(raw_ts)
            if self.state == "TRACKING" and self.target_candidate:
                current_json_data = self.get_2d_target_info(self.target_candidate)

        # 5. ★修正: フィルタリングされたキーがある場合のみ保存
        if target_timestamp_str is not None:
            target_id = self.target_candidate.id if self.target_candidate else 0
            tracks_list = [[target_id] + current_json_data]
            
            self.json_results[target_timestamp_str] = {
                "target_info": current_json_data, 
                "tracks_target_conf_bbox": tracks_list
            }
            
            # 定期保存
            if len(self.json_results) % 50 == 0:
                self.save_results_to_json()

        # ... (画像デバッグ保存などは必要に応じて) ...

        self.publish_visualization(msg.header)
        
    def update_candidates(self, detections, current_time):
        matched_det_indices = set()
        active_ids = set()
        
        for cid, cand in self.candidates.items():
            cand.predict(current_time)

        for cid, cand in self.candidates.items():
            best_dist = 1.0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in matched_det_indices: continue
                dist = np.linalg.norm(det['pos'] - cand.pred_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            if best_idx != -1:
                det = detections[best_idx]
                cand.update_state(det['pos'], det['size'], det['ori'])
                cand.add_points(det['points'])
                matched_det_indices.add(best_idx)
                active_ids.add(cid)
            elif (current_time - cand.last_seen_time) < 1.0:
                active_ids.add(cid)

        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                nid = self.next_candidate_id
                self.next_candidate_id += 1
                self.candidates[nid] = Candidate(nid, det['pos'], det['size'], det['ori'], det['points'])
                active_ids.add(nid)

        self.candidates = {k: v for k, v in self.candidates.items() if k in active_ids}

    def process_initialization(self):
        status_msg = "INITIALIZING"
        if self.target_candidate and self.target_candidate.id in self.candidates:
            self.target_candidate = self.candidates[self.target_candidate.id]
            self.captured_frames_count = len(self.target_candidate.queue)
            status_msg += f": {self.captured_frames_count}/30"
            
            if self.captured_frames_count >= 30:
                feature = self.extract_feature_single(list(self.target_candidate.queue))
                if feature is not None:
                    self.registered_feature = feature
                    self.feature_locked = True
                    self.state = "TRACKING"
                    self.get_logger().info(">>> Feature LOCKED! Tracking Started.")
        else:
             status_msg = "Target Lost during Init"
             
        self.pub_status.publish(String(data=status_msg))

    def process_autonomous_tracking(self, timestamp_nanosec):
        target_found = False
        
        if self.target_candidate and self.target_candidate.id in self.candidates:
            self.target_candidate = self.candidates[self.target_candidate.id]
            target_found = True
            
        if not target_found:
            self.state = "LOST"
            status = "LOST"
            best_sim = -1.0
            best_cand = None
            
            for cid, cand in self.candidates.items():
                if len(cand.queue) < 30: continue
                feat = self.extract_feature_single(list(cand.queue))
                if feat is None: continue
                sim = torch.dot(feat, self.registered_feature).item()
                cand.last_sim = sim 
                if sim > self.REID_SIM_THRESH and sim > best_sim:
                    best_sim = sim
                    best_cand = cand
            
            if best_cand:
                self.get_logger().info(f"ReID RECOVERY! New ID:{best_cand.id} Sim:{best_sim:.2f}")
                self.target_candidate = best_cand
                self.target_candidate.last_sim = best_sim
                self.state = "TRACKING"
                target_found = True
                status = f"RECOVERED ({best_sim:.2f})"
            self.pub_status.publish(String(data=status))
        else:
            self.pub_status.publish(String(data=f"TRACKING ID:{self.target_candidate.id}"))

    def get_2d_target_info(self, candidate):
        pos_3d = candidate.pos
        size_3d = candidate.size
        confidence = candidate.last_sim if candidate.last_sim > 0 else 1.0

        img_w = 1920
        img_h = 960

        x, y, z = pos_3d[0], pos_3d[1], pos_3d[2]
        
        # 1. 角度の算出
        theta = math.atan2(y, x) 
        dist_2d = math.sqrt(x**2 + y**2)
        
        # ★垂直オフセットの調整 (BBoxを下に下げる)
        # 仰角 phi を計算する際、z座標からカメラとLiDARの高さの差（例: 0.2m）を引きます
        # この 0.2 の値を調整することで、BBoxが上下に動きます
        camera_height_offset = 0.4 
        phi = math.atan2(z - camera_height_offset, dist_2d)     

        # 2. ピクセル座標への変換
        u = (0.5 - theta / (2 * math.pi)) * img_w
        v = (0.5 - phi / math.pi) * img_h

        # 3. BBoxサイズの算出とスケール調整
        dist_3d = math.sqrt(x**2 + y**2 + z**2)
        if dist_3d < 0.1: dist_3d = 0.1
        
        # ★人物をしっかり囲むよう、幅(w)と高さ(h)を 1.2倍 に拡大
        w_2d = (size_3d[1] / dist_3d) * (img_w / (2 * math.pi)) * 1.4
        h_2d = (size_3d[2] / dist_3d) * (img_h / math.pi) * 1.4

        # 4. 左上座標の算出
        u_tl = int(u - (w_2d / 2))
        v_tl = int(v - (h_2d / 2))
        
        u_tl = int(u_tl % img_w)
        
        return [u_tl, v_tl, int(w_2d), int(h_2d), confidence]

    def publish_visualization(self, header):
        marker_array = MarkerArray()
        target_id = -1
        if self.target_candidate:
            target_id = self.target_candidate.id

        for cid, cand in self.candidates.items():
            is_target = (target_id != -1 and cid == target_id)
            
            # --- 1. 物体マーカー (Cube/Sphere) ---
            mk = Marker()
            mk.header = header
            mk.ns = "candidates"
            mk.id = cid
            mk.action = Marker.ADD
            mk.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = cand.pos
            q = cand.orientation
            norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
            if norm < 1e-6:
                mk.pose.orientation.w = 1.0
            else:
                mk.pose.orientation.x = q.x / norm
                mk.pose.orientation.y = q.y / norm
                mk.pose.orientation.z = q.z / norm
                mk.pose.orientation.w = q.w / norm

            if is_target:
                mk.type = Marker.CUBE
                mk.scale.x = max(cand.size[0], 0.2)
                mk.scale.y = max(cand.size[1], 0.2)
                mk.scale.z = max(cand.size[2], 0.2)
                if self.feature_locked:
                    mk.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5) 
                else:
                    mk.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.5) 
            else:
                mk.type = Marker.SPHERE
                mk.scale.x = mk.scale.y = mk.scale.z = 0.5
                mk.color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.3) 
            
            marker_array.markers.append(mk)

            # --- 2. ★復活: テキストマーカー ---
            text = Marker()
            text.header = header
            text.ns = "text"
            text.id = cid
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            # テキストの位置調整 (物体の少し上)
            text.pose.position.x = cand.pos[0]
            text.pose.position.y = cand.pos[1]
            text.pose.position.z = cand.pos[2] + (cand.size[2] if is_target else 0.5) + 0.5
            text.scale.z = 0.3
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

            label = f"ID:{cid}"
            if cand.last_sim > 0: label += f"\nSim:{cand.last_sim:.2f}"
            if is_target and not self.feature_locked: label += f"\nInit:{self.captured_frames_count}/30"

            text.text = label
            text.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker_array.markers.append(text)
        
        self.pub_markers.publish(marker_array)

    def save_results_to_json(self):
        filename = 'evaluation_results.json'
        self.get_logger().info(f"Saving JSON results to {filename} ...")
        try:
            with open(filename, 'w') as f:
                json.dump(self.json_results, f, indent=4)
            self.get_logger().info("Save Complete!")
        except Exception as e:
            self.get_logger().error(f"Failed to save JSON: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerClickInitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_results_to_json()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
