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
import threading  # ★追加: 非同期処理用

# --- FilterPy Imports ---
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

# ==========================================
# UKF Functions & Class
# ==========================================

def fx(x, dt):
    """ 状態遷移関数: 等速直線運動モデル """
    # x: [x, y, z, vx, vy, vz]
    F = np.eye(6, dtype=float)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return np.dot(F, x)

def hx(x):
    """ 観測関数: 状態から位置(x,y,z)を取り出す """
    return x[:3]

class KalmanBoxTracker(object):
    def __init__(self, initial_pos):
        # ... (シグマポイントの設定などはそのまま) ...
        points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2.0, kappa=-3) 
        self.kf = UKF(dim_x=6, dim_z=3, dt=0.1, fx=fx, hx=hx, points=points)
        
        self.kf.x[:3] = initial_pos
        self.kf.x[3:] = 0
        
        self.kf.P *= 1.0 
        self.kf.P[3:, 3:] *= 10.0
        
        # ==========================================
        # ★チューニング箇所: すれ違いに強くする設定
        # ==========================================
        
        # 1. 観測ノイズ (R): 少し大きくして、センサの一時的なブレや融合を無視させる
        # デフォルト 0.1 -> 0.5 くらいまで上げると、軌道が滑らかになり吸着しにくくなる
        sensor_noise_std = 0.3
        self.kf.R = np.diag([sensor_noise_std, sensor_noise_std, sensor_noise_std]) ** 2 
        
        # 2. プロセスノイズ (Q): 速度の変化(急ターン・急停止)をあまり許容しないようにする
        # 等速直線運動モデルを強く信じさせる
        self.kf.Q = np.eye(6) * 0.05**2
        
        # 以前は *= 20 でしたが、これを小さくすると「慣性」が強くなります。
        # すれ違い重視なら 1.0 〜 5.0 程度推奨。
        self.kf.Q[3:, 3:] *= 0.1
        
        self.last_timestamp = None

    def predict(self, current_time):
        if self.last_timestamp is None:
            dt = 0.1
        else:
            dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        
        self.kf.predict(dt=dt)
        return self.kf.x[:3]

    def update(self, measurement):
        self.kf.update(measurement)

# ==========================================

class Candidate:
    def __init__(self, id, pos, size, orientation, initial_points):
        self.id = id
        self.size = size
        self.orientation = orientation
        self.queue = collections.deque(maxlen=30)
        self.queue.append(initial_points)
        self.last_sim = 0.0
        self.feature_gallery = collections.deque(maxlen=100)
        
        self.kf = KalmanBoxTracker(pos)
        
        self.pos = pos 
        self.pred_pos = pos 
        self.last_seen_time = time.time()

    def predict(self, current_time):
        self.pred_pos = self.kf.predict(current_time)
        return self.pred_pos

    def update_state(self, pos, size, orientation):
        self.kf.update(pos)
        self.pos = self.kf.kf.x[:3] 
        self.size = size
        self.orientation = orientation
        self.last_seen_time = time.time()

    def add_points(self, points):
        self.queue.append(points)

    def get_feature_distribution(self):
        if len(self.feature_gallery) == 0:
            return None, None
        gallery_tensor = torch.stack(list(self.feature_gallery))
        mean_feature = torch.mean(gallery_tensor, dim=0)
        mean_feature = F.normalize(mean_feature.unsqueeze(0), p=2, dim=1).squeeze(0)
        std_feature = torch.std(gallery_tensor, dim=0)
        return mean_feature, std_feature

    def update_feature_gallery(self, feature_tensor):
        self.feature_gallery.append(feature_tensor.detach().cpu())

class PersonTrackerClickInitNode(Node):
    def __init__(self):
        super().__init__('person_tracker_click_init')
        self.declare_parameter('max_missing_time', 1.0)
        self.declare_parameter('reid_sim_thresh', 0.80)
        self.MAX_MISSING_TIME = self.get_parameter('max_missing_time').value
        self.REID_SIM_THRESH = self.get_parameter('reid_sim_thresh').value
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._load_model()
        self.get_logger().info(f'Model Loaded on {self.device}')

        self.USE_WEIGHTED_SCORE = True

        self.gt_timestamps = []
        gt_path = os.path.expanduser("~/tpt-bench/GTs/0035_dark.json") 
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
                self.gt_timestamps = sorted([int(k) for k in gt_data.keys()])
            self.get_logger().info(f">>> Loaded {len(self.gt_timestamps)} DARK frames from {gt_path}")
        else:
            self.get_logger().error(f"GT file not found: {gt_path}")

        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_bbox = self.create_subscription(Detection3DArray, '/bbox', self.bbox_callback, qos_profile)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/person_reid_markers', 10)
        self.pub_status = self.create_publisher(String, '/target_status', 10)

        self.state = "WAIT_FOR_CLICK"
        self.target_candidate = None    
        self.registered_feature = None 
        self.feature_locked = False     
        self.captured_frames_count = 0 
        self.candidates = {} 
        self.next_candidate_id = 0
        self.json_results = {}
        
        # ★追加: 非同期処理管理用フラグとロック
        self.reid_thread = None
        self.is_reid_running = False
        self.reid_lock = threading.Lock()

        self.get_logger().info(">>> Waiting for Click... Use '2D Goal Pose' in RViz.")
        self.execution_count = 0

    def _load_model(self):
        try:
            net = network.reid3d(1024, num_class=222, stride=1)
            weight_path = f'{HOME_DIR}/ReID3D/reidnet/log/ckpt_best.pth'
            if not os.path.exists(weight_path):
                sys.exit(1)
            state_dict = torch.load(weight_path, map_location=self.device)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
            net.to(self.device)
            net.eval()
            return net
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            sys.exit(1)

    def extract_feature_single(self, points_sequence):
        # ネットワーク推論を行う関数（重い処理）
        if len(points_sequence) < 1: 
            return None
        input_seq = list(points_sequence)
        while len(input_seq) < 30:
            input_seq.append(input_seq[-1])
        if len(input_seq) > 30:
            input_seq = input_seq[-30:]
        seq_np = np.array(input_seq)
        
        tensor = torch.from_numpy(seq_np).float()
        input_tensor = tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.net(input_tensor)
            feature = output['val_bn'][0]
            feature = feature.float()
            feature = F.normalize(feature.unsqueeze(0), p=2, dim=1).squeeze(0)
        return feature

    # =========================================================
    # ★追加: 非同期ReIDワーカー関数
    # =========================================================
    def async_reid_worker(self, target_cand, points_seq, mode="UPDATE"):
        """
        別スレッドで実行される推論処理
        mode: "INIT" (初期登録), "UPDATE" (追跡中の更新), "RECOVERY" (リカバリー探索)
        """
        try:
            # 推論実行 (ここで0.1〜0.5秒かかるが、メインスレッドは止まらない)
            feature = self.extract_feature_single(points_seq)
            
            if feature is None:
                return

            # 結果の適用 (排他制御しながらデータを書き込む)
            with self.reid_lock:
                feat_cpu = feature.cpu()
                
                if mode == "INIT":
                    self.registered_feature = feat_cpu
                    # ギャラリーにも追加
                    target_cand.update_feature_gallery(feat_cpu)
                    self.feature_locked = True
                    self.state = "TRACKING"
                    self.get_logger().info(">>> [Async] Init Complete! Feature LOCKED.")

                elif mode == "UPDATE":
                    # メインスレッド側で「本人だ」と判断されている候補の特徴を更新
                    if self.target_candidate and self.target_candidate.id == target_cand.id:
                        # 類似度チェック（念のため）
                        # 現在の平均と比較
                        mean_feat, _ = target_cand.get_feature_distribution()
                        sim = 0.0
                        if len(target_cand.feature_gallery) < 10:
                             sim = torch.dot(feature, self.registered_feature.to(self.device)).item()
                        elif mean_feat is not None:
                             sim = torch.dot(feature, mean_feat.to(self.device)).item()
                        
                        if sim > 0.6: # 更新時は少し緩めに
                            target_cand.update_feature_gallery(feat_cpu)
                            # Sim値も更新 (表示用)
                            target_cand.last_sim = sim

        except Exception as e:
            self.get_logger().error(f"Async ReID Error: {e}")
        finally:
            self.is_reid_running = False

    # =========================================================

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
        # メインループ: ここは絶対にブロックさせない！
        current_time = time.time()
        if hasattr(msg.header.stamp, 'sec'):
            raw_ts = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        else:
            raw_ts = msg.header.stamp.nanoseconds

        target_timestamp_str = None
        if self.gt_timestamps:
            idx = bisect.bisect_left(self.gt_timestamps, raw_ts)
            candidates = []
            if idx < len(self.gt_timestamps): candidates.append(self.gt_timestamps[idx])
            if idx > 0: candidates.append(self.gt_timestamps[idx - 1])
            if candidates:
                nearest_ts = min(candidates, key=lambda x: abs(x - raw_ts))
                if abs(nearest_ts - raw_ts) < 100000000:
                    target_timestamp_str = str(nearest_ts)

        detections = []
        for detection in msg.detections:
            if detection.source_cloud.width * detection.source_cloud.height == 0: continue
            try:
                raw_points = rnp.point_cloud2.pointcloud2_to_xyz_array(detection.source_cloud)
            except: continue
            if raw_points.shape[0] < 5: continue 
            
            # --- PCAフィルタ (壁対策) ---
            points_xy = raw_points[:, :2]
            mean_xy = np.mean(points_xy, axis=0)
            centered_xy = points_xy - mean_xy
            cov_matrix = np.cov(centered_xy, rowvar=False)
            try:
                eigenvalues, _ = np.linalg.eig(cov_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1] 
            except:
                continue 
            
            lambda1 = max(eigenvalues[0], 1e-6)
            lambda2 = max(eigenvalues[1], 1e-6)
            std_major = np.sqrt(lambda1) 
            std_minor = np.sqrt(lambda2) 
            
            if std_major > 0.45: continue
            ratio = std_major / std_minor
            if ratio > 5.0: continue
            # ---------------------------

            norm_points = normalize_point_cloud(raw_points, num_points=256)
            bbox = detection.bbox
            pos = np.array([bbox.center.position.x, bbox.center.position.y, bbox.center.position.z])
            size = np.array([bbox.size.x, bbox.size.y, bbox.size.z])
            detections.append({'pos': pos, 'size': size, 'ori': bbox.center.orientation, 'points': norm_points})

        # 1. 位置による追跡更新 (常に実行)
        self.update_candidates(detections, current_time)

        # 2. 状態管理とReIDトリガー
        current_json_data = [0, 0, 0, 0, -1]
        
        if self.state == "INITIALIZING":
            self.process_initialization_async() # 非同期版呼び出し
            if self.target_candidate:
                current_json_data = self.get_2d_target_info(self.target_candidate)
                
        elif self.state == "TRACKING" or self.state == "LOST":
            # トラッキングロジック内で、必要に応じて非同期推論をキックする
            self.process_autonomous_tracking(raw_ts)
            if self.state == "TRACKING" and self.target_candidate:
                current_json_data = self.get_2d_target_info(self.target_candidate)

        if target_timestamp_str is not None:
            target_id = self.target_candidate.id if self.target_candidate else 0
            tracks_list = [[target_id] + current_json_data]
            self.json_results[target_timestamp_str] = {
                "target_info": current_json_data, 
                "tracks_target_conf_bbox": tracks_list
            }
            if len(self.json_results) % 50 == 0:
                self.save_results_to_json()

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

    # =========================================================
    # ★変更: 非同期初期化ロジック
    # =========================================================
    def process_initialization_async(self):
        # ターゲットが消えていないかチェック
        if not (self.target_candidate and self.target_candidate.id in self.candidates):
             self.pub_status.publish(String(data="Target Lost during Init"))
             return

        self.target_candidate = self.candidates[self.target_candidate.id]
        self.captured_frames_count = len(self.target_candidate.queue)
        
        status_msg = f"INITIALIZING: {self.captured_frames_count}/30"
        
        # 30フレーム溜まったら、かつスレッドが空いていれば推論開始
        if self.captured_frames_count >= 30 and not self.is_reid_running:
            status_msg += " [Processing...]"
            self.is_reid_running = True
            
            # データのコピーを作成してスレッドに渡す (Queueの中身が変わるため)
            seq_copy = list(self.target_candidate.queue)
            
            self.reid_thread = threading.Thread(
                target=self.async_reid_worker,
                args=(self.target_candidate, seq_copy, "INIT")
            )
            self.reid_thread.start()
            
        self.pub_status.publish(String(data=status_msg))

    # =========================================================
    # ★変更: 非同期トラッキングロジック
    # =========================================================
    def process_autonomous_tracking(self, timestamp_nanosec):
        # ここでは「位置合わせ(UKF)」は既に完了している前提。
        # ReIDの更新やリカバリー判定を行う。
        
        target_found = False
        
        # 現在ターゲットを追跡中か？
        if self.target_candidate and self.target_candidate.id in self.candidates:
            self.target_candidate = self.candidates[self.target_candidate.id]
            target_found = True
            
            # --- ReID更新 (非同期) ---
            # 5フレームに1回、かつスレッドが空いていれば更新を試みる
            self.execution_count += 1
            if (self.execution_count % 5 == 0) and not self.is_reid_running:
                
                # 混雑判定
                is_crowded = False
                for cid, cand in self.candidates.items():
                    if cid == self.target_candidate.id: continue
                    dist = np.linalg.norm(cand.pos - self.target_candidate.pos)
                    if dist < 1.5:
                        is_crowded = True
                        break
                
                if not is_crowded and len(self.target_candidate.queue) >= 1:
                    self.is_reid_running = True
                    seq_copy = list(self.target_candidate.queue)
                    self.reid_thread = threading.Thread(
                        target=self.async_reid_worker,
                        args=(self.target_candidate, seq_copy, "UPDATE")
                    )
                    self.reid_thread.start()

        # --- LOST時の処理 ---
        if not target_found:
            self.state = "LOST"
            self.pub_status.publish(String(data="LOST - Searching..."))
            
            # LOST時のリカバリーは、重いのでメインスレッドでやると詰まる。
            # ここでは簡易的に「位置が予測に近い候補」がいれば暫定復帰させる等の処理を入れるか、
            # もしくはリカバリー自体も非同期にする設計が理想。
            # 今回はシンプルにするため、メインスレッドで同期的に行う(ただし頻度を下げる)
            
            # リカバリー処理 (同期)
            # ※本来はここも非同期にすべきだが、候補数が少なければ高速なので一旦そのまま
            best_sim = -1.0
            best_cand = None
            
            target_mean = None
            if self.target_candidate:
                target_mean, _ = self.target_candidate.get_feature_distribution()
            if target_mean is None:
                target_mean = self.registered_feature.cpu()
            
            target_gallery_len = len(self.target_candidate.feature_gallery) if self.target_candidate else 0

            for cid, cand in self.candidates.items():
                if len(cand.queue) < 5: continue # 点群が少なすぎるのは無視
                
                search_seq = list(cand.queue)
                while len(search_seq) < 30: search_seq.append(search_seq[-1])
                
                # ここだけはGPU推論してしまう(やむなし)。
                # ただしLOST中は頻繁に起きないので許容範囲か。
                feat = self.extract_feature_single(search_seq[:30])
                if feat is None: continue
                feat = feat.cpu()
                
                sim = 0.0
                if target_gallery_len < 10:
                    sim = torch.dot(feat, self.registered_feature.cpu()).item()
                elif self.registered_feature is not None and target_mean is not None:
                    if self.USE_WEIGHTED_SCORE:
                        sim_gallery = torch.dot(feat, target_mean).item()
                        sim_base = torch.dot(feat, self.registered_feature.cpu()).item()
                        sim = (sim_base * 0.3) + (sim_gallery * 0.7)
                    else:
                        sim = torch.dot(feat, target_mean).item()
                else:
                    sim = torch.dot(feat, target_mean).item()

                cand.last_sim = sim 
                if sim > self.REID_SIM_THRESH and sim > best_sim:
                    best_sim = sim
                    best_cand = cand
            
            if best_cand:
                self.get_logger().info(f"ReID RECOVERY! New ID:{best_cand.id} Sim:{best_sim:.2f}")
                if self.target_candidate:
                    old_gallery = self.target_candidate.feature_gallery
                    self.target_candidate = best_cand
                    self.target_candidate.feature_gallery = old_gallery
                else:
                    self.target_candidate = best_cand
                self.target_candidate.last_sim = best_sim
                self.state = "TRACKING"
        else:
            self.pub_status.publish(String(data=f"TRACKING ID:{self.target_candidate.id}"))

    def get_2d_target_info(self, candidate):
        pos_3d = candidate.pos
        size_3d = candidate.size
        confidence = candidate.last_sim if candidate.last_sim > 0 else 1.0
        img_w = 1920
        img_h = 960
        x, y, z = pos_3d[0], pos_3d[1], pos_3d[2]
        theta = math.atan2(y, x) 
        dist_2d = math.sqrt(x**2 + y**2)
        camera_height_offset = 0.4
        phi = math.atan2(z - camera_height_offset, dist_2d)      
        u = (0.5 - theta / (2 * math.pi)) * img_w
        v = (0.5 - phi / math.pi) * img_h
        dist_3d = math.sqrt(x**2 + y**2 + z**2)
        if dist_3d < 0.1: dist_3d = 0.1
        w_2d = (size_3d[1] / dist_3d) * (img_w / (2 * math.pi)) * 1.4
        h_2d = (size_3d[2] / dist_3d) * (img_h / math.pi) * 1.4
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
                    mk.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.5)
            else:
                mk.type = Marker.SPHERE
                mk.scale.x = mk.scale.y = mk.scale.z = 0.5
                mk.color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.3) 
            marker_array.markers.append(mk)
            text = Marker()
            text.header = header
            text.ns = "text"
            text.id = cid
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = cand.pos[0]
            text.pose.position.y = cand.pos[1]
            text.pose.position.z = cand.pos[2] + (cand.size[2] if is_target else 0.5) + 0.5
            text.scale.z = 0.3
            if is_target:
                text.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            else:
                text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label = f"ID:{cid}"
            if cand.last_sim > 0: 
                label += f"\nSim:{cand.last_sim:.2f}"
            elif is_target and self.feature_locked:
                label += "\nSim:Calc..."
            if is_target and not self.feature_locked: 
                label += f"\nInit:{self.captured_frames_count}/30"
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
