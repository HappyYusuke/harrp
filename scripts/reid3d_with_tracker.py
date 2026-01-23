#!/usr/bin/env python3

import os
import sys
import collections
import time
import numpy as np
import torch
import torch.nn.functional as F
import math
import threading

# --- FilterPy Imports ---
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from vision_msgs.msg import Detection3DArray
# ★追加: Point をインポート
from geometry_msgs.msg import PoseStamped, Point

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
# 1. Particle Filter Implementation
# ==========================================
class ParticleFilterTracker(object):
    def __init__(self, initial_pos, num_particles=300):
        self.num_particles = num_particles
        self.particles = np.zeros((num_particles, 6))
        
        # 初期化: ガウス分布で散らす
        self.particles[:, 0] = initial_pos[0] + np.random.randn(num_particles) * 0.1
        self.particles[:, 1] = initial_pos[1] + np.random.randn(num_particles) * 0.1
        self.particles[:, 2] = initial_pos[2] + np.random.randn(num_particles) * 0.05
        self.particles[:, 3:6] = np.random.randn(num_particles, 3) * 0.1
        
        self.weights = np.ones(num_particles) / num_particles
        self.last_timestamp = None
        self.x = np.zeros(6)
        self.x[:3] = initial_pos
        
        # パラメータ (PF用)
        self.process_noise_pos = 0.05
        self.process_noise_vel = 0.1 # 人間用に少し大きく
        self.measurement_noise_std = 0.2

    def predict(self, current_time):
        if self.last_timestamp is None:
            dt = 0.1
        else:
            dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        
        # 等速直線運動
        self.particles[:, 0] += self.particles[:, 3] * dt
        self.particles[:, 1] += self.particles[:, 4] * dt
        self.particles[:, 2] += self.particles[:, 5] * dt
        
        # プロセスノイズ拡散
        self.particles[:, :3] += np.random.randn(self.num_particles, 3) * self.process_noise_pos
        self.particles[:, 3:] += np.random.randn(self.num_particles, 3) * self.process_noise_vel
        
        self.estimate()
        return self.x[:3]

    def update(self, measurement):
        dists = np.linalg.norm(self.particles[:, :3] - measurement, axis=1)
        # 尤度計算
        likelihood = np.exp(-0.5 * (dists / self.measurement_noise_std) ** 2)
        
        if np.sum(likelihood) == 0:
            likelihood[:] = 1.0 / self.num_particles
            
        self.weights *= likelihood
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)
        
        self.estimate()
        self.resample()

    def estimate(self):
        self.x = np.average(self.particles, weights=self.weights, axis=0)

    def resample(self):
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.num_particles / 2.0:
            indices = self.systematic_resample(self.weights)
            self.particles[:] = self.particles[indices]
            self.weights[:] = 1.0 / self.num_particles

    def systematic_resample(self, weights):
        N = len(weights)
        positions = (np.arange(N) + np.random.random()) / N
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

# ==========================================
# 2. Unscented Kalman Filter Implementation
# ==========================================
def fx(x, dt):
    """ UKF 状態遷移関数 """
    F = np.eye(6, dtype=float)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return np.dot(F, x)

def hx(x):
    """ UKF 観測関数 """
    return x[:3]

class KalmanBoxTracker(object):
    def __init__(self, initial_pos):
        points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2.0, kappa=-3) 
        self.kf = UKF(dim_x=6, dim_z=3, dt=0.1, fx=fx, hx=hx, points=points)
        
        self.kf.x[:3] = initial_pos
        self.kf.x[3:] = 0
        self.kf.P *= 1.0 
        self.kf.P[3:, 3:] *= 10.0
        
        sensor_noise_std = 0.5  
        self.kf.R = np.diag([sensor_noise_std, sensor_noise_std, sensor_noise_std]) ** 2 
        self.kf.Q = np.eye(6) * 0.05**2
        self.kf.Q[3:, 3:] *= 2.0 
        
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
    def __init__(self, id, pos, size, orientation, initial_points, algo="UKF"):
        self.id = id
        self.size = size
        self.orientation = orientation
        self.queue = collections.deque(maxlen=30)
        self.queue.append(initial_points)
        self.last_sim = 0.0
        self.feature_gallery = collections.deque(maxlen=100)
        self.algo = algo
        
        if self.algo == "PF":
            self.kf = ParticleFilterTracker(pos, num_particles=500)
        else:
            self.kf = KalmanBoxTracker(pos)
        
        self.pos = pos 
        self.pred_pos = pos 
        self.last_seen_time = time.time()

    def predict(self, current_time):
        self.pred_pos = self.kf.predict(current_time)
        return self.pred_pos

    def update_state(self, pos, size, orientation):
        self.kf.update(pos)
        if self.algo == "PF":
            self.pos = self.kf.x[:3]
        else:
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
        self.declare_parameter('tracking_algo', 'PF') # デフォルトをPFにしておきます 
        
        self.MAX_MISSING_TIME = self.get_parameter('max_missing_time').value
        self.REID_SIM_THRESH = self.get_parameter('reid_sim_thresh').value
        self.TRACKING_ALGO = self.get_parameter('tracking_algo').value
        
        self.get_logger().info(f"Tracking Algorithm: {self.TRACKING_ALGO}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._load_model()
        self.get_logger().info(f'Model Loaded on {self.device}')

        self.USE_WEIGHTED_SCORE = True

        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_bbox = self.create_subscription(Detection3DArray, '/bbox', self.bbox_callback, qos_profile)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.pub_markers = self.create_publisher(MarkerArray, 'reid3d/person_reid_markers', 10)
        self.pub_status = self.create_publisher(String, 'tracker/target_status', 10)
        self.pub_status_marker = self.create_publisher(Marker, 'tracker/status_3d', 10)
        self.pub_target_pose = self.create_publisher(PoseStamped, 'tracker/target_pose', 10)

        self.state = "WAIT_FOR_CLICK"
        self.target_candidate = None    
        self.registered_feature = None 
        self.feature_locked = False      
        self.captured_frames_count = 0 
        self.candidates = {} 
        self.next_candidate_id = 0
        
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

    def async_reid_worker(self, target_cand, points_seq, mode="UPDATE"):
        try:
            feature = self.extract_feature_single(points_seq)
            if feature is None:
                return

            with self.reid_lock:
                feat_cpu = feature.cpu()
                
                if mode == "INIT":
                    self.registered_feature = feat_cpu
                    target_cand.update_feature_gallery(feat_cpu)
                    self.feature_locked = True
                    self.state = "TRACKING"
                    self.get_logger().info(">>> [Async] Init Complete! Feature LOCKED.")

                elif mode == "UPDATE":
                    if self.target_candidate and self.target_candidate.id == target_cand.id:
                        mean_feat, _ = target_cand.get_feature_distribution()
                        sim = 0.0
                        if len(target_cand.feature_gallery) < 10:
                             sim = torch.dot(feature, self.registered_feature.to(self.device)).item()
                        elif mean_feat is not None:
                             sim = torch.dot(feature, mean_feat.to(self.device)).item()
                        
                        if sim > 0.6: 
                            target_cand.update_feature_gallery(feat_cpu)
                            target_cand.last_sim = sim

        except Exception as e:
            self.get_logger().error(f"Async ReID Error: {e}")
        finally:
            self.is_reid_running = False

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
        # GT読み込みとタイムスタンプ処理を削除

        detections = []
        for detection in msg.detections:
            if detection.source_cloud.width * detection.source_cloud.height == 0: continue
            try:
                raw_points = rnp.point_cloud2.pointcloud2_to_xyz_array(detection.source_cloud)
            except: continue
            if raw_points.shape[0] < 5: continue 
            
            # =========================================================
            # ★PCAフィルタ (壁・ポール対策)
            # =========================================================
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
            
            # (A) 壁: 大きすぎる (幅 2.1m以上)
            if std_major > 0.35: 
                continue
            
            # (B) 壁: 細長すぎる (比率 5倍以上)
            ratio = std_major / std_minor
            if ratio > 5.0: 
                continue

            # (C) ポール対策: 形状チェックのみ
            dists_from_center = np.linalg.norm(centered_xy, axis=1)
            std_radial = np.std(dists_from_center)
            if std_radial < 0.04: 
                continue 
            # =========================================================

            norm_points = normalize_point_cloud(raw_points, num_points=256)
            bbox = detection.bbox
            pos = np.array([bbox.center.position.x, bbox.center.position.y, bbox.center.position.z])
            size = np.array([bbox.size.x, bbox.size.y, bbox.size.z])
            detections.append({'pos': pos, 'size': size, 'ori': bbox.center.orientation, 'points': norm_points})

        self.update_candidates(detections, current_time)

        if self.state == "INITIALIZING":
            self.process_initialization_async() 
                
        elif self.state == "TRACKING" or self.state == "LOST":
            self.process_autonomous_tracking()
            
            # ★追加: MPC連携用にターゲット位置をパブリッシュ
            if self.state == "TRACKING" and self.target_candidate:
                pose_msg = PoseStamped()
                pose_msg.header = msg.header # 入力と同じフレームを使用（通常はodom）
                pose_msg.pose.position.x = float(self.target_candidate.pos[0])
                pose_msg.pose.position.y = float(self.target_candidate.pos[1])
                pose_msg.pose.position.z = float(self.target_candidate.pos[2])
                pose_msg.pose.orientation = self.target_candidate.orientation
                self.pub_target_pose.publish(pose_msg)
        
        self.publish_visualization(msg.header)
        self.publish_status_marker_3d(msg.header)        
        
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
                self.candidates[nid] = Candidate(nid, det['pos'], det['size'], det['ori'], det['points'], algo=self.TRACKING_ALGO)
                active_ids.add(nid)
        self.candidates = {k: v for k, v in self.candidates.items() if k in active_ids}

    def process_initialization_async(self):
        if not (self.target_candidate and self.target_candidate.id in self.candidates):
             self.pub_status.publish(String(data="Target Lost during Init"))
             return

        self.target_candidate = self.candidates[self.target_candidate.id]
        self.captured_frames_count = len(self.target_candidate.queue)
        
        status_msg = f"INITIALIZING: {self.captured_frames_count}/30"
        
        if self.captured_frames_count >= 30 and not self.is_reid_running:
            status_msg += " [Processing...]"
            self.is_reid_running = True
            seq_copy = list(self.target_candidate.queue)
            self.reid_thread = threading.Thread(
                target=self.async_reid_worker,
                args=(self.target_candidate, seq_copy, "INIT")
            )
            self.reid_thread.start()
            
        self.pub_status.publish(String(data=status_msg))

    def process_autonomous_tracking(self):
        target_found = False
        
        if self.target_candidate and self.target_candidate.id in self.candidates:
            self.target_candidate = self.candidates[self.target_candidate.id]
            target_found = True
            
            self.execution_count += 1
            if (self.execution_count % 5 == 0) and not self.is_reid_running:
                
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

        if not target_found:
            self.state = "LOST"
            self.pub_status.publish(String(data="LOST - Searching..."))
            
            best_sim = -1.0
            best_cand = None
            
            target_mean = None
            if self.target_candidate:
                target_mean, _ = self.target_candidate.get_feature_distribution()
            if target_mean is None:
                target_mean = self.registered_feature.cpu()
            
            target_gallery_len = len(self.target_candidate.feature_gallery) if self.target_candidate else 0

            for cid, cand in self.candidates.items():
                if len(cand.queue) < 5: continue 
                
                search_seq = list(cand.queue)
                while len(search_seq) < 30: search_seq.append(search_seq[-1])
                
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

    def publish_status_marker_3d(self, header):
        marker = Marker()
        marker.header = header  # 入力点群(LiDAR/Camera)と同じフレームを使用
        marker.ns = "status_3d"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
        
        # センサーの1.5m上、少し前方に表示
        marker.pose.position.x = 1.0 
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.5
        marker.scale.z = 0.5  # 文字の大きさ

        if self.state == "INITIALIZING":
            marker.text = f"INITIALIZING ({self.captured_frames_count}/30)"
            marker.color = ColorRGBA(r=0.3, g=0.3, b=1.0, a=1.0) # 青
        elif self.state == "TRACKING":
            tid = self.target_candidate.id if self.target_candidate else "?"
            sim = self.target_candidate.last_sim if self.target_candidate else 0.0
            marker.text = f"TRACKING ID:{tid}\nSim:{sim:.2f}"
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) # 緑
        elif self.state == "LOST":
            marker.text = "LOST - SEARCHING..."
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) # 赤
        else:
            marker.text = "WAITING FOR CLICK"
            marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0) # 白

        self.pub_status_marker.publish(marker)

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
            
            # ターゲットのみパーティクル描画 (PFの場合)
            if cand.algo == 'PF' and is_target:
                p_mk = Marker()
                p_mk.header = header
                p_mk.ns = f"particles_{cid}"
                p_mk.id = cid
                p_mk.type = Marker.POINTS
                p_mk.action = Marker.ADD
                p_mk.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
                p_mk.scale.x = 0.03
                p_mk.scale.y = 0.03
                p_mk.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)
                for p in cand.kf.particles:
                    pt = Point()
                    pt.x = float(p[0])
                    pt.y = float(p[1])
                    pt.z = float(p[2])
                    p_mk.points.append(pt)
                marker_array.markers.append(p_mk)
                
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
            
            label += f"\n[{cand.algo}]"
            
            text.text = label
            text.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker_array.markers.append(text)
        self.pub_markers.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerClickInitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
