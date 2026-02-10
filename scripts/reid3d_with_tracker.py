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
import struct
from numba import jit

# --- Parallel Processing Imports ---
from concurrent.futures import ThreadPoolExecutor

# --- FilterPy Imports ---
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import PointCloud2

import ros2_numpy as rnp

# ==========================================
# JIT Functions (Fastmath Enabled)
# ==========================================

@jit(nopython=True, cache=True, fastmath=True)
def pf_predict_jit(particles, dt, process_noise_pos, process_noise_vel):
    num = particles.shape[0]
    noise_pos = np.random.randn(num, 2) * process_noise_pos
    noise_z = np.random.randn(num) * (process_noise_pos * 0.1)
    noise_vel = np.random.randn(num, 2) * process_noise_vel
    
    particles[:, 0] += particles[:, 3] * dt + noise_pos[:, 0]
    particles[:, 1] += particles[:, 4] * dt + noise_pos[:, 1]
    particles[:, 2] += particles[:, 5] * dt + noise_z
    particles[:, 3:5] += noise_vel
    particles[:, 5] *= 0.5
    return particles

@jit(nopython=True, cache=True, fastmath=True)
def pf_update_jit(particles, weights, det_pos, det_sim, cov_inv, std_z, sim_thresh):
    N_particles = particles.shape[0]
    N_dets = det_pos.shape[0]
    epsilon = 1.0 / (N_particles * 100.0)
    
    c00 = cov_inv[0, 0]
    c01 = cov_inv[0, 1]
    c10 = cov_inv[1, 0]
    c11 = cov_inv[1, 1]
    std_z_sq = std_z ** 2

    for i in range(N_particles):
        px = particles[i, 0]
        py = particles[i, 1]
        pz = particles[i, 2]
        
        best_likelihood = 0.0
        
        for j in range(N_dets):
            sim = det_sim[j]
            if sim < sim_thresh:
                lik = epsilon
            else:
                dx = px - det_pos[j, 0]
                dy = py - det_pos[j, 1]
                dz = pz - det_pos[j, 2]
                
                mahal_sq = (dx * c00 * dx) + (dx * c01 * dy) + (dy * c10 * dx) + (dy * c11 * dy)
                z_sq = (dz**2) / std_z_sq
                spatial_prob = np.exp(-0.5 * (mahal_sq + z_sq))
                
                lik = sim * spatial_prob
                if lik < epsilon: lik = epsilon

            if lik > best_likelihood:
                best_likelihood = lik
        
        weights[i] *= best_likelihood
        weights[i] += 1.e-300

    w_sum = np.sum(weights)
    if w_sum > 0:
        weights /= w_sum
    else:
        weights[:] = 1.0 / N_particles

    return weights

@jit(nopython=True, cache=True, fastmath=True)
def preprocess_math_jit(raw_points, min_points, pca_wall_thresh, pca_ratio_thresh, pca_radial_thresh, norm_num_points):
    n = raw_points.shape[0]
    if n < min_points:
        return None

    # --- 1. PCA Filter (2D covariance) ---
    sum_x = 0.0
    sum_y = 0.0
    for i in range(n):
        sum_x += raw_points[i, 0]
        sum_y += raw_points[i, 1]
    mean_x = sum_x / n
    mean_y = sum_y / n

    cov_xx = 0.0
    cov_xy = 0.0
    cov_yy = 0.0
    for i in range(n):
        dx = raw_points[i, 0] - mean_x
        dy = raw_points[i, 1] - mean_y
        cov_xx += dx * dx
        cov_xy += dx * dy
        cov_yy += dy * dy
    cov_xx /= (n - 1)
    cov_xy /= (n - 1)
    cov_yy /= (n - 1)

    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    delta = math.sqrt(max(0.0, trace * trace - 4 * det))
    
    l1 = (trace + delta) / 2.0
    l2 = (trace - delta) / 2.0
    
    lambda1 = max(l1, 1e-6)
    lambda2 = max(l2, 1e-6)
    
    std_major = math.sqrt(lambda1)
    std_minor = math.sqrt(lambda2)

    if std_major > pca_wall_thresh: return None
    if (std_major / std_minor) > pca_ratio_thresh: return None

    rad_sq_sum = 0.0
    rad_sum = 0.0
    for i in range(n):
        dx = raw_points[i, 0] - mean_x
        dy = raw_points[i, 1] - mean_y
        d = math.sqrt(dx*dx + dy*dy)
        rad_sum += d
        rad_sq_sum += d*d
    
    rad_mean = rad_sum / n
    rad_var = (rad_sq_sum / n) - (rad_mean * rad_mean)
    std_radial = math.sqrt(max(0.0, rad_var))

    if std_radial < pca_radial_thresh: return None

    # --- 2. Normalize / Sampling ---
    sum_z = 0.0
    for i in range(n):
        sum_z += raw_points[i, 2]
    mean_z = sum_z / n
    
    result = np.zeros((norm_num_points, 3), dtype=np.float32)
    
    if n >= norm_num_points:
        step = n / norm_num_points
        start = np.random.randint(0, int(step) + 1) if step >= 1 else 0
        for i in range(norm_num_points):
            idx = int((i * step + start) % n)
            result[i, 0] = raw_points[idx, 0] - mean_x
            result[i, 1] = raw_points[idx, 1] - mean_y
            result[i, 2] = raw_points[idx, 2] - mean_z
    else:
        for i in range(norm_num_points):
            idx = np.random.randint(0, n)
            result[i, 0] = raw_points[idx, 0] - mean_x
            result[i, 1] = raw_points[idx, 1] - mean_y
            result[i, 2] = raw_points[idx, 2] - mean_z
            
    return result

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
    print("Warning: Could not import 'model.network'. Check paths.")
sys.argv = original_argv

# ==========================================
# 1. Particle Filter Implementation
# ==========================================
class ParticleFilterTracker(object):
    def __init__(self, initial_pos, params):
        self.num_particles = params['pf_num_particles']
        self.particles = np.zeros((self.num_particles, 6)) 
        
        self.particles[:, 0] = initial_pos[0] + np.random.randn(self.num_particles) * 0.1
        self.particles[:, 1] = initial_pos[1] + np.random.randn(self.num_particles) * 0.1
        self.particles[:, 2] = initial_pos[2] + np.random.randn(self.num_particles) * 0.02
        self.particles[:, 3:5] = np.random.randn(self.num_particles, 2) * 0.1
        self.particles[:, 5]   = np.random.randn(self.num_particles) * 0.01 
        
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.last_timestamp = None
        self.x = np.zeros(6)
        self.x[:3] = initial_pos
        
        self.estimated_yaw = 0.0
        
        self.process_noise_pos = params['pf_process_noise_pos']
        self.process_noise_vel = params['pf_process_noise_vel']
        
        self.std_stopped = params['pf_std_stopped']
        self.std_long_max = params['pf_std_long_max']
        self.std_lat_max = params['pf_std_lat_max']
        self.speed_ref = params['pf_max_speed_ref']
        self.std_z = 0.1 

    def predict(self, current_time):
        if self.last_timestamp is None:
            dt = 0.1
        else:
            dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        
        self.particles = pf_predict_jit(
            self.particles, 
            dt, 
            self.process_noise_pos, 
            self.process_noise_vel
        )
        
        self.estimate()
        self._update_estimated_yaw()
        return self.x[:3]

    def _update_estimated_yaw(self):
        vx = self.x[3]
        vy = self.x[4]
        speed = np.hypot(vx, vy)
        if speed > 0.1: 
            self.estimated_yaw = np.arctan2(vy, vx)

    def _get_adaptive_covariance(self):
        vx = self.x[3]
        vy = self.x[4]
        current_speed = np.hypot(vx, vy)
        
        ratio = min(current_speed / self.speed_ref, 1.0)
        curr_long = self.std_stopped + (self.std_long_max - self.std_stopped) * ratio
        curr_lat  = self.std_stopped + (self.std_lat_max  - self.std_stopped) * ratio
        
        yaw = self.estimated_yaw
        c, s = np.cos(yaw), np.sin(yaw)
        
        l2 = curr_long**2
        t2 = curr_lat**2
        
        a = c*c*l2 + s*s*t2
        b = c*s*(l2 - t2)
        d = s*s*l2 + c*c*t2
        
        det = a*d - b*b
        
        if abs(det) < 1e-6:
             cov_inv = np.eye(2) * (1.0 / (self.std_stopped**2))
        else:
             inv_det = 1.0 / det
             cov_inv = np.array([
                 [d * inv_det, -b * inv_det],
                 [-b * inv_det, a * inv_det]
             ])
             
        return cov_inv

    def update(self, detections):
        if not detections: return

        cov_inv = self._get_adaptive_covariance()

        det_pos = np.array([d['pos'] for d in detections], dtype=np.float64)            
        det_sim = np.array([d.get('sim', 1.0) for d in detections], dtype=np.float64)
        
        SIM_THRESH = 0.65

        self.weights = pf_update_jit(
            self.particles, 
            self.weights, 
            det_pos, 
            det_sim, 
            cov_inv, 
            self.std_z, 
            SIM_THRESH
        )
        
        self.estimate()
        self.resample()
        self._update_estimated_yaw()

    def estimate(self):
        self.x = np.average(self.particles, weights=self.weights, axis=0)

    def resample(self):
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.num_particles / 2.0:
            N = self.num_particles
            cumulative_sum = np.cumsum(self.weights)
            cumulative_sum[-1] = 1.0 
            step = 1.0 / N
            r = np.random.uniform(0, step)
            positions = (np.arange(N) * step) + r
            indices = np.searchsorted(cumulative_sum, positions)
            self.particles[:] = self.particles[indices]
            self.weights[:] = 1.0 / N

# ==========================================
# 2. Kalman Tracker
# ==========================================
class KalmanBoxTracker(object):
    def __init__(self, initial_pos): 
        self.pos = initial_pos
        self.x = np.zeros(6)
        self.x[:3] = initial_pos
    def predict(self, current_time): return self.pos
    def update(self, detections):
        if not detections: return
        dists = [np.linalg.norm(self.pos - d['pos']) for d in detections]
        self.pos = detections[np.argmin(dists)]['pos']

# ==========================================
# Candidate Class
# ==========================================
class Candidate:
    def __init__(self, id, pos, size, orientation, initial_points, params):
        self.id = id
        self.size = size
        self.orientation = orientation
        self.queue = collections.deque(maxlen=params['sequence_length'])
        self.queue.append(initial_points)
        self.last_sim = 0.5 
        self.feature_gallery = collections.deque(maxlen=params['gallery_size'])
        
        # ★高速化: 平均特徴量をキャッシュしておく変数
        self.cached_mean_feature = None
        
        self.algo = params['tracking_algo']
        
        if self.algo == "PF":
            self.kf = ParticleFilterTracker(pos, params)
        else:
            self.kf = KalmanBoxTracker(pos)
        
        self.pos = pos 
        self.pred_pos = pos 
        self.last_seen_time = time.time()

    def predict(self, current_time):
        self.pred_pos = self.kf.predict(current_time)
        return self.pred_pos

    def update_state(self, detections, sim_map={}):
        valid_detections = []
        for i, det in enumerate(detections):
            d = det.copy()
            d['sim'] = sim_map.get(i, 0.5) 
            valid_detections.append(d)

        self.kf.update(valid_detections)
        
        if hasattr(self.kf, 'x'):
            self.pos = self.kf.x[:3]
        else:
            self.pos = self.kf.pos

        if valid_detections:
            dists = [np.linalg.norm(self.pos - d['pos']) for d in valid_detections]
            best_idx = np.argmin(dists)
            best_det = valid_detections[best_idx]
            self.size = best_det['size']
            
            if hasattr(self.kf, 'estimated_yaw'):
                yaw = self.kf.estimated_yaw
                cy = math.cos(yaw * 0.5)
                sy = math.sin(yaw * 0.5)
                q = self.orientation
                q.w, q.x, q.y, q.z = cy, 0.0, 0.0, sy
                self.orientation = q
            else:
                self.orientation = best_det['ori']
                
            self.last_seen_time = time.time()

    def add_points(self, points):
        self.queue.append(points)
    
    def get_feature_distribution(self):
        """ キャッシュがあればそれを返す """
        if self.cached_mean_feature is not None:
            return self.cached_mean_feature, None
            
        if len(self.feature_gallery) == 0: return None, None
        
        gallery_tensor = torch.stack(list(self.feature_gallery))
        mean_feature = torch.mean(gallery_tensor, dim=0)
        mean_feature = F.normalize(mean_feature.unsqueeze(0), p=2, dim=1).squeeze(0)
        
        # 結果をCPUでキャッシュ
        self.cached_mean_feature = mean_feature
        return mean_feature, None
    
    def update_feature_gallery(self, feature_tensor):
        # 特徴量を追加
        self.feature_gallery.append(feature_tensor.detach().cpu())
        
        # ★高速化: 追加されたら平均を再計算してCPUキャッシュを更新
        # これにより、リカバリー時に毎回100個のテンソルを計算・GPU転送するのを防ぐ
        gallery_tensor = torch.stack(list(self.feature_gallery))
        mean_feature = torch.mean(gallery_tensor, dim=0)
        mean_feature = F.normalize(mean_feature.unsqueeze(0), p=2, dim=1).squeeze(0)
        self.cached_mean_feature = mean_feature

# ==========================================
# ROS Node
# ==========================================
class PersonTrackerClickInitNode(Node):
    def __init__(self):
        super().__init__('person_tracker_click_init')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('tracking.algo', 'PF'),
                ('tracking.max_missing_time', 1.0),
                ('tracking.match_dist_thresh', 1.0),
                ('tracking.pf_num_particles', 500),
                ('tracking.pf_process_noise_pos', 0.05),
                ('tracking.pf_process_noise_vel', 0.1),
                ('tracking.pf_std_stopped', 0.2), 
                ('tracking.pf_std_long_max', 0.6),
                ('tracking.pf_std_lat_max', 0.25),
                ('tracking.pf_max_speed_ref', 1.5),
                ('reid.weight_path', f'{HOME_DIR}/ReID3D/reidnet/log/ckpt_best.pth'),
                ('reid.sim_thresh', 0.80),
                ('reid.sequence_length', 30),
                ('reid.gallery_size', 100),
                ('reid.feature_dim', 1024),
                ('reid.num_class', 222),
                ('reid.verify_thresh', 0.4),
                ('detection.pca_wall_thresh', 0.35),
                ('detection.pca_ratio_thresh', 5.0),
                ('detection.pca_radial_thresh', 0.04),
                ('detection.min_points', 5),
                ('detection.normalize_num_points', 256),
                ('detection.robot_radius', 0.45),
                ('target_selection.mode', 0),
                ('target_selection.auto_x_min', 0.0),
                ('target_selection.auto_x_max', 3.0),
                ('target_selection.auto_y_min', -0.5),
                ('target_selection.auto_y_max', 0.5),
            ]
        )

        self.params = {
            'tracking_algo': self.get_parameter('tracking.algo').value,
            'max_missing_time': self.get_parameter('tracking.max_missing_time').value,
            'match_dist_thresh': self.get_parameter('tracking.match_dist_thresh').value,
            'pf_num_particles': self.get_parameter('tracking.pf_num_particles').value,
            'pf_process_noise_pos': self.get_parameter('tracking.pf_process_noise_pos').value,
            'pf_process_noise_vel': self.get_parameter('tracking.pf_process_noise_vel').value,
            'pf_std_stopped': self.get_parameter('tracking.pf_std_stopped').value,
            'pf_std_long_max': self.get_parameter('tracking.pf_std_long_max').value,
            'pf_std_lat_max': self.get_parameter('tracking.pf_std_lat_max').value,
            'pf_max_speed_ref': self.get_parameter('tracking.pf_max_speed_ref').value,
            'reid_weight_path': self.get_parameter('reid.weight_path').value,
            'reid_sim_thresh': self.get_parameter('reid.sim_thresh').value,
            'sequence_length': self.get_parameter('reid.sequence_length').value,
            'gallery_size': self.get_parameter('reid.gallery_size').value,
            'reid_feature_dim': self.get_parameter('reid.feature_dim').value,
            'reid_num_class': self.get_parameter('reid.num_class').value,
            'verify_thresh': self.get_parameter('reid.verify_thresh').value,
            'pca_wall_thresh': self.get_parameter('detection.pca_wall_thresh').value,
            'pca_ratio_thresh': self.get_parameter('detection.pca_ratio_thresh').value,
            'pca_radial_thresh': self.get_parameter('detection.pca_radial_thresh').value,
            'min_points': self.get_parameter('detection.min_points').value,
            'normalize_num_points': self.get_parameter('detection.normalize_num_points').value,
            'robot_radius': self.get_parameter('detection.robot_radius').value,
            'choose_target_from_rviz2': self.get_parameter('target_selection.mode').value,
            'auto_x_min': self.get_parameter('target_selection.auto_x_min').value,
            'auto_x_max': self.get_parameter('target_selection.auto_x_max').value,
            'auto_y_min': self.get_parameter('target_selection.auto_y_min').value,
            'auto_y_max': self.get_parameter('target_selection.auto_y_max').value,
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(4)
        self.net = self._load_model()
        self.get_logger().info(f'Model Loaded on {self.device}')

        self.qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_bbox = self.create_subscription(Detection3DArray, '/bbox', self.bbox_callback, self.qos)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.pub_markers = self.create_publisher(MarkerArray, 'reid3d/person_reid_markers', 10)
        self.pub_status = self.create_publisher(String, 'tracker/target_status', 10)
        self.pub_status_marker = self.create_publisher(Marker, 'tracker/status_3d', 10)
        self.pub_target_pose = self.create_publisher(PoseStamped, 'tracker/target_pose', 10)
        self.pub_particles = self.create_publisher(PointCloud2, 'tracker/particles', 10)

        self.state = "WAIT_FOR_CLICK"
        self.target_candidate = None    
        self.registered_feature = None 
        self.feature_locked = False        
        self.captured_frames_count = 0 
        self.candidates = {} 
        self.next_candidate_id = 0
        
        self.reid_thread = None
        self.recovery_thread = None
        self.is_reid_running = False
        self.is_recovery_running = False
        self.is_processing_reid_update = False
        self.reid_lock = threading.Lock()
        
        self.preprocess_executor = ThreadPoolExecutor(max_workers=4)
        self.execution_count = 0

    def destroy_node(self):
        self.preprocess_executor.shutdown()
        super().destroy_node()

    def _load_model(self):
        try:
            net = network.reid3d(self.params['reid_feature_dim'], num_class=self.params['reid_num_class'], stride=1)
            state_dict = torch.load(self.params['reid_weight_path'], map_location=self.device)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
            net.to(self.device)
            
            # FP32 (Single Precision)
            net.eval()
            return net
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            sys.exit(1)

    def extract_features_batch(self, points_seq_list):
        if not points_seq_list: return None
        processed_seqs = []
        seq_len = self.params['sequence_length'] 
        for seq in points_seq_list:
            inp = list(seq)
            while len(inp) < seq_len: inp.append(inp[-1])
            if len(inp) > seq_len: inp = inp[-seq_len:]
            processed_seqs.append(np.array(inp))
        batch_np = np.array(processed_seqs)
        tensor = torch.from_numpy(batch_np).float()
        input_tensor = tensor.to(self.device)
        
        # ★修正: inference_mode() を使用（no_gradよりさらに軽量）
        with torch.inference_mode():
            output = self.net(input_tensor)
            features = output['val_bn']
            features = F.normalize(features.float(), p=2, dim=1) 
        return features
        
    def extract_feature_single(self, points_sequence):
        feats = self.extract_features_batch([points_sequence])
        return feats[0] if feats is not None else None
    
    def async_reid_worker(self, target_cand, points_seq, mode="UPDATE"):
        try:
            feature = self.extract_feature_single(points_seq)
            if feature is None: return
            with self.reid_lock:
                feat_cpu = feature.cpu()
                if mode == "INIT":
                    self.registered_feature = feat_cpu
                    target_cand.update_feature_gallery(feat_cpu)
                    target_cand.last_sim = 1.0 
                    self.feature_locked = True
                    self.state = "TRACKING"
                elif mode == "UPDATE":
                    if self.target_candidate and self.target_candidate.id == target_cand.id:
                        mean_feat, _ = target_cand.get_feature_distribution()
                        sim = 0.0
                        if len(target_cand.feature_gallery) < 10:
                             sim = torch.dot(feature, self.registered_feature.to(self.device).float()).item()
                        elif mean_feat is not None:
                             # 既にCPUキャッシュにある平均特徴量をGPUに送るだけ（1つだけ）
                             sim = torch.dot(feature, mean_feat.to(self.device).float()).item()
                        target_cand.last_sim = sim
                        if sim > 0.6: target_cand.update_feature_gallery(feat_cpu)
        except Exception as e: self.get_logger().error(f"ReID Err: {e}")
        finally: self.is_reid_running = False

    def async_recovery_worker(self, snapshot_candidates):
        try:
            # ★修正: バッチサイズを1から8に増加（GPU効率化）
            BATCH_SIZE = 1
            
            reg_feat = self.registered_feature.to(self.device).float()
            if reg_feat.dim() == 1: reg_feat = reg_feat.unsqueeze(0)

            target_feat = reg_feat 
            
            if self.target_candidate:
                # ★修正: キャッシュされた平均特徴量を直接使う（100個のリストをGPUに送らない）
                mean_feat, _ = self.target_candidate.get_feature_distribution()
                
                if mean_feat is not None:
                    # mean_featは1x1024のTensor (CPU)
                    mean_feat_gpu = mean_feat.to(self.device).float()
                    
                    alpha = 0.3
                    mixed_feat = (alpha * reg_feat) + ((1 - alpha) * mean_feat_gpu)
                    target_feat = F.normalize(mixed_feat, p=2, dim=1)

            found_match = False
            best_match_info = None

            for i in range(0, len(snapshot_candidates), BATCH_SIZE):
                batch_cands = snapshot_candidates[i : i + BATCH_SIZE]
                batch_input = [c['queue'] for c in batch_cands]
                features = self.extract_features_batch(batch_input)
                
                if features is not None:
                    sim_matrix = torch.mm(features, target_feat.T)
                    sim_scores = sim_matrix.flatten().cpu().numpy()
                    
                    if len(sim_scores) > 0:
                        best_idx_local = np.argmax(sim_scores)
                        best_sim = float(sim_scores[best_idx_local])
                        
                        if best_sim > self.params['reid_sim_thresh']:
                            found_match = True
                            best_match_info = (batch_cands[best_idx_local]['id'], best_sim)
                            break
                
                # ★重要修正: empty_cache()を削除！
                # これが最大の遅延原因でした。PyTorchのメモリアロケータに任せます。
                del features, batch_input
            
            if found_match and best_match_info:
                cid, sim = best_match_info
                if cid in self.candidates:
                    cand = self.candidates[cid]
                    self.get_logger().info(f">>> RECOVERY SUCCEEDED! New ID:{cid} (Sim: {sim:.2f})")
                    old_gallery = self.target_candidate.feature_gallery if self.target_candidate else None
                    
                    # 古いキャッシュも引き継ぐ
                    old_cached_mean = self.target_candidate.cached_mean_feature if self.target_candidate else None
                    
                    self.target_candidate = cand
                    self.target_candidate.last_sim = sim
                    if old_gallery:
                        self.target_candidate.feature_gallery = old_gallery
                        self.target_candidate.cached_mean_feature = old_cached_mean # キャッシュ引継ぎ
                    self.state = "TRACKING"

        except Exception as e:
            self.get_logger().error(f"Recovery Err: {e}")
        finally:
            self.is_recovery_running = False

    def async_sim_update_worker(self, reid_jobs):
        try:
            # バッチサイズを微増
            BATCH_SIZE = 8
            reg_feat = self.registered_feature.to(self.device).float()
            if reg_feat.dim() == 1: reg_feat = reg_feat.unsqueeze(0)

            for i in range(0, len(reid_jobs), BATCH_SIZE):
                batch = reid_jobs[i : i + BATCH_SIZE]
                batch_input = [[item[1]] * self.params['sequence_length'] for item in batch]
                
                features = self.extract_features_batch(batch_input)
                if features is not None:
                    sim_matrix = torch.mm(features, reg_feat.T)
                    sim_scores = sim_matrix.flatten().cpu().numpy()
                    
                    for (cid, _), score in zip(batch, sim_scores):
                        if cid in self.candidates:
                            self.candidates[cid].last_sim = float(score)
                            
        except Exception as e:
            self.get_logger().error(f"Sim Update Err: {e}")
        finally:
            self.is_processing_reid_update = False

    def goal_callback(self, msg: PoseStamped):
        if self.params['choose_target_from_rviz2'] == 0: return
        click_x, click_y = msg.pose.position.x, msg.pose.position.y
        closest_cand, min_dist = None, 3.0
        for cid, cand in self.candidates.items():
            dist = math.sqrt((cand.pos[0] - click_x)**2 + (cand.pos[1] - click_y)**2)
            if dist < min_dist: min_dist, closest_cand = dist, cand
        if closest_cand:
            self.target_candidate = closest_cand
            self.captured_frames_count = len(closest_cand.queue)
            self.feature_locked = False
            self.registered_feature = None
            self.state = "INITIALIZING"

    def _preprocess_wrapper(self, det):
        if det.source_cloud.width * det.source_cloud.height == 0: return None
        try:
            raw_points = rnp.point_cloud2.pointcloud2_to_xyz_array(det.source_cloud)
        except: return None

        norm_points = preprocess_math_jit(
            raw_points, 
            self.params['min_points'],
            self.params['pca_wall_thresh'],
            self.params['pca_ratio_thresh'],
            self.params['pca_radial_thresh'],
            self.params['normalize_num_points']
        )
        
        if norm_points is None: return None

        bbox = det.bbox
        pos = np.array([bbox.center.position.x, bbox.center.position.y, bbox.center.position.z])
        size = np.array([bbox.size.x, bbox.size.y, bbox.size.z])
        
        dist_xy = np.hypot(pos[0], pos[1])
        if dist_xy < self.params['robot_radius']: return None

        return {
            'pos': pos,
            'size': size,
            'ori': bbox.center.orientation,
            'points': norm_points,
            'sim': 1.0
        }

    def bbox_callback(self, msg: Detection3DArray):
        current_time = time.time()
        futures = [self.preprocess_executor.submit(self._preprocess_wrapper, det) for det in msg.detections]
        detections = [f.result() for f in futures if f.result() is not None]
        
        self.update_candidates(detections, current_time)

        if self.state == "WAIT_FOR_CLICK":
            if self.params['choose_target_from_rviz2'] == 0: self.process_auto_front_selection()
            else: self.pub_status.publish(String(data="WAITING FOR CLICK..."))
        elif self.state == "INITIALIZING":
            self.process_initialization_async()
        elif self.state == "TRACKING" or self.state == "LOST":
            self.process_autonomous_tracking()
            if self.state == "TRACKING" and self.target_candidate:
                pose_msg = PoseStamped()
                pose_msg.header = msg.header
                pose_msg.pose.position.x = float(self.target_candidate.pos[0])
                pose_msg.pose.position.y = float(self.target_candidate.pos[1])
                pose_msg.pose.position.z = float(self.target_candidate.pos[2])
                pose_msg.pose.orientation = self.target_candidate.orientation
                self.pub_target_pose.publish(pose_msg)
        
        self.publish_visualization(msg.header)
        self.publish_status_marker_3d(msg.header)

    def process_auto_front_selection(self):
        min_dist, best_cand = 100.0, None
        x_min, x_max = self.params['auto_x_min'], self.params['auto_x_max']
        y_min, y_max = self.params['auto_y_min'], self.params['auto_y_max']
        for cid, cand in self.candidates.items():
            if x_min < cand.pos[0] < x_max and y_min < cand.pos[1] < y_max:
                if cand.pos[0] < min_dist: min_dist, best_cand = cand.pos[0], cand
        if best_cand:
            self.target_candidate = best_cand
            self.captured_frames_count = len(best_cand.queue)
            self.feature_locked = False
            self.registered_feature = None
            self.state = "INITIALIZING"
        else: self.pub_status.publish(String(data="SEARCHING FRONT..."))

    def update_candidates(self, detections, current_time):
        for cand in self.candidates.values():
            cand.predict(current_time)
        
        active_ids = set()
        
        if not self.candidates or not detections:
            for i, det in enumerate(detections):
                if not self.candidates: self._create_new_candidate(det)
            if not detections:
                for cid, cand in self.candidates.items():
                    if (current_time - cand.last_seen_time) < self.params['max_missing_time']:
                        active_ids.add(cid)
                self.candidates = {k: v for k, v in self.candidates.items() if k in active_ids}
            return

        cand_ids = list(self.candidates.keys())
        cost_matrix = []
        for cid in cand_ids:
            cand = self.candidates[cid]
            costs = [np.linalg.norm(cand.pred_pos - det['pos']) for det in detections]
            cost_matrix.append(costs)
        
        pairs = []
        for r in range(len(cand_ids)):
            for c in range(len(detections)):
                pairs.append((cost_matrix[r][c], r, c))
        pairs.sort(key=lambda x: x[0])
        
        used_cand_indices = set()
        used_det_indices = set()
        match_dist = self.params['match_dist_thresh']
        
        reid_jobs = [] 
        target_pos = None
        if self.target_candidate:
            target_pos = self.target_candidate.pos

        for dist, r, c in pairs:
            if r in used_cand_indices or c in used_det_indices: continue
            if dist > match_dist: continue
            
            cid = cand_ids[r]
            cand = self.candidates[cid]
            det = detections[c]
            
            used_cand_indices.add(r)
            used_det_indices.add(c)
            
            is_target_initializing = (self.target_candidate and 
                                      self.target_candidate.id == cid and 
                                      not self.feature_locked)
            
            if is_target_initializing:
                cached_sim = 1.0 
            else:
                cached_sim = cand.last_sim if cand.last_sim > 0.001 else 0.5

            cand.update_state([det], sim_map={0: cached_sim})
            cand.add_points(det['points'])
            active_ids.add(cid)
            
            is_target = (self.target_candidate and self.target_candidate.id == cid)
            is_neighbor = False
            if target_pos is not None and not is_target:
                dist_to_target = np.linalg.norm(cand.pos - target_pos)
                if dist_to_target < 2.0: is_neighbor = True

            should_run_reid = (is_target or is_neighbor) and (self.registered_feature is not None)
            
            if should_run_reid:
                if is_target and dist < 0.8:
                    pass 
                else:
                    reid_jobs.append((cid, det['points']))

        if reid_jobs and not self.is_processing_reid_update:
            self.is_processing_reid_update = True
            threading.Thread(target=self.async_sim_update_worker, args=(reid_jobs,)).start()

        for r, cid in enumerate(cand_ids):
            if r not in used_cand_indices:
                cand = self.candidates[cid]
                if (current_time - cand.last_seen_time) < self.params['max_missing_time']:
                    active_ids.add(cid)
        
        for c in range(len(detections)):
            if c not in used_det_indices:
                self._create_new_candidate(detections[c])
        
        keys_to_remove = []
        for cid, cand in self.candidates.items():
            if (current_time - cand.last_seen_time) >= self.params['max_missing_time']:
                keys_to_remove.append(cid)
        for cid in keys_to_remove:
            del self.candidates[cid]

    def _create_new_candidate(self, det):
        nid = self.next_candidate_id
        self.next_candidate_id += 1
        self.candidates[nid] = Candidate(nid, det['pos'], det['size'], det['ori'], det['points'], self.params)

    def process_initialization_async(self):
        seq_len = self.params['sequence_length']
        if not (self.target_candidate and self.target_candidate.id in self.candidates):
             self.pub_status.publish(String(data="Target Lost during Init"))
             if self.params['choose_target_from_rviz2'] == 0: self.state = "WAIT_FOR_CLICK"
             return
        self.target_candidate = self.candidates[self.target_candidate.id]
        self.captured_frames_count = len(self.target_candidate.queue)
        status_msg = f"INITIALIZING: {self.captured_frames_count}/{seq_len}"
        if self.captured_frames_count >= seq_len and not self.is_reid_running:
            status_msg += " [Processing...]"
            self.is_reid_running = True
            seq_copy = list(self.target_candidate.queue)
            self.reid_thread = threading.Thread(target=self.async_reid_worker, args=(self.target_candidate, seq_copy, "INIT"))
            self.reid_thread.start()
        self.pub_status.publish(String(data=status_msg))

    def process_autonomous_tracking(self):
        target_found = False
        
        if self.target_candidate and self.target_candidate.id in self.candidates:
            cand = self.candidates[self.target_candidate.id]
            
            LOST_THRESH = 0.7
            if self.feature_locked and cand.last_sim < LOST_THRESH and cand.last_sim > 0.001:
                target_found = False
                self.get_logger().warn(f"Force LOST: Sim {cand.last_sim:.2f} < {LOST_THRESH}")
            else:
                self.target_candidate = cand
                target_found = True
            
            self.execution_count += 1
            if target_found and (self.execution_count % 30 == 0) and not self.is_reid_running:
                if len(self.target_candidate.queue) >= 1:
                    self.is_reid_running = True
                    seq_copy = list(self.target_candidate.queue)
                    self.reid_thread = threading.Thread(target=self.async_reid_worker, args=(self.target_candidate, seq_copy, "UPDATE"))
                    self.reid_thread.start()

        if target_found:
            self.pub_status.publish(String(data=f"TRACKING ID:{self.target_candidate.id}"))
            self.state = "TRACKING" 
        else:
            self.state = "LOST"
            self.pub_status.publish(String(data="LOST - Searching..."))

            if self.registered_feature is not None and not self.is_recovery_running:
                snapshot_candidates = []
                for cid, cand in self.candidates.items():
                    if len(cand.queue) >= 3: 
                        snapshot_candidates.append({'id': cid, 'queue': list(cand.queue)})
                
                if snapshot_candidates:
                    self.is_recovery_running = True
                    self.recovery_thread = threading.Thread(target=self.async_recovery_worker, args=(snapshot_candidates,))
                    self.recovery_thread.start()

    def publish_status_marker_3d(self, header):
        marker = Marker()
        if not header.frame_id: header.frame_id = "livox_frame"
        marker.header = header; marker.ns = "status_3d"; marker.id = 0; marker.type = Marker.TEXT_VIEW_FACING; marker.action = Marker.ADD
        marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
        marker.pose.position.x = 1.0; marker.pose.position.y = 0.0; marker.pose.position.z = 1.5; marker.pose.orientation.w = 1.0; marker.scale.z = 0.5 
        if self.state == "INITIALIZING": marker.text, marker.color = f"INITIALIZING", ColorRGBA(r=0.3, g=0.3, b=1.0, a=1.0)
        elif self.state == "TRACKING": marker.text, marker.color = f"TRACKING", ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        elif self.state == "LOST": marker.text, marker.color = "LOST", ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        else: marker.text, marker.color = "WAITING", ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        self.pub_status_marker.publish(marker)

    def publish_visualization(self, header):
        marker_array = MarkerArray()
        target_id = self.target_candidate.id if self.target_candidate else -1
        all_particles_xyz = []    
        for cid, cand in self.candidates.items():
            is_target = (target_id != -1 and cid == target_id)
            mk = Marker()
            mk.header = header; mk.ns = "candidates"; mk.id = cid; mk.action = Marker.ADD; mk.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = cand.pos
            mk.pose.orientation = cand.orientation
            if is_target: mk.type, mk.scale.x, mk.scale.y, mk.scale.z, mk.color = Marker.CUBE, max(cand.size[0], 0.2), max(cand.size[1], 0.2), max(cand.size[2], 0.2), ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5)
            else: mk.type, mk.scale.x, mk.scale.y, mk.scale.z, mk.color = Marker.SPHERE, 0.5, 0.5, 0.5, ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.3)
            marker_array.markers.append(mk)
            if cand.algo == 'PF' and hasattr(cand.kf, 'particles') and is_target:
                all_particles_xyz.append(cand.kf.particles[:, :3])
            
            text = Marker(); text.header = header; text.ns = "text"; text.id = cid; text.type = Marker.TEXT_VIEW_FACING; text.action = Marker.ADD
            text.pose.position.x, text.pose.position.y, text.pose.position.z = cand.pos[0], cand.pos[1], cand.pos[2]+1.0
            text.scale.z = 0.3; text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0); text.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            
            label = f"ID:{cid}"
            if cand.last_sim > 0.001:
                label += f"\nSim:{cand.last_sim:.2f}"
            if is_target:
                label += "\n[TARGET]"
                text.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) 
            
            text.text = label
            marker_array.markers.append(text)
            
        self.pub_markers.publish(marker_array)
        if len(all_particles_xyz) > 0:
            points_np = np.vstack(all_particles_xyz)
            red_color_float = np.array([0xFF0000], dtype=np.uint32).view(np.float32)[0]
            data = np.zeros(len(points_np), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)])
            data['x'], data['y'], data['z'], data['rgb'] = points_np[:, 0], points_np[:, 1], points_np[:, 2], red_color_float
            msg = rnp.msgify(PointCloud2, data); msg.header = header
            self.pub_particles.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerClickInitNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
