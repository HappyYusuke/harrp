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
import struct # PointCloud2の色変換用

# --- Parallel Processing Imports ---
from concurrent.futures import ProcessPoolExecutor

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

# ============================
# ReIDモデル設定 (パス解決用)
# ============================
HOME_DIR = os.environ['HOME']
sys.path.insert(0, f'{HOME_DIR}/ReID3D/reidnet/')

original_argv = sys.argv
sys.argv = [original_argv[0]]
try:
    from model import network
except ImportError:
    print("Warning: Could not import 'model.network' at global scope. Check paths.")
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
# 並列処理用のワーカー関数 (クラス外に定義)
# ==========================================
def preprocess_detection_worker(args):
    detection, params = args

    if detection.source_cloud.width * detection.source_cloud.height == 0:
        return None

    try:
        raw_points = rnp.point_cloud2.pointcloud2_to_xyz_array(detection.source_cloud)
    except:
        return None

    if raw_points.shape[0] < params['min_points']:
        return None
    
    # --- PCAフィルタ (壁・ポール対策) ---
    points_xy = raw_points[:, :2]
    mean_xy = np.mean(points_xy, axis=0)
    centered_xy = points_xy - mean_xy
    cov_matrix = np.cov(centered_xy, rowvar=False)
    
    try:
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1] 
    except:
        return None
    
    lambda1 = max(eigenvalues[0], 1e-6)
    lambda2 = max(eigenvalues[1], 1e-6)
    std_major = np.sqrt(lambda1) 
    std_minor = np.sqrt(lambda2) 
    
    if std_major > params['pca_wall_thresh']: return None
    ratio = std_major / std_minor
    if ratio > params['pca_ratio_thresh']: return None
    
    dists_from_center = np.linalg.norm(centered_xy, axis=1)
    std_radial = np.std(dists_from_center)
    if std_radial < params['pca_radial_thresh']: return None
    # -----------------------------------

    norm_points = normalize_point_cloud(raw_points, num_points=params['normalize_num_points'])
    bbox = detection.bbox
    pos = np.array([bbox.center.position.x, bbox.center.position.y, bbox.center.position.z])
    size = np.array([bbox.size.x, bbox.size.y, bbox.size.z])
    
    dist_xy = np.hypot(pos[0], pos[1])
    if dist_xy < params['robot_radius']:
        return None

    return {
        'pos': pos,
        'size': size,
        'ori': bbox.center.orientation,
        'points': norm_points
    }

# ==========================================
# 1. Particle Filter Implementation (With Mahalanobis & ReID Likelihood)
# ==========================================
# ==========================================
# 1. Particle Filter Implementation (Fix: Add missing estimate method)
# ==========================================
class ParticleFilterTracker(object):
    def __init__(self, initial_pos, params):
        self.num_particles = params['pf_num_particles']
        self.particles = np.zeros((self.num_particles, 6))
        
        # 初期化: Z方向のばらつきは小さめに
        self.particles[:, 0] = initial_pos[0] + np.random.randn(self.num_particles) * 0.1
        self.particles[:, 1] = initial_pos[1] + np.random.randn(self.num_particles) * 0.1
        self.particles[:, 2] = initial_pos[2] + np.random.randn(self.num_particles) * 0.02 
        
        # 速度初期値: Z速度はほぼ0にする
        self.particles[:, 3:5] = np.random.randn(self.num_particles, 2) * 0.1
        self.particles[:, 5]   = np.random.randn(self.num_particles) * 0.01 
        
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.last_timestamp = None
        self.x = np.zeros(6)
        self.x[:3] = initial_pos
        
        self.process_noise_pos = params['pf_process_noise_pos']
        self.process_noise_vel = params['pf_process_noise_vel']
        
        # --- マハラノビス距離用パラメータ ---
        self.std_long = params['pf_std_long'] 
        self.std_lat = params['pf_std_lat']
        # Z方向(高さ)の許容誤差
        self.std_z = 0.1 

    def predict(self, current_time):
        if self.last_timestamp is None:
            dt = 0.1
        else:
            dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        
        # 位置更新
        self.particles[:, 0] += self.particles[:, 3] * dt
        self.particles[:, 1] += self.particles[:, 4] * dt
        self.particles[:, 2] += self.particles[:, 5] * dt
        
        # --- プロセスノイズの適用 (Z方向を抑制) ---
        # XYには設定通りのノイズ
        self.particles[:, :2] += np.random.randn(self.num_particles, 2) * self.process_noise_pos
        # Zには非常に小さいノイズのみ与える
        self.particles[:, 2]  += np.random.randn(self.num_particles) * (self.process_noise_pos * 0.1)

        # 速度更新 (XYのみ)
        self.particles[:, 3:5] += np.random.randn(self.num_particles, 2) * self.process_noise_vel
        # Z速度は減衰させる
        self.particles[:, 5] *= 0.5 
        
        self.estimate()
        return self.x[:3]

    def update(self, measurement, reid_sim=1.0, target_yaw=0.0):
        """
        マハラノビス距離(XY) + 単純ガウス(Z) + ReID尤度
        """
        # --- 1. XY平面のマハラノビス距離計算 ---
        cov_local = np.diag([self.std_long**2, self.std_lat**2])
        
        c, s = np.cos(target_yaw), np.sin(target_yaw)
        rot_mat = np.array([[c, -s], 
                            [s,  c]])
        
        cov_global = rot_mat @ cov_local @ rot_mat.T
        
        try:
            cov_inv = np.linalg.inv(cov_global)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(2)
            
        diff_xy = self.particles[:, :2] - measurement[:2]
        
        # XYのマハラノビス距離^2
        mahalanobis_sq_xy = np.einsum('ij,jk,ik->i', diff_xy, cov_inv, diff_xy)
        
        # --- 2. Z方向(高さ)の距離計算 ---
        diff_z = self.particles[:, 2] - measurement[2]
        z_sq = (diff_z / self.std_z) ** 2
        
        # --- 3. 空間尤度の統合 ---
        spatial_likelihood = np.exp(-0.5 * (mahalanobis_sq_xy + z_sq))

        # --- 4. 混合モデルによるReID統合 ---
        uniform_weight = 1.0 / (self.num_particles * 100.0)
        total_likelihood = (reid_sim * spatial_likelihood) + ((1.0 - reid_sim) * uniform_weight)
        
        self.weights *= total_likelihood
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)
        
        self.estimate()
        self.resample()

    def estimate(self):
        """ 重み付き平均で現在の状態(x)を推定する """
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
# 2. Unscented Kalman Filter (Dummy Implementation for compatibility)
# ==========================================
def fx(x, dt):
    F = np.eye(6, dtype=float)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return np.dot(F, x)

def hx(x):
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

    def update(self, measurement, reid_sim=1.0, target_yaw=0.0):
        # UKFは簡易実装のため、ReID/Yawは無視
        self.kf.update(measurement)

# ==========================================

class Candidate:
    def __init__(self, id, pos, size, orientation, initial_points, params):
        self.id = id
        self.size = size
        self.orientation = orientation
        self.queue = collections.deque(maxlen=params['sequence_length'])
        self.queue.append(initial_points)
        self.last_sim = 0.0
        self.feature_gallery = collections.deque(maxlen=params['gallery_size'])
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

    def update_state(self, pos, size, orientation, sim=1.0):
        # QuaternionからYaw角を計算
        q = orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # 拡張されたupdateメソッドを呼び出し
        self.kf.update(pos, reid_sim=sim, target_yaw=yaw)
        
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
        
        # ==========================================
        # パラメータ宣言と取得
        # ==========================================
        self.declare_parameters(
            namespace='',
            parameters=[
                ('tracking.algo', 'PF'),
                ('tracking.max_missing_time', 1.0),
                ('tracking.match_dist_thresh', 1.0),
                ('tracking.pf_num_particles', 500),
                ('tracking.pf_process_noise_pos', 0.05),
                ('tracking.pf_process_noise_vel', 0.1),
                # ↓ 新しいパラメータ (マハラノビス距離用)
                ('tracking.pf_std_long', 0.6), # 進行方向の分散 (大きめ)
                ('tracking.pf_std_lat', 0.2),  # 横方向の分散 (小さめ)

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

        # パラメータを辞書として保持
        self.params = {
            'tracking_algo': self.get_parameter('tracking.algo').value,
            'max_missing_time': self.get_parameter('tracking.max_missing_time').value,
            'match_dist_thresh': self.get_parameter('tracking.match_dist_thresh').value,
            'pf_num_particles': self.get_parameter('tracking.pf_num_particles').value,
            'pf_process_noise_pos': self.get_parameter('tracking.pf_process_noise_pos').value,
            'pf_process_noise_vel': self.get_parameter('tracking.pf_process_noise_vel').value,
            # ↓ 新しいパラメータ
            'pf_std_long': self.get_parameter('tracking.pf_std_long').value,
            'pf_std_lat': self.get_parameter('tracking.pf_std_lat').value,
            
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
        
        self.get_logger().info(f"Tracking Algo: {self.params['tracking_algo']}")
        self.get_logger().info(f"PF Std Long: {self.params['pf_std_long']}, Lat: {self.params['pf_std_lat']}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(4)
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
        self.pub_particles = self.create_publisher(PointCloud2, 'tracker/particles', 10)

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

        self.process_executor = ProcessPoolExecutor(max_workers=4)

        if self.params['choose_target_from_rviz2'] == 1:
            self.get_logger().info(">>> Waiting for Click... Use '2D Goal Pose' in RViz.")
        else:
            self.get_logger().info(f">>> Auto Mode: Searching for target in front...")
            
        self.execution_count = 0

    def destroy_node(self):
        self.process_executor.shutdown()
        super().destroy_node()

    def _load_model(self):
        try:
            feature_dim = self.params['reid_feature_dim']
            num_class = self.params['reid_num_class']
            weight_path = self.params['reid_weight_path']

            net = network.reid3d(feature_dim, num_class=num_class, stride=1)
            
            if not os.path.exists(weight_path):
                self.get_logger().error(f"Weight path does not exist: {weight_path}")
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
        seq_len = self.params['sequence_length']
        
        while len(input_seq) < seq_len:
            input_seq.append(input_seq[-1])
        if len(input_seq) > seq_len:
            input_seq = input_seq[-seq_len:]
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
                        
                        target_cand.last_sim = sim

                        verify_thresh = self.params['verify_thresh']

                        if sim < verify_thresh:
                            self.get_logger().warn(f"Low Similarity Detected! Sim: {sim:.2f} < {verify_thresh}")

                        if sim > 0.6: 
                            target_cand.update_feature_gallery(feat_cpu)

        except Exception as e:
            self.get_logger().error(f"Async ReID Error: {e}")
        finally:
            self.is_reid_running = False

    def goal_callback(self, msg: PoseStamped):
        if self.params['choose_target_from_rviz2'] == 0:
            self.get_logger().warn(">>> Ignore Click: In 'Front Auto' mode (target_selection.mode=0).")
            return

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

        detections = []
        
        process_args = [(det, self.params) for det in msg.detections]
        
        if process_args:
            results = self.process_executor.map(preprocess_detection_worker, process_args)
            for res in results:
                if res is not None:
                    detections.append(res)
        
        self.update_candidates(detections, current_time)

        if self.state == "WAIT_FOR_CLICK":
            if self.params['choose_target_from_rviz2'] == 0:
                self.process_auto_front_selection()
            else:
                self.pub_status.publish(String(data="WAITING FOR CLICK..."))

        elif self.state == "INITIALIZING":
            self.process_initialization_async() 
                
        elif self.state == "TRACKING" or self.state == "LOST":
            self.process_autonomous_tracking()
            
            if self.state == "TRACKING" and self.target_candidate:
                time_since_last_seen = current_time - self.target_candidate.last_seen_time
                
                if time_since_last_seen < 0.2:
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
        min_dist = 100.0
        best_cand = None

        x_min = self.params['auto_x_min']
        x_max = self.params['auto_x_max']
        y_min = self.params['auto_y_min']
        y_max = self.params['auto_y_max']

        for cid, cand in self.candidates.items():
            x = cand.pos[0]
            y = cand.pos[1]
            
            if x_min < x < x_max and y_min < y < y_max:
                dist_to_robot = x
                if dist_to_robot < min_dist:
                    min_dist = dist_to_robot
                    best_cand = cand
        
        if best_cand:
            self.target_candidate = best_cand
            self.captured_frames_count = len(best_cand.queue)
            self.feature_locked = False
            self.registered_feature = None
            self.state = "INITIALIZING"
            self.get_logger().info(f">>> [Auto] Target Selected Front: ID {best_cand.id} at ({best_cand.pos[0]:.2f}, {best_cand.pos[1]:.2f})")
        else:
             self.pub_status.publish(String(data="SEARCHING FRONT..."))

    def update_candidates(self, detections, current_time):
        matched_det_indices = set()
        active_ids = set()
        for cid, cand in self.candidates.items():
            cand.predict(current_time)
        
        for cid, cand in self.candidates.items():
            best_dist = self.params['match_dist_thresh']
            best_idx = -1
            for i, det in enumerate(detections):
                if i in matched_det_indices: continue
                dist = np.linalg.norm(det['pos'] - cand.pred_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx != -1:
                det = detections[best_idx]
                
                # --- ここでReIDスコアも更新に使用する ---
                # last_sim が 0 (初期状態) の場合は 1.0 (信頼) と仮定
                current_sim = cand.last_sim if cand.last_sim > 0.0 else 1.0
                cand.update_state(det['pos'], det['size'], det['ori'], sim=current_sim)
                
                cand.add_points(det['points'])
                matched_det_indices.add(best_idx)
                active_ids.add(cid)
            elif (current_time - cand.last_seen_time) < self.params['max_missing_time']:
                active_ids.add(cid)

        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                nid = self.next_candidate_id
                self.next_candidate_id += 1
                self.candidates[nid] = Candidate(nid, det['pos'], det['size'], det['ori'], det['points'], self.params)
                active_ids.add(nid)
        self.candidates = {k: v for k, v in self.candidates.items() if k in active_ids}

    def process_initialization_async(self):
        seq_len = self.params['sequence_length']
        if not (self.target_candidate and self.target_candidate.id in self.candidates):
             self.pub_status.publish(String(data="Target Lost during Init"))
             if self.params['choose_target_from_rviz2'] == 0:
                 self.state = "WAIT_FOR_CLICK"
             return

        self.target_candidate = self.candidates[self.target_candidate.id]
        self.captured_frames_count = len(self.target_candidate.queue)
        
        status_msg = f"INITIALIZING: {self.captured_frames_count}/{seq_len}"
        
        if self.captured_frames_count >= seq_len and not self.is_reid_running:
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
            seq_len = self.params['sequence_length']

            for cid, cand in self.candidates.items():
                if len(cand.queue) < 5: continue 
                
                search_seq = list(cand.queue)
                while len(search_seq) < seq_len: search_seq.append(search_seq[-1])
                
                feat = self.extract_feature_single(search_seq[:seq_len])
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
                if sim > self.params['reid_sim_thresh'] and sim > best_sim:
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
        
        if not header.frame_id:
            header.frame_id = "livox_frame"
            
        marker.header = header 
        marker.ns = "status_3d"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
        
        marker.pose.position.x = 1.0 
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.5
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.5 

        seq_len = self.params['sequence_length']

        if self.state == "INITIALIZING":
            marker.text = f"INITIALIZING ({self.captured_frames_count}/{seq_len})"
            marker.color = ColorRGBA(r=0.3, g=0.3, b=1.0, a=1.0) 
        elif self.state == "TRACKING":
            tid = self.target_candidate.id if self.target_candidate else "?"
            sim = self.target_candidate.last_sim if self.target_candidate else 0.0
            marker.text = f"TRACKING ID:{tid}\nSim:{sim:.2f}"
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) 
        elif self.state == "LOST":
            marker.text = "LOST - SEARCHING..."
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) 
        else:
            if self.params['choose_target_from_rviz2'] == 1:
                marker.text = "WAITING FOR CLICK"
            else:
                marker.text = "SEARCHING FRONT..."
            marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0) 

        self.pub_status_marker.publish(marker)

    def publish_visualization(self, header):
        marker_array = MarkerArray()
        seq_len = self.params['sequence_length']
        
        target_id = -1
        
        if self.target_candidate:
            target_id = self.target_candidate.id
        
        all_particles_xyz = []    
        
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
            
            # ==========================================
            # パーティクル収集
            # ==========================================
            if cand.algo == 'PF' and hasattr(cand.kf, 'particles'):
                if is_target:
                    xyz = cand.kf.particles[:, :3]
                    all_particles_xyz.append(xyz)
            # ==========================================
            
            text = Marker()
            text.header = header
            text.ns = "text"
            text.id = cid
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = cand.pos[0]
            text.pose.position.y = cand.pos[1]
            text.pose.position.z = cand.pos[2] + (cand.size[2] if is_target else 0.5) + 0.5
            text.pose.orientation.w = 1.0
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
                label += f"\nInit:{self.captured_frames_count}/{seq_len}"
            
            label += f"\n[{cand.algo}]"
            
            text.text = label
            text.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker_array.markers.append(text)
            
        self.pub_markers.publish(marker_array)
        
        if len(all_particles_xyz) > 0:
            points_np = np.vstack(all_particles_xyz)
            
            # --- 赤色データの作成 ---
            red_color_int = 0xFF0000
            red_color_float = np.array([red_color_int], dtype=np.uint32).view(np.float32)[0]

            # NumPy構造化配列 (x, y, z, rgb)
            data = np.zeros(len(points_np), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.float32)
            ])
            data['x'] = points_np[:, 0]
            data['y'] = points_np[:, 1]
            data['z'] = points_np[:, 2]
            data['rgb'] = red_color_float

            msg = rnp.msgify(PointCloud2, data)
            msg.header = header
            
            self.pub_particles.publish(msg)

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
