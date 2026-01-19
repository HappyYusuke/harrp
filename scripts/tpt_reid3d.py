#!/usr/bin/env python3

import os
import sys
import collections
import time
import numpy as np
import torch
import torch.nn.functional as F
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseStamped # クリック受取用

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
# カルマンフィルタ
# ==========================================
class KalmanBoxTracker(object):
    def __init__(self, initial_pos):
        self.x = np.zeros((6, 1))
        self.x[:3, 0] = initial_pos
        self.F = np.eye(6)
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)
        self.P = np.eye(6) * 10.0
        self.Q = np.eye(6) * 0.1
        self.R = np.eye(3) * 1.0 
        self.last_timestamp = time.time()

    def predict(self, current_time):
        dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:3, 0]

    def update(self, measurement):
        z = measurement.reshape(3, 1)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.x.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

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
        self.pos = self.kf.x[:3, 0] 
        self.size = size
        self.orientation = orientation
        self.last_seen_time = time.time()

    def add_points(self, points):
        self.queue.append(points)

class PersonTrackerClickInitNode(Node):
    def __init__(self):
        super().__init__('person_tracker_click_init')

        # --- パラメータ ---
        self.declare_parameter('max_missing_time', 2.0)
        self.declare_parameter('reid_sim_thresh', 0.70)
        
        self.MAX_MISSING_TIME = self.get_parameter('max_missing_time').value
        self.REID_SIM_THRESH = self.get_parameter('reid_sim_thresh').value
        
        # --- モデルロード ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._load_model()
        self.get_logger().info(f'Model Loaded on {self.device}')

        # --- 通信 ---
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_bbox = self.create_subscription(Detection3DArray, '/bbox', self.bbox_callback, qos_profile)
        
        # ★追加: RVizの "2D Goal Pose" を受け取る
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        self.pub_markers = self.create_publisher(MarkerArray, '/person_reid_markers', 10)
        self.pub_status = self.create_publisher(String, '/target_status', 10)

        # --- 状態 ---
        self.state = "WAIT_FOR_CLICK" # 最初はクリック待ち
        self.target_candidate = None   
        self.registered_feature = None 
        self.feature_locked = False    
        self.captured_frames_count = 0 
        
        self.candidates = {} 
        self.next_candidate_id = 0
        
        with open('tracking_results.txt', 'w') as f:
            f.write("") 
            
        self.get_logger().info(">>> Waiting for Click... Use '2D Goal Pose' in RViz to select target.")

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

    # ★クリック時の処理
    def goal_callback(self, msg: PoseStamped):
        click_x = msg.pose.position.x
        click_y = msg.pose.position.y
        self.get_logger().info(f">>> Click Received at ({click_x:.2f}, {click_y:.2f})")
        
        # 一番近い候補を探す
        closest_cand = None
        min_dist = 3.0 # クリック誤差許容範囲
        
        for cid, cand in self.candidates.items():
            dist = math.sqrt((cand.pos[0] - click_x)**2 + (cand.pos[1] - click_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_cand = cand
        
        if closest_cand:
            # リセットして再登録
            self.target_candidate = closest_cand
            self.captured_frames_count = len(closest_cand.queue) # 既存データを引き継ぐ
            self.feature_locked = False
            self.registered_feature = None
            self.state = "INITIALIZING"
            self.get_logger().info(f">>> Target Selected: ID {closest_cand.id}. Gathering frames...")
        else:
            self.get_logger().warn(">>> No candidate found near click position!")

    def bbox_callback(self, msg: Detection3DArray):
        current_time = time.time()
        if hasattr(msg.header.stamp, 'sec'):
            timestamp_nanosec = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        else:
            timestamp_nanosec = msg.header.stamp.nanoseconds

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

        # 2. 候補更新 (KF Matching)
        self.update_candidates(detections, current_time)

        # 3. メインロジック
        if self.state == "INITIALIZING":
            self.process_initialization()
        elif self.state == "TRACKING" or self.state == "LOST":
            self.process_autonomous_tracking(timestamp_nanosec)
        else:
            self.pub_status.publish(String(data="WAITING FOR CLICK..."))

        self.publish_visualization(msg.header)

    def update_candidates(self, detections, current_time):
        matched_det_indices = set()
        active_ids = set()
        
        for cid, cand in self.candidates.items():
            cand.predict(current_time)

        for cid, cand in self.candidates.items():
            best_dist = 2.0 
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
        """クリックされたターゲットのデータが溜まるのを待つ"""
        status_msg = "INITIALIZING"
        
        if self.target_candidate and self.target_candidate.id in self.candidates:
            # ターゲット更新 (Candidateオブジェクトは毎回作り直される可能性があるためIDで追う)
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
        
        # A. 位置ベース
        if self.target_candidate and self.target_candidate.id in self.candidates:
            self.target_candidate = self.candidates[self.target_candidate.id]
            target_found = True
            
        # B. ReID復帰
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

        if target_found and self.target_candidate:
            self.save_log(timestamp_nanosec, self.target_candidate)

    def save_log(self, timestamp, c):
        yaw = get_yaw_from_orientation(c.orientation)
        score = c.last_sim if c.last_sim > 0 else 1.0
        with open('tracking_results.txt', 'a') as f:
             f.write(f"{timestamp} {c.id} Car 0 0 0 -1 -1 -1 -1 "
                    f"{c.size[2]} {c.size[1]} {c.size[0]} " 
                    f"{c.pos[0]} {c.pos[1]} {c.pos[2]} " 
                    f"{yaw} {score}\n")

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
                    mk.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5) # 緑
                else:
                    mk.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.5) # 黄
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
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            
            label = f"ID:{cid}"
            if cand.last_sim > 0: label += f"\nSim:{cand.last_sim:.2f}"
            if is_target and not self.feature_locked: label += f"\nInit:{self.captured_frames_count}/30"
            
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
