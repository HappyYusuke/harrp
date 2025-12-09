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
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from vision_msgs.msg import Detection3DArray

# 点群変換用
import ros2_numpy as rnp

# ============================
# ReIDモデルのインポート設定
# ============================
HOME_DIR = os.environ['HOME']
sys.path.insert(0, f'{HOME_DIR}/ReID3D/reidnet/')

original_argv = sys.argv
sys.argv = [original_argv[0]]
try:
    from model import network
except ImportError:
    print("Error: Could not import 'model.network'. Please check your paths.")
sys.argv = original_argv
# ============================

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

class Candidate:
    """追跡中の候補オブジェクト（データ蓄積用）"""
    def __init__(self, id, pos, size, orientation, initial_points):
        self.id = id
        self.pos = pos
        self.size = size
        self.orientation = orientation
        self.queue = collections.deque(maxlen=30) # 直近30フレームを保存
        self.queue.append(initial_points)
        self.last_seen_time = time.time()
        self.score = 0.0 # ReID類似度スコア
        self.is_reid_checked = False

    def update(self, pos, size, orientation, points):
        self.pos = pos
        self.size = size
        self.orientation = orientation
        self.queue.append(points)
        self.last_seen_time = time.time()
        self.is_reid_checked = False # データ更新されたので再チェック可能に

class PersonTrackerReIDNode(Node):
    def __init__(self):
        super().__init__('person_tracker_reid')

        # --- パラメータ設定 ---
        self.declare_parameter('max_missing_time', 1.0)     # ターゲットを見失ったとみなすまでの時間
        self.declare_parameter('match_dist_thresh', 1.2)    # 同一物体とみなす移動距離閾値(m)
        self.declare_parameter('reid_sim_thresh', 0.85)     # ReID類似度閾値
        self.declare_parameter('registration_duration', 8.0) # 登録にかける時間

        self.MAX_MISSING_TIME = self.get_parameter('max_missing_time').value
        self.MATCH_DIST_THRESH = self.get_parameter('match_dist_thresh').value
        self.REID_SIM_THRESH = self.get_parameter('reid_sim_thresh').value
        self.REG_DURATION = self.get_parameter('registration_duration').value

        # --- モデルロード ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._load_model()
        self.get_logger().info(f'ReID Model Loaded on {self.device}')

        # --- 通信設定 ---
        self.sub_bbox = self.create_subscription(Detection3DArray, '/bbox', self.bbox_callback, 10)
        self.sub_cmd = self.create_subscription(String, '/tracker_cmd', self.cmd_callback, 10)
        self.pub_status = self.create_publisher(String, '/target_status', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/person_reid_markers', 10)

        # --- 内部状態 ---
        self.state = "IDLE" # IDLE, REGISTERING, TRACKING, LOST
        self.registered_feature = None # 登録された特徴量
        
        # ターゲット情報
        self.target_pos = None
        self.target_last_seen = 0.0
        
        # 候補者トラッカー管理 (ID -> Candidate)
        self.candidates = {} 
        self.next_candidate_id = 0
        
        # 登録用一時データ
        self.reg_queue = collections.deque(maxlen=30)
        self.reg_features = []

    def _load_model(self):
        try:
            net = torch.nn.DataParallel(network.reid3d(1024, num_class=222, stride=1))
            weight_path = f'{HOME_DIR}/ReID3D/reidnet/log/ckpt_best.pth'
            if not os.path.exists(weight_path):
                self.get_logger().error(f"Weights not found: {weight_path}")
                sys.exit(1)
            weight = torch.load(weight_path, map_location=self.device)
            net.load_state_dict(weight)
            net.to(self.device)
            net.eval()
            return net
        except Exception as e:
            self.get_logger().error(f"Model load failed: {e}")
            sys.exit(1)

    def cmd_callback(self, msg):
        if msg.data == "start_registration":
            self.state = "REGISTERING"
            self.reg_features = []
            self.reg_queue.clear()
            self.start_reg_time = time.time()
            self.get_logger().info('>>> Start Registration: Walk figure-8.')
        
        elif msg.data == "start_tracking":
            if self.registered_feature is not None:
                self.state = "TRACKING"
                self.target_last_seen = time.time()
                # 現在位置に近い候補がいれば、それを初期位置とする
                # (実装省略: 必要ならここでcandidatesから最も近いものをtarget_posにセット)
                self.get_logger().info('>>> Start Tracking.')
            else:
                self.get_logger().warn('Cannot track: No user registered.')

    def bbox_callback(self, msg: Detection3DArray):
        current_time = time.time()
        
        # 1. 前処理: Detectionリストの作成
        detections = []
        for detection in msg.detections:
            if detection.source_cloud.width * detection.source_cloud.height == 0: continue
            try:
                raw_points = rnp.point_cloud2.pointcloud2_to_xyz_array(detection.source_cloud)
            except: continue
            if raw_points.shape[0] < 10: continue

            norm_points = normalize_point_cloud(raw_points, num_points=256)
            bbox = detection.bbox
            pos = np.array([bbox.center.position.x, bbox.center.position.y, bbox.center.position.z])
            size = np.array([bbox.size.x, bbox.size.y, bbox.size.z])
            detections.append({'pos': pos, 'size': size, 'ori': bbox.center.orientation, 'points': norm_points})

        # 2. 状態別処理
        if self.state == "REGISTERING":
            self.process_registration(detections, current_time)
            
        elif self.state == "TRACKING" or self.state == "LOST":
            self.process_tracking_and_candidates(detections, current_time)

        # 3. 可視化
        self.publish_visualization(msg.header)

    def process_registration(self, detections, current_time):
        """登録フェーズ"""
        # 一番近い検出物を対象とする
        if len(detections) > 0:
            detections.sort(key=lambda d: np.linalg.norm(d['pos'])) # 原点に近い順
            target = detections[0]
            self.reg_queue.append(target['points'])
            self.target_pos = target['pos'] # 表示用

            # 30フレーム溜まるごとに特徴量抽出
            if len(self.reg_queue) == 30:
                feat = self.extract_feature(list(self.reg_queue))
                self.reg_features.append(feat)
                self.get_logger().info(f'Feature captured. Count: {len(self.reg_features)}')
                # キューはクリアせず、スライディングウィンドウ的に使うか、間引くか
                # ここではシンプルに次の30を待つために半分捨てる等の工夫も可だがそのまま

        if (current_time - self.start_reg_time) > self.REG_DURATION:
            if len(self.reg_features) > 0:
                feats_stack = torch.stack(self.reg_features)
                mean_feat = torch.mean(feats_stack, dim=0)
                self.registered_feature = mean_feat / torch.norm(mean_feat)
                self.get_logger().info('Registration COMPLETED.')
                self.state = "IDLE"
            else:
                self.get_logger().warn('Registration FAILED.')
                self.state = "IDLE"

    def process_tracking_and_candidates(self, detections, current_time):
        """
        トラッキング + 候補者管理
        1. 検出と候補(Candidate)の紐付け
        2. ターゲットの追跡 (位置ベース)
        3. ロスト時のReID復帰 (30フレーム溜まった候補に対して実施)
        """
        
        # --- A. データアソシエーション (検出 -> 候補) ---
        # 既存候補の位置予測などは簡易的に「前回位置」とする
        
        matched_det_indices = set()
        active_candidate_ids = set()

        # 既存候補とのマッチング (Greedy)
        for cid, candidate in self.candidates.items():
            best_dist = self.MATCH_DIST_THRESH
            best_idx = -1
            
            for i, det in enumerate(detections):
                if i in matched_det_indices: continue
                dist = np.linalg.norm(det['pos'] - candidate.pos)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            if best_idx != -1:
                # 更新
                det = detections[best_idx]
                candidate.update(det['pos'], det['size'], det['ori'], det['points'])
                matched_det_indices.add(best_idx)
                active_candidate_ids.add(cid)
            elif (current_time - candidate.last_seen_time) < 2.0:
                # マッチしなかったが、まだ消去しない (2秒猶予)
                active_candidate_ids.add(cid)

        # 新規候補の作成
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                new_id = self.next_candidate_id
                self.next_candidate_id += 1
                new_cand = Candidate(new_id, det['pos'], det['size'], det['ori'], det['points'])
                self.candidates[new_id] = new_cand
                active_candidate_ids.add(new_id)

        # 古い候補の削除
        self.candidates = {k: v for k, v in self.candidates.items() if k in active_candidate_ids}

        # --- B. ターゲットの追跡 (Tracking) ---
        target_matched = False
        
        # 現在TRACKINGモードで、直近まで見えていた場合、位置が近い候補をターゲットとみなす
        if self.state == "TRACKING" or (self.state == "LOST" and (current_time - self.target_last_seen) < self.MAX_MISSING_TIME):
            # ターゲット位置に最も近い候補を探す
            best_cand = None
            min_dist = self.MATCH_DIST_THRESH
            
            for cid, cand in self.candidates.items():
                # ターゲットの前回位置との距離
                d = np.linalg.norm(cand.pos - self.target_pos)
                if d < min_dist:
                    min_dist = d
                    best_cand = cand
            
            if best_cand:
                # 追跡継続 (Coastingからの復帰含む)
                self.target_pos = best_cand.pos
                self.target_last_seen = current_time
                self.state = "TRACKING"
                target_matched = True
                
                status = f"TRACKING: {self.target_pos[0]:.1f}, {self.target_pos[1]:.1f} (ID:{best_cand.id})"

        # --- C. ロスト時のReID復帰 (ReID Recovery) ---
        if not target_matched:
            # COASTING時間を過ぎたら LOST モードへ
            if (current_time - self.target_last_seen) > self.MAX_MISSING_TIME:
                self.state = "LOST"
                status = "LOST: Gathering data..."
                
                # 全候補をチェック
                for cid, cand in self.candidates.items():
                    # ★重要: データが30フレーム溜まっている候補のみ判定する
                    if len(cand.queue) == 30:
                        # 毎回計算すると重いので、データ更新時などに制限しても良いが、
                        # ここでは30フレームあるなら毎回チェックする(最新の窓で判定)
                        
                        feat = self.extract_feature(list(cand.queue))
                        sim = F.cosine_similarity(feat.unsqueeze(0), self.registered_feature.unsqueeze(0)).item()
                        cand.score = sim # 可視化用
                        
                        if sim > self.REID_SIM_THRESH:
                            # 発見！
                            self.get_logger().info(f"ReID RECOVERY! ID:{cid} Sim:{sim:.3f}")
                            self.target_pos = cand.pos
                            self.target_last_seen = current_time
                            self.state = "TRACKING"
                            target_matched = True
                            status = f"RECOVERED! ({sim:.2f})"
                            break # 1つ見つかればOK
            else:
                # まだCoasting期間中
                status = f"COASTING: {current_time - self.target_last_seen:.1f}s"

        self.pub_status.publish(String(data=status))

    def extract_feature(self, points_sequence):
        """list of (256, 3) -> tensor -> model -> feature"""
        seq_np = np.array(points_sequence) # (30, 256, 3)
        tensor = torch.from_numpy(seq_np).float()
        input_tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.net(input_tensor)
        return output['val_bn'][0]

    def publish_visualization(self, header):
        marker_array = MarkerArray()

        # 1. ターゲットマーカー
        if self.target_pos is not None:
            # ターゲットBox
            mk = Marker()
            mk.header = header
            mk.ns = "target"
            mk.id = 9999
            mk.type = Marker.CUBE
            mk.action = Marker.ADD
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = self.target_pos
            mk.scale.x, mk.scale.y, mk.scale.z = 0.6, 0.6, 1.6
            
            if self.state == "TRACKING":
                mk.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.6)
            elif self.state == "LOST": # Coasting含む
                mk.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.4)
            else:
                mk.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)
            
            mk.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker_array.markers.append(mk)

        # 2. 候補者マーカー (LOST時のみ詳細表示など)
        for cid, cand in self.candidates.items():
            # ターゲットと位置が被っている(同一人物)なら表示しない制御も可
            if self.target_pos is not None and np.linalg.norm(cand.pos - self.target_pos) < 0.5:
                continue

            mk = Marker()
            mk.header = header
            mk.ns = "candidates"
            mk.id = cid
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = cand.pos
            mk.pose.position.z += 1.0 # 頭上
            mk.scale.x = mk.scale.y = mk.scale.z = 0.3
            
            # データ蓄積具合で色を変える (赤->白)
            progress = len(cand.queue) / 30.0
            mk.color = ColorRGBA(r=1.0, g=progress, b=progress, a=0.8)
            mk.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker_array.markers.append(mk)

            # テキスト (ID と スコア)
            text = Marker()
            text.header = header
            text.ns = "candidates_text"
            text.id = cid
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x, text.pose.position.y, text.pose.position.z = cand.pos
            text.pose.position.z += 1.3
            text.scale.z = 0.2
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text.text = f"ID:{cid}\n{len(cand.queue)}/30"
            if cand.score > 0:
                text.text += f"\nSim:{cand.score:.2f}"
            text.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker_array.markers.append(text)

        self.pub_markers.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerReIDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
