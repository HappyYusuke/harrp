#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs 

import numpy as np
import math
from scipy.optimize import minimize

# マルチスレッド実行用
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

class MPCConfig:
    def __init__(self):
        # --- 制約条件 ---
        self.max_speed = 0.8
        self.min_speed = 0.0
        self.max_yaw_rate = 1.2
        self.max_accel = 0.5
        self.max_dyaw_rate = 2.0
        
        # --- MPCパラメータ ---
        self.dt = 0.2           
        self.horizon = 8        
        
        # --- 重み付け ---
        self.w_dist = 1.0       
        self.w_heading = 0.5    
        self.w_vel = 1.5        
        self.w_obs = 0.8        
        
        # --- その他 ---
        self.robot_radius = 0.3 
        self.goal_tolerance = 0.5
        self.sensor_offset = 0.0 
        
        # PD制御パラメータ
        self.turn_kp = 1.5
        self.turn_kd = 0.5
        self.turn_yaw_tolerance = 0.1
        
        # Livoxフィルタ
        self.min_height = -0.05
        self.max_height = 0.1
        self.min_dist = 0.2
        
        # ロスト判定時間 (秒)
        self.lost_timeout = 1.0

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')
        self.config = MPCConfig()
        
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_speed', self.config.max_speed),
                ('min_speed', self.config.min_speed),
                ('max_yaw_rate', self.config.max_yaw_rate),
                ('max_accel', self.config.max_accel),
                ('max_dyaw_rate', self.config.max_dyaw_rate),
                ('dt', self.config.dt),
                ('horizon', self.config.horizon),
                ('w_dist', self.config.w_dist),
                ('w_heading', self.config.w_heading),
                ('w_vel', self.config.w_vel),
                ('w_obs', self.config.w_obs),
                ('robot_radius', self.config.robot_radius),
                ('goal_tolerance', self.config.goal_tolerance),
                ('sensor_offset', self.config.sensor_offset),
                ('turn_kp', self.config.turn_kp),
                ('turn_kd', self.config.turn_kd),
                ('turn_yaw_tolerance', self.config.turn_yaw_tolerance),
                ('lost_timeout', self.config.lost_timeout),
            ]
        )
        self.update_config_from_params()
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'mpc/predict_path', 10)
        self.obs_pub = self.create_publisher(Marker, 'mpc/obstacles', 10)
        self.robot_radius_pub = self.create_publisher(Marker, 'mpc/robot_radius', 10)
        self.target_marker_pub = self.create_publisher(Marker, 'mpc/target_debug', 10)
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            Odometry, 'odom', self.odom_callback, qos_profile, 
            callback_group=self.cb_group
        )
        self.create_subscription(
            PoseStamped, 'tracker/target_pose', self.target_callback, 10, 
            callback_group=self.cb_group
        )
        # PointCloud2はデータ量が多いため、QoSを調整
        self.create_subscription(
            PointCloud2, '/livox/lidar', self.lidar_callback, 10, 
            callback_group=self.cb_group
        )
        self.create_subscription(
            String, 'tracker/target_status', self.status_callback, 10,
            callback_group=self.cb_group
        )
        
        self.current_vel = 0.0
        self.current_yaw_rate = 0.0
        
        self.target_local = None 
        self.target_odom = None  
        self.last_target_time = None
        self.sensor_frame_id = "livox_frame"
        self.is_tracker_lost = False
        
        self.obstacles_local = np.array([])
        
        self.prev_solution = np.zeros(self.config.horizon * 2)
        self.prev_yaw_error = None
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.timer = self.create_timer(
            0.1, self.control_loop, 
            callback_group=self.cb_group
        )

    def update_config_from_params(self):
        self.config.max_speed = self.get_parameter('max_speed').value
        self.config.min_speed = self.get_parameter('min_speed').value
        self.config.max_yaw_rate = self.get_parameter('max_yaw_rate').value
        self.config.max_accel = self.get_parameter('max_accel').value
        self.config.max_dyaw_rate = self.get_parameter('max_dyaw_rate').value
        self.config.dt = self.get_parameter('dt').value
        self.config.horizon = self.get_parameter('horizon').value
        self.config.w_dist = self.get_parameter('w_dist').value
        self.config.w_heading = self.get_parameter('w_heading').value
        self.config.w_vel = self.get_parameter('w_vel').value
        self.config.w_obs = self.get_parameter('w_obs').value
        self.config.robot_radius = self.get_parameter('robot_radius').value
        self.config.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.config.sensor_offset = self.get_parameter('sensor_offset').value
        self.config.turn_kp = self.get_parameter('turn_kp').value
        self.config.turn_kd = self.get_parameter('turn_kd').value
        self.config.turn_yaw_tolerance = self.get_parameter('turn_yaw_tolerance').value
        self.config.lost_timeout = self.get_parameter('lost_timeout').value

    def parameter_callback(self, params):
        horizon_changed = False
        for param in params:
            if hasattr(self.config, param.name):
                setattr(self.config, param.name, param.value)
                if param.name == 'horizon':
                    horizon_changed = True
        if horizon_changed:
            self.prev_solution = np.zeros(self.config.horizon * 2)
        return SetParametersResult(successful=True)

    def odom_callback(self, msg):
        self.current_vel = msg.twist.twist.linear.x
        self.current_yaw_rate = msg.twist.twist.angular.z

    def status_callback(self, msg):
        if "LOST" in msg.data:
            self.is_tracker_lost = True
        else:
            self.is_tracker_lost = False

    def target_callback(self, msg):
        self.target_local = np.array([msg.pose.position.x, msg.pose.position.y])
        self.sensor_frame_id = msg.header.frame_id
        self.last_target_time = self.get_clock().now()
        
        self.is_tracker_lost = False

        try:
            transform_timeout = rclpy.duration.Duration(seconds=0.05)
            self.target_odom = self.tf_buffer.transform(msg, 'odom', timeout=transform_timeout)
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass

    def lidar_callback(self, msg):
        """
        NumPyを使った高速かつ堅牢な点群処理
        point_stepがfloat32(4byte)の倍数でない場合でもズレないように修正
        """
        self.sensor_frame_id = msg.header.frame_id
        self.lidar_header_stamp = msg.header.stamp
        
        try:
            point_step = msg.point_step
            # 1. バイト列として一括読み込み
            raw_uint8 = np.frombuffer(msg.data, dtype=np.uint8)
            
            # 2. ポイント数を確認してリシェイプ [N, point_step]
            # これにより、1点ごとの区切りをバイト単位で正確に守ります
            if len(raw_uint8) % point_step != 0:
                self.get_logger().warn("PointCloud data size is not a multiple of point_step!")
                return
            
            num_points = len(raw_uint8) // point_step
            points_data = raw_uint8.reshape(num_points, point_step)
            
            # 3. x, y, z のバイト列を抽出して float32 に変換
            # オフセットは通常 0, 4, 8 です
            # .copy() を入れてメモリを連続させてから view() します
            x_bytes = points_data[:, 0:4].copy().view(dtype=np.float32).reshape(-1)
            y_bytes = points_data[:, 4:8].copy().view(dtype=np.float32).reshape(-1)
            z_bytes = points_data[:, 8:12].copy().view(dtype=np.float32).reshape(-1)
            
            # 4. (N, 3) の座標配列を作成
            xyz = np.stack((x_bytes, y_bytes, z_bytes), axis=1)
            
            # --- 以下フィルタリング (既存) ---
            
            # NaN/Inf 除去
            mask_valid = np.isfinite(xyz).all(axis=1)
            xyz = xyz[mask_valid]
            
            if xyz.shape[0] == 0:
                self.obstacles_local = np.array([])
                return

            # 高さフィルタ
            mask_z = (xyz[:, 2] > self.config.min_height) & (xyz[:, 2] < self.config.max_height)
            xyz = xyz[mask_z]
            
            # 距離フィルタ
            dists = np.linalg.norm(xyz[:, :2], axis=1)
            mask_dist = (dists > self.config.robot_radius) & (dists < 5.0)
            xyz = xyz[mask_dist]
            
            # 間引き
            xyz = xyz[::5] 
            
            if xyz.shape[0] > 0:
                self.obstacles_local = xyz[:, :2].astype(np.float64)
            else:
                self.obstacles_local = np.array([])
                
        except Exception as e:
            self.get_logger().warn(f"Lidar Processing Error: {e}")
            self.obstacles_local = np.array([])

    def get_target_for_mpc(self):
        # 最後に受信してから一定時間経ったらリセットする処理
        if self.last_target_time is None:
            return None

        elapsed = (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9
        is_lost_condition = self.is_tracker_lost or (elapsed > self.config.lost_timeout)
        
        # --- A. トラッキング中 (通常) ---
        if not is_lost_condition:
            # センサー座標系(livox_frame)などの相対座標をそのまま使う
            return self.target_local

        # --- B. 見失った場合 (Odom座標から復元) ---
        else:
            if self.target_odom is None:
                return None
            
            try:
                # ★修正ポイント: 
                # 過去のPoseStampedをそのまま使うと、TFは「過去のロボット位置」基準で計算してしまう。
                # 「現在のロボット位置」基準で計算させるため、タイムスタンプを最新(0)にする。
                
                # 1. データをコピー (元のself.target_odomを書き換えないため)
                #    ※ copyモジュールがない場合は import copy してください
                #    簡易的に新しいPoseStampedを作ります
                current_target_in_odom = PoseStamped()
                current_target_in_odom.header.frame_id = self.target_odom.header.frame_id # 'odom'
                current_target_in_odom.header.stamp = rclpy.time.Time().to_msg() # ★時刻を「最新」にする
                current_target_in_odom.pose = self.target_odom.pose # 座標は過去に覚えた絶対位置

                # 2. 現在の base_link 基準に変換
                target_recovered = self.tf_buffer.transform(
                    current_target_in_odom, 
                    'base_link',  # 現在のロボット中心基準
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                
                return np.array([target_recovered.pose.position.x, target_recovered.pose.position.y])

            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"Transform Error in get_target: {e}")
                return None

    def predict_next_state(self, state, v, w, dt):
        x, y, yaw, _, _ = state
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += w * dt
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        return np.array([x, y, yaw, v, w])

    # ★修正: obstacles引数を追加 (スナップショットを受け取る)
    def cost_function(self, u_flat, current_target, current_obstacles):
        cost = 0.0
        curr_state = np.array([0.0, 0.0, 0.0, self.current_vel, self.current_yaw_rate])
        u = u_flat.reshape(self.config.horizon, 2)
        
        for i in range(self.config.horizon):
            v, w = u[i]
            curr_state = self.predict_next_state(curr_state, v, w, self.config.dt)
            px, py, pyaw, _, _ = curr_state
            
            dx = current_target[0] - px
            dy = current_target[1] - py
            dist = math.hypot(dx, dy)
            cost += self.config.w_dist * dist
            
            target_yaw = math.atan2(dy, dx)
            yaw_diff = abs(target_yaw - pyaw)
            yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
            cost += self.config.w_heading * abs(yaw_diff)
            
            target_v = self.config.max_speed if dist > 1.0 else 0.0
            cost += self.config.w_vel * abs(target_v - v)
            
            # ★修正: クラス変数の代わりに引数の current_obstacles を使用
            if len(current_obstacles) > 0:
                dx_obs = current_obstacles[:, 0] - px
                dy_obs = current_obstacles[:, 1] - py
                obs_dists = np.hypot(dx_obs, dy_obs)
                min_obs_dist = np.min(obs_dists)
                if min_obs_dist < self.config.robot_radius:
                    cost += self.config.w_obs * (1.0 / (min_obs_dist + 0.01)) * 100
                elif min_obs_dist < self.config.robot_radius + 0.5:
                    cost += self.config.w_obs * (1.0 / min_obs_dist)
        return cost

    def run_mpc(self):
        target = self.get_target_for_mpc()
        if target is None:
            return 0.0, 0.0, []

        # ★修正: 障害物情報のコピーを作成してスナップショット化 (競合回避)
        obstacles_snapshot = self.obstacles_local.copy()

        dx = target[0]
        dy = target[1]
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx) 

        # PD制御 (旋回)
        if dist < self.config.goal_tolerance:
            yaw_diff = target_yaw
            while yaw_diff > math.pi: yaw_diff -= 2 * math.pi
            while yaw_diff < -math.pi: yaw_diff += 2 * math.pi

            if self.prev_yaw_error is None:
                derivative = 0.0
            else:
                derivative = (yaw_diff - self.prev_yaw_error) / 0.1
            self.prev_yaw_error = yaw_diff

            if abs(yaw_diff) > self.config.turn_yaw_tolerance:
                w_cmd = (self.config.turn_kp * yaw_diff) + (self.config.turn_kd * derivative)
                w_cmd = max(min(w_cmd, self.config.max_yaw_rate), -self.config.max_yaw_rate)
                return 0.0, w_cmd, []
            else:
                return 0.0, 0.0, []
        else:
            self.prev_yaw_error = None

        # MPC
        bounds = []
        for _ in range(self.config.horizon):
            bounds.append((self.config.min_speed, self.config.max_speed))
            bounds.append((-self.config.max_yaw_rate, self.config.max_yaw_rate))

        x0 = np.roll(self.prev_solution, -2)
        x0[-2:] = 0.0

        # ★修正: obstacles_snapshot を渡す
        res = minimize(
            self.cost_function, 
            x0, 
            args=(target, obstacles_snapshot),
            method='SLSQP', 
            bounds=bounds, 
            tol=1e-3
        )
        
        if res.success:
            self.prev_solution = res.x
            v_cmd = res.x[0]
            w_cmd = res.x[1]
            predict_path = []
            curr_state = np.array([0.0, 0.0, 0.0, self.current_vel, self.current_yaw_rate])
            u = res.x.reshape(self.config.horizon, 2)
            for i in range(self.config.horizon):
                curr_state = self.predict_next_state(curr_state, u[i][0], u[i][1], self.config.dt)
                predict_path.append([curr_state[0], curr_state[1]])
            return v_cmd, w_cmd, predict_path
        else:
            self.get_logger().warn("MPC Optimization Failed")
            return 0.0, 0.0, []

    def publish_path(self, trajectory):
        if not trajectory: return
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.sensor_frame_id
        for point in trajectory:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        
    def publish_obstacles(self):
        if len(self.obstacles_local) == 0: return
        marker = Marker()
        marker.header.frame_id = self.sensor_frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacles"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color.r = 1.0; marker.color.a = 1.0
        
        # 配列が空でないか再確認
        if self.obstacles_local.shape[0] > 0:
            # ★修正: floatへの明示的キャスト (AssertionError対策)
            # Numpyの配列をPythonのリストに変換してからイテレートする方が安全
            for obs in self.obstacles_local:
                p = Point()
                p.x = float(obs[0]) # ここが重要
                p.y = float(obs[1]) # ここが重要
                marker.points.append(p)
        
        self.obs_pub.publish(marker)
        
    def publish_robot_radius(self):
        marker = Marker()
        marker.header.frame_id = self.sensor_frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "safety_margin"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        diameter = self.config.robot_radius * 2.0
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.3
        self.robot_radius_pub.publish(marker)

    def publish_target_marker(self):
        target = self.get_target_for_mpc()
        if target is None: return
        
        marker = Marker()
        
        # ★修正: トラッキング中かロスト中かで、座標系が変わるため frame_id を切り替える
        elapsed = 0.0
        if self.last_target_time is not None:
            elapsed = (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9
        
        is_lost = self.is_tracker_lost or (elapsed > self.config.lost_timeout)

        if is_lost:
            # ロスト時は get_target_for_mpc が 'base_link' 基準の値を返す
            marker.header.frame_id = "base_link"
        else:
            # トラッキング中は target_callback で受け取ったセンサー座標系 (例: livox_frame)
            marker.header.frame_id = self.sensor_frame_id

        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mpc_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(target[0])
        marker.pose.position.y = float(target[1])
        marker.pose.position.z = 0.5 
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        if self.is_tracker_lost:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        else:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            
        marker.color.a = 1.0
        self.target_marker_pub.publish(marker)

    def control_loop(self):
        v, w, path = self.run_mpc()
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_vel_pub.publish(msg)
        self.publish_path(path)
        self.publish_obstacles()
        self.publish_robot_radius()
        self.publish_target_marker()

def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
