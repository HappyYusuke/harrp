#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
# 変更点1: PointCloud2 と読み込み用ライブラリをインポート
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker

import numpy as np
import math
from scipy.optimize import minimize

class MPCConfig:
    def __init__(self):
        # --- 制約条件 ---
        self.max_speed = 0.8
        self.min_speed = 0.0
        self.max_yaw_rate = 1.2
        self.max_accel = 0.5
        self.max_dyaw_rate = 2.0
        
        # --- MPCパラメータ ---
        self.dt = 0.2           # 予測の刻み時間 (s)
        self.horizon = 8        # 何ステップ先まで読むか
        
        # --- 重み付け (Cost Function) ---
        self.w_dist = 1.0       # ゴールへ近づくこと
        self.w_heading = 0.5    # ゴールを向くこと
        self.w_vel = 1.5        # 速度を出すこと
        self.w_obs = 0.8        # 障害物を避けること
        
        # --- その他 ---
        self.robot_radius = 0.3 # 安全マージン
        self.goal_tolerance = 0.5
        self.sensor_offset = 0.0 # Lidar位置補正
        
        # --- Livox用フィルタ設定 (追加) ---
        self.min_height = -0.5  # これより低い点は無視（床対策）
        self.max_height = 0.5   # これより高い点は無視
        self.min_dist = 0.2     # 近すぎるノイズ除去

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')
        self.config = MPCConfig()
        
        # --- 全パラメータの宣言 ---
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
            ]
        )
        self.update_config_from_params()
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'mpc/predict_path', 10)
        self.obs_pub = self.create_publisher(Marker, 'mpc/obstacles', 10)
        self.robot_radius_pub = self.create_publisher(Marker, 'mpc/robot_radius', 10)
        
        # QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile)
        self.create_subscription(PoseStamped, 'target_pose', self.target_callback, 10)
        self.create_subscription(PointCloud2, '/livox/lidar', self.lidar_callback, 10)
        
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.target = None
        self.obstacles = []
        self.prev_solution = np.zeros(self.config.horizon * 2)
        
        self.timer = self.create_timer(0.1, self.control_loop)

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

    def parameter_callback(self, params):
        horizon_changed = False
        for param in params:
            if hasattr(self.config, param.name):
                setattr(self.config, param.name, param.value)
                if param.name == 'horizon':
                    horizon_changed = True
                self.get_logger().info(f'Parameter updated: {param.name} = {param.value}')

        if horizon_changed:
            self.get_logger().warn(f'Horizon changed. Resetting buffer.')
            self.prev_solution = np.zeros(self.config.horizon * 2)

        return SetParametersResult(successful=True)

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw, 
                               msg.twist.twist.linear.x, msg.twist.twist.angular.z])

    def target_callback(self, msg):
        self.target = np.array([msg.pose.position.x, msg.pose.position.y])

    # 変更点3: Livox用の新しいコールバック関数
    def lidar_callback(self, msg):
        obs = []
        
        # オドメトリから現在のロボット姿勢を取得
        rx, ry, ryaw = self.state[0], self.state[1], self.state[2]
        c_yaw = math.cos(ryaw)
        s_yaw = math.sin(ryaw)

        # 点群データの読み込み (x, y, z のジェネレータ)
        # skip_n: データが多すぎる場合、間引く数を増やす (例: 5点に1点採用)
        skip_n = 5
        point_generator = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        for i, point in enumerate(point_generator):
            if i % skip_n != 0: continue

            x, y, z = point

            # フィルタリング 1: 高さ (Z軸)
            # ロボットの高さに合わせて調整してください。床を含めないように注意。
            if z < self.config.min_height or z > self.config.max_height:
                continue
            
            # フィルタリング 2: 距離 (近すぎるノイズを除去)
            dist = math.hypot(x, y)
            if dist < self.config.min_dist:
                continue

            # 座標変換: ロボット座標系(Lidar) -> ワールド座標系(Odom)
            # ※ Lidarがロボットの中心(またはsensor_offset分前方)にあると仮定
            # 回転
            gx = x * c_yaw - y * s_yaw
            gy = x * s_yaw + y * c_yaw
            
            # 並進 (sensor_offset を考慮)
            # オフセット分だけ、現在の向きに座標をずらす
            offset_x = self.config.sensor_offset * c_yaw
            offset_y = self.config.sensor_offset * s_yaw

            ox = gx + rx + offset_x
            oy = gy + ry + offset_y

            obs.append([ox, oy])
        
        if len(obs) > 0:
            self.obstacles = np.array(obs)
        else:
            self.obstacles = np.array([])

    def predict_next_state(self, state, v, w, dt):
        x, y, yaw, _, _ = state
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += w * dt
        # 角度正規化
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        return np.array([x, y, yaw, v, w])

    def cost_function(self, u_flat):
        cost = 0.0
        curr_state = self.state.copy()
        u = u_flat.reshape(self.config.horizon, 2)
        
        for i in range(self.config.horizon):
            v, w = u[i]
            curr_state = self.predict_next_state(curr_state, v, w, self.config.dt)
            px, py, pyaw, _, _ = curr_state
            
            dx = self.target[0] - px
            dy = self.target[1] - py
            dist = math.hypot(dx, dy)
            cost += self.config.w_dist * dist
            
            target_yaw = math.atan2(dy, dx)
            yaw_diff = abs(target_yaw - pyaw)
            yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi # 正規化
            cost += self.config.w_heading * abs(yaw_diff)
            
            target_v = self.config.max_speed if dist > 1.0 else 0.0
            cost += self.config.w_vel * abs(target_v - v)
            
            # 障害物回避コスト (障害物がある場合のみ計算)
            if len(self.obstacles) > 0:
                # すべての障害物との距離を計算するのは重いので、
                # ここでは単純化のために直近のデータを使うか、または
                # scipy.spatial.KDTreeなどを使うのが望ましいが、元のロジックを踏襲する
                
                # 計算量削減のため、予測位置周辺の障害物だけを考慮するのが一般的だが
                # ここではNumPyのブロードキャストで一括計算
                dx_obs = self.obstacles[:, 0] - px
                dy_obs = self.obstacles[:, 1] - py
                obs_dists = np.hypot(dx_obs, dy_obs)
                
                min_obs_dist = np.min(obs_dists)
                
                if min_obs_dist < self.config.robot_radius:
                    cost += self.config.w_obs * (1.0 / (min_obs_dist + 0.01)) * 100
                elif min_obs_dist < self.config.robot_radius + 0.5:
                    cost += self.config.w_obs * (1.0 / min_obs_dist)

        return cost

    def run_mpc(self):
        if self.target is None:
            return 0.0, 0.0, []

        dx = self.target[0] - self.state[0]
        dy = self.target[1] - self.state[1]
        if math.hypot(dx, dy) < self.config.goal_tolerance:
            self.get_logger().info("Goal Reached")
            return 0.0, 0.0, []

        bounds = []
        for _ in range(self.config.horizon):
            bounds.append((self.config.min_speed, self.config.max_speed))
            bounds.append((-self.config.max_yaw_rate, self.config.max_yaw_rate))

        x0 = np.roll(self.prev_solution, -2)
        x0[-2:] = 0.0

        res = minimize(self.cost_function, x0, method='SLSQP', bounds=bounds, tol=1e-3)
        
        if res.success:
            self.prev_solution = res.x
            v_cmd = res.x[0]
            w_cmd = res.x[1]
            
            predict_path = []
            curr_state = self.state.copy()
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
        path_msg.header.frame_id = "odom"
        for point in trajectory:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        
    def publish_obstacles(self):
        if len(self.obstacles) == 0: return
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacles"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05 # サイズ調整
        marker.scale.y = 0.05
        marker.color.r = 1.0; marker.color.a = 1.0
        
        # 表示する障害物点群を間引く（描画負荷軽減）
        display_step = 5
        for i, obs in enumerate(self.obstacles):
            if i % display_step == 0:
                p = Point(); p.x = obs[0]; p.y = obs[1]
                marker.points.append(p)
        self.obs_pub.publish(marker)
        
    def publish_robot_radius(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "safety_margin"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = self.state[0]
        marker.pose.position.y = self.state[1]
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

    def control_loop(self):
        v, w, path = self.run_mpc()
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_vel_pub.publish(msg)
        self.publish_path(path)
        self.publish_obstacles()
        self.publish_robot_radius()

def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
