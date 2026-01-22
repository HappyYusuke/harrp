#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from scipy.optimize import minimize # 最適化計算用

class MPCConfig:
    def __init__(self):
        # --- 制約条件 ---
        self.max_speed = 0.8
        self.min_speed = 0.0
        self.max_yaw_rate = 1.2
        self.max_accel = 0.5
        self.max_dyaw_rate = 2.0
        
        # --- MPCパラメータ ---
        self.dt = 0.2          # 予測の刻み時間 (s)
        self.horizon = 8       # 何ステップ先まで読むか (8 * 0.2 = 1.6秒先まで予測)
        
        # --- 重み付け (Cost Function) ---
        # ゴールへ近づくことの重要性
        self.w_dist = 1.0
        # ゴールを向くことの重要性
        self.w_heading = 0.5
        # 速度を出すことの重要性
        self.w_vel = 1.5
        # 障害物を避けることの重要性 (非常に大きくする)
        self.w_obs = 0.8
        
        # 安全マージン
        self.robot_radius = 0.3
        self.goal_tolerance = 0.5
        
        # Lidar位置補正
        self.sensor_offset = 0.0

class MPCController(Node):
    def __init__(self):
        super().__init__('dwa_controller') # ノード名は互換性維持
        self.config = MPCConfig()
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_speed', self.config.max_speed),
                ('horizon', self.config.horizon),
            ]
        )
        self.update_config_from_params()
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'mpc_predict_path', 10) # 予測軌跡
        self.obs_pub = self.create_publisher(Marker, 'mpc_obstacles', 10)
        
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, 'target_pose', self.target_callback, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # x, y, yaw, v, w
        self.target = None
        self.obstacles = []
        self.is_reached = False
        
        # 前回の最適化結果を保存（次の初期値にするため＝ウォームスタート）
        # [v0, w0, v1, w1, ...] という1次元配列
        self.prev_solution = np.zeros(self.config.horizon * 2)
        
        self.timer = self.create_timer(0.1, self.control_loop)

    def update_config_from_params(self):
        self.config.max_speed = self.get_parameter('max_speed').value
        self.config.horizon = self.get_parameter('horizon').value

    def parameter_callback(self, params):
        for param in params:
            if hasattr(self.config, param.name):
                setattr(self.config, param.name, param.value)
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

    def scan_callback(self, msg):
        obs = []
        # angle変数の初期化は不要になります
        ignore_radius = 0.1
        
        # 計算負荷軽減のための間引き設定
        # PCスペックに余裕があれば 2 や 1 にすると精度が上がります
        step = 3 
        
        for i, r in enumerate(msg.ranges):
            # 間引き処理
            if i % step != 0: continue
            
            if not math.isinf(r) and not math.isnan(r):
                if ignore_radius < r < msg.range_max:
                    # 【修正箇所】インデックスから正確な角度を計算
                    current_angle = msg.angle_min + i * msg.angle_increment
                    
                    # 反転補正
                    fixed_angle = -current_angle 
                    
                    # 座標変換
                    ox = r * math.cos(fixed_angle + self.state[2]) + self.state[0]
                    oy = r * math.sin(fixed_angle + self.state[2]) + self.state[1]
                    obs.append([ox, oy])
                    
        self.obstacles = np.array(obs)

    # --- 予測モデル (Motion Model) ---
    def predict_next_state(self, state, v, w, dt):
        x, y, yaw, _, _ = state
        # 差動2輪モデル
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += w * dt
        
        # 角度正規化
        while yaw > math.pi: yaw -= 2*math.pi
        while yaw < -math.pi: yaw += 2*math.pi
            
        return np.array([x, y, yaw, v, w])

    # --- コスト関数 (ここがMPCの脳みそ) ---
    def cost_function(self, u_flat):
        # u_flat: 最適化変数が1列に並んだもの [v0, w0, v1, w1, ...]
        cost = 0.0
        curr_state = self.state.copy()
        
        # 配列を (Horizon, 2) に変形 -> [[v0, w0], [v1, w1], ...]
        u = u_flat.reshape(self.config.horizon, 2)
        
        for i in range(self.config.horizon):
            v, w = u[i]
            
            # 次の状態を予測
            curr_state = self.predict_next_state(curr_state, v, w, self.config.dt)
            px, py, pyaw, _, _ = curr_state
            
            # 1. ゴールとの距離コスト
            dx = self.target[0] - px
            dy = self.target[1] - py
            dist = math.hypot(dx, dy)
            cost += self.config.w_dist * dist
            
            # 2. 向きのコスト (ゴールを向いているか)
            target_yaw = math.atan2(dy, dx)
            yaw_diff = abs(target_yaw - pyaw)
            while yaw_diff > math.pi: yaw_diff -= 2*math.pi
            cost += self.config.w_heading * abs(yaw_diff)
            
            # 3. 速度コスト (速いほうがいい)
            # ただしゴール付近では速度を落とすために、distに比例させると良い
            target_v = self.config.max_speed if dist > 1.0 else 0.0
            cost += self.config.w_vel * abs(target_v - v)
            
            # 4. 障害物コスト (最重要)
            if len(self.obstacles) > 0:
                # ロボット位置と全障害物の距離を計算
                # (高速化のため本来は近傍探索すべきだが、ここでは全探索)
                obs_dists = np.hypot(self.obstacles[:,0] - px, self.obstacles[:,1] - py)
                min_obs_dist = np.min(obs_dists)
                
                # 安全圏内に入ったらペナルティを爆増させる (ソフト制約)
                if min_obs_dist < self.config.robot_radius:
                    cost += self.config.w_obs * (1.0 / (min_obs_dist + 0.01)) * 100
                elif min_obs_dist < self.config.robot_radius + 0.5:
                    cost += self.config.w_obs * (1.0 / min_obs_dist)

        return cost

    def run_mpc(self):
        if self.target is None:
            return 0.0, 0.0, []

        # ゴール判定
        dx = self.target[0] - self.state[0]
        dy = self.target[1] - self.state[1]
        if math.hypot(dx, dy) < self.config.goal_tolerance:
            return 0.0, 0.0, []

        # --- 最適化の準備 ---
        # 変数の範囲 (Bounds)
        bounds = []
        for _ in range(self.config.horizon):
            bounds.append((self.config.min_speed, self.config.max_speed)) # v
            bounds.append((-self.config.max_yaw_rate, self.config.max_yaw_rate)) # w

        # 初期推定値 (前回の解をずらして使うと収束が早い)
        # [v0, w0, v1, w1, ...] -> [v1, w1, ..., 0, 0]
        x0 = np.roll(self.prev_solution, -2)
        x0[-2:] = 0.0 # 末尾はゼロ埋め

        # --- 最適化実行 (SLSQP法) ---
        # ここが重い処理。Pythonなので時間がかかるが、Horizonが短ければなんとかなる
        res = minimize(self.cost_function, x0, method='SLSQP', bounds=bounds, tol=1e-3)
        
        if res.success:
            self.prev_solution = res.x
            # 最初の1手目だけを採用して実行する
            v_cmd = res.x[0]
            w_cmd = res.x[1]
            
            # 予測軌跡の可視化用に座標を計算しなおす
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
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.r = 1.0; marker.color.a = 1.0
        for obs in self.obstacles:
            p = Point(); p.x = obs[0]; p.y = obs[1]
            marker.points.append(p)
        self.obs_pub.publish(marker)

    def control_loop(self):
        v, w, path = self.run_mpc()
        
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_vel_pub.publish(msg)
        
        self.publish_path(path)
        self.publish_obstacles()

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
