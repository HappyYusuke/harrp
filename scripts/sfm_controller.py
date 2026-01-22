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

class SFMConfig:
    def __init__(self):
        # --- 基本速度設定 ---
        self.max_speed = 0.6
        self.max_yaw_rate = 1.0    # 旋回も少しマイルドに
        self.goal_tolerance = 0.5
        
        # --- SFM(力)の重みパラメータ ---
        self.attraction_gain = 1.0
        
        # --- 【変更】 斥力パラメータ (指数関数用) ---
        self.repulsion_gain = 0.5      # 力の強さ (A)
        self.repulsion_decay = 0.5     # 力の減衰率 (B): 大きいほど遠くまで影響する
        self.tangential_gain = 0.2     # 【新規】障害物を回る力の強さ (接線力)
        
        self.repulsion_radius = 2.0    # 計算対象にする範囲
        
        self.turn_kp = 1.0
        self.sensor_offset = 0.0

class SFMController(Node):
    def __init__(self):
        super().__init__('dwa_controller') # ノード名は互換性のため維持
        self.config = SFMConfig()
        
        # パラメータ宣言
        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_speed', self.config.max_speed),
                ('attraction_gain', self.config.attraction_gain),
                ('repulsion_gain', self.config.repulsion_gain),
                ('repulsion_radius', self.config.repulsion_radius),
            ]
        )
        self.update_config_from_params()
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Publishers / Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.vector_pub = self.create_publisher(Marker, 'sfm_force_vector', 10) # 力の向きを可視化
        self.obs_pub = self.create_publisher(Marker, 'sfm_obstacles', 10)
        
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, 'target_pose', self.target_callback, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # x, y, yaw, v, w
        self.target = None
        self.obstacles = []
        self.is_reached = False
        
        self.timer = self.create_timer(0.1, self.control_loop)

    def update_config_from_params(self):
        self.config.max_speed = self.get_parameter('max_speed').value
        self.config.attraction_gain = self.get_parameter('attraction_gain').value
        self.config.repulsion_gain = self.get_parameter('repulsion_gain').value
        self.config.repulsion_radius = self.get_parameter('repulsion_radius').value

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
        # 以前修正した「反転対応」込みの障害物検知
        obs = []
        angle = msg.angle_min
        ignore_radius = 0.1
        for r in msg.ranges:
            if not math.isinf(r) and not math.isnan(r):
                if ignore_radius < r < msg.range_max:
                    # 左右反転 (-angle)
                    fixed_angle = -angle 
                    ox = r * math.cos(fixed_angle + self.state[2]) + self.state[0]
                    oy = r * math.sin(fixed_angle + self.state[2]) + self.state[1]
                    obs.append([ox, oy])
            angle += msg.angle_increment
        self.obstacles = np.array(obs)

    # --- SFMの核となる計算 ---
    def calc_sfm_action(self):
        if self.target is None:
            return 0.0, 0.0, [0.0, 0.0]

        rx, ry, ryaw = self.state[0], self.state[1], self.state[2]
        tx, ty = self.target[0], self.target[1]

        # 1. 引力 (Attraction)
        dx = tx - rx
        dy = ty - ry
        dist_to_goal = math.hypot(dx, dy)
        
        if dist_to_goal < self.config.goal_tolerance:
            return 0.0, 0.0, [0.0, 0.0]

        att_x = (dx / dist_to_goal) * self.config.attraction_gain
        att_y = (dy / dist_to_goal) * self.config.attraction_gain

        # 2. 斥力 (Repulsion) & 接線力 (Tangential)
        rep_x, rep_y = 0.0, 0.0
        tan_x, tan_y = 0.0, 0.0
        
        if len(self.obstacles) > 0:
            for obs in self.obstacles:
                ox, oy = obs[0], obs[1]
                dist_obs = math.hypot(rx - ox, ry - oy)
                
                # 安全マージン（ロボットの半径分）を考慮して実質距離を計算
                # 0.3mくらい近づいたら最大反発にする
                effective_dist = max(dist_obs - 0.3, 0.0) 

                if dist_obs < self.config.repulsion_radius:
                    # --- 【改良 1】 指数関数ポテンシャル ---
                    # 距離が縮まるにつれてなめらかに強くなる
                    # Force = A * exp(-dist / B)
                    potential = self.config.repulsion_gain * math.exp(-effective_dist / self.config.repulsion_decay)
                    
                    # 障害物からロボットへの向き単位ベクトル (Normal Vector)
                    nx = (rx - ox) / dist_obs
                    ny = (ry - oy) / dist_obs
                    
                    # 斥力を加算
                    rep_x += nx * potential
                    rep_y += ny * potential
                    
                    # --- 【改良 2】 接線力 (Tangential Force) ---
                    # 障害物の周りを「回る」力。
                    # 引力(ゴール方向)との内積を見て、左右どちらに避けるべきか決める
                    
                    # ゴール方向と障害物方向の外積(z成分)
                    cross_product = (dx * ny - dy * nx)
                    
                    # ゴールが障害物の「左」にあれば左回り、「右」なら右回りの接線ベクトルを作る
                    # (-ny, nx) は左回転, (ny, -nx) は右回転
                    if cross_product > 0:
                        tx_vec, ty_vec = -ny, nx
                    else:
                        tx_vec, ty_vec = ny, -nx
                        
                    tan_x += tx_vec * potential * self.config.tangential_gain
                    tan_y += ty_vec * potential * self.config.tangential_gain

        # 3. 合力 (Resultant Force)
        # 引力 + 斥力 + 接線力
        total_fx = att_x + rep_x + tan_x
        total_fy = att_y + rep_y + tan_y
        
        # 4. 力のベクトルをロボットの操作量(v, w)に変換
        desired_yaw = math.atan2(total_fy, total_fx)
        
        yaw_error = desired_yaw - ryaw
        while yaw_error > math.pi: yaw_error -= 2 * math.pi
        while yaw_error < -math.pi: yaw_error += 2 * math.pi

        cmd_w = self.config.turn_kp * yaw_error
        cmd_w = max(min(cmd_w, self.config.max_yaw_rate), -self.config.max_yaw_rate)

        speed_factor = max(0.0, math.cos(yaw_error))
        force_magnitude = math.hypot(total_fx, total_fy)
        
        # 力が強すぎるとき（壁際など）は逆に慎重に進むため、斥力が強いときは速度を抑える工夫
        # 力の大きさそのものを速度にするのではなく、MAX速度にspeed_factorを掛ける形が安定します
        target_speed = min(force_magnitude, self.config.max_speed)
        
        cmd_v = target_speed * speed_factor

        return cmd_v, cmd_w, [total_fx, total_fy]
    def publish_force_vector(self, vector):
        # 合力を矢印で表示 (RVizデバッグ用)
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "force_vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # 始点 (ロボット位置)
        p1 = Point()
        p1.x = self.state[0]
        p1.y = self.state[1]
        
        # 終点 (ロボット位置 + ベクトル)
        p2 = Point()
        # ベクトルが大きすぎると見にくいのでスケーリングしても良い
        scale = 0.5
        p2.x = self.state[0] + vector[0] * scale
        p2.y = self.state[1] + vector[1] * scale
        
        marker.points.append(p1)
        marker.points.append(p2)
        
        marker.scale.x = 0.1 # 軸の太さ
        marker.scale.y = 0.2 # 頭の太さ
        marker.scale.z = 0.2
        
        marker.color.r = 0.0
        marker.color.g = 1.0 # 緑色の矢印
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.vector_pub.publish(marker)

    def publish_obstacles(self):
        # 障害物を赤い点で表示
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
        v, w, force_vector = self.calc_sfm_action()
        
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_vel_pub.publish(msg)
        
        self.publish_force_vector(force_vector)
        self.publish_obstacles()

def main(args=None):
    rclpy.init(args=args)
    node = SFMController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
