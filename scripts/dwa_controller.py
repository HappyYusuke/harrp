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

class DWAConfig:
    def __init__(self):
        self.max_speed = 0.5
        self.min_speed = 0.0
        self.max_yaw_rate = 1.0
        self.max_accel = 0.5
        self.max_dyaw_rate = 2.0
        self.v_reso = 0.02
        self.yaw_reso = 0.05
        self.dt = 0.1
        self.predict_time = 3.0
        
        self.heading_score_gain = 0.1
        self.dist_score_gain = 2.0
        self.velocity_score_gain = 1.0
        
        self.robot_radius = 0.25
        self.goal_tolerance = 0.8
        
        # --- 変更: PD制御用ゲイン ---
        self.turn_p_gain = 1.0  # Pゲイン (少し下げる: 1.5 -> 1.0)
        self.turn_d_gain = 0.5  # Dゲイン (新規追加)
        # ------------------------

class DWAController(Node):
    def __init__(self):
        super().__init__('dwa_controller')
        self.config = DWAConfig()
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_speed', self.config.max_speed),
                ('min_speed', self.config.min_speed),
                ('max_yaw_rate', self.config.max_yaw_rate),
                ('max_accel', self.config.max_accel),
                ('max_dyaw_rate', self.config.max_dyaw_rate),
                ('v_reso', self.config.v_reso),
                ('yaw_reso', self.config.yaw_reso),
                ('dt', self.config.dt),
                ('predict_time', self.config.predict_time),
                ('heading_score_gain', self.config.heading_score_gain),
                ('dist_score_gain', self.config.dist_score_gain),
                ('velocity_score_gain', self.config.velocity_score_gain),
                ('robot_radius', self.config.robot_radius),
                ('goal_tolerance', self.config.goal_tolerance),
                # --- 変更 ---
                ('turn_p_gain', self.config.turn_p_gain),
                ('turn_d_gain', self.config.turn_d_gain),
            ]
        )
        self.update_config_from_params()
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'dwa_path', 10)
        self.candidate_pub = self.create_publisher(MarkerArray, 'dwa_candidate_paths', 10)
        # --- 追加: 障害物可視化用 ---
        self.obs_pub = self.create_publisher(Marker, 'dwa_obstacles', 10)
        # ------------------------
        
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, 'target_pose', self.target_callback, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 
        self.target = None 
        self.obstacles = []
        self.is_reached = False
        
        # --- 追加: PD制御用の前回誤差 ---
        self.prev_yaw_error = 0.0
        # -----------------------------
        
        # --- 追加: 振り向きモード管理用フラグ ---
        self.is_turning_mode = False 
        # ------------------------------------
        
        self.timer = self.create_timer(0.1, self.control_loop)

    def update_config_from_params(self):
        self.config.max_speed = self.get_parameter('max_speed').value
        self.config.min_speed = self.get_parameter('min_speed').value
        self.config.max_yaw_rate = self.get_parameter('max_yaw_rate').value
        self.config.max_accel = self.get_parameter('max_accel').value
        self.config.max_dyaw_rate = self.get_parameter('max_dyaw_rate').value
        self.config.v_reso = self.get_parameter('v_reso').value
        self.config.yaw_reso = self.get_parameter('yaw_reso').value
        self.config.dt = self.get_parameter('dt').value
        self.config.predict_time = self.get_parameter('predict_time').value
        self.config.heading_score_gain = self.get_parameter('heading_score_gain').value
        self.config.dist_score_gain = self.get_parameter('dist_score_gain').value
        self.config.velocity_score_gain = self.get_parameter('velocity_score_gain').value
        self.config.robot_radius = self.get_parameter('robot_radius').value
        self.config.goal_tolerance = self.get_parameter('goal_tolerance').value
        # --- 変更 ---
        self.config.turn_p_gain = self.get_parameter('turn_p_gain').value
        self.config.turn_d_gain = self.get_parameter('turn_d_gain').value

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
        self.state = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw, msg.twist.twist.linear.x, msg.twist.twist.angular.z])

    def target_callback(self, msg):
        self.target = np.array([msg.pose.position.x, msg.pose.position.y])

    def scan_callback(self, msg):
        obs = []
        angle = msg.angle_min
        ignore_radius = 0.1 
        for r in msg.ranges:
            if not math.isinf(r) and not math.isnan(r):
                if ignore_radius < r < msg.range_max:
                    # --- 変更点: angle を -angle にして左右反転 ---
                    # シミュレータ上でLidarの回転方向が逆、または上下逆の場合の補正です
                    fixed_angle = -angle 
                    
                    # グローバル座標系への変換
                    # ロボットの向き(state[2]) + Lidarの角度(fixed_angle)
                    ox = r * math.cos(fixed_angle + self.state[2]) + self.state[0]
                    oy = r * math.sin(fixed_angle + self.state[2]) + self.state[1]
                    obs.append([ox, oy])
            angle += msg.angle_increment
        self.obstacles = np.array(obs)

    def motion(self, x, u, dt):
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    def calc_dynamic_window(self, x):
        current_v = max(0.0, x[3])
        Vs = [self.config.min_speed, self.config.max_speed,
              -self.config.max_yaw_rate, self.config.max_yaw_rate]
        Vd = [current_v - self.config.max_accel * self.config.dt,
              current_v + self.config.max_accel * self.config.dt,
              x[4] - self.config.max_dyaw_rate * self.config.dt,
              x[4] + self.config.max_dyaw_rate * self.config.dt]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw

    def predict_trajectory(self, x_init, v, w, predict_time):
        x = np.array(x_init)
        traj = np.array(x)
        time = 0
        while time <= predict_time:
            x = self.motion(x, [v, w], self.config.dt)
            traj = np.vstack((traj, x))
            time += self.config.dt
        return traj

    def calc_heading_score(self, trajectory):
        dx = self.target[0] - trajectory[-1, 0]
        dy = self.target[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        diff = error_angle - trajectory[-1, 2]
        return math.pi - abs(math.atan2(math.sin(diff), math.cos(diff)))

    def calc_dist_score(self, trajectory):
        if len(self.obstacles) == 0: return 2.0
        ox = self.obstacles[:, 0]
        oy = self.obstacles[:, 1]
        min_r = float("inf")
        for i in range(0, len(trajectory), 2):
            dx = trajectory[i, 0] - ox
            dy = trajectory[i, 1] - oy
            r = np.hypot(dx, dy)
            min_r = min(min_r, np.min(r))
        if min_r <= self.config.robot_radius:
            return -float("inf")
        return min_r

    def dwa_control(self):
        if self.target is None:
            return 0.0, 0.0, None, []

        dx = self.target[0] - self.state[0]
        dy = self.target[1] - self.state[1]
        dist_to_goal = math.hypot(dx, dy)
        
        # --- 追加: ターゲットとの角度差を計算 ---
        target_yaw = math.atan2(dy, dx)
        yaw_diff = target_yaw - self.state[2]
        while yaw_diff > math.pi: yaw_diff -= 2 * math.pi
        while yaw_diff < -math.pi: yaw_diff += 2 * math.pi
        
        # --- 変更: ヒステリシス付きの振り向きモード判定 ---
        
        # 閾値の設定 (ラジアン変換)
        START_TURN_THRESHOLD = math.radians(60) # 誤差60度で開始
        END_TURN_THRESHOLD   = math.radians(10) # 誤差10度になるまで続ける
        
        if self.is_turning_mode:
            # 現在振り向きモード中なら、十分正面を向く(10度以内)まで解除しない
            if abs(yaw_diff) < END_TURN_THRESHOLD:
                self.is_turning_mode = False
        else:
            # 通常モード中なら、誤差が大きくなった時(60度以上)に振り向きモードへ
            if abs(yaw_diff) > START_TURN_THRESHOLD:
                self.is_turning_mode = True

        # 決定したモードに基づいてゲインを設定
        if self.is_turning_mode:
            current_heading_gain = 2.0  # 強力に旋回
            current_velocity_gain = 0.0 # 前進禁止
        else:
            current_heading_gain = self.config.heading_score_gain
            current_velocity_gain = self.config.velocity_score_gain
        # ----------------------------------------------------
                
        # 停止判定ロジック (変更なし)
        if self.is_reached:
            # ... (中略) ...
            pass # 省略
        else:
            self.prev_yaw_error = 0.0
            if dist_to_goal < self.config.goal_tolerance:
                self.is_reached = True
                return 0.0, 0.0, None, []

        # 予測時間の短縮 (変更なし)
        time_to_goal = dist_to_goal / self.config.max_speed
        active_predict_time = min(self.config.predict_time, time_to_goal * 0.6)
        active_predict_time = max(active_predict_time, 0.5)

        dw = self.calc_dynamic_window(self.state)
        eval_db = [] 
        
        for v in np.arange(dw[0], dw[1], self.config.v_reso):
            for w in np.arange(dw[2], dw[3], self.config.yaw_reso):
                trajectory = self.predict_trajectory(self.state, v, w, active_predict_time)
                
                heading_score = self.calc_heading_score(trajectory)
                dist_score = self.calc_dist_score(trajectory)
                vel_score = float(v)
                
                if dist_score == -float("inf"):
                    continue

                eval_db.append([v, w, heading_score, dist_score, vel_score, trajectory])

        if not eval_db:
            return 0.0, 0.0, None, []

        # 正規化の準備 (変更なし)
        max_heading = max([e[2] for e in eval_db]) + 1e-6
        max_dist    = max([e[3] for e in eval_db]) + 1e-6
        max_vel     = max([e[4] for e in eval_db]) + 1e-6
        
        best_u = [0.0, 0.0]
        best_traj = None
        max_total_score = -float("inf")
        all_candidate_trajectories = []
        
        for e in eval_db:
            norm_heading = e[2] / max_heading
            norm_dist    = e[3] / max_dist
            norm_vel     = e[4] / max_vel
            
            # --- 変更: ここで動的に決定したゲインを使う ---
            s_head = current_heading_gain * norm_heading
            s_dist = self.config.dist_score_gain * norm_dist
            s_vel  = current_velocity_gain * norm_vel
            # ----------------------------------------
            
            total_score = s_head + s_dist + s_vel
            
            all_candidate_trajectories.append(e[5])

            if total_score > max_total_score:
                max_total_score = total_score
                best_u = [e[0], e[1]]
                best_traj = e[5]
                
                # ベストパスのスコア詳細を保存
                debug_scores = {
                    "v": e[0], "w": e[1],
                    "raw_head": e[2], "norm_head": norm_heading, "final_head": s_head,
                    "raw_dist": e[3], "norm_dist": norm_dist,    "final_dist": s_dist,
                    "raw_vel":  e[4], "norm_vel":  norm_vel,     "final_vel":  s_vel,
                    "total": total_score
                }

        # --- 追加: 評価関数の詳細ログ出力 ---
        if debug_scores:
            self.get_logger().info(
                f"\n--- Best Path Evaluation ---"
                f"\n  CMD      : v={debug_scores['v']:.2f}, w={debug_scores['w']:.2f}"
                f"\n  Heading  : raw={debug_scores['raw_head']:.2f} -> norm={debug_scores['norm_head']:.3f} -> weighted={debug_scores['final_head']:.3f}"
                f"\n  Dist     : raw={debug_scores['raw_dist']:.2f} -> norm={debug_scores['norm_dist']:.3f} -> weighted={debug_scores['final_dist']:.3f}"
                f"\n  Velocity : raw={debug_scores['raw_vel']:.2f} -> norm={debug_scores['norm_vel']:.3f} -> weighted={debug_scores['final_vel']:.3f}"
                f"\n  TOTAL    : {debug_scores['total']:.4f}"
            )
        # ----------------------------------
        
        return best_u[0], best_u[1], best_traj, all_candidate_trajectories
        
    def publish_candidate_paths(self, trajectories):
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        step = 1
        for i, traj in enumerate(trajectories[::step]):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "candidates"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.01
            marker.color.r = 0.7
            marker.color.g = 0.7
            marker.color.b = 0.7
            marker.color.a = 0.5 
            for point in traj:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = 0.0
                marker.points.append(p)
            marker_array.markers.append(marker)
        self.candidate_pub.publish(marker_array)
        
    def publish_obstacles(self):
        if len(self.obstacles) == 0:
            return
            
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacles"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # 点の大きさ
        marker.scale.y = 0.1
        
        marker.color.r = 1.0  # 赤色
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for obs in self.obstacles:
            p = Point()
            p.x = obs[0]
            p.y = obs[1]
            p.z = 0.0
            marker.points.append(p)
            
        self.obs_pub.publish(marker)

    def publish_path(self, trajectory):
        if trajectory is None:
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = "odom"
            self.path_pub.publish(path_msg)
            return
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "odom"
        for point in trajectory:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def control_loop(self):
        v, w, best_traj, candidates = self.dwa_control()
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_vel_pub.publish(msg)
        self.publish_path(best_traj)
        if candidates:
            self.publish_candidate_paths(candidates)
            
        self.publish_obstacles()

def main(args=None):
    rclpy.init(args=args)
    node = DWAController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
