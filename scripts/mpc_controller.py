#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped, Point, TransformStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs

import numpy as np
import math
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from numba import jit, prange

# ==========================================
# JIT Functions (Fastmath + nogil + Parallel)
# ==========================================

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def predict_sequence_jit(state, v_seq, w_seq, dt):
    horizon = len(v_seq)
    traj = np.zeros((horizon, 5), dtype=np.float64)
    curr_x, curr_y, curr_yaw = state[0], state[1], state[2]
    
    for i in range(horizon):
        v = v_seq[i]; w = w_seq[i]
        curr_x += v * np.cos(curr_yaw) * dt
        curr_y += v * np.sin(curr_yaw) * dt
        curr_yaw += w * dt
        if curr_yaw > np.pi: curr_yaw -= 2 * np.pi
        elif curr_yaw < -np.pi: curr_yaw += 2 * np.pi
        traj[i, 0] = curr_x; traj[i, 1] = curr_y; traj[i, 2] = curr_yaw
        traj[i, 3] = v; traj[i, 4] = w
    return traj

@jit(nopython=True, cache=True, fastmath=True, nogil=True, parallel=True)
def evaluate_samples_jit(samples_v, samples_w, current_state, target, obstacles, 
                         dt, w_dist, w_heading, w_vel, w_obs, w_smooth,
                         max_speed, robot_radius):
    num_samples = samples_v.shape[0]
    horizon = samples_v.shape[1]
    costs = np.zeros(num_samples, dtype=np.float64)
    
    robot_radius_sq = robot_radius ** 2
    collision_dist_sq = (robot_radius + 0.15) ** 2 
    num_obs = len(obstacles)
    
    # 現在の速度・角速度 (平滑化用)
    curr_v = current_state[3]
    curr_w = current_state[4]

    for k in prange(num_samples):
        cost = 0.0
        cx, cy, cyaw = current_state[0], current_state[1], current_state[2]
        last_v = curr_v
        last_w = curr_w
        collision = False
        
        for t in range(horizon):
            if collision:
                cost += 10000.0
                continue

            v = samples_v[k, t]
            w = samples_w[k, t]
            
            # --- モデル更新 ---
            cx += v * np.cos(cyaw) * dt
            cy += v * np.sin(cyaw) * dt
            cyaw += w * dt
            
            # --- コスト計算 ---
            dx = target[0] - cx
            dy = target[1] - cy
            dist_sq = dx*dx + dy*dy
            
            # 1. 距離コスト (ゴールに近づくことを推奨)
            cost += w_dist * np.sqrt(dist_sq)
            
            # 2. 向きコスト
            target_yaw = np.arctan2(dy, dx)
            yaw_diff = np.abs(target_yaw - cyaw)
            if yaw_diff > np.pi: yaw_diff = 2*np.pi - yaw_diff
            cost += w_heading * yaw_diff

            # 3. 速度コスト (なるべく最高速度で)
            # ゴールまで0.5m以上なら最高速度、近ければ減速
            target_v = max_speed if dist_sq > 0.25 else 0.0
            cost += w_vel * np.abs(target_v - v)
            
            # 4. 平滑化コスト (急激な変化を抑制 ★重要)
            # これを入れると左右の振動が減る
            cost += w_smooth * (np.abs(v - last_v) + np.abs(w - last_w))
            last_v = v
            last_w = w

            # 5. 障害物コスト
            if num_obs > 0:
                min_d_sq = 1000.0
                for o in range(num_obs):
                    odx = obstacles[o, 0] - cx
                    ody = obstacles[o, 1] - cy
                    d_sq = odx*odx + ody*ody
                    if d_sq < min_d_sq: min_d_sq = d_sq
                    if d_sq < robot_radius_sq:
                        collision = True
                        break
                
                if collision:
                    cost += w_obs * 10000.0
                elif min_d_sq < collision_dist_sq:
                    d_val = np.sqrt(min_d_sq)
                    cost += w_obs * (1.0 / d_val) * 10.0
        
        costs[k] = cost

    return np.argmin(costs)

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def process_lidar_jit(xyz, sensor_offset, robot_radius, min_height, max_height):
    num_in = xyz.shape[0]
    temp_obs = np.zeros((num_in, 2), dtype=np.float64)
    count = 0
    robot_radius_sq = robot_radius ** 2
    max_dist_limit_sq = 3.0 ** 2 
    
    for i in range(num_in):
        x = xyz[i, 0]; y = xyz[i, 1]; z = xyz[i, 2]
        if np.isnan(x) or np.isnan(y) or np.isnan(z): continue
        
        # 高さフィルタ
        if z < min_height or z > max_height: continue
            
        bx = x + sensor_offset; by = y
        dist_sq = bx*bx + by*by
        
        if dist_sq > robot_radius_sq and dist_sq < max_dist_limit_sq:
            temp_obs[count, 0] = bx; temp_obs[count, 1] = by
            count += 1
            
    return temp_obs[:count]

# ==========================================

class MPCConfig:
    def __init__(self):
        self.max_speed = 0.8
        self.min_speed = 0.0
        self.max_yaw_rate = 1.0
        self.max_accel = 1.0
        self.dt = 0.1
        self.horizon = 15      
        self.num_samples = 150 
        
        # --- パラメータ調整 ---
        self.w_dist = 0.8      # ゴールへ向かう力を強化 (0.5 -> 0.8)
        self.w_heading = 0.3   # 向きの重み
        self.w_vel = 0.2       # 速度維持
        self.w_obs = 0.8       # 障害物回避
        self.w_smooth = 0.3    # ★追加: 動きの滑らかさ (振動抑制)
        
        self.robot_radius = 0.23
        self.goal_tolerance = 0.6 
        self.sensor_offset = 0.156
        self.turn_kp = 0.5
        self.turn_kd = 0.2
        self.turn_yaw_tolerance = 0.1
        
        # ★修正: 床誤検知対策
        self.min_height = 0.1  # 0.05 -> 0.1 (10cm以下の点は無視)
        self.max_height = 0.1
        self.lost_timeout = 1.0

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')
        self.config = MPCConfig()
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameters(namespace='', parameters=[
            ('max_speed', self.config.max_speed), ('min_speed', self.config.min_speed),
            ('max_yaw_rate', self.config.max_yaw_rate), ('dt', self.config.dt),
            ('horizon', self.config.horizon), ('w_dist', self.config.w_dist),
            ('w_heading', self.config.w_heading), ('w_vel', self.config.w_vel),
            ('w_obs', self.config.w_obs), ('w_smooth', self.config.w_smooth),
            ('robot_radius', self.config.robot_radius),
            ('sensor_offset', self.config.sensor_offset), ('lost_timeout', self.config.lost_timeout),
        ])
        self.update_config_from_params()

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'mpc/predict_path', 10)
        self.obs_pub = self.create_publisher(Marker, 'mpc/obstacles', 10)
        self.target_marker_pub = self.create_publisher(Marker, 'mpc/target_debug', 10)
        self.robot_radius_pub = self.create_publisher(Marker, 'mpc/robot_radius', 10)
        
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(Odometry, '/kachaka/odometry/odometry', self.odom_callback, qos_profile, callback_group=self.cb_group)
        self.create_subscription(PoseStamped, 'tracker/target_pose', self.target_callback, 10, callback_group=self.cb_group)
        self.create_subscription(PointCloud2, '/livox/lidar', self.lidar_callback, 10, callback_group=self.cb_group)
        self.create_subscription(String, 'tracker/target_status', self.status_callback, 10, callback_group=self.cb_group)
        
        self.current_state = np.zeros(5) 
        self.target_local = None 
        self.target_pose_in_odom = None  
        self.last_target_time = None
        self.is_tracker_lost = False
        self.obstacles_local = np.zeros((0, 2), dtype=np.float64)
        self.sensor_frame_id = "livox_frame"
        
        self.prev_v_seq = np.zeros(self.config.horizon)
        self.prev_w_seq = np.zeros(self.config.horizon)
        self.prev_yaw_error = None
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.timer = self.create_timer(0.1, self.control_loop, callback_group=self.cb_group)

    def update_config_from_params(self):
        self.config.max_speed = self.get_parameter('max_speed').value
        self.config.max_yaw_rate = self.get_parameter('max_yaw_rate').value
        self.config.horizon = self.get_parameter('horizon').value
        # w_smoothも更新可能にする
        if self.has_parameter('w_smooth'):
            self.config.w_smooth = self.get_parameter('w_smooth').value

    def odom_callback(self, msg):
        self.current_state[3] = msg.twist.twist.linear.x
        self.current_state[4] = msg.twist.twist.angular.z
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'; t.child_frame_id = 'base_link'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def status_callback(self, msg):
        self.is_tracker_lost = "LOST" in msg.data

    def target_callback(self, msg):
        self.last_target_time = self.get_clock().now()
        self.is_tracker_lost = False
        try:
            target_time = rclpy.time.Time()
            msg_tmp = PoseStamped()
            msg_tmp.header = msg.header
            msg_tmp.header.stamp = target_time.to_msg()
            msg_tmp.pose = msg.pose
            
            target_in_base = self.tf_buffer.transform(msg_tmp, 'base_link', timeout=rclpy.duration.Duration(seconds=0.1))
            self.target_local = np.array([target_in_base.pose.position.x, target_in_base.pose.position.y])

            self.target_pose_in_odom = self.tf_buffer.transform(msg_tmp, 'odom', timeout=rclpy.duration.Duration(seconds=0.1))
        except (LookupException, ConnectivityException, ExtrapolationException): pass

    def lidar_callback(self, msg):
        self.sensor_frame_id = msg.header.frame_id
        try:
            point_step = msg.point_step
            raw_data = np.frombuffer(msg.data, dtype=np.uint8)
            num_points = len(raw_data) // point_step
            if num_points == 0: return

            float_view = np.frombuffer(msg.data, dtype=np.float32)
            xyz = np.empty((num_points, 3), dtype=np.float32)
            float_step = point_step // 4
            xyz[:, 0] = float_view[0::float_step][:num_points]
            xyz[:, 1] = float_view[1::float_step][:num_points]
            xyz[:, 2] = float_view[2::float_step][:num_points]
            
            self.obstacles_local = process_lidar_jit(
                xyz.astype(np.float64), 
                self.config.sensor_offset, self.config.robot_radius,
                self.config.min_height, self.config.max_height
            )
        except Exception as e:
            self.get_logger().debug(f"Lidar Error: {e}")

    def get_target_pos(self):
        if self.last_target_time is None: return None
        elapsed = (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9
        is_lost = self.is_tracker_lost or (elapsed > self.config.lost_timeout)
        if not is_lost: return self.target_local
        elif self.target_pose_in_odom is not None:
            try:
                tm = PoseStamped()
                tm.header.frame_id = 'odom'; tm.header.stamp = rclpy.time.Time().to_msg()
                tm.pose = self.target_pose_in_odom.pose
                tb = self.tf_buffer.transform(tm, 'base_link', timeout=rclpy.duration.Duration(seconds=0.1))
                return np.array([tb.pose.position.x, tb.pose.position.y])
            except: return None
        return None

    def run_mpc_sampling(self, target):
        H = self.config.horizon; N = self.config.num_samples
        self.prev_v_seq = np.roll(self.prev_v_seq, -1); self.prev_v_seq[-1] = 0
        self.prev_w_seq = np.roll(self.prev_w_seq, -1); self.prev_w_seq[-1] = 0
        
        # サンプリング時のノイズを少し減らして安定化
        samples_v = self.prev_v_seq + np.random.normal(0, 0.2, (N, H))
        samples_w = self.prev_w_seq + np.random.normal(0, 0.4, (N, H))
        
        samples_v = np.clip(samples_v, self.config.min_speed, self.config.max_speed)
        samples_w = np.clip(samples_w, -self.config.max_yaw_rate, self.config.max_yaw_rate)
        
        obs_base = self.obstacles_local
        sim_state = np.array([0.0, 0.0, 0.0, self.current_state[3], self.current_state[4]], dtype=np.float64)
        
        best_idx = evaluate_samples_jit(
            samples_v, samples_w, sim_state, target, obs_base,
            self.config.dt, self.config.w_dist, self.config.w_heading, self.config.w_vel, self.config.w_obs, self.config.w_smooth,
            self.config.max_speed, self.config.robot_radius
        )
        
        best_v_seq = samples_v[best_idx]
        best_w_seq = samples_w[best_idx]
        self.prev_v_seq = best_v_seq
        self.prev_w_seq = best_w_seq
        
        path_points = predict_sequence_jit(sim_state, best_v_seq, best_w_seq, self.config.dt)
        return best_v_seq[0], best_w_seq[0], path_points

    def control_loop(self):
        self.publish_obstacles()
        self.publish_target_marker()
        self.publish_robot_radius()

        target = self.get_target_pos()
        if target is None:
            self.cmd_vel_pub.publish(Twist())
            self.publish_path([])
            return
            
        dist = math.hypot(target[0], target[1])
        if dist < 0.5:
            yaw_diff = math.atan2(target[1], target[0])
            while yaw_diff > math.pi: yaw_diff -= 2 * math.pi
            while yaw_diff < -math.pi: yaw_diff += 2 * math.pi
            twist = Twist()
            if abs(yaw_diff) > self.config.turn_yaw_tolerance:
                twist.angular.z = float(np.clip(yaw_diff * 1.5, -0.5, 0.5))
            elif dist > 0.2:
                twist.linear.x = float(np.clip(dist * 0.5, 0.0, 0.3))
                twist.angular.z = float(np.clip(yaw_diff * 1.0, -0.3, 0.3))
            self.cmd_vel_pub.publish(twist)
            self.publish_path([])
            return

        v, w, path = self.run_mpc_sampling(target)
        msg = Twist(); msg.linear.x = float(v); msg.angular.z = float(w)
        self.cmd_vel_pub.publish(msg)
        self.publish_path(path)

    def publish_path(self, trajectory):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "base_link"
        for i in range(len(trajectory)):
            pose = PoseStamped()
            pose.pose.position.x = trajectory[i, 0]
            pose.pose.position.y = trajectory[i, 1]
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def publish_obstacles(self):
        if len(self.obstacles_local) == 0: return
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacles"; marker.id = 0; marker.type = Marker.POINTS; marker.action = Marker.ADD
        marker.scale.x = 0.05; marker.scale.y = 0.05; marker.color.b = 1.0; marker.color.a = 1.0
        for obs in self.obstacles_local:
            p = Point(); p.x = float(obs[0]); p.y = float(obs[1])
            marker.points.append(p)
        self.obs_pub.publish(marker)
        
    def publish_robot_radius(self):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "safety_margin"; marker.id = 0; marker.type = Marker.CYLINDER; marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        diameter = self.config.robot_radius * 2.0
        marker.scale.x = diameter; marker.scale.y = diameter; marker.scale.z = 0.05
        marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 0.3
        self.robot_radius_pub.publish(marker)

    def publish_target_marker(self):
        if self.target_pose_in_odom is None: return
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mpc_target"; marker.id = 0; marker.type = Marker.SPHERE; marker.action = Marker.ADD
        marker.pose = self.target_pose_in_odom.pose
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
        marker.color.g = 1.0; marker.color.a = 1.0
        self.target_marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
