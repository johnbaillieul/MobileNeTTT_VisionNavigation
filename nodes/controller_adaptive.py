#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import generic_filter, gaussian_filter
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from mobilenettt_visionnavigation.msg import TTTValues
from cv_bridge import CvBridge


def yaw_from_quaternion(qx, qy, qz, qw):
    s = 2.0 * (qw * qz + qx * qy)
    c = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(s, c))


class Controller(Node):
    def __init__(self):
        super().__init__('controller')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_topic', "/camera/image"),
                ('ttt_topic', "/ttt_values"),
                ('velocity_topic', "/cmd_vel"),
                ('odom_topic', "/odom"),
                ('enable_visualization', True),

                ('fov_deg', 90.0),
                ('tau_min', 0.20),
                ('depth_scale', 4.0),

                ('heat_sigma', 0.3),

                ('map_size', 240),
                ('map_res', 0.05),
                ('map_decay', 0.01),

                ('L_occ', 0.85),
                ('L_free', -0.35),
                ('L_min', -5.0),
                ('L_max', 5.0),

                ('scan_row_start', 3),
                ('scan_row_end', 7),
            ]
        )

        # Parameters
        self.camera_topic = self.get_parameter('camera_topic').value
        self.ttt_topic = self.get_parameter('ttt_topic').value
        self.velocity_topic = self.get_parameter('velocity_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.enable_visualization = self.get_parameter('enable_visualization').value

        self.fov_deg = float(self.get_parameter('fov_deg').value)
        self.tau_min = float(self.get_parameter('tau_min').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.heat_sigma = float(self.get_parameter('heat_sigma').value)

        self.map_size = int(self.get_parameter('map_size').value)
        self.map_res = float(self.get_parameter('map_res').value)
        self.map_decay = float(self.get_parameter('map_decay').value)

        self.L_occ = float(self.get_parameter('L_occ').value)
        self.L_free = float(self.get_parameter('L_free').value)
        self.L_min = float(self.get_parameter('L_min').value)
        self.L_max = float(self.get_parameter('L_max').value)

        self.scan_row_start = int(self.get_parameter('scan_row_start').value)
        self.scan_row_end = int(self.get_parameter('scan_row_end').value)

        # ROS
        self.pub_velocity = self.create_publisher(TwistStamped, self.velocity_topic, 2)
        self.sub_camera = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.sub_ttt = self.create_subscription(TTTValues, self.ttt_topic, self.ttt_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)

        self.bridge = CvBridge()

        # Robot state
        self.velocity = 0.4
        self.have_odom = False
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0

        # Map
        self.log_odds = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.map_center = self.map_size // 2
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_origin_initialized = False

        if self.enable_visualization:
            self.setup_visualization()

    # ------------------------------------------------------------
    # Odom
    # ------------------------------------------------------------
    def odom_callback(self, msg):
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)

        q = msg.pose.pose.orientation
        self.odom_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.have_odom = True

        if not self.map_origin_initialized:
            self.map_origin_x = self.odom_x
            self.map_origin_y = self.odom_y
            self.map_origin_initialized = True

    # ------------------------------------------------------------
    # Visualization Setup
    # ------------------------------------------------------------
    def setup_visualization(self):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 5))
        gs = self.fig.add_gridspec(1, 2, wspace=0.25)

        # Heatmap
        self.ax_hm = self.fig.add_subplot(gs[0, 0])
        self.ax_hm.set_title("Smoothed Closeness Heatmap")
        self.img_hm = self.ax_hm.imshow(
            np.zeros((10, 20)),
            cmap='turbo',
            aspect='auto',
            interpolation='bilinear',
            vmin=0,
            vmax=1
        )
        plt.colorbar(self.img_hm, ax=self.ax_hm)

        # Occupancy map
        self.ax_map = self.fig.add_subplot(gs[0, 1])
        self.ax_map.set_title("TTT Occupancy Map")
        self.ax_map.set_aspect('equal', 'box')

        self.img_map = self.ax_map.imshow(
            0.5 * np.ones((self.map_size, self.map_size)),
            cmap='gray',
            origin='lower',
            vmin=0,
            vmax=1,
            interpolation='nearest'
        )

        # Robot circle (real radius)
        robot_radius_m = 0.15
        robot_radius_cells = robot_radius_m / self.map_res

        self.robot_patch = Circle(
            (self.map_center, self.map_center),
            robot_radius_cells,
            facecolor='red',
            edgecolor='black',
            linewidth=2,
            zorder=10
        )
        self.ax_map.add_patch(self.robot_patch)

        self.heading_arrow = None

        plt.tight_layout()
        plt.show(block=False)

    # ------------------------------------------------------------
    # Fill zeros
    # ------------------------------------------------------------
    def fill_zeros_with_neighbors(self, grid):
        filled = grid.copy()

        def robust_fill(values):
            valid = values[values > 0.0]
            return np.mean(valid) if len(valid) > 0 else 0.0

        for _ in range(3):
            mask = filled == 0.0
            if not mask.any():
                break
            result = generic_filter(filled, robust_fill, size=5, mode='constant', cval=0.0)
            filled[mask] = result[mask]

        return filled

    # ------------------------------------------------------------
    # Closeness
    # ------------------------------------------------------------
    def compute_closeness(self, ttt_filled):
        depth_proxy = self.velocity * ttt_filled
        closeness = 1.0 / (depth_proxy + 1e-3)

        c_min = np.percentile(closeness, 5)
        c_max = np.percentile(closeness, 95)
        closeness = (closeness - c_min) / (c_max - c_min + 1e-6)

        return np.clip(closeness, 0, 1)

    # ------------------------------------------------------------
    # Grid → Scan
    # ------------------------------------------------------------
    def ttt_grid_to_scan(self, ttt_filled):
        band = ttt_filled[self.scan_row_start:self.scan_row_end, :].copy()
        band[band <= 0.0] = np.nan
        T_cols = np.nanmedian(band, axis=0)

        cols = ttt_filled.shape[1]
        fov = np.deg2rad(self.fov_deg)
        angles = np.linspace(-fov / 2.0, fov / 2.0, cols)

        ranges = self.depth_scale * self.velocity * T_cols
        ranges[(np.isnan(T_cols)) | (T_cols < self.tau_min)] = np.nan

        return angles, ranges

    # ------------------------------------------------------------
    # World → Grid
    # ------------------------------------------------------------
    def world_to_grid(self, xw, yw):
        dx = xw - self.map_origin_x
        dy = yw - self.map_origin_y
        gx = int(self.map_center + dx / self.map_res)
        gy = int(self.map_center + dy / self.map_res)
        return gx, gy

    # ------------------------------------------------------------
    # Update Map
    # ------------------------------------------------------------
    def update_occupancy_from_scan(self, angles, ranges):
        if not self.have_odom:
            self.odom_x = 0.0
            self.odom_y = 0.0
            self.odom_yaw = 0.0

        if self.map_decay > 0:
            self.log_odds *= (1.0 - self.map_decay)

        cy = np.cos(self.odom_yaw)
        sy = np.sin(self.odom_yaw)

        for a, r in zip(angles, ranges):
            if np.isnan(r):
                continue

            dx_r = np.cos(a)
            dy_r = np.sin(a)

            dx_w = cy * dx_r - sy * dy_r
            dy_w = sy * dx_r + cy * dy_r

            steps = int(r / self.map_res)
            if steps <= 1:
                continue

            for s in range(steps - 2):
                dist = s * self.map_res
                x = self.odom_x + dist * dx_w
                y = self.odom_y + dist * dy_w
                gx, gy = self.world_to_grid(x, y)
                if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                    self.log_odds[gy, gx] += self.L_free

            x_end = self.odom_x + r * dx_w
            y_end = self.odom_y + r * dy_w
            gx, gy = self.world_to_grid(x_end, y_end)
            if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                self.log_odds[gy, gx] += self.L_occ

        self.log_odds = np.clip(self.log_odds, self.L_min, self.L_max)

    # ------------------------------------------------------------
    # Visualization Update
    # ------------------------------------------------------------
    def update_visualization(self, ttt_grid):
        ttt_filled = self.fill_zeros_with_neighbors(ttt_grid)

        closeness = self.compute_closeness(ttt_filled)
        closeness_smooth = gaussian_filter(closeness, sigma=self.heat_sigma)
        self.img_hm.set_data(closeness_smooth)

        angles, ranges = self.ttt_grid_to_scan(ttt_filled)
        self.update_occupancy_from_scan(angles, ranges)

        prob = 1.0 / (1.0 + np.exp(-self.log_odds))
        self.img_map.set_data(prob)

        rx, ry = self.world_to_grid(self.odom_x, self.odom_y)

        self.robot_patch.center = (rx, ry)

        # Update heading arrow
        if self.heading_arrow:
            self.heading_arrow.remove()

        arrow_len = 0.4 / self.map_res
        dx = arrow_len * np.cos(self.odom_yaw)
        dy = arrow_len * np.sin(self.odom_yaw)

        self.heading_arrow = self.ax_map.arrow(
            rx, ry,
            dx, dy,
            head_width=5,
            head_length=5,
            fc='red',
            ec='black',
            zorder=11
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ------------------------------------------------------------
    # TTT Callback
    # ------------------------------------------------------------
    def ttt_callback(self, data):
        ttt_grid = np.array(data.num).reshape(10, 20)

        if self.enable_visualization:
            self.update_visualization(ttt_grid)

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(self.velocity)
        msg.twist.angular.z = 0.0
        self.pub_velocity.publish(msg)

    def image_callback(self, msg):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    plt.close('all')


if __name__ == "__main__":
    main()
