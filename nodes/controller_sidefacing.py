#!/usr/bin/env python3
"""
Left Wall Alignment Controller (Side-Facing Camera)
Simple wall following using only the left side-facing camera.
Control law from paper: u = k * (Tau_2 - Tau_1)

Camera Setup:
- Left camera at (0, 0.2, 0.2), facing LEFT (90° rotation)
- Views the left wall from the side
"""
import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from mobilenettt_visionnavigation.msg import TTTValues
from cv_bridge import CvBridge


class LeftWallAlignmentController(Node):

    def __init__(self):
        super().__init__('controller_left_alignment')
        self.declare_parameters(namespace='',
        parameters=[
            ('left_camera_topic', "/left_camera/image"),
            ('left_ttt_topic', "/left_ttt_values"),
            ('velocity_topic', "/cmd_vel"),
            ('use_sim_time', True)
            ])
        
        self.left_camera_topic = self.get_parameter('left_camera_topic').get_parameter_value().string_value
        self.left_ttt_topic = self.get_parameter('left_ttt_topic').get_parameter_value().string_value
        self.velocity_topic = self.get_parameter('velocity_topic').get_parameter_value().string_value

        # Publisher
        self.pub_velocity = self.create_publisher(TwistStamped, self.velocity_topic, 2)
        
        # Subscribers
        self.sub_left_camera = self.create_subscription(Image, self.left_camera_topic, self.left_image_callback, 10)
        self.sub_left_ttt = self.create_subscription(TTTValues, self.left_ttt_topic, self.left_ttt_callback, 10)
        
        self.bridge = CvBridge()

        # Frame
        self.left_frame = None
        
        # TTT values
        self.left_tau_1 = 0.0   # Left half of grid
        self.left_tau_2 = 0.0   # Right half of grid
        
        # Control parameters
        self.velocity = 0.3
        self.k = 0.5  # Control gain (same as paper)
        self.plot_histogram = True
        
        self.get_logger().info('Left Wall Alignment Controller Started')
        self.get_logger().info('Using only left camera for alignment control')
    
    def left_image_callback(self, msg):
        try:
            self.left_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert left image: {e}')

    def draw_histogram_left(self, col_means_ttt, left_thr, right_thr):
        if self.left_frame is None:
            return
        img = self.left_frame.copy()
        height, width, _ = img.shape
        n_bars = len(col_means_ttt)
        bar_width = width // n_bars
        max_val_scaled = 20
        scale = height / max_val_scaled

        for i, val in enumerate(col_means_ttt):
            x = i * bar_width
            bar_height = int(val * 10 * scale)
            if i <= left_thr:
                color = (255, 0, 0)  # Blue - Tau_1 (left half)
            elif i >= right_thr:
                color = (0, 165, 255)  # Orange - Tau_2 (right half)
            else:
                color = (0, 255, 0)  # Green
            y1 = height
            y2 = y1 - bar_height
            cv2.rectangle(img, (x, y1), (x + bar_width - 2, y2), color, -1)

        # Draw vertical line at midpoint
        mid_x = width // 2
        cv2.line(img, (mid_x, 0), (mid_x, height), (255, 255, 255), 3)
        cv2.putText(img, "Tau_1 | Tau_2", (mid_x - 100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display tau values
        tau_text = f"T1={self.left_tau_1:.3f}  T2={self.left_tau_2:.3f}"
        cv2.putText(img, tau_text, (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Left Wall Alignment", img)
        cv2.waitKey(1)
        
    def left_ttt_callback(self, data):
        ttt_array = np.array(data.num).reshape(10, 20)
        ttt_array[ttt_array == 0] = np.nan 
        col_means_ttt = np.nanmean(ttt_array, axis=0)
        mid_point = len(col_means_ttt) // 2
        left_grid = col_means_ttt[:mid_point]
        right_grid = col_means_ttt[mid_point:]
        
        # Normalize by velocity
        self.left_tau_1 = np.nanmean(left_grid) / self.velocity
        self.left_tau_2 = np.nanmean(right_grid) / self.velocity
        
        # Display histogram
        if self.plot_histogram:
            self.draw_histogram_left(col_means_ttt, mid_point-1, mid_point)
        
        # ═══════════════════════════════════════════════════════
        # LEFT WALL ALIGNMENT CONTROL
        # ═══════════════════════════════════════════════════════
        
        # Apply control law from paper: u = k * (Tau_2 - Tau_1)
        u = self.k * (self.left_tau_2 - self.left_tau_1)
        
        # Saturate
        max_angular = 1.0
        u = max(-max_angular, min(max_angular, u))
        
        # Determine direction
        if u > 0.05:
            direction = "→ RIGHT (away from wall)"
        elif u < -0.05:
            direction = "← LEFT (toward wall)"
        else:
            direction = "↑ STRAIGHT (aligned)"
        
        print(f'\n=== LEFT WALL ALIGNMENT ===')
        print(f'Tau_1 (left half):  {self.left_tau_1:.3f}')
        print(f'Tau_2 (right half): {self.left_tau_2:.3f}')
        print(f'Difference: {self.left_tau_2 - self.left_tau_1:.3f}')
        print(f'Control: u = {u:.3f}')
        print(f'Steering: {direction}')
        print(f'===========================\n')
        
        # Publish
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(self.velocity)
        msg.twist.angular.z = float(u)
        
        self.pub_velocity.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = LeftWallAlignmentController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
