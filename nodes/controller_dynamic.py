#!/usr/bin/env python3
import os

import numpy as np
import cv2
import os
import re
import shutil
import time
import sys
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from IPython import display

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image, CameraInfo
from mobilenettt_visionnavigation.msg import TTTValues


class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        self.declare_parameters(namespace='',
        parameters=[
            ('camera_topic', "/camera/image"),
            ('camera_info_topic', "/camera_color/camera_info"), 
            ('ttt_topic', "/ttt_values"),
            ('velocity_topic', "/cmd_vel")
            ])
        
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.ttt_topic = self.get_parameter('ttt_topic').get_parameter_value().string_value
        self.velocity_topic = self.get_parameter('velocity_topic').get_parameter_value().string_value

        
        self.pub_velocity = self.create_publisher(TwistStamped, self.velocity_topic, 2)
        self.sub_camera = self.create_subscription(Image,self.camera_topic,self.image_callback,10)
        self.sub_camera
        #self.sub_camera_info = self.create_subscription(CameraInfo,self.camera_info_topic,self.info_callback,10)
        #self.sub_camera_info
        self.sub_ttt = self.create_subscription(TTTValues,self.ttt_topic,self.ttt_callback,10)
        self.sub_ttt
        
        self.bridge = CvBridge()

        self.frame = None
        # Initialization arrays of TTT values for each ROI during sense phase
        self.final_right_e = np.array([])
        self.final_left_e = np.array([])
        self.final_right = np.array([])
        self.final_left = np.array([])
        self.tau_center_values = np.array([])
        
        self.col_means_ttt = np.array([])
        
        self.x_percent = 10 
        self.constant = 6        
        # Initialization average TTT values for each ROI during sense phase
        self.mean_tau_er = 0
        self.mean_tau_el = 0
        self.mean_tau_r = 0
        self.mean_tau_l = 0
        self.mean_tau_c = 0
        
        self.obstacle = False
        self.extreme_right = False
        self.extreme_left = False
        self.right = False
        self.left = False
        self.center = False
        
        self.plot_histogram = False
        
        self.max_u = 1
        self.kp_e = 0.15
        self.kp = 0.25
        
        #self.kp_e = 0.3
        #self.kp = 0.2
        self.max_control = 1.0
        self.velocity = 0.4
        
        
    def find_obstacle(self, mean, t):

        if self.center and (mean <= t):
            self.obstacle = True
        else:
            self.obstacle = False
        
    def threshold(self, value, limit):
        if value >= limit:
            value = limit
        elif value <= -limit:
            value = -limit
        return value
        
    def find_neighbors(self, vector):
        vector = np.array(vector)*10
        vector = vector/self.velocity
        max_val = np.max(vector)
        max_idx = np.argmax(vector)

        print(f'The vector is: {vector}')
        print(f'The max value is {max_val} and the position in the vector is {max_idx}')
    
        threshold = max_val * (1 - self.x_percent / 100.0)
    
        # Search left of max
        if max_idx > 0:
            left_vals = vector[:max_idx]
            left_candidates = left_vals[left_vals <= threshold]
            left_max = np.max(left_candidates) if left_candidates.size > 0 else None
            if left_max != None:
                left_max_idx = np.where(left_vals == left_max)[0][0]
            else:
                left_max_idx = None
                left_candidates = np.zeros(5)
        else:
            left_max = None
            left_max_idx = None
            left_candidates = np.zeros(5)

        print(f'The left candidates are: {left_candidates}')
        print(f'The left max values is {left_max}')
        print(f'The left max values idx is {left_max_idx}')

        # Search right of max
        if max_idx < len(vector) - 1:
            #print('Searching the boundary on the right')
            right_vals = vector[max_idx+1:]
            right_candidates = right_vals[right_vals <= threshold]
            right_max = np.max(right_candidates) if right_candidates.size > 0 else None
            if right_max != None:
                right_max_idx = np.where(right_vals == right_max)[0][0]
                right_max_idx = right_max_idx + max_idx+1
            else:
                right_max_idx = None
                right_candidates = np.zeros(5)
        else:
            right_max = None
            right_max_idx = None
            right_candidates = np.zeros(5)
            
        print(f'The right candidates are: {right_candidates}')
        print(f'The right max values is {right_max}')
        print(f'The right max values idx is {right_max_idx}')

        return max_val, left_max, right_max, max_idx, left_max_idx, right_max_idx, left_candidates, right_candidates
        
    def draw_histogram_opencv(self, col_means_ttt, left_thr, right_thr, left_max_idx=None, right_max_idx=None, max_idx=None, left_max=None, right_max=None, max_val=None):
        # Setup
        width = 800
        height = 400
        margin = 50
        n_bars = len(col_means_ttt)
        bar_width = (width - 2 * margin) // n_bars
        max_val_scaled = 20

        # Create white canvas
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Normalize values to height
        scale = (height - 2 * margin) / max_val_scaled

        # Draw bars
        for i, val in enumerate(col_means_ttt):
            x = margin + i * bar_width
            bar_height = int(val * 10 * scale)

            # Determine color
            if i <= left_thr:
                color = (255, 0, 0)  # Blue in BGR
            elif i >= right_thr:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green

            y1 = height - margin
            y2 = y1 - bar_height
            cv2.rectangle(img, (x, y1), (x + bar_width - 2, y2), color, -1)
        
        # Optional: Show image
        cv2.imshow("Histogram (OpenCV)", img)
        cv2.waitKey(10)
        #cv2.destroyAllWindows()

    # Draw vertical lines
    def draw_line(idx, color, label):
        if idx is None:
            return
        x = margin + idx * bar_width + bar_width // 2
        cv2.line(img, (x, height - margin), (x, margin), color, 2)
        cv2.putText(img, label, (x + 5, margin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if left_max_idx is not None:
            draw_line(left_max_idx, (255, 200, 100), f'Left Max: {left_max}')
        if right_max_idx is not None:
            draw_line(right_max_idx, (0, 255, 255), f'Right Max: {right_max}')
        if max_idx is not None:
            draw_line(max_idx, (0, 200, 0), f'Max: {max_val}')

        # Draw axis line
        cv2.line(img, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 1)

        # Optional: Show image
        cv2.imshow("Histogram (OpenCV)", img)
        cv2.waitKey(10)
        #cv2.destroyAllWindows()
        
    def draw_histogram_opencv_on_img(self, col_means_ttt, left_thr, right_thr, left_max_idx=None, right_max_idx=None, max_idx=None, left_max=None, right_max=None, max_val=None):
        if self.frame is None:
            return

        # Make a copy of the original frame to draw on
        img = self.frame.copy()

        height, width, _ = img.shape
        n_bars = len(col_means_ttt)
        bar_width = width // n_bars
        max_val_scaled = 20
        scale = height / max_val_scaled

        for i, val in enumerate(col_means_ttt):
            x = i * bar_width
            bar_height = int(val * 10 * scale)

            if i <= left_thr:
                color = (255, 0, 0)
            elif i >= right_thr:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            y1 = height
            y2 = y1 - bar_height
            cv2.rectangle(img, (x, y1), (x + bar_width - 2, y2), color, -1)

        # Add vertical lines for indices (optional)
        def draw_line(idx, color, label):
            if idx is not None:
                x = int(idx) * bar_width + bar_width // 2
                cv2.line(img, (x, height), (x, height - 100), color, 2)
                cv2.putText(img, label, (x + 5, height - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        #draw_line(left_max_idx, (255, 200, 100), f'LMax: {left_max:.1f}' if left_max else '')
        #draw_line(right_max_idx, (0, 255, 255), f'RMax: {right_max:.1f}' if right_max else '')
        #draw_line(max_idx, (0, 200, 0), f'Max: {max_val:.1f}' if max_val else '')

        cv2.imshow("Histogram Overlay", img)
        cv2.waitKey(1)
    
    
    def ttt_callback(self, data):
        
        ttt_array = np.array(data.num).reshape(10, 20)
        
        ttt_array = ttt_array
        
        ttt_array[ttt_array == 0] = np.nan 
        
        # Grabs means of each of the 20 columns
        col_means_ttt = np.nanmean(ttt_array, axis=0)
 
        
        max_val, left_max, right_max, max_idx, left_max_idx, right_max_idx, left_candidates, right_candidates = self.find_neighbors(col_means_ttt)
        
        #final_tau_center = np.mean(col_means_ttt[int(left_max_idx):int(right_max_idx)])
        #final_tau_right_e = np.mean(right_candidates)
        #final_tau_left_e = np.mean(left_candidates)
        
        # FIXED: Use max_val instead of 0 when there's an opening (no boundary detected)
        # This represents "free space" or "far away obstacle"
        if left_max is not None:
            left_thr = int(left_max_idx)
            final_tau_left_e = np.mean(left_candidates)
        else:
            left_thr = -1
            final_tau_left_e = max_val  # FIXED: Was 0, now max_val (represents opening)
            
        if right_max is not None:
            right_thr = int(right_max_idx)
            final_tau_right_e = np.mean(right_candidates)
        else:
            right_thr = 30
            final_tau_right_e = max_val  # FIXED: Was 0, now max_val (represents opening)
        
        if right_max is not None and left_max is not None:
            final_tau_center = np.mean(col_means_ttt[int(left_max_idx):int(right_max_idx)])
        elif left_max is not None:
            final_tau_center = np.mean(col_means_ttt[left_thr:])
        elif right_max is not None:
            final_tau_center = np.mean(col_means_ttt[0:right_thr])
        else:
            final_tau_center = 0
        
        if self.plot_histogram:
        	self.draw_histogram_opencv_on_img(col_means_ttt, left_thr, right_thr, left_max_idx, right_max_idx, max_idx, left_max, right_max, max_val)
        

        print(f'Tau Left Extreme: {final_tau_left_e}')
        print(f'Tau Right Extreme: {final_tau_right_e}')
        print(f'Tau Center: {final_tau_center}')
        
        # FIXED: Now we check if the value is significantly high (opening) rather than just != 0
        if final_tau_right_e > 0.1:  # Small threshold to avoid numerical errors
            self.extreme_right = True
            self.final_right_e = final_tau_right_e
        if final_tau_left_e > 0.1:  # Small threshold to avoid numerical errors
            self.extreme_left = True
            self.final_left_e = final_tau_left_e
        if final_tau_center > 0.1:  # Small threshold to avoid numerical errors
            self.center = True
            self.final_center = final_tau_center
        
        control_e = 0
        control = 0
        
        if self.extreme_left and self.extreme_right:
            print("Complete Extreme control!")
            control_e = self.kp_e*(self.final_left_e - self.final_right_e)
###########################################################################################
            if self.center and (self.final_center<0.55) and (self.final_center > 0.3):
                if self.final_left_e < self.final_right_e:
                    control_e = -0.55
                else:
                    control_e = 0.5
            else: 
                print("Complete Extreme control!")
                control_e = self.kp_e*(self.final_left_e - self.final_right_e)
############################################################################################
        elif self.extreme_left:
            print("Extreme Left control!")
            control_e = self.kp_e*(self.final_left_e - self.constant)
            if self.center and (self.final_center<0.55) and (self.final_center > 0.3):
                if self.final_left_e < self.final_right_e:
                    control_e = -0.5
                else:
                    control_e = 0.5
            else: 
                print("Extreme Left control!")
                control_e = self.kp_e*(self.final_left_e - self.constant)
        elif self.extreme_right:
            print("Extreme Right control!")
            control_e = -self.kp_e*(self.final_right_e - self.constant)
            if self.center and (self.final_center<0.55) and (self.final_center > 0.3):
                if self.final_left_e < self.final_right_e:
                    control_e = -0.5
                else:
                    control_e = 0.5
            else: 
                print("Extreme Right control!")
                control_e = -self.kp_e*(self.final_right_e - self.constant)
            
        control = control_e
        control = self.threshold(control, self.max_control)
        print(f'The control action is: {control}')
        
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(self.velocity)
        msg.twist.angular.z = float(control)
        
        self.pub_velocity.publish(msg)
        
        self.extreme_right = False
        self.extreme_left = False
        self.center = False
        
    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            self.frame = None

def main(args=None):
    rclpy.init(args=args)

    ml_algorithm = Controller()

    try:
        rclpy.spin(ml_algorithm)
    except KeyboardInterrupt:
        ml_algorithm.destroy_node()

    rclpy.shutdown()

if __name__=="__main__":
    main()
