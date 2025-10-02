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

        
        #self.pub_velocity = self.create_publisher(Twist, self.velocity_topic, 2)
        self.pub_velocity = self.create_publisher(TwistStamped,"jackal_velocity_controller/cmd_vel",10)
        #self.sub_camera = self.create_subscription(Image,self.camera_topic,self.image_callback,10)
        #self.sub_camera
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
        
        self.max_u = 1
        self.kp_e = 0.4
        self.kp = 0.3
        
        #self.kp_e = 0.3
        #self.kp = 0.2
        self.max_control = 1.0
        self.velocity = 0.2
        
        
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
	
    def ttt_callback(self, data):
        
        ttt_array = np.array(data.num).reshape(10, 20)
        
        # Data acquisition from TauValues topic
        #final_tau_right_e = np.mean(ttt_array[:, 15:])/self.velocity
        #final_tau_left_e = np.mean(ttt_array[:, :5])/self.velocity
        #final_tau_right = np.mean(ttt_array[:, 11:15])/self.velocity
        #final_tau_left = np.mean(ttt_array[:, 5:9])/self.velocity
        #final_tau_center = np.mean(ttt_array[3:7, 9:11])/self.velocity
        #final_tau_center = np.mean(ttt_array[3:7, 7:16])/self.velocity
        
        final_tau_right_e = np.mean(ttt_array[:, 17:])/self.velocity
        final_tau_left_e = np.mean(ttt_array[:, :3])/self.velocity
        final_tau_right = np.mean(ttt_array[:, 13:17])/self.velocity
        final_tau_left = np.mean(ttt_array[:, 3:7])/self.velocity
        #final_tau_center = np.mean(ttt_array[3:7, 9:11])/self.velocity
        final_tau_center = np.mean(ttt_array[3:7, 7:16])/self.velocity
        
        print(f'Tau Left Extreme: {final_tau_left_e}')
        print(f'Tau Right Extreme: {final_tau_right_e}')
        print(f'Tau Left: {final_tau_left}')
        print(f'Tau Right: {final_tau_right}')
        print(f'Tau Center: {final_tau_center}')
        
        if final_tau_right_e != 0:
            self.extreme_right = True
            self.final_right_e = final_tau_right_e
        if final_tau_left_e != 0:
            self.extreme_left = True
            self.final_left_e = final_tau_left_e
        if final_tau_right != 0:
            self.right = True
            self.final_right = final_tau_right
        if final_tau_left != 0:
            self.left = True
            self.final_left = final_tau_left
        if final_tau_center != 0:
            self.center = True
            self.final_center = final_tau_center
        
        # Tau vector for every ROI
        #if final_tau_right_e >= 0:
        #    self.final_right_e = np.append(self.final_right_e, final_tau_right_e)
        #if final_tau_left_e >= 0:
        #    self.final_left_e = np.append(self.final_left_e, final_tau_left_e)
        #if final_tau_right >= 0:
        #    self.final_right = np.append(self.final_right, final_tau_right)
        #if final_tau_left >= 0:
        #    self.final_left = np.append(self.final_left, final_tau_left)
        #if final_tau_center >= 0:
        #    self.tau_center_values = np.append(self.tau_center_values, final_tau_center)
        
        # Check if each ROI has a valid value
        #if np.size(self.final_right_e) > 0:
        #    self.mean_tau_er = np.sum(self.final_right_e) / np.size(self.final_right_e)
        #else:
        #    self.extreme_right = False
            
        #if np.size(self.final_left_e) > 0:
        #    self.mean_tau_el = np.sum(self.final_left_e) / np.size(self.final_left_e)
        #else:
        #    self.extreme_left = False
            
        #if np.size(self.final_right) > 0:
        #    self.mean_tau_r = np.sum(self.final_right) / np.size(self.final_right)
        #else:
        #    self.right = False
            
        #if np.size(self.final_left) > 0:
        #    self.mean_tau_l = np.sum(self.final_left) / np.size(self.final_left)
        #else:
        #    self.left = False
            
        #if np.size(self.tau_center_values) > 0:
        #    self.mean_tau_c = np.sum(self.tau_center_values) / np.size(self.tau_center_values)
        #else:
        #    self.center = False
        
        
        control_e = 0
        control_m = 0
        control = 0
        
        if self.extreme_left and self.extreme_right:
            print("Complete Extreme control!")
            control_e = self.kp_e*(self.final_left_e - self.final_right_e)
###########################################################################################
            if self.center and (self.final_center<0.6) and (self.final_center > 0.3):
                if self.final_left_e < self.final_right_e:
                    control_e = -0.2
                else:
                    control_e = 0.2
            else: 
                print("Complete Extreme control!")
                control_e = self.kp_e*(self.final_left_e - self.final_right_e)
############################################################################################
        elif self.extreme_left:
            print("Extreme Left control!")
            control_e = self.kp_e(self.final_left_e - self.constant)
        elif self.extreme_right:
            print("Extreme Right control!")
            control_e = -self.kp_e(self.final_right_e - self.constant)
            
        if self.left and self.right:
            print("Complete Median control!")
            control_m = self.kp*(self.final_left - self.final_right)
        elif self.left:
            print("Median Left control!")
            control_m = self.kp_e(self.final_left - self.constant)
        elif self.right:
            print("Median Right control!")
            control_m = -self.kp_e(self.final_right - self.constant)
            
        control = control_e + control_m
        control = self.threshold(control, self.max_control)
        print(control)
        
        msg = TwistStamped()
        msg.twist.linear.x = float(self.velocity)
        msg.twist.angular.z = float(control)
        
        self.pub_velocity.publish(msg)
        
        self.extreme_right = False
        self.extreme_left = False
        self.right = False
        self.left = False
        self.center = False

            
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
