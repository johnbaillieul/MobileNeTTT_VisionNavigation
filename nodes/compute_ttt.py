#!/usr/bin/env python3
import os

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import re
import shutil
import time
import sys

from cv_bridge import CvBridge
from PIL import Image as PILImage
from cbam import CBAM

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image, CameraInfo
from mobilenettt_visionnavigation.msg import TTTValues
from ament_index_python.packages import get_package_share_directory
gpu_no = 0
assert torch.cuda.is_available()
#device = torch.device(f'cuda:{gpu_no}')
device = torch.device("cpu")

class TauValues(Node):

    def __init__(self):
        super().__init__('tttvalues')
        self.declare_parameters(namespace='',
		parameters=[
			('camera_topic', "/camera/image"),
			('camera_info_topic', "/camera_color/camera_info"), 
			('ttt_topic', "/ttt_values")
            ])
        
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.ttt_topic = self.get_parameter('ttt_topic').get_parameter_value().string_value

        self.pub_ttt = self.create_publisher(TTTValues, self.ttt_topic, 2)
        self.sub_camera = self.create_subscription(Image,self.camera_topic,self.image_callback,10)
        self.sub_camera
        #self.sub_camera_info = self.create_subscription(CameraInfo,self.camera_info_topic,self.info_callback,10)
        #self.sub_camera_info
        
        self.bridge = CvBridge()

        self.frame = None
        self.prev_image = None
        self.curr_image = None
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
        
        self.velocity = 0.5
        
        self.first_time = False
        self.attention = True
        self.spatial_att_modified = False
        self.plot_grid = True
        
    def draw_grid_text(self, image_grid, img_h, img_w, data, rows, cols, deltaY, line_color=(0, 0, 0)):
        height, width = img_h, img_w
       
        # Calculate the spacing between grid lines
        row_spacing = (height-2*deltaY) // rows
        col_spacing = width // cols
       
        # Draw horizontal grid lines
        for i in range(0, rows + 1):
            y = i * row_spacing
            cv2.line(image_grid, (0, int(y)+deltaY), (width, int(y)+deltaY), line_color, 1)

        # Draw vertical grid lines
            for j in range(cols):
                x = j * col_spacing
                x_text = (j+1)*col_spacing
                if i < data.shape[0] and j < data.shape[1]:
                    cell_data = np.around(data[i,j]/self.velocity, decimals=1)
                    text_position = ((x + col_spacing // 2), (y + deltaY + row_spacing // 2))
                    cv2.putText(image_grid, str(cell_data), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
               
            # Draw vertical grid lines
        for j in range(1, cols + 1):
            x = j * col_spacing
            cv2.line(image_grid, (int(x), deltaY), (int(x), height-deltaY), line_color, 1)

        cv2.namedWindow('Grid Representation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Grid Representation', (600, 600))
        cv2.imshow('Grid Representation', image_grid)
        cv2.waitKey(10)
        
    def normalize_img(self, img0, img1):
        img0 = (img0/255)
        img1 = (img1/255)
        return img0, img1
            
    def mobilenet_network(self):
        debug_dimensions = False
        mobilenet = mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
        mobilenet.features[0][0] = torch.nn.Conv2d(6, 16, kernel_size=7, stride=2, padding=3, bias=False)
        mobilenet_without_head = torch.nn.Sequential(*list(mobilenet.children())[:-2])
        conv_layer1 = torch.nn.Conv2d(576, 288, kernel_size=2, stride=1, padding=0)
        conv_layer2 = torch.nn.Conv2d(288, 144, kernel_size=(3,2), stride=1, padding=1)
        conv_layer3 = torch.nn.Conv2d(144, 72, kernel_size=2, stride=1, padding=1)
        conv_layer4 = torch.nn.Conv2d(72, 2, kernel_size=2, stride=1, padding=0)
        
        head = torch.nn.Sequential( 
            conv_layer1, 
            torch.nn.ReLU(inplace=True),
            conv_layer2, 
            torch.nn.ReLU(inplace=True), 
            conv_layer3, 
            torch.nn.ReLU(inplace=True), 
            conv_layer4, 
            torch.nn.ReLU(inplace=True)  
            )
            
        model = torch.nn.Sequential(
            mobilenet_without_head, 
            head
            )
            
        return model
        
    def mobilenet_attention_network(self):
        debug_dimensions = False
        mobilenet = mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
        mobilenet.features[0][0] = torch.nn.Conv2d(64, 16, kernel_size=7, stride=2, padding=3, bias=False)
        mobilenet_without_head = torch.nn.Sequential(*list(mobilenet.children())[:-2])
        conv_layer1 = torch.nn.Conv2d(576, 288, kernel_size=2, stride=1, padding=0)
        conv_layer2 = torch.nn.Conv2d(288, 144, kernel_size=(3,2), stride=1, padding=1)
        conv_layer3 = torch.nn.Conv2d(144, 72, kernel_size=2, stride=1, padding=1)
        conv_layer4 = torch.nn.Conv2d(72, 2, kernel_size=2, stride=1, padding=0)
        
        head = torch.nn.Sequential( 
            conv_layer1, 
            torch.nn.ReLU(inplace=True),
            conv_layer2, 
            torch.nn.ReLU(inplace=True), 
            conv_layer3, 
            torch.nn.ReLU(inplace=True), 
            conv_layer4, 
            torch.nn.ReLU(inplace=True)  
            )
            
        model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=6, out_channels=64, kernel_size=1, stride=1),
            CBAM(gate_channels=64, reduction_ratio=4, no_channel=not(self.attention), no_spatial=not(self.spatial_att_modified)),
            mobilenet_without_head, 
            head
            )
            
        return model
		
	
    def image_callback(self, data):
        self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.first_time = False
        if self.prev_image is None:
            self.prev_image = self.frame
            self.prev_image = PILImage.fromarray(self.frame)
            self.first_time = True
            return
        if not self.first_time:
            self.curr_image = self.frame
            crop_params = (70, 0, 340, 640)
            self.curr_image = PILImage.fromarray(self.curr_image)
            print(f'Previous Image: {self.prev_image.size}')
            print(f'Current Image: {self.curr_image.size}')
            img1 = transforms.functional.crop(self.prev_image, *crop_params)
            img2 = transforms.functional.crop(self.curr_image, *crop_params)
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            
            img1.to(device)
            img2.to(device)
            img1, img2 = self.normalize_img(img1, img2)
            input_imgs = torch.cat([img1, img2], dim=0)
            input_imgs = input_imgs.unsqueeze(0)
            input_imgs = input_imgs.to(device)
            
            package_name = 'mobilenettt_visionnavigation'
            package_dir = get_package_share_directory(package_name)

            # Construct path to your weights file
            weights_path = os.path.join(package_dir, 'trained_weights', 'mobilenetV3_small_channel_lossgeneralnorm2.pth')
            weights_path2 = os.path.join(package_dir, 'trained_weights', 'mobilenetV3_small_lossgeneralnorm2.pth') 
            if self.attention:
                model = self.mobilenet_attention_network()
                model.load_state_dict(torch.load(weights_path))
            else:
                model = self.mobilenet_network()
                model.load_state_dict(torch.load(weights_path2))
            model.to(device)
            
            result = model(input_imgs)[0]
            
            ttt = result[0,:,:].cpu().detach().numpy()
            conf = result[1,:,:].cpu().detach().numpy()
            
            mask = np.where(conf < 0.5, 0, 1)
            ttt_mask = ttt*mask
            
            
            msg = TTTValues()
            msg.num = ttt_mask.flatten().tolist()
            self.pub_ttt.publish(msg)
            
            if self.plot_grid:
            	self.draw_grid_text(self.frame, 480, 640, ttt_mask, 10, 20, 70)
            self.prev_image = self.curr_image
            #print(ttt)
            
                        
def main(args=None):
    rclpy.init(args=args)

    ml_algorithm = TauValues()

    try:
        rclpy.spin(ml_algorithm)
    except KeyboardInterrupt:
        ml_algorithm.destroy_node()

    rclpy.shutdown()

if __name__=="__main__":
	main()
