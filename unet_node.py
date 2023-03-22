#!/usr/bin/env python

# Import required Python code.
import rospy
from nps_uw_multibeam_sonar.msg import SonarImgHeat

import numpy as np
from cv_bridge import CvBridge
import cv2 as cv
import torch

import matplotlib.pyplot as plt

from model import UNet
from inference import predict_img

class SonarObjectDetector:

    # Constructor
    def __init__(self, model) :
        self.bridge = CvBridge()
        
        print('Loading the model')
        self.net = UNet(n_channels=3, n_classes=1, bilinear=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device=self.device)
        state_dict = torch.load(model, map_location=self.device)
        state_dict.pop('mask_values', [0, 1])
        self.net.load_state_dict(state_dict)
        print(f'Model = {model} loaded successfully!')

    # Create a callback function for the subscriber.
    def callback(self, sonar_heat_msg):

        # Writing the 2 images and then comparing the real one with the predicted
        cv2_img = self.bridge.imgmsg_to_cv2(sonar_heat_msg.image, "bgr8")
        cv2_heat = self.bridge.imgmsg_to_cv2(sonar_heat_msg.heat, "mono8")
        cv2_img_rbg = cv.cvtColor(cv2_img, cv.COLOR_BGR2RGB)

        cv2_img_rbg = cv2_img_rbg.transpose(2, 0, 1) 
        tensor_img = torch.from_numpy(cv2_img_rbg)
        #print(f'Img Shape = {cv2_img.shape}')
        label = predict_img(net=self.net,
                            full_img=tensor_img,
                            device=self.device,
                            scale_factor=0.5).astype(cv2_heat.dtype)

        print(f"Idx Max Label = {label.max()}")
        print(f"Idx Max GT = {cv2_heat.max()}")

        cv.imshow("Sonar Img", cv2_img)
        cv.imshow("Heat", cv2_heat)
        cv.imshow("Predicted Heat", label)
        cv.waitKey(10)

    # This ends up being the main while loop.
    def listener(self):

        # Get the ~private namespace parameters from command line or launch file.
        topic = '/sonar/img_heat'

        # Create a subscriber with appropriate topic, custom message and name of callback function.
        rospy.Subscriber(topic, SonarImgHeat, self.callback)
        
        print(f'Waiting on topic = {topic}')

        # Wait for messages on topic, go to callback function when new messages arrive.
        rospy.spin()

# Main function.
if __name__ == '__main__':

    # Initialize the node and name it.
    rospy.init_node('ObjectDetector', anonymous = True)
    
    model = "/home/slam-emix/Workspace/Underwater/Saipem/underwater_sim/src/nps_uw_multibeam_sonar/scripts/sonar_net/checkpoints/checkpoint_epoch300.pth"
    s_obj_dt = SonarObjectDetector(model)

    # Go to the main loop.
    s_obj_dt.listener()