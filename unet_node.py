#!/usr/bin/env python

# Import required Python code.
import rospy
from nps_uw_multibeam_sonar.msg import SonarImgHeat

import numpy as np
import math
from cv_bridge import CvBridge
import cv2 as cv
import torch

from model import UNet
from inference import predict_img

class SonarObjectDetector:

    # Constructor
    def __init__(self, model) :
        self.bridge = CvBridge()
        
        print('Loading the model')
        self.net = UNet(n_channels=1, n_classes=1, bilinear=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device=self.device)
        state_dict = torch.load(model, map_location=self.device)
        state_dict.pop('mask_values', [0, 1])
        self.net.load_state_dict(state_dict)
        self.err_x = 0.0
        self.err_y = 0.0
        self.counter = 0
        print(f'Model = {model} loaded successfully!')

    # Create a callback function for the subscriber.
    def callback(self, sonar_heat_msg):

        # Writing the 2 images and then comparing the real one with the predicted
        cv2_img = self.bridge.imgmsg_to_cv2(sonar_heat_msg.image, "bgr8")
        cv2_heat = self.bridge.imgmsg_to_cv2(sonar_heat_msg.heat, "mono8")
        cv2_img_gray = cv.cvtColor(cv2_img, cv.COLOR_BGR2GRAY)

        tensor_img = torch.from_numpy(cv2_img_gray)
        #print(f'Img Shape = {cv2_img.shape}')
        predicted_label = predict_img(net=self.net,
                                      full_img=tensor_img,
                                      device=self.device,
                                      scale_factor=0.5).astype(cv2_heat.dtype)

        self.counter += 1

        # Given the predicted label it is needed to estimate 
        # the center of the rectangle and its orientation
        center_pred, angle_pred = self.findCenterAndAngle(predicted_label)
        center_gt, angle_gt_comp = self.findCenterAndAngle(cv2_heat)
        gt_angle = sonar_heat_msg.angle
        
        print(f"Predicted Values : C({center_pred[0]}, {center_pred[1]}) | alpha = {angle_pred}")
        print(f"GT Values : C({center_gt[0]}, {center_gt[1]}) | alpha = {angle_gt_comp} | alpha_msg = {gt_angle}")

        self.err_y += (center_pred[0] - center_gt[0]) * (center_pred[0] - center_gt[0])
        self.err_x += (center_pred[1] - center_gt[1]) * (center_pred[1] - center_gt[1])

        if self.counter % 5 == 0 :
            print(f"MSE = {math.sqrt((self.err_x + self.err_y) / self.counter)}")

        cv.imshow("Sonar Img", cv2_img)
        cv.imshow("Heat", cv2_heat)
        cv.imshow("Predicted Heat", predicted_label)
        cv.waitKey(10)


    def findCenterAndAngle(self, img) :

        # Finding the 2 opposite corners of the rectangle
        rgb = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        corners = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        detected_corners = cv.goodFeaturesToTrack(img, maxCorners=4, qualityLevel=0.5, minDistance=150)

        for idx, cn in enumerate(detected_corners) :
            x, y = cn.ravel()
            corners[idx][0] = x
            corners[idx][1] = y
            cv.circle(rgb,(x,y),8,(255,120,255),-1)
            print("({}, {})".format(x,y))
        
        # Visualization only for debugging
        cv.imshow("image", rgb)
        cv.waitKey(20)

        c1 = corners[0]

        # Create an array with the distances from c1
        # Furthest is the opposite
        # Second longest is "big side"
        # Remaining one is the "short side"
        dtype = [('index', int), ('distance', float)]
        distances = np.array([[1, 0.0], [2, 0.0], [3, 0.0]], dtype=dtype)
        for idx in range(1, corners.shape[0]) :
            distances[idx - 1]['distance'] = np.linalg.norm(c1 - corners[idx])

        np.sort(distances, order='distances')
        c3 = corners[distances[2]['index']]
        c2 = corners[distances[1]['index']]
        c4 = corners[distances[0]['index']]

        center = np.array([(c1[0] + c3[0]) / 2, 
                           (c1[1] + c3[1]) / 2])

        angle = np.arctan2(c1[1] - c4[1],
                           c1[0] - c4[0])

        print(f"Center = ({center[0]}, {center[1]}), Angle = {angle}")
        
        return center, angle

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