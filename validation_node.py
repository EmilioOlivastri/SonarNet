#!/usr/bin/env python

# Import required Python code.
import rospy
from pf_localization.srv import SonarObjObservation, SonarObjObservationResponse

import numpy as np
import math
from cv_bridge import CvBridge
import cv2 as cv
import torch

from model import SonarNet
from inference import predict_img

class SonarObjectDetector:

    # Constructor
    def __init__(self, model) :
        self.bridge = CvBridge()
        
        print('Loading the model')
        self.net = SonarNet(n_channels=3, n_classes=1, n_angles=18)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device=self.device)
        state_dict = torch.load(model, map_location=self.device)
        state_dict.pop('mask_values', [0, 1])
        self.net.load_state_dict(state_dict)
        self.range = 15.0
        self.angle_step = 10 * math.pi / 180.0
        print(f'Model = {model} loaded successfully!')

    # Create a callback function for the subscriber.
    def particle_callback(self, request):

        # Writing the 2 images and then comparing the real one with the predicted
        sonar_img = self.bridge.imgmsg_to_cv2(request.sonar_img, "bgr8")
        sonar_img = sonar_img.transpose(2, 0, 1)        
        tensor_img = torch.from_numpy(sonar_img)

        print(f'Img Shape = {sonar_img.shape}')
        mask, angle = predict_img(net=self.net,
                                  full_img=tensor_img,
                                  device=self.device,
                                  scale_factor=0.5)
        mask = mask.astype(np.uint8)
        angle = angle.astype(float)

        # Finding max
        max_idxs = [0, 0]
        max_val = 0.0
        for idx in range(mask.shape[0]) :
            for idy in range(mask.shape[1]) :
                new_val = mask[idx][idy]
                if ( max_val < new_val ) :
                    max_val = new_val
                    max_idxs[0] = idx
                    max_idxs[1] = idy

        idx_angle = angle.argmax()

        step_x = mask.shape[0] / self.range
        step_y = mask.shape[1] / self.range
        pos_x = (mask.shape[0] - max_idxs[0]) / step_x
        pos_y = (max_idxs[1] - mask.shape[1]/2) / step_y
        s1 = self.angle_step * idx_angle
        yaws = [s1, s1 + math.pi]

        print(f'Estimated pose in sonar frame is = {pos_x, pos_y, yaws}')

        hypo_img = np.copy(mask)
        hypo_img.fill(0)

        for idx in range(0, 5) :
            hypo_img[max_idxs[0] + idx][max_idxs[1] + idx] = 254
            hypo_img[max_idxs[0] - idx][max_idxs[1] - idx] = 254
            hypo_img[max_idxs[0]][max_idxs[1] - idx] = 254
            hypo_img[max_idxs[0]][max_idxs[1] + idx] = 254
            hypo_img[max_idxs[0] + idx][max_idxs[1]] = 254
            hypo_img[max_idxs[0] - idx][max_idxs[1]] = 254
        

        cv.imshow("Predicted Heat", mask)
        cv.imshow("Selected Max", hypo_img)
        cv.waitKey(10)
        return SonarObjObservationResponse(request.sonar_img.header, pos_x, pos_y, yaws[0], yaws[1])

    # This ends up being the main while loop.
    def start(self):

        # Get the ~private namespace parameters from command line or launch file.
        topic = 'sonar_obj_detect'

        s = rospy.Service(topic, SonarObjObservation, lambda msg: self.particle_callback(msg))
    
        # Wait for messages on topic, go to callback function when new messages arrive.
        rospy.spin()

# Main function.
if __name__ == '__main__':

    # Initialize the node and name it.
    rospy.init_node('ObjectDetector', anonymous = True)
    
    model = "/home/slam-emix/Datasets/UnderwaterDeep/SonarNet/weights/best_angle.pth"
    s_obj_dt = SonarObjectDetector(model)

    # Go to the main loop.
    s_obj_dt.start()