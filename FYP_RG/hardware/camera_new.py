import logging


import sys
 # setting path
# sys.path.append('/home/nsrie/working/graspgen/FYP_RG')

import matplotlib.pyplot as plt
import numpy as np
#import pyrealsense2 as rs
import cv2
from matplotlib.animation import FuncAnimation
from dept_estimator.dpt_midas import DeptEstimator
from hardware.device import get_device

logger = logging.getLogger(__name__)


class CustomCamera:
    def __init__(self,
                 device_id,
                 width=640,
                 height=480,
                 dpt=False
                 ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.dpt = dpt
        
        if self.dpt:
            self.dpt_estimator = DeptEstimator(mode=1, device=device)

        self.pipeline = None
        self.scale = None
        self.intrinsics = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.device_id)

    def get_image_bundle(self):
        
        _, frame = self.cap.read()
        
        dim = (self.width,self.height)
        color_image = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        if self.dpt:
            depth_image = self.dpt_estimator.dpt_pred(
                                rgb=color_image,
                                save=False,
                                vis=False)
        else:
            depth_image = np.zeros_like(color_image)

        return {
            'rgb': color_image,
            'aligned_depth': depth_image,
        }
    

    def plot_image_bundle(self):
        
        images = self.get_image_bundle()
        
        rgb = images['rgb']
        depth = images['aligned_depth']
        
        img = np.hstack((rgb, depth))        
        
        cv2.imshow("test",img)
        cv2.waitKey(1)
        



if __name__ == '__main__':
    device = get_device(False)
    cam = CustomCamera(device_id="http://192.168.1.110:8080/video", dpt=True)
    cam.connect()
    print("camera connected")

    while True:
        cam.plot_image_bundle()

