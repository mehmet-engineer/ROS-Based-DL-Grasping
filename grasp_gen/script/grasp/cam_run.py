#!/usr/bin/env python3

import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '/home/nsrie/working/graspgen/FYP_RG')
 
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import CompressedImage # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
from std_srvs.srv import Empty, EmptyResponse  
from grasp_gen.srv import depth, depthResponse

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torch


from hardware.camera_new import CustomCamera
from hardware.device import get_device
from models.common import post_process_output
from utils.data.camera_data import CameraData, CameraData_new
from utils.visualisation.plot import save_results, plot_results
from dept_estimator.dpt_midas import DeptEstimator

def grasp_pred(req):
 
  rospy.sleep(rospy.Duration(1))
  data = rospy.wait_for_message('/camera1/rgb/image_raw/compressed', CompressedImage, timeout=5)
  #br = CvBridge()
 
  # Output debugging information to the terminal
  rospy.loginfo("receiving video frame")
  
  np_arr = np.frombuffer(data.data, np.uint8)
  image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
  # Convert ROS Image message to OpenCV image
  
  rospy.loginfo("dept estimation")
  dpt_depth = dpt_estimator.dpt_pred(rgb=image_np)
  dpt_depth = np.array(dpt_depth)
  
  #print(dpt_depth.shape)
  # Display image
  # cv2.imshow("pred", dpt_depth)
  # rospy.loginfo("show result...")
  # cv2.waitKey(0)
  # cv2.destroyWindow("pred")
  # cv2.waitKey(1)
  
  #### Create CompressedIamge ####
  msg = CompressedImage()
  msg.header.stamp = rospy.Time.now()
  msg.format = "jpeg"
  msg.data = np.array(cv2.imencode('.jpg', dpt_depth)[1]).tostring()
  rospy.loginfo("msg sent...")
  return msg
      
def receive_message():
 
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name.
  
  rospy.init_node('grasp_server', anonymous=True)
  rospy.loginfo("Starting grasp_server...")
  # Node is subscribing to the video_frames topic
  my_service = rospy.Service('/grasp_service', depth, grasp_pred) # create the Service called my_service with the defined callback
  rospy.spin() 

  # Close down the video stream when done
  cv2.destroyAllWindows()



device = get_device(False)
cam_data = CameraData_new(include_depth=1,
                            include_rgb=1,
                            output_size=1)
# Load Network
rospy.loginfo('Loading grasp detector...')
#net = torch.load("/home/nsrie/working/graspgen/models/10_m_rgbd_epoch_50_thresh_44_iou_80_r_35.pt")
rospy.loginfo('Loading dept Estimator...')
dpt_estimator = DeptEstimator(mode=3, device=device)
rospy.loginfo('Done') 



if __name__ == '__main__':
  
    
  receive_message()