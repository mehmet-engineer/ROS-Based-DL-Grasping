#!/usr/bin/env python3
# Description:
# - Subscribes to real-time streaming video from your built-in webcam.
#
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
# https://automaticaddison.com/working-with-ros-and-opencv-in-ros-noetic/


# Import the necessary libraries


import rospy # Python library for ROS
from sensor_msgs.msg import Image, CompressedImage # Image is the message type
import cv2 # OpenCV library
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
from grasp_gen.srv import *
from std_srvs.srv import Empty, EmptyResponse
import numpy as np

from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from gazebo_msgs.srv import SpawnModel
from grasp.grasp_utils import img_2_world, GraspPredictor

class GraspServer:
  
  """
  GraspServer has two function:
    1. Broadcasting camera view
    2. Grasp prediction by request
  
  """
  
  
  def __init__(self,
               
               ):
    self.init()
    self.current_frame = None
    
    rospy.loginfo('System ready')
    rospy.spin()
    cv2.destroyAllWindows()
    
  def init(self):
    
    # start broadcasting camera
    img_topic = '/cam_scene/rgb/image_raw/compressed'
    rospy.loginfo(f'Start Subscriber to {img_topic}')
    self.subscriber  = rospy.Subscriber(img_topic, CompressedImage, self.cam_subscriber_callback)
    
    
    # self.grasp_predictor = GraspPredictor(n_grasp=5)
    
    # start grasp server
    service_name = '/grasp_service'
    rospy.loginfo(f'Start Service at {service_name}')
    self.service = rospy.Service(service_name, Grasp, self.grasp_service_callback)
    
  def cam_subscriber_callback(self, data):
    
    np_arr = np.frombuffer(data.data, np.uint8)
    self.current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    cv2.imshow("camera", self.current_frame)
    cv2.waitKey(1)

  def grasp_service_callback(self,req):
    
    rospy.loginfo("===== Request received =====")
    
    rgb = self.current_frame
    dpt_depth = self.grasp_predictor.depth_pred(rgb)
    img, grasps = self.grasp_predictor.grasp_pred(rgb, dpt_depth)
    
    g_world = self.grasp_2_world(grasps)
    
    #### process respond ####
    
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', img)[1]).tostring()
    
    #print(g_world)
    
    g_world = np.reshape(g_world,(-1))
    g_world = g_world.tolist()
    
    res = GraspResponse()
    res.img = msg
    res.grasp = g_world
    
    rospy.loginfo("===== Sending respond =====")
    return res
  
  def grasp_2_world(self,grasps):
    """
     Transform pixel coordinate to world position
     
     return list  [world_x, world_y, obj_avg_h, angle, length, width]
    """
    
    obj_avg_h = 0.678169
    g_world = []
    for g in grasps:
      pix_x, pix_y, angle, length, width = g.center[1],g.center[0],g.angle,g.length,g.width
      rospy.loginfo(f'Grasp detected{pix_x,pix_y,angle,length,width}')
      world_x,world_y = img_2_world(pix_x, pix_y, obj_avg_h) # --> x,y
      
      g_world.append([world_x,world_y,obj_avg_h,angle, length, width])
    
    return g_world


if __name__ == '__main__':
  node = rospy.init_node('gs', anonymous=True)
  gs = GraspServer()