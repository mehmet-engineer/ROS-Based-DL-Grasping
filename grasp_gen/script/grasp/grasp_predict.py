#!/usr/bin/env python3
# Description:
# - Subscribes to real-time streaming video from your built-in webcam.
#
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
# https://automaticaddison.com/working-with-ros-and-opencv-in-ros-noetic/
# http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber

# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image, CompressedImage # Image is the message type
import cv2 # OpenCV library

from grasp_gen.srv import *
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
from grasp.grasp_utils import MarkerSpawner,GraspBroadcaster

class GraspClient():
  
  def __init__(self):
    self.marker_spawner = MarkerSpawner()
    
    rospy.wait_for_service('/grasp_service')
    rospy.loginfo('Waiting for /grasp_service')
    self.client = rospy.ServiceProxy('/grasp_service', Grasp)
    
    self.wait_display = False
  
  def display(self,data):
    rospy.loginfo("Processing respond")
    np_arr = np.frombuffer(data.img.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    grasp = np.reshape(data.grasp,(-1,6)) 
    
    rospy.loginfo("Displaying Result and Markers")
    self.marker_spawner.spawn_marker(grasp)
    self.g_tf = GraspBroadcaster(grasp)
  
    cv2.imshow("test", image_np)
    
    wait = 0 if self.wait_display else 1
    cv2.waitKey(wait)
        
  def predict(self):
    req = GraspRequest()
    rospy.loginfo("===== Sending request =====")
    result = self.client(req)
    self.display(result)
  
  def stop_display(self):
    cv2.destroyAllWindows()
    self.g_tf.stop()
    self.marker_spawner.despawn()
    rospy.loginfo("===== Stop displaying... =====")
  
    
  
if __name__ == '__main__':
  rospy.init_node('grasp_client')
  gc = GraspClient()
  gc.wait_display = True
  gc.predict()
  gc.stop_display()