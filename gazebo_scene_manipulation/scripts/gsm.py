#!/usr/bin/python

import rospy
from geometry_msgs.msg import Pose, PoseStamped
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler

from gazebo_scene_manipulation import *
#from fusion_server.srv import CaptureScene

import os, sys
from glob import glob
import numpy as np

rospy.init_node('gsm_wrapper')


def upright_pose(x,y,z, q):
  pose = Pose()
  pose.position.x = x
  pose.position.y = y
  pose.position.z = z
  pose.orientation.x = q[0]
  pose.orientation.y = q[1]
  pose.orientation.z = q[2]
  pose.orientation.w = q[3]
  return pose

def random_pose_in_area(xlims, ylims, zlims,
                        roll_lims, pitch_lims, yaw_lims):
  x = np.random.uniform(xlims[0], xlims[1])
  y = np.random.uniform(ylims[0], ylims[1])
  z = np.random.uniform(zlims[0], zlims[1])
  roll  = np.random.uniform(roll_lims[0], roll_lims[1])
  pitch = np.random.uniform(pitch_lims[0], pitch_lims[1])
  yaw   = np.random.uniform(yaw_lims[0], yaw_lims[1])
  q = quaternion_from_euler(roll, pitch, yaw)
  pose = upright_pose(x,y,z, q)
  return pose

xlims = [-0.2,0.2]
ylims = [-0.2,0.2]
zlims = [0.75,0.8]
roll_lims  = [0,0]#[-np.pi, np.pi]
pitch_lims = [0,0]#[-np.pi, np.pi]
yaw_lims   = [-2*np.pi, 2*np.pi]

models = ["coke_can", "beer", "cricket_ball",
          "gear_part", "wood_cube_10cm", "coffee_box",
          ]

models = os.listdir(rospy.get_param("~models_dir", "$(find gazebo_scene_manipulation)/models/"))
models.remove("tools")
models.append("hammer")

gsm = GSM()

obj_list = []

for i in range(3):
  model_name = np.random.choice(models)
  instance_name = f"{model_name}_{i}"
  obj_list.append(instance_name)
  pose = random_pose_in_area(xlims, ylims, zlims,
                        roll_lims, pitch_lims, yaw_lims)
  success = gsm.spawn(model_name,
                      instance_name,
                      pose)
                       
  if success:
    rospy.loginfo("success")

  rospy.sleep(rospy.Duration(1))

#rospy.sleep(rospy.Duration(5))

# rospy.loginfo("despawning..")
# for obj in obj_list:
#   gsm.despawn(obj)

rospy.sleep(rospy.Duration(1))

