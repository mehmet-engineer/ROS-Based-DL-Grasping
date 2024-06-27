#!/usr/bin/env python3

import rospy 
import numpy as np
from std_msgs.msg import Float32MultiArray, String
from fyp.srv import RobotServer, RobotServerResponse
from FYPRobot import FYPRobot

"""
    Robot server as a service
    when the service is called, the robot will perform the command received
    
    service message:
        request
            string: command type
            floatarray: values (pose number/grasp/joints)
        
        response
            string: task status (success/fail)
            string: etc.
            floatarray: data
            
    
    robot server will be client

"""



class RobotCommandServer():
    def __init__(self):

        
         # start grasp server
        service_name = '/robot_server'
        rospy.loginfo(f'Start Service at {service_name}')
        self.service = rospy.Service(service_name, RobotServer, self.service_callback)

        self.robot = FYPRobot()
    
    def service_callback(self,req):
        
        self.command = req.command
        self.data = req.data
        
        rospy.loginfo(f"Command : {self.command}") 
        self.task_status = True
        
        self.robot.bz = True
        
        extra = ""
        curr_joint = []
        try:
            if self.command == "grasp":
                
                grasp = [self.data]
                print(grasp)
                
                
                curr_joint = self.robot.grasp(grasp)
                # signal to robot
            
            elif self.command == "joint":
                
                joint_val = self.data
                # print(joint_val)
                curr_joint = self.robot.joint(joint_val)
                # signal to robot
            
            elif self.command == "pose":
                
                pose_idx = int(self.data[0])
                # print(pose_idx)
                
                # signal to robot

                curr_joint = self.robot.pose(pose_idx)
                    
            elif self.command == "start":
                
                # signal to robot 
                # get preset poses.
                pose_list, curr_joint = self.robot.start()
                pose_list_string = "\n".join(pose_list)
                extra = pose_list_string
        except Exception as e:
                print(e)
                self.task_status = False
                self.robot.bz = False
                rospy.loginfo(f"Task Succeed : {self.task_status}") 
                
        while self.robot.bz: continue
        
        res = RobotServerResponse()
        res.status = "SUCCESS" if self.task_status else "FAIL"
        res.extra = extra
        res.data = curr_joint
        
        return res   
    
            
if __name__ == '__main__':
  node = rospy.init_node('robot', anonymous=True)
  robot = RobotCommandServer()
  rospy.spin() 
        
        
