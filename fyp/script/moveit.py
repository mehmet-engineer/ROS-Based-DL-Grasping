#!/usr/bin/env python

from __future__ import print_function
from six.moves import input

from grasp.grasp_predict import GraspClient
from grasp.robot_server import RobotCommandServer
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

import numpy as np
import math
import tf, tf2_ros

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

# from spatialmath import SE3

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

## END_SUB_TUTORIAL


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class MoveGroupPythonInterface(object):

    def __init__(self):
        super(MoveGroupPythonInterface, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)

        # self.grasp_client = GraspClient()
        self.grasp_client = RobotCommandServer()
        
        robot = moveit_commander.RobotCommander()

        
        scene = moveit_commander.PlanningSceneInterface()

        
        group_name = "arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        
        ee_name = "gripper"
        gripper_group = moveit_commander.MoveGroupCommander(ee_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

       
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)
        
        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())
        
        active_joints = move_group.get_active_joints()
        print("============ arm_active_joints:\n%s" % active_joints)
        print("============ get_pose_reference_frame:\n", move_group.get_pose_reference_frame())
        print("============ get_named_targets:\n", move_group.get_named_targets())
        print("============ get_current_joint_values:\n", move_group.get_current_joint_values())
        print("============ get_current_pose:\n", move_group.get_current_pose().pose)

        gripper_active_joints = gripper_group.get_active_joints()
        print("============ gripper_active_joints: %s" % gripper_active_joints)
        print("============ get_pose_reference_frame:\n", gripper_group.get_pose_reference_frame())
        print("============ get_named_targets:\n", gripper_group.get_named_targets())
        print("============ get_current_joint_values:\n", gripper_group.get_current_joint_values())
        

        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.gripper_group = gripper_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        
        self.target = "grasp_0"
        
        #self.move_group.set_planning_time(20)
        self.move_group.set_num_planning_attempts(3)
        self.move_group.set_max_velocity_scaling_factor(0.3)
        self.move_group.allow_replanning(1)

    
    def set_traget(self, target):
        
        target_int = int(target)
        self.target = f"grasp_{target_int}"
    
    def get_pose(self):
        print(self.move_group.get_current_pose().pose)



    def go_to_obj(self, step):

        move_group = self.move_group
        # self.move_group.set_goal_orientation_tolerance(np.deg2rad(5))
        # self.move_group.set_goal_position_tolerance(0.005)
        
        target = self.target
        trans,rot = self.get_tf(ref="world",tar=target)

        pose_goal = self.move_group.get_current_pose().pose

        waypoints = []
        
        print(f"======== go to obj step:{step}")
        
        margin = 0 if step else 0.10  # 0 if step == 1

        pose_goal.position.x = trans[0]
        pose_goal.position.y = trans[1]
        pose_goal.position.z = trans[2] + margin
        
        if step == 0:
            move_group.set_pose_target(pose_goal) 
            success = move_group.go(wait=True)
            
            print(f"======== plan to tf => {success}")

        else:
            waypoints.append(copy.deepcopy(pose_goal))

            (plan, fraction) = move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )  # jump_threshold

            success = move_group.execute(plan, wait=True)
            print(f"======== compute_cartesian_path => {success}")   


        move_group.stop()
        move_group.clear_pose_targets()
        
        
    def gripper(self,mode):
        
        gripper_val = [-0.1502983257076549, 0.5113476836458775]
        
        joint_goal = self.gripper_group.get_current_joint_values()
        joint_goal[0] = gripper_val[mode]
        
        mode_n = "open" if mode else "close"
        print(f"======== gripper [{mode_n}]")
        success = self.gripper_group.go(joint_goal, wait=True)
        self.gripper_group.stop()
        
        print(f"======== gripper =>{success}")

    
    def get_tf(self,ref='world',tar="tool0",verbose=0):
        
        ref_frame = ref
        target_frame = tar
        
        listener=tf.TransformListener()
        listener.waitForTransform(target_frame,ref_frame,rospy.Time(), rospy.Duration(1.0))
        (trans,rot)=listener.lookupTransform(ref_frame,target_frame,rospy.Time())
        
        if verbose:
            print(f"""==> Transform {target_frame}->{ref_frame}:\n======> Translation:\n{trans}\n======> Rotation:\n{rot}"""
                  )
        return trans,rot

    def wait_for_state_update(
        self, box_is_known=False, box_is_attached=False, timeout=4
    ):
        box_name = self.box_name
        scene = self.scene

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = box_name in scene.get_known_object_names()

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False
        ## END_SUB_TUTORIAL
    
    def ee_angle_prep(self):
        
        
        trans,rot = self.get_tf(ref="world",tar=self.target)
        print("============ rot g: ",rot)
        # trans,rot = self.get_tf(ref="world",tar="tool0")     
        # print("============ rot t: ",rot)
        
        rad_val = tf.transformations.euler_from_quaternion(rot)
        rad_val = np.deg2rad(90) - rad_val[2]
        print(f"============ Turn ee [{rad_val}] rad")
        
        current_joint = self.move_group.get_current_joint_values()
        current_joint[-1] = rad_val
        target_joint = current_joint
        #print(target_joint)
        
        self.move_group.go(target_joint, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()
        
   
  

    def set_pose(self, pos_name):
        
        pos_list = {"pre_grasp":[1.5974109821149494, -1.5708413681533495,
                            1.5707239437027747, -1.5707588027140745,
                            -1.570762731403538, 0],
                    "place":[-1.61343183480954, -1.3727402441839103,
                             1.997539276005213, -2.706063866055354,
                             -1.5797915925203546, 1.2346094946456532],
                    "home":[ 0, -1.5972622762816973,
                            1.4198577706381794, -2.7061245022568494,
                            -1.649864420073071, 1.5708976445354415],
                    }

        print(f"==== go to [{pos_name}] pose")
        self.move_group.go(pos_list[pos_name], wait=True)
        self.move_group.stop()
        
    def add_table(self, timeout=4):
        
        box_name = "table"
        scene = self.scene
        
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "world"
        
        quat = tf.transformations.quaternion_from_euler(0,0,0)
        box_pose.pose.orientation.x = quat[0]
        box_pose.pose.orientation.y = quat[1]
        box_pose.pose.orientation.z = quat[2]
        box_pose.pose.orientation.w = quat[3]
        
        box_pose.pose.position.x = -0.054718 
        box_pose.pose.position.y = -0.185332
        box_pose.pose.position.z = 0.314283

        scene.add_box(box_name, box_pose, size=(1.172340, 1.676260, 0.628566))
        
        self.box_name = box_name
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)
    
    def add_plate(self, timeout=4):
        
        box_name = "plate"
        scene = self.scene
        
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "world"
        
        quat = tf.transformations.quaternion_from_euler(0,0,0)
        box_pose.pose.orientation.x = quat[0]
        box_pose.pose.orientation.y = quat[1]
        box_pose.pose.orientation.z = quat[2]
        box_pose.pose.orientation.w = quat[3]
        
        box_pose.pose.position.x = 0
        box_pose.pose.position.y = 0
        box_pose.pose.position.z = 0.642773

        scene.add_box(box_name, box_pose, size=(0.77, 0.994407, 0.028835))
        
        self.box_name = box_name
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)
    
    def move(self):
        self.add_table()
        self.add_plate()
        print("x"*100)
        while True:
            if self.grasp_client.command == "start":
                self.set_pose("home")
                self.gripper(0)
                self.grasp_client.command = ""
                self.grasp_client.working = False
            
            # tutorial.grasp_client.predict()
            if self.grasp_client.command == "grasp":
                self.set_traget(self.grasp_client.selected_grasp)
                self.set_pose("pre_grasp")
                
                self.go_to_obj(0)
                self.gripper(1)
                self.ee_angle_prep()
                
                self.go_to_obj(1)
                self.gripper(0)
                self.go_to_obj(2)

                self.set_pose("place")
                self.gripper(1)
                self.grasp_client.stop_display()
                self.grasp_client.command = "start"
                self.grasp_client.working = False
                
            # tutorial.grasp_client.stop_display()
            rospy.sleep(0.5)
            
        print("x"*100)
    
    
def main():
    rospy.init_node("move_group_python_interface_tutorial", anonymous=True)
    try:
        tutorial = MoveGroupPythonInterface()
        tutorial.move()
        
        
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
