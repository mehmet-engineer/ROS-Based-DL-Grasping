#!/usr/bin/env python3
import rospy, sys
import moveit_commander
import geometry_msgs.msg
"""
 @Author: Mehmet Kahraman
 @Date: 22.12.2022
"""

def initialization():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("moveit_simulation", anonymous=True)
    
    group_name = "ur_arm"
    ur_group = moveit_commander.MoveGroupCommander(group_name)
    group_name = "hand"
    eef_group = moveit_commander.MoveGroupCommander(group_name)
    rospy.loginfo("robot initialized")
    rospy.sleep(1)
    
    return (ur_group, eef_group)

def go_to_target(ur_group, joint_angles):
    joints = ur_group.get_current_joint_values()
    print(joints)
    joint_values = joint_angles
    plan_exec = ur_group.go(joint_values, wait=True)
    if plan_exec != True:
        print("plan couldn't be executed")
        quit()
    rospy.loginfo("reached target position \n")
    rospy.sleep(1)
    
def gripper_command(eef_group, command):
    joint_values = eef_group.get_current_joint_values()
    print(joint_values)
    if command == "open":
        joint_values[0] = 1.0
    elif command == "close":
        joint_values[0] = -0.15
    else:
        rospy.loginfo("gripper command failed!")
    plan_exec = eef_group.go(joint_values, wait=True)
    if plan_exec != True:
        print("gripper couldn't be executed")
    rospy.loginfo("gripper command executed \n")
    rospy.sleep(1)
    
def algorithm(ur_group, eef_group):
    
    # GO TO HOME
    go_to_target(ur_group, joint_angles=[0, -1.57, 0, -1.57, 0, 0])
    
    # TARGET OBJECT
    go_to_target(ur_group, joint_angles=[-1.6, -1.0, 1.1, -2.2, -1.57, 0])
    gripper_command(eef_group, command="open")
    go_to_target(ur_group, joint_angles=[-1.6, -1.0, 1.48, -2.2, -1.57, 0])
    gripper_command(eef_group, command="close")
    
    # GO TO HOME
    go_to_target(ur_group, joint_angles=[0, -1.57, 0, -1.57, 0, 0])

def test(ur_group, eef_group):
    """
        0.111437 -0.712785 0.545245
    """
    
    
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.w = 1.0
    pose_goal.position.x = 0.15
    pose_goal.position.y = -0.75
    pose_goal.position.z = 0.4

    ur_group.set_pose_target(pose_goal)
    
    # `go()` returns a boolean indicating whether the planning and execution was successful.
    success = ur_group.go(wait=True)
    # Calling `stop()` ensures that there is no residual movement
    ur_group.stop()
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets().
    ur_group.clear_pose_targets()

def spawnbox(ur_group, eef_group):
    robot = moveit_commander.RobotCommander()
    eef_link = eef_group.get_end_effector_link()
    scene = moveit_commander.PlanningSceneInterface()
    scene.remove_world_object("box")
    
    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = "world"
    box_pose.pose.orientation.w = 1.0
    box_pose.pose.position.x = 0.111437  # above the panda_hand frame
    box_pose.pose.position.y = -0.712785
    box_pose.pose.position.z = 0.045245
    box_name = "box"
    scene.add_box(box_name, box_pose, size=(0.06, 0.06, 0.06))
    
    grasping_group = "hand"
    touch_links = robot.get_link_names(group=grasping_group)
    scene.attach_box(eef_link, box_name, touch_links=touch_links)

if __name__ == '__main__':
    ur_group, eef_group = initialization()
    spawnbox(ur_group, eef_group)
    #algorithm(ur_group, eef_group)
    #test(ur_group, eef_group)
    
