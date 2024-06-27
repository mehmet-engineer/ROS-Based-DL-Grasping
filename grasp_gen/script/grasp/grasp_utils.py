import sys
import os
# adding Folder to the system path
sys.path.insert(0, '/home/nsrie/working/graspgen/FYP_RG')

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torch


from utils.dataset_processing.grasp import detect_grasps
from hardware.camera_new import CustomCamera
from hardware.device import get_device
from models.common import post_process_output
from utils.data.camera_data import CameraData, CameraData_new
from utils.visualisation.plot import save_results, plot_results, return_results
from dept_estimator.dpt_midas import DeptEstimator

import threading

import geometry_msgs.msg
from tf.transformations import quaternion_from_euler
from gazebo_msgs.srv import SpawnModel, DeleteModel

import numpy as np
import rospy
import cv2
import tf2_ros, tf
import rospkg

rospack = rospkg.RosPack()

def img_2_world(pix_x,pix_y, obj_avg_h,
                cam_height = 1.5,
                width = 480,
                height = 480,
                fx = 415.69101839339027,
                fy = 415.69101839339027,
                rot_mat = np.array([[0,-1],[-1,0]])
                ):

    """
    param   - image dimension (width,height)
            - Z distance (from cam to the plane)
            - camera focal length (fx,fy)
            - object pix (pix_x,pix_y)
    """
    
    # D: [0.0, 0.0, 0.0, 0.0, 0.0]
    # K: [415.69101839339027, 0.0, 240.5, 0.0, 415.69101839339027, 240.5, 0.0, 0.0, 1.0]
    # R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    # P: [415.69101839339027, 0.0, 240.5, -0.0, 0.0, 415.69101839339027, 240.5, 0.0, 0.0, 0.0, 1.0, 0.0]

    # z cam to object  (camera - avg height of objects)
    z = cam_height - obj_avg_h 

    # image dimension
    dimension = np.array([width,
                    height])
    
    #image center
    pp = dimension/2 

    #camera focal length
    f = np.array([fx,fy])
    
    #object coordinate
    pix = np.array([pix_x,pix_y])

    # camera is fixed at xyz = [0,0,z] and rpy = [0,-90,0]
    cam_2_world_mat = rot_mat

    res = np.matmul(cam_2_world_mat,(pix - pp)) * (z/f)

    return res #[x,y]


class MarkerSpawner():
    
    def __init__(self):
        self.active_markers = []
    
    def spawn_marker(self, grasps):
        rospy.loginfo(f'Spawn markers....')
        for i,g in enumerate(grasps):
            world_x,world_y,obj_avg_h,angle, length, width = g
            
            pose = self.upright_pose(world_x,
                                    world_y,                                    
                                    obj_avg_h,
                                    angle,)
            
            name = f"marker_{i}"
            self.spawn(name,pose)
            self.active_markers.append(name)
    
    def spawn(self, instance_name, pose):
        pkg_path = rospack.get_path('gazebo_scene_manipulation')
        path_to_model = os.path.join(pkg_path,"models/tools/marker_v2/model.sdf")
        sdff = open(path_to_model).read()
        
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_client(
            model_name=instance_name,
            model_xml=sdff,
            robot_namespace='/marker',
            initial_pose=pose,
            reference_frame='world'
        )
    
    def despawn(self):
        
        if len(self.active_markers) > 0 :
            rospy.loginfo(f'Despawn markers....')
            
            for n in self.active_markers:
                delete_model = rospy.ServiceProxy('/gazebo/delete_model', 
                                                    DeleteModel)
                delete_model(n)

            self.active_markers = []
        
        
    def upright_pose(self,x,y,z,angle):
        q = quaternion_from_euler(0, 0, angle)
        pose = geometry_msgs.msg.Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose

class GraspBroadcaster(threading.Thread):
    
    def __init__(self, grasp, static=False):
        super(GraspBroadcaster, self).__init__()
        
        self.static = static
        if self.static:
            self.br = tf2_ros.StaticTransformBroadcaster()
        else:
            self.br = tf2_ros.TransformBroadcaster()
        
        self.grasp = grasp
        self.done = False
        rospy.loginfo('GraspBroadcaster created')
        self.start()
        #self.run()
    
    def run(self):
        rospy.loginfo('Broadcasting grasps\' tf')
        rate = rospy.Rate(10.0)
        while not self.done:            
            for i,g in enumerate(self.grasp):
                g_name = f"grasp_{i}"
                self.broadcast_tf(g,g_name)
                rate.sleep()
            
            if self.static:
                #sendTransform once
                self.done = True
        rospy.loginfo('Finish GraspBroadcasting')
    
    def stop(self):
        self.done = True
        
    def broadcast_tf(self,g,child = "grasp_me"):
      
      world_x,world_y,obj_avg_h,angle, length, width = g

      xyz = [world_x,world_y,obj_avg_h]
      
      parent = "world"
      
      t = geometry_msgs.msg.TransformStamped()
      quat = tf.transformations.quaternion_from_euler(0,0,angle)

      t.header.frame_id = parent
      t.child_frame_id = child
      t.transform.translation.x = xyz[0]
      t.transform.translation.y = xyz[1]
      t.transform.translation.z = xyz[2]
      t.transform.rotation.x = quat[0]
      t.transform.rotation.y = quat[1]
      t.transform.rotation.z = quat[2]
      t.transform.rotation.w = quat[3]

      t.header.stamp = rospy.Time.now()
      self.br.sendTransform(t)
    
    



class GraspPredictor():
    
    def __init__(self,
                 n_grasp = 3,
                 frame_width = 480,
                 frame_height = 480):
        self.n_grasp = n_grasp
        self.init_grasp_gen()
        
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def init_grasp_gen(self):
        self.device = get_device(False)
        self.cam_data = CameraData_new(
                                width=self.frame_width,
                                height=self.frame_height,
                                include_depth=1,
                                include_rgb=1,
                                output_size=480,
                                crop = True)
        
        rospy.loginfo('Loading grasp detector...')
        self.net = torch.load("/home/nsrie/working/graspgen/models/10_m_rgbd_epoch_50_thresh_44_iou_80_r_35.pt", map_location='cuda:0')
        
        rospy.loginfo('Loading dept Estimator...')
        self.dpt_estimator = DeptEstimator(mode=3, device=self.device)
    
    def depth_pred(self, rgb):    
        rospy.loginfo("Estimating depth...")
        dpt_depth = self.dpt_estimator.dpt_pred(rgb=rgb)
        dpt_depth = np.array(dpt_depth)
        rospy.loginfo("Depth estimated")
        return dpt_depth
  
    def grasp_pred(self, rgb, dpt_depth, return_single=True):
        
        dpt_depth_new = np.expand_dims(np.array(dpt_depth), axis=2)

        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, 
                                                    depth=dpt_depth_new, 
                                                    dpt_mode="dpt")

        dpt_img = np.squeeze(self.cam_data.get_depth(dpt_depth_new, dpt_mode="dpt"))
        rgb_img = self.cam_data.get_rgb(rgb, False)
        
        with torch.no_grad():
            rospy.loginfo('Predicting grasp...')
            xc = x.to(self.device)
            pred = self.net.forward(xc)

            q_img, ang_img, width_img = post_process_output(pred[0], pred[1],pred[2], pred[3])
            grasps = detect_grasps(q_img, ang_img, width_img, no_grasps=self.n_grasp)
            rospy.loginfo('Visualizing result...')
            if return_single:
                fig = plt.figure(figsize=(10, 10))
                plot_results(fig=fig,
                            rgb_img=rgb_img,
                            grasp_q_img=q_img,
                            grasp_angle_img=ang_img,
                            no_grasps=self.n_grasp,
                            depth_img= dpt_img,
                            grasp_width_img=width_img)
                
                img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
                rospy.loginfo('Grasp predicted')
                return img, grasps
            else:
                fig = plt.figure(figsize=(10, 10))
                res_list = ["original", "depth", "q",
                            "angle", "width", "grasps"]
                
                result = return_results(fig=fig,
                            rgb_img=rgb_img,
                            grasp_q_img=q_img,
                            grasp_angle_img=ang_img,
                            no_grasps=self.n_grasp,
                            depth_img= dpt_img,
                            grasp_width_img=width_img)
                
                res = { n:i for n,i in zip(res_list,result)}
                
                rospy.loginfo('Grasp predicted')
                return res, grasps



        


if __name__ == '__main__':
    pass