import cv2
from sensor_msgs.msg import Image,CompressedImage
from cv_bridge import CvBridge
import numpy as np
import rospy
import threading

from std_msgs.msg import Float32MultiArray
from grasp.grasp_utils import img_2_world, GraspPredictor
from fyp.srv import RobotServer, RobotServerRequest

class ROSBackend():
    
    def __init__(self,
                 debug_bool=False,
                 gazebo=True):
        
        self.queue = {"object":[],
                      "scene":[],}
        self.obj_avg_h = 0.678169
        self.prev = {"object":np.zeros((480,480,3), np.uint8) ,
                      "scene":np.zeros((480,480,3), np.uint8),}
        self.gazebo = gazebo
        
        if self.gazebo :
            self.sub = rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.callback_object)
            self.sub = rospy.Subscriber('/cam_scene/rgb/image_raw/compressed', CompressedImage, self.callback_scene)
        else:
            self.cam_object = cv2.VideoCapture(id=0)
            self.cam_scene = self.cam_object #cv2.VideoCapture(id=0)
            _, cam_frame = self.cam_object.read()
            
        self.robot_client = rospy.ServiceProxy('/robot_server', RobotServer)
        
        
        if not debug_bool:
            self.grasp_predictor = GraspPredictor(n_grasp=5,
                                                frame_width = cam_frame.shape[0] if not gazebo else 480,
                                                frame_height = cam_frame.shape[1] if not gazebo else 480)
            
        self.current_frame = None
        self.results = None
        
        if debug_bool:
            self.curr_n_grasps = [0,1,2,3] #None
            self.robot_poses = ["home","stand","pregrasp"] #None
            self.current_joint = [0,1,2,3]
        else:
            self.curr_n_grasps = None
            self.robot_poses = None
            self.current_joint = None
    
    def callback_object(self,image):
        
        np_arr = np.frombuffer(image.data, np.uint8)
        self.current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.queue["object"].append(self.current_frame)
    
    def callback_scene(self,image):
        
        np_arr = np.frombuffer(image.data, np.uint8)
        self.current_scene = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.queue["scene"].append(self.current_scene)
        
    def cam_cv2(self):
        _, self.current_frame = self.cam_object.read()
        self.queue["object"].append(self.current_frame)
        
        _, self.current_scene = self.cam_scene.read()
        self.queue["scene"].append(self.current_scene)
    
    def get_frame(self, feed):
        
        if not self.gazebo: self.cam_cv2()
        
        if not self.queue[feed]:
            frame = self.prev[feed]
        else:
            frame = self.queue[feed].pop(0)
            self.prev[feed] = frame

        return cv2.imencode('.jpg', frame)[1].tobytes()
    
    def send_grasp_command(self, which_grasp):
        
        self.g_world = self.grasp_2_world(self.grasps)
        
        # sent only selected grasp
        selected_g_world = self.g_world[which_grasp]
        
        msg = RobotServerRequest()
        msg.command = "grasp"
        msg.data = selected_g_world
        result = self.robot_client(msg)
        
        status = result.status
        
        return status
    
    def send_start_command(self):
        msg = RobotServerRequest()
        msg.command = "start"
        result = self.robot_client(msg)
        
        status = result.status
        poses = result.extra
        pose_list = poses.split("\n")
        self.current_joint = result.data
        
        self.robot_poses = pose_list
        
        return status
    
    
    def send_pose_command(self, pose_idx):
        msg = RobotServerRequest()
        msg.command = "pose"
        msg.data = [pose_idx]
        result = self.robot_client(msg)
        
        status = result.status
        self.current_joint = result.data
        return status
    
    def send_joint_command(self, joint_val):
        
        joint_val = [np.deg2rad(i) for i in joint_val]
        
        msg = RobotServerRequest()
        msg.command = "joint"
        msg.data = joint_val
        result = self.robot_client(msg)
        
        status = result.status
        self.current_joint = result.data
        
        return status


    def predict_grasps(self):
        rgb = self.current_frame
        dpt_depth = self.grasp_predictor.depth_pred(rgb)
        self.results, self.grasps = self.grasp_predictor.grasp_pred(rgb, dpt_depth, return_single=False)
        self.curr_n_grasps = [ i for i in range(len(self.grasps)) ]

    def reset_grasp(self):
        self.curr_n_grasps = None
        self.results = None

    def grasp_2_world(self,grasps):
        
        """
        Transform pixel coordinate to world position
        
        return list  [world_x, world_y, obj_avg_h, angle, length, width]
        """
        
        obj_avg_h = self.obj_avg_h
        g_world = []
        for g in grasps:
            pix_x, pix_y, angle, length, width = g.center[1],g.center[0],g.angle,g.length,g.width
            #rospy.loginfo(f'Grasp detected{pix_x,pix_y,angle,length,width}')
            world_x,world_y = img_2_world(pix_x, pix_y, 
                                        obj_avg_h,
                                        cam_height = 1.5,
                                        fx = 415.69101839339027,
                                        fy = 415.69101839339027,
                                        rot_mat = np.array([[0,-1],[-1,0]])
                                        ) # --> x,y
            
            g_world.append([world_x,world_y,obj_avg_h,angle, length, width])
        
        return g_world