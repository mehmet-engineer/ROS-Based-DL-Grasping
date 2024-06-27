import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torch

import cv2
from hardware.camera_new import CustomCamera
from hardware.device import get_device
from models.common import post_process_output
from utils.data.camera_data import  CameraData_new
from utils.visualisation.plot import  plot_results
from dept_estimator.dpt_midas import DeptEstimator

log = logging.getLogger()
log.setLevel(logging.INFO)
#log.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='saved_data/cornell_rgbd_iou_0.96',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--input-size', type=int, default=480,
                        help='input_size to network')
    parser.add_argument('--description', type=str, default='',
                         help='Description')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # Get the compute device
    device = get_device(False)

    # Connect to Camera
    log.info('Connecting to camera...')
    cam = CustomCamera(device_id="http://192.168.1.110:8080/video")
    cam.connect()
    
    image_bundle = cam.get_image_bundle()
    rgb = image_bundle['rgb']
    
    
    cam_data = CameraData_new(
                            width = rgb.shape[0],
                            height = rgb.shape[1],
                            include_depth=args.use_depth,
                            include_rgb=args.use_rgb,
                            output_size=args.input_size
                            )
    log.info('Camera connected...')
    # Load Network
    logging.info('Loading grasp detector...')
    net = torch.load(args.network)
    logging.info('Loading dept Estimator...')
    dpt_estimator = DeptEstimator(mode=3, device=device)
    logging.info('Done')
    
    dpt_depth = None
    vis = True
    
    try:
        fig = plt.figure(figsize=(10, 10))
        logging.info('Starting...')
        while True:
            image_bundle = cam.get_image_bundle()
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
            
            
            k = cv2.waitKey(1)
            if k==ord('a'):    # Esc key to stop
                logging.info('Predicting Depth...')
                dpt_depth = dpt_estimator.dpt_pred(rgb=rgb)
                dpt_depth_new = np.expand_dims(np.array(dpt_depth), axis=2)
                
                x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=dpt_depth_new, dpt_mode=args.description)
                with torch.no_grad():
                    logging.info('Predicting Grasp...')
                    xc = x.to(device)
                    pred = net.forward(xc)

                    q_img, ang_img, width_img = post_process_output(pred[0], pred[1],
                    pred[2], pred[3])
                    
                    dpt_img = np.squeeze(cam_data.get_depth(dpt_depth_new, dpt_mode=args.description)) if args.use_depth else None
                    
                    fig = plt.figure(figsize=(10, 10))
                    plot_results(fig=fig,
                                rgb_img=cam_data.get_rgb(rgb, False),
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                no_grasps=args.n_grasps,
                                depth_img= dpt_img,
                                grasp_width_img=width_img)
                    if vis:
                        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
                        cv2.imshow("plot",img)
                    
                    #fig.savefig(out_name)
                    logging.info('Grasp predicted...')
            elif k==ord('q'):  # normally -1 returned,so don't print it
                break
            
            
        
            
            if dpt_depth is None:
                img = np.hstack((rgb, depth))
            else: 
                img = np.hstack((rgb, dpt_depth))        

            cv2.imshow("test",img)
    finally:
        pass
    
    cv2.destroyAllWindows()