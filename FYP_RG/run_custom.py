import argparse
import logging
import time 
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image

from hardware.device import get_device
from models.common import post_process_output
from utils.data.camera_data import CameraData_new
from utils.visualisation.plot import plot_results, save_results


logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str,
                        help='Path to saved network to evaluate')
    parser.add_argument('--rgb_path', type=str, default='cornell/08/pcd0845r.png',
                        help='RGB Image path')
    parser.add_argument('--depth_path', type=str, default='cornell/08/pcd0845d.tiff',
                        help='Depth Image path')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--save', type=int, default=0,
                        help='Save the results')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--description', type=str, default='',
                         help='Description')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')                 

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Load image
    logging.info('Loading image...')
    rgb = None
    depth = None
    
    if args.use_rgb:
      pic = Image.open(args.rgb_path, 'r')
      rgb = np.array(pic)

    if args.use_depth:
      pic = Image.open(args.depth_path, 'r')
      depth = np.expand_dims(np.array(pic), axis=2)


    model_name = "".join(args.network.split("/")[-1].split(".")[0])
    out_dir = f"./inference/{model_name}"
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)


    f_name = args.rgb_path.split("/")[-1].split(".")[0]
    out_name = f"{out_dir}/{f_name}_{args.description}_depth_{args.use_depth}_img_result.png"
    save_path = f"{out_dir}/{f_name}"
    # Get the compute device
    device = get_device(args.force_cpu)
    
    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network, map_location=device)
    logging.info('Done')

    t1 = time.time()

    
    img_data = CameraData_new(include_depth=args.use_depth,
                              include_rgb=args.use_rgb,
                              output_size=args.input_size)
    
    #img_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb, output_size=args.input_size)

    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth, dpt_mode=args.description)

    vis = True
    with torch.no_grad():
        xc = x.to(device)
        pred = net.forward(xc)

        q_img, ang_img, width_img = post_process_output(pred[0], pred[1],
         pred[2], pred[3])
        
        dpt_img = np.squeeze(img_data.get_depth(depth, dpt_mode=args.description)) if args.use_depth else None
        if args.save:
            save_results(
                rgb_img= img_data.get_rgb(rgb, False),
                depth_img=dpt_img,
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                no_grasps=args.n_grasps,
                grasp_width_img=width_img,
                path = save_path
            )
        else:
            fig = plt.figure(figsize=(10, 10))
            plot_results(fig=fig,
                         rgb_img=img_data.get_rgb(rgb, False),
                         grasp_q_img=q_img,
                         grasp_angle_img=ang_img,
                         no_grasps=args.n_grasps,
                         depth_img= dpt_img,
                         grasp_width_img=width_img)
            if vis:
              fig.canvas.draw()
              fig.canvas.flush_events()
              plt.show()
              # img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
              # cv2.imshow("plot",img)
              # cv2.waitKey(0)
            fig.savefig(out_name)
        #cv2.destroyAllWindows()
        logging.info(f'Result is save as {out_name}')
        t2 = time.time()
        print("time:", (t2 - t1)," sec")