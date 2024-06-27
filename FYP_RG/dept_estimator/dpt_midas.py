
"""
credit to https://colab.research.google.com/github/olaviinha/NeuralDepthPrediction/blob/main/MonocularDepthMapPrediction_timm.ipynb

"""

import sys
 # setting path
# sys.path.append('/home/nsrie/working/graspgen/FYP_RG')

import os
import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from hardware.device import get_device
import logging
import cv2

logging.basicConfig(level=logging.INFO)


class DeptEstimator:
    def __init__(self,
                 mode=1,
                 device="cpu",
                 ):
      self.mode = mode
      self.device = device
      
      self.load_dpt_model()

    def load_dpt_model(self):
        #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        
        model_type_dict = {3:"DPT_Large",
                      2:"DPT_Hybrid",
                      1:"MiDaS_small"}
        
        model_type = model_type_dict[self.mode]
        
        self.midas = torch.hub.load("intel-isl/MiDaS",model_type)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.midas.to(self.device)
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        logging.info('MIDAS loaded..')


    def dpt_pred(self, out_path ='./dept_results',
                rgb=None, save=False, vis=False):
      
      if not os.path.exists(out_path):
          os.mkdir(out_path)
          
      input = rgb
      auto_adjust = True
      
      if isinstance(rgb,str):
        rgb_name = rgb.split("/")[-1].split(".")[0]
        out_path = os.path.join(out_path,rgb_name+'_depthmap.png')
        
        img = cv2.imread(input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      else:
          img = input
      
      input_batch = self.transform(img).to(self.device)
      with torch.no_grad():
        prediction = self.midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

      output = prediction.cpu().numpy()
      if auto_adjust is True:
        output = ((output - output.min()) / (output.max()-output.min())) * 255
        output = output.astype(np.uint8)

      im = Image.fromarray(~output)
      im = im.convert('RGB')
      
      if vis:
        plt.imshow(im)
        plt.show()
      
      if save and isinstance(rgb,str):
        im.save(out_path)

        if os.path.isfile(out_path) is True:
            logging.info(f'Depth map saved as {out_path}')
        else:
            logging.info('Error occurred.')
      
      else:
          return im




if __name__ == '__main__':
    device = get_device(False)
    dpt_estimator = DeptEstimator(mode=1, device=device)
    im = dpt_estimator.dpt_pred(
             out_path ='./dept_results',
             rgb="/home/nsrie/working/graspgen/new_dataset/42_top_color.png",#./experiment/cup-gd80a1f031_1920.jpg",
             save=True,
             vis=True)
