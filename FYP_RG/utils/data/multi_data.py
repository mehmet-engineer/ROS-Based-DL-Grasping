import os
import glob

from .grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image
import numpy as np

class CornellMultiDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """
    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CornellMultiDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, '*top_annotations.txt'))
        #print(os.path.join(file_path, '*_annotations.txt'))
        self.grasp_files.sort()
        #print(len(graspf))
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        #depthf = [f.replace('annotations.txt', 'depth.tiff') for f in graspf]
        #rgbf = [f.replace('depth.tiff', 'color.png') for f in depthf]
        self.depth_files = [f.replace('annotations.txt', 'color_depthmap.png') for f in self.grasp_files]
        self.rgb_files = [f.replace('color_depthmap.png', 'color.png') for f in self.depth_files]


    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left)) # together with img crop
        gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        depth_img.img = depth_img.img[:,:,0]

        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        #depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        #if normalise:
            #rgb_img.normalise()
            #rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        rgb_img.img = np.reshape(rgb_img.img, (3,self.output_size,self.output_size))
        return rgb_img.img
