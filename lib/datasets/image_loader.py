import torch
import yaml
import glob
import os
import numpy as np
import cv2
import scipy.io as sio
import obj
import posecnn_cuda
from scipy.spatial.transform import Rotation as Rot
from fcn.config import cfg

class ImageLoader():

    def __init__(self, images_color, images_depth,
                 device='cuda:0',
                 cap_depth=False):
        """ TODO(ywchao): complete docstring.
        Args:
        device: A torch.device string argument. The specified device is used only
          for certain data loading computations, but not storing the loaded data.
          Currently the loaded data is always stored as numpy arrays on cpu.
         """
        assert device in ('cuda', 'cpu') or device.split(':')[0] == 'cuda'
        self._images_color = images_color
        self._images_depth = images_depth
        self._device = torch.device(device)
        self._cap_depth = cap_depth

        self._num_frames = len(images_color)
        self._h = cfg.TRAIN.SYN_HEIGHT
        self._w = cfg.TRAIN.SYN_WIDTH
        self._depth_bound = 20.0

        # tex coord
        y, x = torch.meshgrid(torch.arange(self._h), torch.arange(self._w))
        x = x.float()
        y = y.float()
        s = torch.stack((x / (self._w - 1), y / (self._h - 1)), dim=2)
        self._pcd_tex_coord = [s.numpy()]

        # colored point cloud 
        self._pcd_rgb = [np.zeros((self._h, self._w, 3), dtype=np.uint8)]
        self._pcd_vert = [np.zeros((self._h, self._w, 3), dtype=np.float32)]
        self._pcd_mask = [np.zeros((self._h, self._w), dtype=np.bool)]
        self._frame = 0
        self._num_cameras = 1

        self._intrinsic_matrix = np.array([[524.7917885754071, 0, 332.5213232846151],
                                          [0, 489.3563960810721, 281.2339855172282],
                                          [0, 0, 1]])
        self._master_intrinsics = self._intrinsic_matrix

    def load_frame_rgbd(self, i):
        color_file = self._images_color[i]
        color = cv2.imread(color_file)
        color = color[:, :, ::-1]
        depth_file = self._images_depth[i]
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        print(color_file)
        print(depth_file)
        return color, depth

    def deproject_depth_and_filter_points(self, depth):

        # backproject depth
        depth = depth.astype(np.float32) / 1000.0
        depth = torch.from_numpy(depth).to(self._device)
        fx = self._intrinsic_matrix[0, 0]
        fy = self._intrinsic_matrix[1, 1]
        px = self._intrinsic_matrix[0, 2]
        py = self._intrinsic_matrix[1, 2]
        im_pcloud = posecnn_cuda.backproject_forward(fx, fy, px, py, depth)[0]

        m = depth < self._depth_bound
        p = im_pcloud.cpu().numpy()
        m = m.cpu().numpy()
        return p, m

    def step(self):
        self._frame = (self._frame + 1) % self._num_frames
        self.update_pcd()

    def update_pcd(self):
        rgb, d = self.load_frame_rgbd(self._frame)
        p, m = self.deproject_depth_and_filter_points(d)
        self._pcd_rgb[0] = rgb
        self._pcd_vert[0] = p
        self._pcd_mask[0] = m

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def num_cameras(self):
        return self._num_cameras

    @property
    def dimensions(self):
        return self._w, self._h

    @property
    def master_intrinsics(self):
        return self._master_intrinsics

    @property
    def pcd_rgb(self):
        return self._pcd_rgb

    @property
    def pcd_vert(self):
        return self._pcd_vert

    @property
    def pcd_tex_coord(self):
        return self._pcd_tex_coord

    @property
    def pcd_mask(self):
        return self._pcd_mask
