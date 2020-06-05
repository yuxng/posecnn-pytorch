import torch
import torch.utils.data as data

import os, math
import sys
import os.path as osp
from os.path import *
import numpy as np
import numpy.random as npr
import cv2
try:
    import cPickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle as cPickle
import scipy.io
import glob

import datasets
from fcn.config import cfg
from utils.blob import pad_im, chromatic_transform, add_noise, add_noise_cuda, add_noise_depth_cuda
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import euler2quat
from utils.se3 import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def get_bb3D(points):

    num = points.shape[1]
    bb_all = np.zeros((3, num * 864), dtype=np.float32)
    count = 0

    interval = 30
    for yaw in range(-180, 180, interval):
        for pitch in range(-90, 90, interval):
            for roll in range(-180, 180, interval):
                qt = euler2quat(roll * math.pi / 180.0, pitch * math.pi / 180.0, yaw * math.pi / 180.0, 'syxz')
                R = quat2mat(qt)
                bb_all[:, num*count:num*count+num] = np.matmul(R, points)
                count += 1

    return bb_all


def optimize_depths(width, height, points, intrinsic_matrix):

    # extract 3D points
    x3d = np.ones((4, points.shape[1]), dtype=np.float32)
    x3d[:3, :] = points

    # optimization
    x0 = 2.0
    res = minimize(objective_depth, x0, args=(width, height, x3d, intrinsic_matrix), method='nelder-mead', options={'xtol': 1e-1, 'disp': False})
    return res.x


def objective_depth(x, width, height, x3d, intrinsic_matrix):

    # project points
    RT = np.zeros((3, 4), dtype=np.float32)
    RT[:3, :3] = np.eye(3, dtype=np.float32)
    RT[2, 3] = x
    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

    roi_pred = np.zeros((4, ), dtype=np.float32)
    roi_pred[0] = np.min(x2d[0, :])
    roi_pred[1] = np.min(x2d[1, :])
    roi_pred[2] = np.max(x2d[0, :])
    roi_pred[3] = np.max(x2d[1, :])
    w = roi_pred[2] - roi_pred[0]
    h = roi_pred[3] - roi_pred[1]

    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(x2d[0, :], x2d[1, :])
    print(width, height)
    print(w, h)
    plt.show()
    #'''

    return np.abs(w - width)
    # return np.abs(w * h - width * height)


class DOCSObject(data.Dataset, datasets.imdb):
    def __init__(self, image_set, docs_object_path = None):

        self._name = 'docs_object_' + image_set
        self._image_set = image_set
        self._docs_object_path = self._get_default_path() if docs_object_path is None \
                            else docs_object_path
        self._data_path = os.path.join(self._docs_object_path, 'data')
        self._model_path = os.path.join(datasets.ROOT_DIR, 'data', 'models')
        self.root_path = self._docs_object_path
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).cuda().float()
        self._classes_all = ('__background__', 'foreground')
        self._classes = self._classes_all

        # list all the 3D models
        self.model_mesh_paths = []
        self.model_texture_paths = []
        self.model_scales = []

        subdirs = os.listdir(self._model_path)
        for i in range(len(subdirs)):
            subdir = subdirs[i]
            if subdir == 'industrial_dolly' or subdir == 'fusion_duplo_dude' or 'block' in subdir:
                continue
            filename = os.path.join(self._model_path, subdir, 'textured_simple.obj')
            if not os.path.exists(filename):
                filename = os.path.join(self._model_path, subdir, 'google_16k', 'textured_simple.obj')
            self.model_mesh_paths.append(filename)
            self.model_texture_paths.append('')
            self.model_scales.append(1.0)

        self.model_num = len(self.model_mesh_paths)
        self.model_list = np.random.permutation(self.model_num)
        self.model_index = 0
        print('%d 3D models' % (self.model_num))

        # random model colors
        self.model_colors = []
        for i in range(self.model_num):
            self.model_colors.append(np.random.uniform(size=3))

        self._width = cfg.TRAIN.SYN_WIDTH
        self._height = cfg.TRAIN.SYN_HEIGHT
        self._intrinsic_matrix = np.array([[524.7917885754071, 0, 332.5213232846151],
                                          [0, 489.3563960810721, 281.2339855172282],
                                          [0, 0, 1]])
        self._size = cfg.TRAIN.SYNNUM
        self._build_uniform_poses()
        self.render_depths = None

        self.lb_shift = -0.1
        self.ub_shift = 0.1
        self.lb_scale = 0.8
        self.ub_scale = 1.2

        assert os.path.exists(self._docs_object_path), \
                'docs_object path does not exist: {}'.format(self._docs_object_path)


    def compute_render_depths(self, renderer):

        cache_file = os.path.join(self.cache_path, self.name + '_render_depths.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                render_depths = cPickle.load(fid)
            print('{} render_depths loaded from {}'.format(self.name, cache_file))
            print(render_depths)
            self.render_depths = render_depths
            return render_depths

        print('computing canonical depths')
        extents = renderer.extents
        vertices = renderer.vertices
        render_depths = np.zeros((len(extents), ), dtype=np.float32)
        for i in range(len(extents)):
            vertices = renderer.vertices[i]
            perm = np.random.permutation(np.arange(vertices.shape[0]))
            index = perm[:3000]
            pcloud = vertices[index, :]
            points = get_bb3D(np.transpose(pcloud))
            render_depths[i] = abs(optimize_depths(self._width / 4, self._height / 4, points, self._intrinsic_matrix))
            print(self.model_mesh_paths[i], render_depths[i])

        with open(cache_file, 'wb') as fid:
            cPickle.dump(render_depths, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote render_depths to {}'.format(cache_file))
        self.render_depths = render_depths


    def _render_item(self):

        height = cfg.TRAIN.SYN_HEIGHT
        width = cfg.TRAIN.SYN_WIDTH
        fx = self._intrinsic_matrix[0, 0]
        fy = self._intrinsic_matrix[1, 1]
        px = self._intrinsic_matrix[0, 2]
        py = self._intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.01

        # sample a target object
        if self.model_index >= self.model_num:
            self.model_index = 0
            self.model_list = np.random.permutation(self.model_num)
        cls_index = self.model_list[self.model_index]
        self.model_index += 1
        render_depth = abs(self.render_depths[int(cls_index)])

        # render two poses
        num = 2
        image_blob = torch.cuda.FloatTensor(num, 3, height, width).fill_(0)
        mask_blob = torch.cuda.FloatTensor(num, 3, height, width).fill_(0)
        label_blob = torch.cuda.FloatTensor(num, 2, height, width).fill_(0)
        affine_blob = torch.cuda.FloatTensor(num, 2, 3).fill_(0)
        for i in range(num):

            while 1:
                poses_all = []
                qt = np.zeros((7, ), dtype=np.float32)
                # rotation
                cls = 0
                if self.pose_indexes[cls] >= len(self.pose_lists[cls]):
                    self.pose_indexes[cls] = 0
                    self.pose_lists[cls] = np.random.permutation(np.arange(len(self.eulers)))
                yaw = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][0] + 15 * np.random.randn()
                pitch = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][1] + 15 * np.random.randn()
                pitch = np.clip(pitch, -90, 90)
                roll = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][2] + 15 * np.random.randn()
                qt[3:] = euler2quat(yaw * math.pi / 180.0, pitch * math.pi / 180.0, roll * math.pi / 180.0, 'syxz')
                self.pose_indexes[cls] += 1

                # translation
                bound = cfg.TRAIN.SYN_BOUND
                qt[0] = np.random.uniform(-bound, bound)
                qt[1] = np.random.uniform(-bound, bound)
                qt[2] = np.random.uniform(0.4 * render_depth, 0.8 * render_depth)
                poses_all.append(qt)

                cfg.renderer.set_poses(poses_all)

                # sample lighting
                cfg.renderer.set_light_pos(np.random.uniform(-0.5, 0.5, 3))
                intensity = np.random.uniform(0.8, 2)
                light_color = intensity * np.random.uniform(0.9, 1.1, 3)
                cfg.renderer.set_light_color(light_color)
            
                # rendering
                cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
                image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
                seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
                pc_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
                cfg.renderer.render([cls_index], image_tensor, seg_tensor, pc2_tensor=pc_tensor)
                image_tensor = image_tensor.flip(0)
                seg_tensor = seg_tensor.flip(0)
                pc_tensor = pc_tensor.flip(0)

                # foreground mask
                seg = seg_tensor[:,:,2] + 256*seg_tensor[:,:,1] + 256*256*seg_tensor[:,:,0]
                mask = (seg != 0).unsqueeze(0).repeat((3, 1, 1)).float()
                mask_blob[i] = mask

                index = torch.nonzero(mask[0])
                if index.shape[0] > 10:
                    break

            # RGB to BGR order
            im = image_tensor.cpu().numpy()
            im = np.clip(im, 0, 1)
            im = im[:, :, (2, 1, 0)] * 255
            im = im.astype(np.uint8)

            # chromatic transform
            if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
                im = chromatic_transform(im)

            im_cuda = torch.from_numpy(im).cuda().float() / 255.0
            if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
                im_cuda = add_noise_cuda(im_cuda)
            im_cuda -= self._pixel_mean
            im_cuda = im_cuda.permute(2, 0, 1)
            image_blob[i] = im_cuda

            # binary label blob
            label_blob[i, 0, :, :] = 1.0
            label_blob[i, 1, :, :] = 0.0
            index = torch.nonzero(mask[0])
            if index.shape[0] > 0:
                label_blob[i, 1, index[:, 0], index[:, 1]] = 1.0
                label_blob[i, 0, index[:, 0], index[:, 1]] = 0.0

            # affine transformation
            shift = np.float32([np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)])
            scale = np.random.uniform(self.lb_scale, self.ub_scale)
            affine_matrix = np.float32([[scale, 0, shift[0]], [0, scale, shift[1]]])
            affine_blob[i] = torch.from_numpy(affine_matrix).cuda()

            '''
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(im[:, :, (2, 1, 0)])
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(label_blob[i, 1].cpu().numpy())
            plt.show()
            #'''

        sample = {'image_color': image_blob,
                  'label': label_blob,
                  'mask': mask_blob,
                  'affine': affine_blob}

        return sample


    def __getitem__(self, index):

        is_syn = 0
        if ((cfg.MODE == 'TRAIN' and cfg.TRAIN.SYNTHESIZE) or (cfg.MODE == 'TEST' and cfg.TEST.SYNTHESIZE)) and \
           (cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD' or (index % cfg.TRAIN.SYN_RATIO != 0)):
            is_syn = 1

        is_syn = 1

        if is_syn:
            return self._render_item()
        else:
            return self._compose_item()


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where docs_object is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'YCB_Object')
