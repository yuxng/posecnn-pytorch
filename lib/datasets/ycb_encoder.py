import os, sys
import os.path
import cv2
import random
import glob
import torch
import torch.utils.data as data
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
import datasets
from fcn.config import cfg
from transforms3d.quaternions import *
from transforms3d.euler import *
from scipy.optimize import minimize
from utils.blob import pad_im, chromatic_transform, add_noise, add_noise_cuda, add_noise_depth_cuda


def get_bb3D(extent):

    orders = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])
    num = orders.shape[0]
    bb = np.zeros((3, 8 * num), dtype=np.float32)
    
    for i in range(num):
        xi = orders[i, 0]
        yi = orders[i, 1]
        zi = orders[i, 2]
        xHalf = extent[xi] * 0.5
        yHalf = extent[yi] * 0.5
        zHalf = extent[zi] * 0.5
        bb[:, i*num+0] = [xHalf, yHalf, zHalf]
        bb[:, i*num+1] = [-xHalf, yHalf, zHalf]
        bb[:, i*num+2] = [xHalf, -yHalf, zHalf]
        bb[:, i*num+3] = [-xHalf, -yHalf, zHalf]
        bb[:, i*num+4] = [xHalf, yHalf, -zHalf]
        bb[:, i*num+5] = [-xHalf, yHalf, -zHalf]
        bb[:, i*num+6] = [xHalf, -yHalf, -zHalf]
        bb[:, i*num+7] = [-xHalf, -yHalf, -zHalf]
    return bb


def optimize_depths(width, height, points, intrinsic_matrix):

    # extract 3D points
    x3d = np.ones((4, points.shape[1]), dtype=np.float32)
    x3d[:3, :] = points

    # optimization
    x0 = 2.0
    res = minimize(objective_depth, x0, args=(width, height, x3d, intrinsic_matrix), method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
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
    return np.abs(w * h - width * height)


class YCBEncoder(data.Dataset, datasets.imdb):

    def __init__(self, image_set, ycb_object_path = None):

        self._name = 'ycb_encoder_' + image_set
        self._image_set = image_set
        self._ycb_object_path = self._get_default_path() if ycb_object_path is None \
                            else ycb_object_path
        self.root_path = self._ycb_object_path

        # define all the classes
        self._classes_all = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick', 'holiday_cup1', 'holiday_cup2', 'sanning_mug', \
                         '001_chips_can')
        self._num_classes_all = len(self._classes_all)
        self._class_colors_all = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (0, 64, 0), (64, 0, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
                              (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192), (32, 0, 0)]
        self._extents_all = self._load_object_extents()

        self._width = 128
        self._height = 128
        self._intrinsic_matrix = np.array([[500, 0, 64],
                                          [0, 500, 64],
                                          [0, 0, 1]])

        # select a subset of classes
        self._classes = [self._classes_all[i] for i in cfg.TRAIN.CLASSES]
        self._num_classes = len(self._classes)
        self._class_colors = [self._class_colors_all[i] for i in cfg.TRAIN.CLASSES]
        self._extents = self._extents_all[cfg.TRAIN.CLASSES]

        # other classes as occluder
        self._classes_other = []
        for i in range(1, self._num_classes_all):
            if i not in cfg.TRAIN.CLASSES:
                # do not use clamp
                if i == 19 and 20 in cfg.TRAIN.CLASSES:
                    continue
                if i == 20 and 19 in cfg.TRAIN.CLASSES:
                    continue
                self._classes_other.append(i)
        self._num_classes_other = len(self._classes_other)

        # 3D model paths
        self.model_mesh_paths = ['{}/models/{}/textured_simple.obj'.format(self._ycb_object_path, cls) for cls in self._classes_all[1:]]
        self.model_texture_paths = ['{}/models/{}/texture_map.png'.format(self._ycb_object_path, cls) for cls in self._classes_all[1:]]
        self.model_colors = [np.array(self._class_colors_all[i]) / 255.0 for i in range(1, len(self._classes_all))]

        self.model_mesh_paths_target = ['{}/models/{}/textured_simple.obj'.format(self._ycb_object_path, cls) for cls in self._classes]
        self.model_texture_paths_target = ['{}/models/{}/texture_map.png'.format(self._ycb_object_path, cls) for cls in self._classes]
        self.model_colors_target = [np.array(self._class_colors_all[i]) / 255.0 for i in cfg.TRAIN.CLASSES]

        self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
        self._size = cfg.TRAIN.SYNNUM
        if cfg.MODE == 'TRAIN' or (cfg.MODE == 'TEST' and cfg.TEST.SYNTHESIZE == True):
            self._build_background_images()
        self._build_uniform_poses()

        assert os.path.exists(self._ycb_object_path), \
                'ycb_object path does not exist: {}'.format(self._ycb_object_path)

        # compute the canonical distance to render
        self.render_depths = np.zeros((self._extents.shape[0], ), dtype=np.float32)
        margin = 20
        for i in range(self._extents.shape[0]):
            extent = self._extents[i, :]
            points = get_bb3D(extent)
            self.render_depths[i] = optimize_depths(self._width - margin, self._height - margin, points, self._intrinsic_matrix)
        print('depth for rendering:')
        print(self.render_depths)

        self.lb_shift = -5.0
        self.ub_shift = 5.0
        self.lb_scale = 0.975
        self.ub_scale = 1.025


    def _render_item(self):

        height = cfg.TRAIN.SYN_HEIGHT
        width = cfg.TRAIN.SYN_WIDTH
        fx = self._intrinsic_matrix[0, 0]
        fy = self._intrinsic_matrix[1, 1]
        px = self._intrinsic_matrix[0, 2]
        py = self._intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.01
        qt = np.zeros((7, ), dtype=np.float32)
        image_target_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
        classes = np.array(cfg.TRAIN.CLASSES)

        # sample target object
        cls_indexes = []
        cls_target = np.random.randint(len(cfg.TRAIN.CLASSES), size=1)[0]
        cls_indexes.append(cfg.TRAIN.CLASSES[cls_target]-1)

        # sample target pose
        poses_all = []
        cls = int(cls_indexes[0])
        extent_target = np.mean(self._extents_all[cls+1, :])
        if self.pose_indexes[cls] >= len(self.pose_lists[cls]):
            self.pose_indexes[cls] = 0
            self.pose_lists[cls] = np.random.permutation(np.arange(len(self.eulers)))
        yaw = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][0] + 5 * np.random.randn()
        pitch = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][1] + 5 * np.random.randn()
        roll = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][2] + 5 * np.random.randn()
        qt[3:] = euler2quat(roll * math.pi / 180.0, pitch * math.pi / 180.0, yaw * math.pi / 180.0, 'syxz')
        self.pose_indexes[cls] += 1

        qt[0] = 0
        qt[1] = 0
        qt[2] = self.render_depths[cls_target]

        # render target with constant lighting
        poses_all.append(qt.copy())
        cfg.renderer.set_poses(poses_all)
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1.0, 1.0, 1.0])
        cfg.renderer.render(cls_indexes, image_target_tensor, seg_tensor)
        image_target_tensor = image_target_tensor.flip(0)
        image_target_tensor = image_target_tensor[:, :, (2, 1, 0)]
        seg_tensor = seg_tensor.flip(0)
        seg = torch.sum(seg_tensor[:, :, :3], dim=2)
        mask_background = (seg == 0)
        image_target_tensor[mask_background, :] = 0.5
        mask = (seg != 0).cpu().numpy()

        if np.random.rand(1) < 0.3:
            # sample an occluder
            cls_indexes.append(0)
            poses_all.append(np.zeros((7, ), dtype=np.float32))
            while 1:

                ind = np.random.randint(self._num_classes_other, size=1)[0]
                cls_occ = self._classes_other[ind]
                cls_indexes[1] = cls_occ - 1

                # sample poses
                cls = int(cls_indexes[1])
                if self.pose_indexes[cls] >= len(self.pose_lists[cls]):
                    self.pose_indexes[cls] = 0
                    self.pose_lists[cls] = np.random.permutation(np.arange(len(self.eulers)))
                yaw = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][0] + 5 * np.random.randn()
                pitch = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][1] + 5 * np.random.randn()
                roll = self.eulers[self.pose_lists[cls][self.pose_indexes[cls]]][2] + 5 * np.random.randn()
                qt[3:] = euler2quat(roll * math.pi / 180.0, pitch * math.pi / 180.0, yaw * math.pi / 180.0, 'syxz')
                self.pose_indexes[cls] += 1

                # translation, sample an object nearby
                object_id = 0
                extent = np.maximum(np.mean(self._extents_all[cls+1, :]), extent_target)

                flag = np.random.randint(0, 2)
                if flag == 0:
                    flag = -1
                qt[0] = poses_all[object_id][0] + flag * extent * np.random.uniform(0.25, 0.5)

                flag = np.random.randint(0, 2)
                if flag == 0:
                    flag = -1
                qt[1] = poses_all[object_id][1] + flag * extent * np.random.uniform(0.25, 0.5)

                qt[2] = poses_all[object_id][2] - extent * np.random.uniform(1.0, 2.0)
                poses_all[1] = qt
                cfg.renderer.set_poses(poses_all)

                # rendering
                cfg.renderer.set_light_pos(np.random.uniform(-0.5, 0.5, 3))
                intensity = np.random.uniform(0.8, 2)
                light_color = intensity * np.random.uniform(0.9, 1.1, 3)
                cfg.renderer.set_light_color(light_color)
                cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)

                seg_tensor = seg_tensor.flip(0)
                im_label = seg_tensor.cpu().numpy()
                im_label = im_label[:, :, (2, 1, 0)] * 255
                im_label = np.round(im_label).astype(np.uint8)
                im_label = np.clip(im_label, 0, 255)
                im_label_only, im_label = self.process_label_image(im_label)

                # compute occlusion percentage
                mask_target = (im_label == cls_indexes[0]+1).astype(np.int32)

                per_occ = 1.0 - np.sum(mask & mask_target) / np.sum(mask)
                if per_occ < 0.5:
                    break
        else:
            # rendering
            cfg.renderer.set_light_pos(np.random.uniform(-0.5, 0.5, 3))
            intensity = np.random.uniform(0.8, 2)
            light_color = intensity * np.random.uniform(0.9, 1.1, 3)
            cfg.renderer.set_light_color(light_color)
            cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)

            seg_tensor = seg_tensor.flip(0)
            im_label = seg_tensor.cpu().numpy()
            im_label = im_label[:, :, (2, 1, 0)] * 255
            im_label = np.round(im_label).astype(np.uint8)
            im_label = np.clip(im_label, 0, 255)
            im_label_only, im_label = self.process_label_image(im_label)

        # RGB to BGR order
        image_tensor = image_tensor.flip(0)
        im = image_tensor.cpu().numpy()
        im = np.clip(im, 0, 1)
        im = im[:, :, (2, 1, 0)] * 255
        im = im.astype(np.uint8)

        # affine transformation
        shift = np.float32([np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)])
        scale = np.random.uniform(self.lb_scale, self.ub_scale)
        affine_matrix = np.float32([[scale, 0, shift[0] / width], [0, scale, shift[1] / height]])

        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)

        im_cuda = torch.from_numpy(im).cuda().float() / 255.0
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im_cuda = add_noise_cuda(im_cuda)
        im_cuda = torch.clamp(im_cuda, min=0.0, max=1.0)

        # im is pytorch tensor in gpu
        sample = {'image_input': im_cuda.permute(2, 0, 1),
                  'image_target': image_target_tensor.permute(2, 0, 1),
                  'image_label': torch.from_numpy(im_label).unsqueeze(2).repeat(1, 1, 3).permute(2, 0, 1).float().cuda(),
                  'affine_matrix': torch.from_numpy(affine_matrix).cuda()}

        return sample


    def __getitem__(self, index):

        return self._render_item()


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where ycb_object is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'YCB_Object')


    def _load_object_extents(self):

        extent_file = os.path.join(self._ycb_object_path, 'extents.txt')
        assert os.path.exists(extent_file), \
                'Path does not exist: {}'.format(extent_file)

        extents = np.zeros((self._num_classes_all, 3), dtype=np.float32)
        extents[1:, :] = np.loadtxt(extent_file)

        return extents


    def process_label_image(self, label_image):
        """
        change label image to label index
        """
        height = label_image.shape[0]
        width = label_image.shape[1]
        labels = np.zeros((height, width), dtype=np.int32)
        labels_all = np.zeros((height, width), dtype=np.int32)

        # label image is in BGR order
        index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
        for i in xrange(1, len(self._class_colors_all)):
            color = self._class_colors_all[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            labels_all[I[0], I[1]] = i

            ind = np.where(np.array(cfg.TRAIN.CLASSES) == i)[0]
            if len(ind) > 0:
                labels[I[0], I[1]] = ind+1

        return labels, labels_all
