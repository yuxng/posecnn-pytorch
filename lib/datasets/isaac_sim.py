from __future__ import print_function

import torch
import torch.utils.data as data

import os, math
import sys
import os.path as osp
from os.path import *
import numpy as np
import numpy.random as npr
import cv2


import scipy.io

import datasets
from fcn.config import cfg
from utils.blob import pad_im, chromatic_transform, add_noise, add_noise_cuda
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import euler2quat
from utils.se3 import *

from datasets.isaac_sim_streamer import IsaacSimStreamer, processed_q


# The following params should match with IsaacSim.
_HORIZONTAL_FOV = 60.
_VERTICAL_FOV = 47.

def get_projection_matrix():
    width = 640.
    height = 480.
    f_x = width / (2 * math.tan(np.radians(_HORIZONTAL_FOV / 2)))
    f_y = height / (2 * math.tan(np.radians(_VERTICAL_FOV / 2)))
    c_x = width / 2
    c_y = height / 2
    output = np.eye(3, dtype=np.float32)
    output[0][0] = f_x
    output[1][1] = f_y
    output[0][2] = c_x
    output[1][2] = c_y

    return output


class IsaacSim(data.Dataset, datasets.imdb):
    def __init__(self, image_set, ycb_object_path = None):

        self._name = 'isaac_sim_' + image_set
        self._image_set = image_set
        self._ycb_object_path = self._get_default_path() if ycb_object_path is None \
                            else ycb_object_path
        self.root_path = self._ycb_object_path

        # define all the classes
        self._classes_all = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick', 'holiday_cup1', 'holiday_cup2', 'sanning_mug')
        self._num_classes_all = len(self._classes_all)
        self._class_colors_all = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (0, 64, 0), (64, 0, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
                              (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192)]
        self._isaac_sim_classes = ['__background__', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', '010_potted_meat_can', 'sanning_mug', '024_bowl']
        
        self._extents_all = self._load_object_extents()

        self._width = 640
        self._height = 480
        self._intrinsic_matrix = get_projection_matrix()

        # select a subset of classes
        self._classes = [self._classes_all[i] for i in cfg.TRAIN.CLASSES]
        
        self._isaac_to_all_index = {}
        self._isaac_to_classes_index = {}
        for cindex, selected_class in enumerate(self._classes):
            all_index = self._classes_all.index(selected_class)
            isaac_index = self._isaac_sim_classes.index(selected_class)
            assert(isaac_index != -1, '{} is not supported in IsaacSim yet'.format(selected_class))
            self._isaac_to_all_index[isaac_index] = all_index
            self._isaac_to_classes_index[isaac_index] = cindex

            
        self._num_classes = len(self._classes)
        self._class_colors = [self._class_colors_all[i] for i in cfg.TRAIN.CLASSES]
        self._symmetry = np.array(cfg.TRAIN.SYMMETRY).astype(np.float32)
        self._extents = self._extents_all[cfg.TRAIN.CLASSES]
        self._points, self._points_all, self._point_blob = self._load_object_points()
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).cuda().float()

        # 3D model paths
        self.model_mesh_paths = ['{}/models/{}/textured_simple.obj'.format(self._ycb_object_path, cls) for cls in self._classes_all[1:]]
        self.model_texture_paths = ['{}/models/{}/texture_map.png'.format(self._ycb_object_path, cls) for cls in self._classes_all[1:]]
        self.model_colors = [np.array(self._class_colors_all[i]) / 255.0 for i in range(1, len(self._classes_all))]

        self.model_mesh_paths_target = ['{}/models/{}/textured_simple.obj'.format(self._ycb_object_path, cls) for cls in self._classes[1:]]
        self.model_texture_paths_target = ['{}/models/{}/texture_map.png'.format(self._ycb_object_path, cls) for cls in self._classes[1:]]
        self.model_colors_target = [np.array(self._class_colors_all[i]) / 255.0 for i in cfg.TRAIN.CLASSES[1:]]

        self._size = cfg.TRAIN.SYNNUM
        
        assert(os.path.exists(self._ycb_object_path), \
               'ycb_object path does not exist: {}'.format(self._ycb_object_path))
        assert(cfg.TRAIN.SYN_HEIGHT == 480)
        assert(cfg.TRAIN.SYN_WIDTH == 640)

        height = cfg.TRAIN.SYN_HEIGHT
        width = cfg.TRAIN.SYN_WIDTH

        self._listener = IsaacSimStreamer(
            [np.uint8, np.uint8, np.float32], [[height, width, 3], [height, width], [len(self._isaac_sim_classes) - 1, 4, 4]], processed_q)
        self._listener.start()


    def _remap_poses(self, input_poses, index_dict):
        poses = np.concatenate([np.zeros((1, 4, 4), dtype=np.float32), input_poses], axis=0)
        #('remap_poses {}'.format(poses.shape))
        output = np.zeros(poses.shape, poses.dtype)
        for isaac_index, ycb_index in index_dict.items():
            output[ycb_index, :, :] = poses[isaac_index, :, :]
        
        return output.copy()[:1, :, :]


    def _remap_segmentation_image(self, img, index_dict):
        assert(len(img.shape) == 2)
        output = np.zeros(img.shape, dtype=np.uint8)
        class_present = [0. for _ in range(self.num_classes)]
        for isaac_index, ycb_index in index_dict.items():
            if isaac_index == 0:
                continue

            mask = img == isaac_index
            visible = 0.
            if np.any(mask):
                output[mask] = ycb_index
                visible=1.
            
            class_present[isaac_index] = visible

        class_present[0] = 0 # remove background category from visible categories.

        
        return output, class_present
    

    def _get_centers(self, RT):
        T = np.reshape(RT[:3, 3], [3, 1])
        proj = np.matmul(self._intrinsic_matrix, T)

        return [proj[0]/proj[2], proj[1]/proj[2]]
    

    def _render_item(self):
        fx = self._intrinsic_matrix[0, 0]
        fy = self._intrinsic_matrix[1, 1]
        px = self._intrinsic_matrix[0, 2]
        py = self._intrinsic_matrix[1, 2]
        

        isaac_data = self._listener.get_training_data() # (image, segmentation, poses nx4x4)
        im = isaac_data[0]
        im_label, visible_classes = self._remap_segmentation_image(isaac_data[1], self._isaac_to_classes_index)
        poses = isaac_data[2]
        centers = np.asarray([self._get_centers(RT) for RT in poses], dtype=np.float32)
        
        
        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)

        im_cuda = torch.from_numpy(im).cuda().float() / 255.0
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im_cuda = add_noise_cuda(im_cuda, cfg.TRAIN.NOISE_LEVEL)
        im_cuda -= self._pixel_mean
        im_cuda = im_cuda.permute(2, 0, 1)

        # label blob
        classes = np.array(range(self.num_classes))
        label_blob = np.zeros((self.num_classes, self._height, self._width), dtype=np.float32)
        label_blob[0, :, :] = 1.0
        for i in range(1, self.num_classes):
            I = np.where(im_label == classes[i])
            if len(I[0]) > 0:
                label_blob[i, I[0], I[1]] = 1.0
                label_blob[0, I[0], I[1]] = 0.0

        # poses and boxes
        pose_blob = np.zeros((self.num_classes, 9), dtype=np.float32)
        gt_boxes = np.zeros((self.num_classes, 5), dtype=np.float32)
        indexes_target = np.where(visible_classes)[0]
        #print('indexes_target = {}'.format(indexes_target))
        #print('visible_classes = {}'.format(visible_classes))


        poses_all = []
        remapped_centers = []

        for i, isaac_cls in enumerate(indexes_target):
            cls = self._isaac_to_classes_index[isaac_cls]
            pose_blob[i, 0] = 1
            pose_blob[i, 1] = cls

            T = poses[isaac_cls - 1][:3, 3]
            q = mat2quat(poses[isaac_cls - 1][:3, :3])
            
            full_pose = np.zeros((7,), dtype=np.float32)
            full_pose[:3] = T
            full_pose[3:] = q
            poses_all.append(full_pose.copy())

            # egocentric to allocentric
            q_allocentric = egocentric2allocentric(q, T)
            if q_allocentric[0] < 0:
              q_allocentric = -1 * q_allocentric
            pose_blob[i, 2:6] = q_allocentric
            pose_blob[i, 6:] = T.copy() 

            # compute box
            x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
            x3d[0, :] = self._points_all[cls,:,0]
            x3d[1, :] = self._points_all[cls,:,1]
            x3d[2, :] = self._points_all[cls,:,2]
            RT = poses[isaac_cls - 1][:3,:]
            x2d = np.matmul(self._intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        
            gt_boxes[i, 0] = np.min(x2d[0, :])
            gt_boxes[i, 1] = np.min(x2d[1, :])
            gt_boxes[i, 2] = np.max(x2d[0, :])
            gt_boxes[i, 3] = np.max(x2d[1, :])
            gt_boxes[i, 4] = cls

            remapped_centers.append(centers[isaac_cls - 1])


        remapped_centers = np.asarray(remapped_centers)

        # construct the meta data
        """
        format of the meta_data
        intrinsic matrix: meta_data[0 ~ 8]
        inverse intrinsic matrix: meta_data[9 ~ 17]
        """
        K = self._intrinsic_matrix
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros(18, dtype=np.float32)
        meta_data_blob[0:9] = K.flatten()
        meta_data_blob[9:18] = Kinv.flatten()

        # vertex regression target
        if cfg.TRAIN.VERTEX_REG:
            vertex_targets, vertex_weights = self._generate_vertex_targets(im_label, indexes_target, remapped_centers, poses_all, classes, self.num_classes)
        else:
            vertex_targets = []
            vertex_weights = []

        im_info = np.array([im.shape[1], im.shape[2], cfg.TRAIN.SCALES_BASE[0]], dtype=np.float32)

        sample = {'image': im_cuda,
                  'label': label_blob,
                  'meta_data': meta_data_blob,
                  'poses': pose_blob,
                  'extents': self._extents,
                  'points': self._point_blob,
                  'symmetry': self._symmetry,
                  'gt_boxes': gt_boxes,
                  'im_info': im_info}

        if cfg.TRAIN.VERTEX_REG:
            sample['vertex_targets'] = vertex_targets
            sample['vertex_weights'] = vertex_weights

        return sample



    def __getitem__(self, index):

        return self._render_item()


    def __len__(self):
        return self._size


    # compute the voting label image in 2D
    def _generate_vertex_targets(self, im_label, cls_indexes, center, poses, classes, num_classes):

        width = im_label.shape[1]
        height = im_label.shape[0]
        vertex_targets = np.zeros((3 * num_classes, height, width), dtype=np.float32)
        vertex_weights = np.zeros((3 * num_classes, height, width), dtype=np.float32)

        c = np.zeros((2, 1), dtype=np.float32)
        center_index = -1
        for i in range(1, num_classes):
            y, x = np.where(im_label == classes[i])
            if len(x) > 0:
                center_index += 1
                c[0] = center[center_index, 0]
                c[1] = center[center_index, 1]
                z = poses[center_index][2]
                R = np.tile(c, (1, len(x))) - np.vstack((x, y))
                # compute the norm
                N = np.linalg.norm(R, axis=0) + 1e-10
                # normalization
                R = np.divide(R, np.tile(N, (2,1)))
                # assignment
                vertex_targets[3*i+0, y, x] = R[0,:]
                vertex_targets[3*i+1, y, x] = R[1,:]
                vertex_targets[3*i+2, y, x] = math.log(z)

                vertex_weights[3*i+0, y, x] = cfg.TRAIN.VERTEX_W_INSIDE
                vertex_weights[3*i+1, y, x] = cfg.TRAIN.VERTEX_W_INSIDE
                vertex_weights[3*i+2, y, x] = cfg.TRAIN.VERTEX_W_INSIDE

        return vertex_targets, vertex_weights


    def _get_default_path(self):
        """
        Return the default path where ycb_object is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'YCB_Object')


    def _load_object_points(self):

        points = [[] for _ in range(len(self._classes))]
        num = np.inf

        for i in range(1, len(self._classes)):
            point_file = os.path.join(self._ycb_object_path, 'models', self._classes[i], 'points.xyz')
            print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)
            if points[i].shape[0] < num:
                num = points[i].shape[0]

        points_all = np.zeros((self._num_classes, num, 3), dtype=np.float32)
        for i in range(1, len(self._classes)):
            points_all[i, :, :] = points[i][:num, :]

        # rescale the points
        point_blob = points_all.copy()
        for i in range(1, self._num_classes):
            # compute the rescaling factor for the points
            weight = 10.0 / np.amax(self._extents[i, :])
            if weight < 10:
                weight = 10
            if self._symmetry[i] > 0:
                point_blob[i, :, :] = 4 * weight * point_blob[i, :, :]
            else:
                point_blob[i, :, :] = weight * point_blob[i, :, :]

        return points, points_all, point_blob


    def _load_object_extents(self):

        extent_file = os.path.join(self._ycb_object_path, 'extents.txt')
        assert os.path.exists(extent_file), \
                'Path does not exist: {}'.format(extent_file)

        extents = np.zeros((self._num_classes_all, 3), dtype=np.float32)
        extents[1:, :] = np.loadtxt(extent_file)

        return extents


    def labels_to_image(self, labels):

        height = labels.shape[0]
        width = labels.shape[1]
        im_label = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(self.num_classes):
            I = np.where(labels == i)
            im_label[I[0], I[1], :] = self._class_colors[i]

        return im_label


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
        for i in range(1, len(self._class_colors_all)):
            color = self._class_colors_all[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            labels_all[I[0], I[1]] = i

            ind = np.where(np.array(cfg.TRAIN.CLASSES) == i)[0]
            if len(ind) > 0:
                labels[I[0], I[1]] = ind

        return labels, labels_all

