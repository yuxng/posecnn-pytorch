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
import scipy
import cPickle
from fcn.config import cfg
from transforms3d.quaternions import *
from transforms3d.euler import *
from scipy.optimize import minimize
from utils.blob import pad_im, chromatic_transform, add_noise, add_noise_cuda, add_noise_depth_cuda
from utils.se3 import *
from layers.roi_align import ROIAlign


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
    '''

    return np.abs(w - width)
    # return np.abs(w * h - width * height)


class YCBEncoderSelfSupervision(data.Dataset, datasets.imdb):

    def __init__(self, image_set, ycb_object_path = None):

        self._name = 'ycb_encoder_' + image_set
        self._image_set = image_set
        self._ycb_object_path = self._get_default_path() if ycb_object_path is None \
                            else ycb_object_path
        self.root_path = self._ycb_object_path
        self._data_path = os.path.join(self._ycb_object_path, 'data')

        # define all the classes
        self._classes_all = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick', 'holiday_cup1', 'holiday_cup2', 'sanning_mug', \
                         '001_chips_can', 'block_red', 'block_green', 'block_blue', 'block_yellow')
        self._num_classes_all = len(self._classes_all)
        self._class_colors_all = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (0, 64, 0), (64, 0, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), \
                              (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192), (32, 0, 0), \
                              (150, 0, 0), (0, 150, 0), (0, 0, 150), (150, 150, 0)]
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
        self._points = self._load_object_points()

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
        self.model_sdf_paths = ['{}/models/{}/textured_simple_low_res.pth'.format(self._ycb_object_path, cls) for cls in self._classes_all[1:22]]
        self.model_colors = [np.array(self._class_colors_all[i]) / 255.0 for i in range(1, len(self._classes_all))]

        self.model_mesh_paths = []
        for cls in self._classes_all[1:]:
            filename = '{}/models/{}/textured_simple.obj'.format(self._ycb_object_path, cls)
            if os.path.exists(filename):
                self.model_mesh_paths.append(filename)
                continue
            filename = '{}/models/{}/textured_simple.ply'.format(self._ycb_object_path, cls)
            if os.path.exists(filename):
                self.model_mesh_paths.append(filename)

        self.model_texture_paths = []
        for cls in self._classes_all[1:]:
            filename = '{}/models/{}/texture_map.png'.format(self._ycb_object_path, cls)
            if os.path.exists(filename):
                self.model_texture_paths.append(filename)
            else:
                self.model_texture_paths.append('')

        # target meshes
        self.model_colors_target = [np.array(self._class_colors_all[i]) / 255.0 for i in cfg.TRAIN.CLASSES[1:]]
        self.model_mesh_paths_target = []
        for cls in self._classes[1:]:
            filename = '{}/models/{}/textured_simple.obj'.format(self._ycb_object_path, cls)
            if os.path.exists(filename):
                self.model_mesh_paths_target.append(filename)
                continue
            filename = '{}/models/{}/textured_simple.ply'.format(self._ycb_object_path, cls)
            if os.path.exists(filename):
                self.model_mesh_paths_target.append(filename)

        self.model_texture_paths_target = []
        for cls in self._classes[1:]:
            filename = '{}/models/{}/texture_map.png'.format(self._ycb_object_path, cls)
            if os.path.exists(filename):
                self.model_texture_paths_target.append(filename)
            else:
                self.model_texture_paths_target.append('')

        self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
        self._image_ext = '.png'
        self._image_index, size = self._load_image_set_index(image_set)

        if size == 0:
            self._size = cfg.TRAIN.SYNNUM
        else:
            self._size = size * (cfg.TRAIN.SYN_RATIO+1)

        self._roidb = self.gt_roidb()
        self._perm = [[] for i in range(self._num_classes)]
        if cfg.MODE == 'TRAIN' or cfg.TEST.VISUALIZE:
            for i in range(self._num_classes):
                self._perm[i] = np.random.permutation(np.arange(len(self._roidb[i])))
        else:
            for i in range(self._num_classes):
                self._perm[i] = np.arange(len(self._roidb[i]))
        self._cur = np.zeros((self._num_classes, ), dtype=np.int32)

        self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
        self._build_uniform_poses()
        self._losses_pose = np.zeros((self._num_classes, cfg.TRAIN.SYNNUM), dtype=np.float32)

        assert os.path.exists(self._ycb_object_path), \
                'ycb_object path does not exist: {}'.format(self._ycb_object_path)

        # compute the canonical distance to render
        self.render_depths = np.zeros((self._extents.shape[0], ), dtype=np.float32)
        margin = 20
        self.render_depths = self.compute_render_depths(margin)
        self.lb_shift = -margin
        self.ub_shift = margin
        self.lb_scale = 0.8
        self.ub_scale = 1.2
        self.cls_target = 0


    def compute_render_depths(self, margin):

        prefix = '_class'
        for i in range(len(cfg.TRAIN.CLASSES)):
            prefix += '_%d' % cfg.TRAIN.CLASSES[i]
        cache_file = os.path.join(self.cache_path, self.name + prefix + '_render_depths.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                render_depths = cPickle.load(fid)
            print('{} render_depths loaded from {}'.format(self.name, cache_file))
            print(render_depths)
            return render_depths

        print('computing canonical depths')
        render_depths = np.zeros((self._extents.shape[0], ), dtype=np.float32)
        for i in range(self._extents.shape[0]):
            points = get_bb3D(np.transpose(self._points[i]))
            render_depths[i] = optimize_depths(self._width - margin, self._height - margin, points, self._intrinsic_matrix)
            print(self._classes[i], render_depths[i])

        with open(cache_file, 'wb') as fid:
            cPickle.dump(render_depths, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote render_depths to {}'.format(cache_file)

        return render_depths


    def _render_item(self, sample_index):

        image_target_tensor = torch.cuda.FloatTensor(self._height, self._width, 4).detach()
        seg_target_tensor = torch.cuda.FloatTensor(self._height, self._width, 4).detach()
        image_tensor = torch.cuda.FloatTensor(self._height, self._width, 4).detach()
        seg_tensor = torch.cuda.FloatTensor(self._height, self._width, 4).detach()
        qt = np.zeros((7, ), dtype=np.float32)
        if cfg.MODE == 'TEST' and cfg.TEST.BUILD_CODEBOOK:
            interval = 0
        else:
            interval = cfg.TRAIN.UNIFORM_POSE_INTERVAL

        # initialize renderer
        fx = self._intrinsic_matrix[0, 0]
        fy = self._intrinsic_matrix[1, 1]
        px = self._intrinsic_matrix[0, 2]
        py = self._intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.01
        cfg.renderer.set_projection_matrix(self._width, self._height, fx, fy, px, py, znear, zfar)

        # sample target object (train one object only)
        cls_indexes = []
        cls_target = self.cls_target
        cls_indexes.append(cfg.TRAIN.CLASSES[cls_target]-1)

        # sample target pose
        poses_all = []
        cls = int(cls_indexes[0])
        extent_target = np.mean(self._extents_all[cls+1, :])
        if self.pose_indexes[cls] >= len(self.pose_lists[cls]):
            self.pose_indexes[cls] = 0
            self.pose_lists[cls] = np.random.permutation(np.arange(len(self.eulers)))
        index_euler = self.pose_lists[cls][self.pose_indexes[cls]]

        # use hard pose
        if (sample_index + 1) % cfg.TRAIN.IMS_PER_BATCH == 0:
            index_euler = np.argmax(self._losses_pose[cls_target, :])

        yaw = self.eulers[index_euler][0] + interval * np.random.randn()
        pitch = self.eulers[index_euler][1] + interval * np.random.randn()
        roll = self.eulers[index_euler][2] + interval * np.random.randn()
        qt[3:] = euler2quat(roll * math.pi / 180.0, pitch * math.pi / 180.0, yaw * math.pi / 180.0, 'syxz')
        self.pose_indexes[cls] += 1

        qt[0] = 0
        qt[1] = 0
        qt[2] = self.render_depths[cls_target]
        pose_target = qt.copy()

        # render target with constant lighting
        poses_all.append(qt.copy())
        cfg.renderer.set_poses(poses_all)
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([2.0, 2.0, 2.0])
        cfg.renderer.render(cls_indexes, image_target_tensor, seg_target_tensor)
        image_target_tensor = image_target_tensor.flip(0)
        image_target_tensor = image_target_tensor[:, :, (2, 1, 0)]
        seg_target_tensor = seg_target_tensor.flip(0)
        image_target_tensor = torch.clamp(image_target_tensor, min=0.0, max=1.0)
        seg_target = seg_target_tensor[:,:,2] + 256*seg_target_tensor[:,:,1] + 256*256*seg_target_tensor[:,:,0]

        # set background color here
        # image_target_tensor[seg_target == 0] = 0.5

        mask_target = (seg_target != 0).unsqueeze(0).repeat((3, 1, 1)).float()
        seg_target = seg_target.cpu().numpy()

        # render input image
        if cfg.MODE == 'TRAIN' or cfg.TEST.BUILD_CODEBOOK == False:

            while 1:
                # sample occluders
                if np.random.rand(1) < 0.8:
                    num_occluder = 3

                    if len(cls_indexes) == 1:
                        for i in range(num_occluder):
                            cls_indexes.append(0)
                            poses_all.append(np.zeros((7, ), dtype=np.float32))

                    for i in range(num_occluder):
                        while 1:
                            cls_occ = np.random.randint(1, self._num_classes_all, size=1)[0]
                            if cls_occ != cfg.TRAIN.CLASSES[cls_target]:
                                cls_indexes[i + 1] = cls_occ - 1
                                break

                        # sample poses
                        cls = int(cls_indexes[i + 1])
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
                        extent = np.mean(self._extents_all[cls+1, :])
                        qt[0] = poses_all[object_id][0] + np.random.uniform(-0.1, 0.1)
                        qt[1] = poses_all[object_id][1] + np.random.uniform(-0.1, 0.1)
                        qt[2] = poses_all[object_id][2] - np.random.uniform(extent, extent+0.2)

                        poses_all[i + 1] = qt.copy()
                        cfg.renderer.set_poses(poses_all)

                # rendering
                # light pose
                theta = np.random.uniform(-np.pi/2, np.pi/2)
                phi = np.random.uniform(0, np.pi/2)
                r = np.random.uniform(0.25, 3.0)
                light_pos = [r * np.sin(theta) * np.sin(phi), r * np.cos(phi) + np.random.uniform(-2, 2), r * np.cos(theta) * np.sin(phi)]
                cfg.renderer.set_light_pos(light_pos)

                # light color
                intensity = np.random.uniform(0.3, 3.0)
                light_color = intensity * np.random.uniform(0.2, 1.8, 3)
                cfg.renderer.set_light_color(light_color)
                cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)

                seg_tensor = seg_tensor.flip(0)
                seg = seg_tensor[:,:,2] + 256*seg_tensor[:,:,1] + 256*256*seg_tensor[:,:,0]
                seg_input = seg.clone().cpu().numpy()
            
                non_occluded = np.sum(np.logical_and(seg_target > 0, seg_target == seg_input)).astype(np.float)
                occluded_ratio = 1 - non_occluded / np.sum(seg_target>0).astype(np.float)

                if occluded_ratio < 0.8:
                    break
                else:
                    cls_indexes = cls_indexes[:1]
                    poses_all = poses_all[:1]

            # foreground mask
            mask = (seg != 0).unsqueeze(0).repeat((3, 1, 1)).float()

            # RGB to BGR order
            image_tensor = image_tensor.flip(0)
            im = image_tensor.cpu().numpy()
            im = np.clip(im, 0, 1)
            im = im[:, :, (2, 1, 0)] * 255
            im = im.astype(np.uint8)

            # affine transformation
            shift = np.float32([np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)])
            scale = np.random.uniform(self.lb_scale, self.ub_scale)
            affine_matrix = np.float32([[scale, 0, shift[0] / self._width], [0, scale, shift[1] / self._height]])

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
                  'mask': mask,
                  'cls_index': torch.from_numpy(np.array([cls_target]).astype(np.float32)),
                  'index_euler': torch.from_numpy(np.array([index_euler])),
                  'affine_matrix': torch.from_numpy(affine_matrix).cuda()}
        else:
            sample = {'image_input': image_target_tensor.permute(2, 0, 1),
                  'image_target': image_target_tensor.permute(2, 0, 1),
                  'mask': mask_target,
                  'cls_index': torch.from_numpy(np.array([cls_target]).astype(np.float32)),
                  'pose_target': pose_target}

        return sample


    def __getitem__(self, index):

        is_syn = 0
        ind = self.cls_target
        if len(self._image_index[ind]) == 0 or index % (cfg.TRAIN.SYN_RATIO+1) != 0:
            is_syn = 1

        if is_syn:
            return self._render_item(index)

        if self._cur[ind] >= len(self._roidb[ind]):
            self._perm[ind] = np.random.permutation(np.arange(len(self._roidb[ind])))
            self._cur[ind] = 0
        db_ind = self._perm[ind][self._cur[ind]]
        roidb = self._roidb[ind][db_ind]
        self._cur[ind] += 1

        random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
        sample, flag_bad = self._get_image_item(roidb, random_scale_ind)
        if flag_bad:
            return self._render_item(index)
        else:
            return sample


    def _get_image_item(self, roidb, scale_ind):

        # rgba
        rgba = pad_im(cv2.imread(roidb['image'], cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]
        if im_scale != 1.0:
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        height = im.shape[0]
        width = im.shape[1]

        if roidb['flipped']:
            im = im[:, ::-1, :]

        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)

        im_cuda = torch.from_numpy(im).cuda().float() / 255.0
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im_cuda = add_noise_cuda(im_cuda)
        im_cuda = torch.clamp(im_cuda, min=0.0, max=1.0)

        # load the annotation
        meta_data = scipy.io.loadmat(roidb['meta_data'])
        meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
        intrinsic_matrix = np.matrix(meta_data['intrinsic_matrix'])
        fx_data = intrinsic_matrix[0, 0]
        fy_data = intrinsic_matrix[1, 1]
        px_data = intrinsic_matrix[0, 2]
        py_data = intrinsic_matrix[1, 2]

        # initialize renderer
        fx = self._intrinsic_matrix[0, 0]
        fy = self._intrinsic_matrix[1, 1]
        px = self._intrinsic_matrix[0, 2]
        py = self._intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.01
        cfg.renderer.set_projection_matrix(self._width, self._height, fx, fy, px, py, znear, zfar)

        # poses
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))
        if roidb['flipped']:
            poses = _flip_poses(poses, meta_data['intrinsic_matrix'], width)

        # sample target object (train one object only)
        cls_indexes = []
        cls_target = self.cls_target
        cls_indexes.append(cfg.TRAIN.CLASSES[cls_target]-1)

        # render poses to get the target image
        poses_all = []
        qt = np.zeros((7, ), dtype=np.float32)
        ind = np.where(meta_data['cls_indexes'] == cfg.TRAIN.CLASSES[cls_target])[0]
        RT = poses[:, :, ind]
        qt[0] = 0
        qt[1] = 0
        qt[2] = self.render_depths[cls_target]
        qt[3:] = mat2quat(RT[:, :3])
        poses_all.append(qt)
        pose_target = qt.copy()
            
        # rendering
        cfg.renderer.set_poses(poses_all)
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([2.0, 2.0, 2.0])
        image_target_tensor = torch.cuda.FloatTensor(self._height, self._width, 4).detach()
        seg_target_tensor = torch.cuda.FloatTensor(self._height, self._width, 4).detach()
        cfg.renderer.render(cls_indexes, image_target_tensor, seg_target_tensor)
        image_target_tensor = image_target_tensor.flip(0)
        image_target_tensor = image_target_tensor[:, :, (2, 1, 0)]
        seg_target_tensor = seg_target_tensor.flip(0)
        image_target_tensor = torch.clamp(image_target_tensor, min=0.0, max=1.0)
        seg_target = seg_target_tensor[:,:,2] + 256*seg_target_tensor[:,:,1] + 256*256*seg_target_tensor[:,:,0]
        mask_target = (seg_target != 0).unsqueeze(0).repeat((3, 1, 1)).float()
        mask_target[:, :, :] = 1.0
        seg_target = seg_target.cpu().numpy()

        # project the 3D translation to get the center
        uv = np.zeros((1, 3), dtype=np.float32)
        uv[0, 0] = fx_data * RT[0, 3] / RT[2, 3] + px_data
        uv[0, 1] = fy_data * RT[1, 3] / RT[2, 3] + py_data
        uv[0, 2] = 1
        z = np.zeros((1, 1), dtype=np.float32)
        z[0, 0] = RT[2, 3]
        render_dist = self.render_depths[cls_target]

        # crop the rois from input image
        im_roi_cuda, scale_roi = self.trans_zoom_uvz_cuda(im_cuda, uv, z, fx_data, fy_data, render_dist)
        if torch.sum(im_roi_cuda) == 0:
            flag_bad = 1
        else:
            flag_bad = 0

        # affine transformation
        shift = np.float32([np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)])
        scale = np.random.uniform(self.lb_scale, self.ub_scale)
        affine_matrix = np.float32([[scale, 0, shift[0] / self._width], [0, scale, shift[1] / self._height]])

        if cfg.MODE == 'TRAIN' or cfg.TEST.BUILD_CODEBOOK == False:
            sample = {'image_input': im_roi_cuda[0],
                  'image_target': image_target_tensor.permute(2, 0, 1),
                  'mask': mask_target,
                  'cls_index': torch.from_numpy(np.array([cls_target]).astype(np.float32)),
                  'index_euler': torch.from_numpy(np.array([-1])),
                  'affine_matrix': torch.from_numpy(affine_matrix).cuda()}
        else:
            sample = {'image_input': im_roi_cuda[0],
                  'image_target': image_target_tensor.permute(2, 0, 1),
                  'mask': mask_target,
                  'cls_index': torch.from_numpy(np.array([cls_target]).astype(np.float32)),
                  'pose_target': pose_target}

        return sample, flag_bad


    def trans_zoom_uvz_cuda(self, image, uvs, zs, pf_fu, pf_fv, target_distance=2.5, out_size=128):
        image = image.permute(2, 0, 1).float().unsqueeze(0).cuda()

        bbox_u = target_distance * (1 / zs) / self._intrinsic_matrix[0, 0] * pf_fu * out_size / image.size(3)
        bbox_u = torch.from_numpy(bbox_u).cuda().float().squeeze(1)
        bbox_v = target_distance * (1 / zs) / self._intrinsic_matrix[1, 1] * pf_fv * out_size / image.size(2)
        bbox_v = torch.from_numpy(bbox_v).cuda().float().squeeze(1)

        center_uvs = torch.from_numpy(uvs).cuda().float()
        center_uvs[:, 0] /= image.size(3)
        center_uvs[:, 1] /= image.size(2)

        boxes = torch.zeros(center_uvs.size(0), 5).cuda()
        boxes[:, 1] = (center_uvs[:, 0] - bbox_u/2) * float(image.size(3))
        boxes[:, 2] = (center_uvs[:, 1] - bbox_v/2) * float(image.size(2))
        boxes[:, 3] = (center_uvs[:, 0] + bbox_u/2) * float(image.size(3))
        boxes[:, 4] = (center_uvs[:, 1] + bbox_v/2) * float(image.size(2))

        out = self.CropAndResizeFunction(image, boxes)
        uv_scale = target_distance * (1 / zs) / self._intrinsic_matrix[0, 0] * pf_fu

        '''
        for i in range(out.shape[0]):
            roi = out[i].permute(1, 2, 0).cpu().numpy()
            roi = np.clip(roi, 0, 1)
            im = roi * 255
            im = im.astype(np.uint8)
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.imshow(im[:, :, (2, 1, 0)])
            plt.show()
        #'''

        return out, uv_scale


    def CropAndResizeFunction(self, image, rois):
        return ROIAlign((128, 128), 1.0, 0)(image, rois)


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where ycb_self_supervision is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'YCB_Self_Supervision')


    def _load_image_set_index(self, image_set):
        """
        Load the indexes of images in the data folder
        """

        image_set_file = os.path.join(self._ycb_object_path, image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        subdirs = []
        with open(image_set_file) as f:
            for x in f.readlines():
                subdirs.append(x.rstrip('\n'))

        # each class has a index list
        image_index = [[] for i in range(len(cfg.TRAIN.CLASSES))]

        for i in range(len(subdirs)):
            subdir = subdirs[i]
            filename = os.path.join(self._data_path, subdir, '*.mat')
            files = glob.glob(filename)
            for j in range(len(files)):
                filename = files[j]
                head, name = os.path.split(filename)
                index = subdir + '/' + name[:-9]

                # load the annotation to see if the target object is in the image
                meta_data = scipy.io.loadmat(filename)
                cls_indexes = meta_data['cls_indexes'].flatten()
                for k in range(len(cls_indexes)):
                    cls_index = int(cls_indexes[k])
                    ind = np.where(np.array(cfg.TRAIN.CLASSES) == cls_index)[0]
                    if len(ind) > 0:
                        image_index[ind[0]].append(index)

        print('=======================================================')
        size = 0
        for i in range(len(cfg.TRAIN.CLASSES)):
            size = max(size, len(image_index[i]))
            print('%s: %d image' % (self._classes[i], len(image_index[i])))
        print('in %s' % (self._data_path))
        print('=======================================================')
        return image_index, size


    def _load_object_extents(self):

        extent_file = os.path.join(self._ycb_object_path, 'extents.txt')
        assert os.path.exists(extent_file), \
                'Path does not exist: {}'.format(extent_file)

        extents = np.zeros((self._num_classes_all, 3), dtype=np.float32)
        extents[1:, :] = np.loadtxt(extent_file)

        return extents


    def _load_object_points(self):

        points = [[] for _ in xrange(len(self._classes))]
        for i in xrange(len(self._classes)):
            point_file = os.path.join(self._ycb_object_path, 'models', self._classes[i], 'points.xyz')
            print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)

        return points

    # image
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """

        image_path = os.path.join(self._data_path, index + '_color' + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # depth
    def depth_path_at(self, i):
        """
        Return the absolute path to depth i in the image sequence.
        """
        return self.depth_path_from_index(self.image_index[i])

    def depth_path_from_index(self, index):
        """
        Construct an depth path from the image's "index" identifier.
        """
        depth_path = os.path.join(self._data_path, index + '_depth' + self._image_ext)
        assert os.path.exists(depth_path), \
                'Path does not exist: {}'.format(depth_path)
        return depth_path

    # camera pose
    def metadata_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.metadata_path_from_index(self.image_index[i])

    def metadata_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        metadata_path = os.path.join(self._data_path, index + '_meta.mat')
        assert os.path.exists(metadata_path), \
                'Path does not exist: {}'.format(metadata_path)
        return metadata_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        gt_roidb = [[] for i in range(len(cfg.TRAIN.CLASSES))]
        for i in range(len(cfg.TRAIN.CLASSES)):
            gt_roidb[i] = [self._load_ycb_encoder_self_supervision_annotation(index)
                for index in self._image_index[i]]

        return gt_roidb


    def _load_ycb_encoder_self_supervision_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path = self.image_path_from_index(index)

        # depth path
        depth_path = self.depth_path_from_index(index)

        # metadata path
        metadata_path = self.metadata_path_from_index(index)
        
        return {'image': image_path,
                'depth': depth_path,
                'meta_data': metadata_path,
                'flipped': False}


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
