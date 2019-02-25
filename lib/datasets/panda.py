import torch
import torch.utils.data as data

import os, math, sys
import os.path as osp
from os.path import *
import numpy as np
import numpy.random as npr
import cv2
import cPickle
import scipy.io
import threading
import datetime
import datasets
from numpy.linalg import inv
from fcn.config import cfg
from utils.blob import pad_im, chromatic_transform, add_noise, add_noise_cuda
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import mat2euler, euler2mat, euler2quat
from utils.pose_error import *
from utils.se3 import *
from robotPose.robot_pykdl import *

class panda(data.Dataset, datasets.imdb):
    def __init__(self, image_set, panda_path = None):

        # initialize robot
        self.robot = robot_kinematics('panda_arm')

        self._name = 'panda_' + image_set #set panda path
        self._image_set = image_set
        self._panda_path = self._get_default_path() if panda_path is None \
                            else panda_path

        # define all the robot parts
        self._classes_all = ['__background__', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand', 'finger', 'finger']
        self._num_classes_all = len(self._classes_all)
        self._class_colors_all = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0)]
        self._symmetry_all = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32)
        self._extents_all = self._load_object_extents()
        self._diameters_all = np.linalg.norm(self._extents_all,axis=1)

        self._width = 640
        self._height = 480
        self._intrinsic_matrix = np.array([[524.7917885754071, 0, 332.5213232846151],
                                          [0, 489.3563960810721, 281.2339855172282],
                                          [0, 0, 1]])

        #render the entire arm
        self._base_link = 'panda_link0'
        self._classes_idx = cfg.TRAIN.CLASSES
        self._classes = [self._classes_all[i] for i in self._classes_idx]
        print 'panda dataset classes: ', self._classes 

        self._num_classes = len(self._classes)
        self._class_colors = [self._class_colors_all[idx] for idx in self._classes_idx] 
        self._diameters = self._diameters_all[self._classes_idx] 
        self._symmetry = self._symmetry_all[self._classes_idx]
        self._extents = self._extents_all[self._classes_idx]
        self._points, self._points_all, self._point_blob = self._load_object_points()

        # 3D model paths, all models
        self.model_mesh_paths = ['{}/{}.DAE'.format(self._panda_path, cls) for cls in self._classes_all[1:]]
        self.model_texture_paths = ['' for cls in self._classes_all[1:]]
        self.model_colors = [np.array(self._class_colors_all[i]) / 255.0 for i in range(1, len(self._classes_all))]

        self.model_mesh_paths_target = ['{}/{}.DAE'.format(self._panda_path, cls) for cls in self._classes[1:]]
        self.model_texture_paths_target = ['' for cls in self._classes[1:]]
        self.model_colors_target = [np.array(self._class_colors_all[i]) / 255.0 for i in cfg.TRAIN.CLASSES[1:]]

        self._image_ext = '.png'
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).cuda().float() # as in linemod
        self._size = cfg.TRAIN.SYNNUM
        if cfg.MODE == 'TRAIN' or (cfg.MODE == 'TEST' and cfg.TEST.SYNTHESIZE == True):
            self._build_background_images()

        assert os.path.exists(self._panda_path), \
                'panda path does not exist: {}'.format(self._panda_path)

    
    def _render_item(self):

        height = cfg.TRAIN.SYN_HEIGHT
        width = cfg.TRAIN.SYN_WIDTH
        fx = self._intrinsic_matrix[0, 0]
        fy = self._intrinsic_matrix[1, 1]
        px = self._intrinsic_matrix[0, 2]
        py = self._intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.01
        bound = 0.2
        image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

        while 1:

            # sample target poses
            robot_all, _ = self.robot.gen_rand_pose(self._base_link, base_pose=np.eye(4))
            # offset the center in model coordinates
            robot_all = self.robot.offset_pose_center(robot_all, dir='off', base_link=self._base_link)
            extrinsic = self._sample_camera(robot_all, self._classes_idx[1]-1)
            cfg.renderer.set_camera_default()
            cfg.renderer.set_light_pos(np.random.uniform(-0.5, 0.5, 3))
            intensity = np.random.uniform(0.8, 2)
            light_color = intensity * np.random.uniform(0.9, 1.1, 3)          
            cfg.renderer.set_light_color(light_color)

            poses_all = []
            cls_indexes = []
            poses_target = []
            cls_indexes_target = []
            for i in range(len(self._classes_all) - 1):
                pose = extrinsic.dot(robot_all[i])
                quat = mat2quat(pose[:3, :3])
                trans = pose[:3, 3]
                tq = np.hstack((trans, quat))
                poses_all.append(tq)
                cls_indexes.append(i)

                ind = np.where(np.array(cfg.TRAIN.CLASSES[1:]) == i + 1)[0]
                if len(ind) > 0:
                    poses_target.append(tq)
                    cls_indexes_target.append(i)

            # render target
            cfg.renderer.set_poses(poses_target)
            cfg.renderer.render(cls_indexes_target, image_tensor, seg_tensor)

            seg_tensor = seg_tensor.flip(0)
            seg = torch.sum(seg_tensor[:, :, :3], dim=2)
            mask = (seg != 0).cpu().numpy()

            # render the entire arm
            cfg.renderer.set_poses(poses_all)
            cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
            image_tensor = image_tensor.flip(0)
            seg_tensor = seg_tensor.flip(0)

            im_label = seg_tensor.cpu().numpy()
            im_label = im_label[:, :, (2, 1, 0)] * 255
            im_label = np.round(im_label).astype(np.uint8)
            im_label = np.clip(im_label, 0, 255)
            im_label, im_label_all = self.process_label_image(im_label)

            # compute occlusion percentage
            mask_target = (im_label > 0).astype(np.int32)
            per_occ = 1.0 - np.sum(mask & mask_target) / np.sum(mask)
            if per_occ < 0.5:
                '''
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1)
                im = image_tensor.cpu().numpy()
                im = np.clip(im, 0, 1)
                im = im[:, :, (2, 1, 0)] * 255
                im = im.astype(np.uint8)
                plt.imshow(im[:, :, (2, 1, 0)])
                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(mask)
                print per_occ
                plt.show()
                '''
                break
        
        cls_indexes = np.array(range(len(self._classes_all) - 1)).astype(np.int32)

        # RGB to BGR order
        im = image_tensor.cpu().numpy()
        im = np.clip(im, 0, 1)
        im = im[:, :, (2, 1, 0)] * 255
        im = im.astype(np.uint8)

        # part centers
        rcenters = cfg.renderer.get_centers()
        num = len(rcenters)
        centers = np.zeros((num, 2), dtype=np.float32)
        for i in range(num):
            centers[i, 0] = rcenters[i][1] * width
            centers[i, 1] = rcenters[i][0] * height

        # add background to the image
        ind = np.random.randint(len(self._backgrounds), size=1)[0]
        filename = self._backgrounds[ind]
        background = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        try:
            # randomly crop a region as background
            bw = background.shape[1]
            bh = background.shape[0]
            x1 = npr.randint(0, int(bw/3))
            y1 = npr.randint(0, int(bh/3))
            x2 = npr.randint(int(2*bw/3), bw)
            y2 = npr.randint(int(2*bh/3), bh)
            background = background[y1:y2, x1:x2]
            background = cv2.resize(background, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
        except:
            background = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            print 'bad background image'

        if len(background.shape) != 3:
            background = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            print 'bad background image'

        # paste objects on background
        I = np.where(im_label_all == 0)
        im[I[0], I[1], :] = background[I[0], I[1], :3]
        im = im.astype(np.uint8)
        margin = 10
        for i in range(num):
            I = np.where(im_label_all == cls_indexes[i]+1)
            if len(I[0]) > 0:
                y1 = np.max((np.round(np.min(I[0])) - margin, 0))
                x1 = np.max((np.round(np.min(I[1])) - margin, 0))
                y2 = np.min((np.round(np.max(I[0])) + margin, self._height-1))
                x2 = np.min((np.round(np.max(I[1])) + margin, self._width-1))
                foreground = im[y1:y2, x1:x2].astype(np.uint8)
                mask = 255 * np.ones((foreground.shape[0], foreground.shape[1]), dtype=np.uint8)
                background = cv2.seamlessClone(foreground, background, mask, ((x1+x2)/2, (y1+y2)/2), cv2.NORMAL_CLONE)
        im = background

        '''
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(im[:, :, (2, 1, 0)])
        for i in range(num):
            plt.plot(centers[i, 0], centers[i, 1], 'yo')
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(im_label)
        plt.show()
        '''

        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)

        im_cuda = torch.from_numpy(im).cuda().float() / 255.0
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im_cuda = add_noise_cuda(im_cuda)
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
        idx = self._classes_idx[1:]
        for i in range(len(idx)):
            cls = i + 1
            ind = np.where(cls_indexes == int(idx[i]) - 1)[0]
            pose_blob[i, 0] = 1
            pose_blob[i, 1] = cls
            T = poses_all[int(ind)][:3]
            qt = poses_all[int(ind)][3:]

            # egocentric to allocentric
            qt_allocentric = egocentric2allocentric(qt, T)
            if qt_allocentric[0] < 0:
                qt_allocentric = -1 * qt_allocentric
            pose_blob[i, 2:6] = qt_allocentric
            pose_blob[i, 6:] = T

            # compute box
            x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
            x3d[0, :] = self._points_all[cls,:,0]
            x3d[1, :] = self._points_all[cls,:,1]
            x3d[2, :] = self._points_all[cls,:,2]
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(qt)
            RT[:, 3] = T
            x2d = np.matmul(self._intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        
            gt_boxes[i, 0] = np.min(x2d[0, :])
            gt_boxes[i, 1] = np.min(x2d[1, :])
            gt_boxes[i, 2] = np.max(x2d[0, :])
            gt_boxes[i, 3] = np.max(x2d[1, :])
            gt_boxes[i, 4] = cls
        
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
            vertex_targets, vertex_weights = self._generate_vertex_targets(im_label, cls_indexes, centers, poses_all, classes, self.num_classes)
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
        return self._render_item() #no real dataset

    def __len__(self):
        return self._size

    def _get_default_path(self):
        """
        Return the default path where panda is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'ycb_render','robotPose','panda_arm_models')


    # compute the voting label image in 2D
    def _generate_vertex_targets(self, im_label, cls_indexes, center, poses, classes, num_classes):

        width = im_label.shape[1]
        height = im_label.shape[0]
        vertex_targets = np.zeros((3 * num_classes, height, width), dtype=np.float32)
        vertex_weights = np.zeros((3 * num_classes, height, width), dtype=np.float32)

        c = np.zeros((2, 1), dtype=np.float32)
        for i in xrange(1, num_classes):
            y, x = np.where(im_label == classes[i])
            I = np.where(im_label == classes[i])
            idx = self._classes_idx[i]
            ind = np.where(cls_indexes == idx - 1)[0]
            if len(x) > 0 and len(ind) > 0:
                c[0] = center[ind, 0]
                c[1] = center[ind, 1]
                z = poses[int(ind)][2]
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


    def _load_object_points(self):

        points = [[] for _ in xrange(len(self._classes))]
        num = np.inf

        for i in xrange(1, len(self._classes)):
            point_file = os.path.join(self._panda_path, self._classes[i] + '.xyz')
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)
            if points[i].shape[0] < num:
                num = points[i].shape[0]

        points_all = np.zeros((self._num_classes, num, 3), dtype=np.float32)
        for i in xrange(1, len(self._classes)):
            points_all[i, :, :] = points[i][:num, :]

        # rescale the points
        point_blob = points_all.copy()
        for i in xrange(1, self._num_classes):
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

        extent_file = os.path.join(self._panda_path, 'extents.txt')
        assert os.path.exists(extent_file), \
                'Path does not exist: {}'.format(extent_file)

        extents = np.zeros((self._num_classes_all, 3), dtype=np.float32)
        extents[1:, :] = np.loadtxt(extent_file)
        return extents


    def _sample_camera(self, poses, target_id):
        # sample a camera extrinsics
        target = poses[target_id][:3, 3].T
        count = 0
        while 1:
            theta = np.random.uniform(low=0, high=np.pi)
            phi = np.random.uniform(low=-np.pi/2, high=np.pi/2) #top sphere
            r = np.random.uniform(low=cfg.TRAIN.SYN_TNEAR, high=cfg.TRAIN.SYN_TFAR) #sphere radius
            pos = np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])
            n = np.linalg.norm(pos-target)
            if n > cfg.TRAIN.SYN_TNEAR and n < cfg.TRAIN.SYN_TFAR and self._valid_camera_pos(poses, pos):
                break
            count += 1
            if count > 50:
                break
        ref = pos + (pos - target) + np.random.uniform(-0.15, 0.15, 3) #so that the target not always at image center
        cfg.renderer.set_camera(pos, ref, [0, 0, -1])
        return cfg.renderer.V


    def _valid_camera_pos(self, arm_pose, camera_pos):
        # avoid sampling camera position too close to arm
        for pose in arm_pose:
            if np.linalg.norm(camera_pos-pose[:3, 3].T) < 0.4:
                return False
        return True


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
        for i in xrange(1, len(self._class_colors_all)):
            color = self._class_colors_all[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            labels_all[I[0], I[1]] = i

            ind = np.where(np.array(cfg.TRAIN.CLASSES) == i)[0]
            if len(ind) > 0:
                labels[I[0], I[1]] = ind

        return labels, labels_all
