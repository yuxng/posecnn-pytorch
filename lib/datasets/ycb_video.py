import torch
import torch.utils.data as data

import os, math
import sys
import os.path as osp
from os.path import *
import numpy as np
import numpy.random as npr
import cv2
import cPickle
import scipy.io

import datasets
from fcn.config import cfg
from utils.blob import pad_im, chromatic_transform, add_noise
from transforms3d.quaternions import mat2quat, quat2mat

class YCBVideo(data.Dataset, datasets.imdb):
    def __init__(self, image_set, ycb_video_path = None):

        self._name = 'ycb_video_' + image_set
        self._image_set = image_set
        self._ycb_video_path = self._get_default_path() if ycb_video_path is None \
                            else ycb_video_path
        self._data_path = os.path.join(self._ycb_video_path, 'data')

        # define all the classes
        self._classes_all = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
        self._num_classes_all = len(self._classes_all)
        self._class_colors_all = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
                              (192, 0, 0), (0, 192, 0), (0, 0, 192)]
        self._symmetry_all = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).astype(np.float32)
        self._extents_all = self._load_object_extents()

        self._width = 640
        self._height = 480
        self._intrinsic_matrix = np.array([[1.066778e+03, 0.000000e+00, 3.129869e+02],
                                          [0.000000e+00, 1.067487e+03, 2.413109e+02],
                                          [0.000000e+00, 0.000000e+00, 1.000000e+00]])

        # select a subset of classes
        self._classes = [self._classes_all[i] for i in cfg.TRAIN.CLASSES]
        self._num_classes = len(self._classes)
        self._class_colors = [self._class_colors_all[i] for i in cfg.TRAIN.CLASSES]
        self._symmetry = self._symmetry_all[cfg.TRAIN.CLASSES]
        self._extents = self._extents_all[cfg.TRAIN.CLASSES]
        self._points, self._points_all, self._point_blob = self._load_object_points()

        self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()

        if (cfg.MODE == 'TRAIN' and cfg.TRAIN.SYNTHESIZE) or (cfg.MODE == 'TEST' and cfg.TEST.SYNTHESIZE):
            self._size = len(self._image_index) * (cfg.TRAIN.SYN_RATIO+1)
        else:
            self._size = len(self._image_index)
        self._roidb = self.gt_roidb()
        self._build_background_images()

        assert os.path.exists(self._ycb_video_path), \
                'ycb_video path does not exist: {}'.format(self._ycb_video_path)
        assert os.path.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)


    def _render_item(self):

        fx = self._intrinsic_matrix[0, 0]
        fy = self._intrinsic_matrix[1, 1]
        px = self._intrinsic_matrix[0, 2]
        py = self._intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.25
        if cfg.TRAIN.ITERS % 100 == 0:
            is_display = 1
        else:
            is_display = 0

        parameters = np.zeros((17, ), dtype=np.float32)
        parameters[0] = self._width
        parameters[1] = self._height
        parameters[2] = fx
        parameters[3] = fy
        parameters[4] = px
        parameters[5] = py
        parameters[6] = znear
        parameters[7] = zfar
        parameters[8] = cfg.TRAIN.SYN_TNEAR
        parameters[9] = cfg.TRAIN.SYN_TFAR
        parameters[10] = cfg.TRAIN.SYN_MIN_OBJECT
        parameters[11] = cfg.TRAIN.SYN_MAX_OBJECT
        parameters[12] = cfg.TRAIN.SYN_STD_ROTATION
        parameters[13] = cfg.TRAIN.SYN_STD_TRANSLATION
        parameters[14] = cfg.TRAIN.SYN_SAMPLE_OBJECT
        parameters[15] = cfg.TRAIN.SYN_SAMPLE_POSE
        parameters[16] = is_display

        # render image
        im = np.zeros((self._height, self._width, 3), dtype=np.float32)
        vertmap = np.zeros((self._height, self._width, 3), dtype=np.float32)
        class_indexes = np.zeros((self.num_classes, ), dtype=np.float32)
        poses = np.zeros((self.num_classes, 7), dtype=np.float32)
        centers = np.zeros((self.num_classes, 2), dtype=np.float32)
        cfg.synthesizer.render_python(parameters, im, vertmap, class_indexes, poses, centers)

        index = np.where(class_indexes > 0)[0]
        class_indexes = class_indexes[index]
        poses = poses[index, :]
        centers = centers[index, :]
        im_label = np.round(vertmap[:, :, 0])

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

        # add background
        I = np.where(im_label == 0)
        im[I[0], I[1], :] = background[I[0], I[1], :3]
        im = im.astype(np.uint8)

        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN':
            im = chromatic_transform(im)

        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN':
            im = add_noise(im)

        im = im.astype(np.float32)
        im -= cfg.PIXEL_MEANS
        im = np.transpose(im / 255.0, (2, 0, 1))

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
        num = poses.shape[0]
        pose_blob = np.zeros((self.num_classes, 9), dtype=np.float32)
        gt_boxes = np.zeros((self.num_classes, 5), dtype=np.float32)
        for i in xrange(num):
            cls = int(class_indexes[i])
            pose_blob[i, 0] = 1
            pose_blob[i, 1] = cls
            qt = poses[i, :4]
            if qt[0] < 0:
                qt = -1 * qt
            pose_blob[i, 2:6] = qt
            pose_blob[i, 6:] = poses[i, 4:]

            # compute box
            x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
            x3d[0, :] = self._points_all[cls,:,0]
            x3d[1, :] = self._points_all[cls,:,1]
            x3d[2, :] = self._points_all[cls,:,2]
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(pose_blob[i, 2:6])
            RT[:, 3] = pose_blob[i, 6:]
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
            vertex_targets, vertex_weights = self._generate_vertex_targets(im_label, class_indexes, centers, poses, classes, self.num_classes)
        else:
            vertex_targets = []
            vertex_weights = []

        im_info = np.array([im.shape[1], im.shape[2], cfg.TRAIN.SCALES_BASE[0]], dtype=np.float32)

        sample = {'image': im,
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

        is_syn = 0
        if ((cfg.MODE == 'TRAIN' and cfg.TRAIN.SYNTHESIZE) or (cfg.MODE == 'TEST' and cfg.TEST.SYNTHESIZE)) and (index % (cfg.TRAIN.SYN_RATIO+1) != 0):
            is_syn = 1

        if is_syn:
            return self._render_item()

        if (cfg.MODE == 'TRAIN' and cfg.TRAIN.SYNTHESIZE) or (cfg.MODE == 'TEST' and cfg.TEST.SYNTHESIZE):
            index = int((index % self._size) / (cfg.TRAIN.SYN_RATIO+1))
        else:
            index = int(index % self._size)
        roidb = self._roidb[index]

        # Get the input image blob
        random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
        im_blob, im_scale, height, width = self._get_image_blob(roidb, random_scale_ind)

        # build the label blob
        label_blob, meta_data_blob, pose_blob, gt_boxes, vertex_targets, vertex_weights \
            = self._get_label_blob(roidb, self._num_classes, im_scale, height, width)

        im_info = np.array([im_blob.shape[1], im_blob.shape[2], im_scale], dtype=np.float32)

        sample = {'image': im_blob,
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


    def _get_image_blob(self, roidb, scale_ind):    

        # rgba
        rgba = pad_im(cv2.imread(roidb['image'], cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        # chromatic transform
        if cfg.TRAIN.CHROMATIC:
            im = chromatic_transform(im)

        if cfg.TRAIN.ADD_NOISE:
            im = add_noise(im)

        if roidb['flipped']:
            im = im[:, ::-1, :]

        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS	
        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        height = im.shape[0]
        width = im.shape[1]
        im = np.transpose(im / 255.0, (2, 0, 1))

        return im, im_scale, height, width


    def _get_label_blob(self, roidb, num_classes, im_scale, height, width):
        """ build the label blob """

        meta_data = scipy.io.loadmat(roidb['meta_data'])
        meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
        classes = np.array(cfg.TRAIN.CLASSES)

        # read label image
        im_label = pad_im(cv2.imread(roidb['label'], cv2.IMREAD_UNCHANGED), 16)
        if roidb['flipped']:
            if len(im_label.shape) == 2:
                im_label = im_label[:, ::-1]
            else:
                im_label = im_label[:, ::-1, :]
        im_label = cv2.resize(im_label, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        label_blob = np.zeros((num_classes, height, width), dtype=np.float32)
        label_blob[0, :, :] = 1.0
        for i in range(1, num_classes):
            I = np.where(im_label == classes[i])
            if len(I[0]) > 0:
                label_blob[i, I[0], I[1]] = 1.0
                label_blob[0, I[0], I[1]] = 0.0

        # bounding boxes
        boxes = meta_data['box'].copy()
        num = boxes.shape[0]
        if roidb['flipped']:
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = width - oldx2 - 1
            boxes[:, 2] = width - oldx1 - 1

        # poses
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))
        if roidb['flipped']:
            poses = _flip_poses(poses, meta_data['intrinsic_matrix'], width)

        num = poses.shape[2]
        pose_blob = np.zeros((num_classes, 9), dtype=np.float32)
        gt_boxes = np.zeros((num_classes, 5), dtype=np.float32)
        count = 0
        for i in xrange(num):
            cls = int(meta_data['cls_indexes'][i])
            ind = np.where(classes == cls)[0]
            if len(ind) > 0:
                R = poses[:, :3, i]
                T = poses[:, 3, i]
                pose_blob[count, 0] = 1
                pose_blob[count, 1] = ind
                qt = mat2quat(R)
                if qt[0] < 0:
                    qt = -1 * qt
                pose_blob[count, 2:6] = qt
                pose_blob[count, 6:] = T
                gt_boxes[count, :4] =  boxes[i, :] * im_scale
                gt_boxes[count, 4] =  ind
                count += 1

        # construct the meta data
        """
        format of the meta_data
        intrinsic matrix: meta_data[0 ~ 8]
        inverse intrinsic matrix: meta_data[9 ~ 17]
        """
        K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros(18, dtype=np.float32)
        meta_data_blob[0:9] = K.flatten()
        meta_data_blob[9:18] = Kinv.flatten()

        # vertex regression target
        if cfg.TRAIN.VERTEX_REG:
            center = meta_data['center']
            if roidb['flipped']:
                center[:, 0] = width - center[:, 0]
            vertex_targets, vertex_weights = self._generate_vertex_targets(im_label, meta_data['cls_indexes'], center, poses, classes, num_classes)
        else:
            vertex_targets = []
            vertex_weights = []

        return label_blob, meta_data_blob, pose_blob, gt_boxes, vertex_targets, vertex_weights


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
            ind = np.where(cls_indexes == classes[i])[0]
            if len(x) > 0 and len(ind) > 0:
                c[0] = center[ind, 0]
                c[1] = center[ind, 1]
                if len(poses.shape) == 3:
                    z = poses[2, 3, ind]
                else:
                    z = poses[ind, -1]
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


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where YCB_Video is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'YCB_Video')


    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._ycb_video_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        image_index = []
        video_ids_selected = set([])
        video_ids_not = set([])
        count = np.zeros((self.num_classes, ), dtype=np.int32)

        with open(image_set_file) as f:
            for x in f.readlines():
                index = x.rstrip('\n')
                pos = index.find('/')
                video_id = index[:pos]

                if not video_id in video_ids_selected and not video_id in video_ids_not:
                    filename = os.path.join(self._data_path, video_id, '000001-meta.mat')
                    meta_data = scipy.io.loadmat(filename)
                    cls_indexes = meta_data['cls_indexes'].flatten()
                    flag = 0
                    for i in range(len(cls_indexes)):
                        cls_index = int(cls_indexes[i])
                        ind = np.where(np.array(cfg.TRAIN.CLASSES) == cls_index)[0]
                        if len(ind) > 0:
                            count[ind] += 1
                            flag = 1
                    if flag:
                        video_ids_selected.add(video_id)
                    else:
                        video_ids_not.add(video_id)

                if video_id in video_ids_selected:
                    image_index.append(index)

        for i in range(1, self.num_classes):
            print('%d %s [%d/%d]' % (i, self.classes[i], count[i], len(list(video_ids_selected))))

        # sample a subset for training
        if self._image_set == 'train':
            image_index = image_index[::10]

        return image_index


    def _load_object_points(self):

        points = [[] for _ in xrange(len(self._classes))]
        num = np.inf

        for i in xrange(1, len(self._classes)):
            point_file = os.path.join(self._ycb_video_path, 'models', self._classes[i], 'points.xyz')
            print point_file
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

        extent_file = os.path.join(self._ycb_video_path, 'extents.txt')
        assert os.path.exists(extent_file), \
                'Path does not exist: {}'.format(extent_file)

        extents = np.zeros((self._num_classes_all, 3), dtype=np.float32)
        extents[1:, :] = np.loadtxt(extent_file)

        return extents


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

        image_path = os.path.join(self._data_path, index + '-color' + self._image_ext)
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
        depth_path = os.path.join(self._data_path, index + '-depth' + self._image_ext)
        assert os.path.exists(depth_path), \
                'Path does not exist: {}'.format(depth_path)
        return depth_path

    # label
    def label_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.label_path_from_index(self.image_index[i])

    def label_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        label_path = os.path.join(self._data_path, index + '-label' + self._image_ext)
        assert os.path.exists(label_path), \
                'Path does not exist: {}'.format(label_path)
        return label_path

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
        metadata_path = os.path.join(self._data_path, index + '-meta.mat')
        assert os.path.exists(metadata_path), \
                'Path does not exist: {}'.format(metadata_path)
        return metadata_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        prefix = '_class'
        for i in range(len(cfg.TRAIN.CLASSES)):
            prefix += '_%d' % cfg.TRAIN.CLASSES[i]
        cache_file = os.path.join(self.cache_path, self.name + prefix + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_ycb_video_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def _load_ycb_video_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path = self.image_path_from_index(index)

        # depth path
        depth_path = self.depth_path_from_index(index)

        # label path
        label_path = self.label_path_from_index(index)

        # metadata path
        metadata_path = self.metadata_path_from_index(index)

        # parse image name
        pos = index.find('/')
        video_id = index[:pos]
        
        return {'image': image_path,
                'depth': depth_path,
                'label': label_path,
                'meta_data': metadata_path,
                'video_id': video_id,
                'flipped': False}


    def labels_to_image(self, labels):

        height = labels.shape[0]
        width = labels.shape[1]
        im_label = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(self.num_classes):
            I = np.where(labels == i)
            im_label[I[0], I[1], :] = self._class_colors[i]

        return im_label
