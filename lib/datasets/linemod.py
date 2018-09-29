import torch
import torch.utils.data as data

import os, math
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

class linemod(data.Dataset):
    def __init__(self, image_set, linemod_path = None):

        self._name = 'linemod_' + image_set
        self._image_set = image_set
        self._linemod_path = self._get_default_path() if linemod_path is None \
                            else linemod_path
        self._data_path = os.path.join(self._linemod_path, 'data')

        self._classes = ('__background__', 'ape', 'benchvise', 'bowl', 'camera', 'can', \
                         'cat', 'cup', 'driller', 'duck', 'eggbox', \
                         'glue', 'holepuncher', 'iron', 'lamp', 'phone')
        self._num_classes = len(self._classes)

        self._class_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), \
                              (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (64, 0, 0), (0, 64, 0), (0, 0, 64)]

        self._class_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self._symmetry = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        self._extents = self._load_object_extents()
        self._points, self._points_all, self._point_blob = self._load_object_points()

        self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        self._size = len(self._image_index)
        self._roidb = self.gt_roidb()

        assert os.path.exists(self._linemod_path), \
                'linemod path does not exist: {}'.format(self._linemod_path)
        assert os.path.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)

    @property
    def name(self):
        return self._name

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(datasets.ROOT_DIR, 'data', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def __getitem__(self, index):

        index = index % self._size
        roidb = self._roidb[index]

        # Get the input image blob
        random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
        im_blob, im_scale, height, width = self._get_image_blob(roidb, random_scale_ind)

        # build the label blob
        label_blob, meta_data_blob, pose_blob, gt_boxes \
            = self._get_label_blob(roidb, self._num_classes, im_scale, height, width)

        im_info = np.array([im_blob.shape[1], im_blob.shape[2], im_scale], dtype=np.float32)

        sample = {'image': im_blob,
                  'label': label_blob,
                  'meta_data': meta_data_blob,
                  'poses': pose_blob,
                  'extents': self._extents,
                  'points': self._point_blob,
                  'gt_boxes': gt_boxes,
                  'im_info': im_info}

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

        return im, im_scale, height, width


    def _get_label_blob(self, roidb, num_classes, im_scale, blob_height, blob_width):
        """ build the label blob """

        meta_data = scipy.io.loadmat(roidb['meta_data'])
        meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()

        # read label image
        im = pad_im(cv2.imread(roidb['label'], cv2.IMREAD_UNCHANGED), 16)
        if roidb['flipped']:
            if len(im.shape) == 2:
                im = im[:, ::-1]
            else:
                im = im[:, ::-1, :]
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        label_blob = im.astype(np.int32)

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
        for j in xrange(num):
            R = poses[:, :3, j]
            T = poses[:, 3, j]

            pose_blob[j, 0] = 1
            pose_blob[j, 1] = meta_data['cls_indexes'][j]
            pose_blob[j, 2:6] = mat2quat(R)
            pose_blob[j, 6:] = T

            gt_boxes[j, :4] =  boxes[j, :] * im_scale
            gt_boxes[j, 4] =  meta_data['cls_indexes'][j]

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
    
        return label_blob, meta_data_blob, pose_blob, gt_boxes


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where linemod is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'LINEMOD')


    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._linemod_path, 'indexes', self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]
        return image_index


    def _load_object_points(self):

        points = [[] for _ in xrange(len(self._classes))]
        num = np.inf

        for i in xrange(1, len(self._classes)):
            point_file = os.path.join(self._linemod_path, 'models', self._classes[i] + '.xyz')
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
            # weight = 2.0 / np.amax(self._extents[i, :])
            weight = 1.0
            point_blob[i, :, :] = weight * point_blob[i, :, :]

        return points, points_all, point_blob


    def _load_object_extents(self):

        extent_file = os.path.join(self._linemod_path, 'extents.txt')
        assert os.path.exists(extent_file), \
                'Path does not exist: {}'.format(extent_file)

        extents = np.zeros((self._num_classes, 3), dtype=np.float32)
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
        metadata_path = os.path.join(self._data_path, index + '-meta-one.mat')
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(self._data_path, index + '-meta.mat')
            assert os.path.exists(metadata_path), \
                'Path does not exist: {}'.format(metadata_path)
        return metadata_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_linemod_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def _load_linemod_annotation(self, index):
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
