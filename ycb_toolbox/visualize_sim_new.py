import torch
import cv2
import numpy as np
import glob
import scipy.io

import _init_paths
from transforms3d.quaternions import mat2quat, quat2mat
from ycb_renderer import YCBRenderer
from utils.se3 import *


if __name__ == '__main__':

    size = 'small'
    model_path = '/capri/YCB_Video_Dataset'
    data_path = '/capri/YCB_Self_Supervision/data/blocks_' + size + '/scene_00/'
    width = 640
    height = 480

    files = glob.glob(data_path + '*.mat')
    files.sort()

    renderer = YCBRenderer(width=width, height=height, render_marker=False)
    models = ['block_red_' + size, 'block_green_' + size, 'block_blue_' + size, 'block_yellow_' + size]
    if size == 'big':
        class_index = [26, 27, 28, 29]
    elif size == 'median':
        class_index = [34, 35, 36, 37]
    elif size == 'small':
        class_index = [30, 31, 32, 33]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]

    obj_paths = ['{}/models/{}/textured_simple.obj'.format(model_path, item) for item in models]
    texture_paths = ['' for item in models]
    renderer.load_objects(obj_paths, texture_paths, colors)
    renderer.set_camera_default()
    renderer.set_light_pos([0, 0, 0])
    renderer.set_light_color([1, 1, 1])

    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()

    for file_path in files:

        print(file_path)
        metadata = scipy.io.loadmat(file_path)
        indexes = metadata['cls_indexes'].flatten()
        intrinsic_matrix = metadata['intrinsic_matrix']
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.01
        renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
        print(metadata)
        print(fx, fy, px, py)

        # read image
        filename = file_path[:-8] + 'color.jpg'
        rgb = cv2.imread(filename)

        cls_indexes = []
        poses = []
        for i in range(len(indexes)):
            cls = indexes[i]
            print(i, cls)

            cls_index = np.where(class_index == cls)[0][0]
            cls_indexes.append(cls_index)
            RT = metadata['poses'][:, :, i]
            print(RT)
            qt = np.zeros((7, ), dtype=np.float32)
            qt[3:] = mat2quat(RT[:3, :3])
            qt[:3] = RT[:, 3]
            poses.append(qt)

            print(fx * qt[0] / qt[2] + px, fy * qt[1] / qt[2] + py)

        renderer.set_poses(poses)
        print(cls_indexes)
        renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)

        # RGB to BGR order
        im = image_tensor.cpu().numpy()
        im = np.clip(im, 0, 1)
        im = im[:, :, (2, 1, 0)] * 255
        im = im.astype(np.uint8)

        im_label = seg_tensor.cpu().numpy()
        im_label = im_label[:, :, (2, 1, 0)] * 255
        im_label = np.round(im_label).astype(np.uint8)
        im_label = np.clip(im_label, 0, 255)
    
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(rgb[:, :, (2, 1, 0)])

        ax = fig.add_subplot(2, 2, 3)
        plt.imshow(im[:, :, (2, 1, 0)])

        ax = fig.add_subplot(2, 2, 4)
        plt.imshow(im_label[:, :, (2, 1, 0)])
        plt.show()
