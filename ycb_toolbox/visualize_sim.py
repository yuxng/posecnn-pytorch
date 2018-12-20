import torch
import cv2
import numpy as np
import glob
from transforms3d.quaternions import mat2quat, quat2mat

import _init_paths
from ycb_renderer import YCBRenderer
from utils.se3 import *


if __name__ == '__main__':

    model_path = '/capri/YCB_Video_Dataset'
    width = 640
    height = 480
    files = glob.glob('debug_data/*.npy')

    renderer = YCBRenderer(width=width, height=height, render_marker=True)
    # models = ['003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', '010_potted_meat_can']
    # colors = [[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.5, 0.5, 0]]

    models = ['003_cracker_box']
    colors = [[0, 1, 0]]


    obj_paths = [
        '{}/models/{}/textured_simple.obj'.format(model_path, item) for item in models]
    texture_paths = [
        '{}/models/{}/texture_map.png'.format(model_path, item) for item in models]
    renderer.load_objects(obj_paths, texture_paths, colors)

    renderer.set_fov(40)
    renderer.set_light_pos([0, 0, 0])

    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    RT_object = np.zeros((3, 4), dtype=np.float32)
    RT_camera = np.zeros((3, 4), dtype=np.float32)

    for file_path in files[1:]:
      
        data = np.load(file_path).item()
        cls_indexes = []
        poses = []

        print('object_labels', data['object_labels'])
        print('fov', data['horizontal_fov'])

        for i, object_name in enumerate(data['object_labels']):

            cls_index = -1
            for j in range(len(models)):
                if object_name in models[j]:
                    cls_index = j
                    break

            if cls_index >= 0:
                cls_indexes.append(cls_index)

                w = data['absolute_poses'][i][6]
                x = data['absolute_poses'][i][3]
                y = data['absolute_poses'][i][4]
                z = data['absolute_poses'][i][5]	
                # RT_object[:3, :3] = quat2mat([w, x, y, z])
                RT_object[:3, :3] = np.eye(3)

                x = data['absolute_poses'][i][0]
                y = data['absolute_poses'][i][2]
                z = data['absolute_poses'][i][1]
                RT_object[:, 3] = [x, y, z]
                print 'RT_object'
                print RT_object

                RT = se3_mul(se3_inverse(RT_camera), RT_object)
                print 'RT'
                print RT

                qt = np.zeros((7, ), dtype=np.float32)
                qt[3:] = mat2quat(RT[:3, :3])
                qt[:3] = RT[:, 3]
                poses.append(qt)
            
            if object_name == 'camera':

                w = data['absolute_poses'][i][6]
                x = data['absolute_poses'][i][3]
                y = data['absolute_poses'][i][4]
                z = data['absolute_poses'][i][5]
                # RT_camera[:3, :3] = quat2mat([w, x, y, z])
                RT_camera[:3, :3] = np.eye(3)

                x = data['absolute_poses'][i][0]
                y = data['absolute_poses'][i][2]
                z = data['absolute_poses'][i][1]

                RT_camera[:, 3] = [x, y, z]
                renderer.set_camera([x, z, y], [0, 0, 0], [0, 1, 0])
                print 'RT_camera'
                print RT_camera

            print('object_name: {}, relative_qt = {}, absolute_qt = {}'.format(data['object_labels'][i], data['relative_poses'][i], data['absolute_poses'][i]))

        renderer.set_poses(poses)

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
        plt.imshow(data['rgb'][:, :, (2, 1, 0)])

        ax = fig.add_subplot(2, 2, 2)
        mask = np.squeeze(data['segmentation'], -1).astype(np.uint8)
        mask *= 40
        plt.imshow(mask)

        ax = fig.add_subplot(2, 2, 3)
        plt.imshow(im[:, :, (2, 1, 0)])

        ax = fig.add_subplot(2, 2, 4)
        plt.imshow(im_label[:, :, (2, 1, 0)])
        plt.show()
