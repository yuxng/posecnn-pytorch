import os
import os.path
import torch
import cv2
import numpy as np
import glob
import random
import math
from transforms3d.quaternions import mat2quat, quat2mat
import _init_paths
import three
import posecnn_cuda
from pathlib import Path
from datasets import rendering
from datasets import ShapeNetObject
from datasets.shapenet_utils import *
from pyrender import RenderFlags


if __name__ == '__main__':
    shapenet = ShapeNetObject('train')

    cfg.TRAIN.SYN_HEIGHT = 224
    cfg.TRAIN.SYN_WIDTH = 224
    shapenet._width = 224
    shapenet._height = 224
    focal = 250
    shapenet._intrinsic_matrix = np.array([[focal, 0, 112],
                                           [0, focal, 112],
                                           [0, 0, 1]])
    shapenet.Kinv = np.linalg.inv(shapenet._intrinsic_matrix)
    shapenet.INTRINSIC = [[focal, 0, 112],
                          [0, focal, 112],
                          [0, 0, 1]]
    shapenet.z_bound = (0.8, 1.2)
    shapenet.restrict_table = 0.3

    renderer = rendering.Renderer(width=shapenet._width, height=shapenet._height)
    intrinsic = torch.tensor(shapenet.INTRINSIC)
    max_size = 2e7
    root_dir = '/capri/ShapeNetCore-crop/training_set'
    is_save = True
    num_scenes = 40000
    num_views = 7
    min_object = 3
    max_object = 5

    # for each scene
    for k in range(num_scenes):

        folder = os.path.join(root_dir, 'scene_%05d' % (k))
        print(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # sample a table and a 3D object model
        num = np.random.randint(min_object, max_object+1)
        context = shapenet.random_model(num)

        # samples poses
        in_translations, in_quaternions = shapenet.random_poses(
            num_views, constrained=shapenet.use_constrained_cameras, disk_sample=shapenet.disk_sample_cameras)

        images = []
        depths = []
        for i in range(num_views):
            translation = in_translations[i]
            quaternion = in_quaternions[i]
            context.randomize_lights(shapenet.min_lights, shapenet.max_lights)
            context.set_pose(translation, quaternion)
            image_tensor, depth, mask = renderer.render(context)

            # convert image
            im = image_tensor.numpy()
            im = np.clip(im, 0, 1) * 255
            im = im.astype(np.uint8)
            images.append(im)

            # convert depth
            depth_32 = depth.numpy() * 1000
            depth = np.array(depth_32, dtype=np.uint16)
            depths.append(depth)

        # render segmentation mask
        for i in range(len(context.object_nodes)):
            object_node = context.object_nodes[i]
            for primitive in object_node.mesh.primitives:
                instance_color = np.array(shapenet.class_colors_all[i]) / 255.0
                primitive.material_old = primitive.material
                primitive.material = shapenet.get_color_material(instance_color)

        im_labels = []
        labels = []
        for i in range(num_views):
            translation = in_translations[i]
            quaternion = in_quaternions[i]
            context.set_pose(translation, quaternion)
            seg_tensor, _, _ = renderer.render(context, renderer._render_flags | RenderFlags.FLAT)

            im_label = seg_tensor.numpy() * 255
            im_label = np.round(im_label).astype(np.uint8)
            im_label = np.clip(im_label, 0, 255)
            im_labels.append(im_label)
            label = shapenet.process_label_image(im_label, num)
            labels.append(label)

        # save image in BGR order
        if is_save:
            for i in range(num_views):
                # rgb
                filename = os.path.join(folder, 'rgb_%05d.jpg' % (i))
                cv2.imwrite(filename, images[i][:, :, (2, 1, 0)])
                # depth
                filename = os.path.join(folder, 'depth_%05d.png' % (i))
                cv2.imwrite(filename, depths[i])
                # label
                filename = os.path.join(folder, 'label_%05d.png' % (i))
                cv2.imwrite(filename, labels[i])
        else:
            # visualization
            import matplotlib.pyplot as plt
            for i in range(num_views):
                fig = plt.figure()
                ax = fig.add_subplot(2, 2, 1)
                plt.imshow(images[i])
                ax = fig.add_subplot(2, 2, 2)
                plt.imshow(depths[i])
                ax = fig.add_subplot(2, 2, 3)
                plt.imshow(im_labels[i])
                ax = fig.add_subplot(2, 2, 4)
                plt.imshow(labels[i])
                plt.show()

        del context
