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

    shapenet._width = 128
    shapenet._height = 128
    focal = 250
    shapenet._intrinsic_matrix = np.array([[focal, 0, 64],
                                           [0, focal, 64],
                                           [0, 0, 1]])
    shapenet.Kinv = np.linalg.inv(shapenet._intrinsic_matrix)
    shapenet.INTRINSIC = [[focal, 0, 64],
                          [0, focal, 64],
                          [0, 0, 1]]
    canonical_dis = 2.5

    renderer = rendering.Renderer(width=shapenet._width, height=shapenet._height)
    intrinsic = torch.tensor(shapenet.INTRINSIC)
    max_size = 2e7
    root_dir = '/capri/ShapeNetCore-render'
    is_save = False

    # for each object
    num = len(shapenet.shape_paths)
    for k in range(num):
        object_path = shapenet.shape_paths[k]
        if object_path.stat().st_size > max_size:
            continue

        # load object
        try:
            context = rendering.SceneContext(intrinsic)
            obj, _ = rendering.load_object(object_path, load_materials=shapenet.use_model_materials)
            context.add_obj(obj)
        except ValueError as e:
            print('exception while loading mesh', object_path)
            continue

        # make dir
        parts = object_path.parts
        ind = parts.index('shapenet')
        folder = os.path.join(root_dir, parts[ind+1], parts[ind+2])
        print('%d/%d: %s' % (k, num, folder))
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Assign random materials.
        for i in range(len(context.object_nodes)):
            object_node = context.object_nodes[i]
            for primitive in object_node.mesh.primitives:
                if np.random.rand(1) < 0.5:
                    primitive.material = shapenet.get_random_material(is_coco=False)
                else:
                    primitive.material = shapenet.get_random_material(is_coco=True)

        # render poses
        interval = 45
        count = 0
        for azimuth in range(0, 360, interval):
            for elevation in [-45, 45]:
                for roll in range(0, 360, interval):

                    a = azimuth * math.pi / 180
                    e = elevation * math.pi / 180
                    r = roll * math.pi / 180

                    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
                    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
                    Ry = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
                    R = np.dot(Ry, np.dot(Rx, Rz))
                    quaternion = torch.from_numpy(mat2quat(R)).float()
                    translation = torch.tensor([0, 0, canonical_dis]).float()
                    context.set_pose(translation, quaternion)

                    context.set_lighting(intensity=10)
                    image_tensor, depth, mask = renderer.render(context)

                    # convert image
                    im = image_tensor.cpu().numpy()
                    im = np.clip(im, 0, 1) * 255
                    im = im.astype(np.uint8)

                    # save image in BGR order
                    if is_save:
                        filename = os.path.join(folder, '%06d.jpg' % (count))
                        cv2.imwrite(filename, im[:, :, (2, 1, 0)])
                        count += 1
                    else:
                        # visualization
                        import matplotlib.pyplot as plt
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        plt.imshow(im)
                        plt.show()

        del context
