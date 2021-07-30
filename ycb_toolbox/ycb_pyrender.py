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
from datasets import YCBObject
from datasets import ShapeNetObject
from pyrender import RenderFlags


if __name__ == '__main__':
    shapenet = ShapeNetObject('train')
    ycb = YCBObject('train')

    ycb._width = 640
    ycb._height = 480
    focal = 579.4113
    ycb._intrinsic_matrix = np.array([[focal, 0, 320],
                                      [0, focal, 240],
                                      [0, 0, 1]])
    ycb.Kinv = np.linalg.inv(ycb._intrinsic_matrix)
    ycb.INTRINSIC = [[focal, 0, 320],
                     [0, focal, 240],
                     [0, 0, 1]]
    canonical_dis = 2.5

    # switch Y and Z axis
    rot = np.array([[1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]])

    renderer = rendering.Renderer(width=ycb._width, height=ycb._height)
    intrinsic = torch.tensor(ycb.INTRINSIC)
    root_dir = '/capri/YCB-render/ycb-with-background'
    is_save = False

    # for each object
    num = 21
    for k in range(1, num):
        cls = ycb._classes_all[k + 1]
        object_path = ycb.model_mesh_paths[k]
        print(object_path)

        # load object
        try:
            context = rendering.SceneContext(intrinsic)
            obj, _ = rendering.load_object(object_path, load_materials=True)
            context.add_obj(obj)
        except ValueError as e:
            print('exception while loading mesh', object_path)
            continue

        # make dir
        folder = os.path.join(root_dir, cls)
        print('%d/%d: %s' % (k, num, folder))
        if not os.path.exists(folder):
            os.makedirs(folder)

        # render poses
        interval = 45
        count = 0
        for azimuth in range(0, 360, interval):
            for elevation in [-45, 45]:
                for roll in range(0, 360, interval):

                    a = azimuth * math.pi / 180
                    e = -elevation * math.pi / 180
                    r = roll * math.pi / 180

                    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
                    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
                    Ry = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
                    R1 = np.dot(Ry, np.dot(Rx, Rz))
                    R = np.dot(rot, R1)
                    quaternion = torch.from_numpy(mat2quat(R)).float()
                    translation = torch.tensor([0, 0, canonical_dis]).float()
                    context.set_pose(translation, quaternion)

                    # lighting
                    context.set_lighting(intensity=100)
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
