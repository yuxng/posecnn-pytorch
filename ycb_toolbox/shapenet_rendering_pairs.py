import os
import os.path
import torch
import torch.nn as nn
import cv2
import numpy as np
import glob
import random
import math
from transforms3d.quaternions import mat2quat, quat2mat
import _init_paths
import three
from pathlib import Path
from fcn.config import cfg
from datasets import rendering
from datasets import ShapeNetEncoder
from datasets.factory import get_dataset
from datasets.shapenet_utils import *
from pyrender import RenderFlags


# render views
def render_item(shapenet, renderer, obj, translation, quaternion, materials=None):

    # create context
    try:
        context = rendering.SceneContext(intrinsic)
        context.add_obj(obj)
    except ValueError as e:
        return None, materials

    # assign random materials.
    if materials is None:
        materials = []
        for i in range(len(context.object_nodes)):
            object_node = context.object_nodes[i]
            for primitive in object_node.mesh.primitives:
                if np.random.rand(1) < 0.5:
                    primitive.material = shapenet.get_random_material(is_coco=False)
                else:
                    primitive.material = shapenet.get_random_material(is_coco=True)
                materials.append(primitive.material)
    else:
        count = 0
        for i in range(len(context.object_nodes)):
            object_node = context.object_nodes[i]
            for primitive in object_node.mesh.primitives:
                primitive.material = materials[count]
                count += 1

    # render target
    context.set_pose(translation, quaternion)
    context.set_lighting(intensity=30)
    image_tensor_target, depth_target, mask_target = renderer.render(context)
    seg_target = np.around(mask_target.numpy())
    if np.sum(seg_target) < 100:
        return None, materials

    while True:

        # sample occluders
        if np.random.rand(1) < 0.8:
            num_occluder = 3
            context = shapenet.add_occluders(context, num_occluder)
        else:
            num_occluder = 0

        # randomize lighting
        context.randomize_lights(shapenet.min_lights, shapenet.max_lights)
        context.set_pose(translation, quaternion)
        image_tensor, depth, mask = renderer.render(context)

        # render segmentation mask
        for i in range(len(context.object_nodes)):
            object_node = context.object_nodes[i]
            for primitive in object_node.mesh.primitives:
                instance_color = np.array(shapenet.class_colors_all[i]) / 255.0
                primitive.material_old = primitive.material
                primitive.material = shapenet.get_color_material(instance_color)
        seg_tensor, _, _ = renderer.render(context, renderer._render_flags | RenderFlags.FLAT)

        im_label = seg_tensor.numpy() * 255
        im_label = np.round(im_label).astype(np.uint8)
        im_label = np.clip(im_label, 0, 255)
        label = shapenet.process_label_image(im_label, num_occluder+1)

        # compute occlusion percentage
        non_occluded = np.sum(np.logical_and(seg_target > 0, seg_target == label)).astype(np.float)
        occluded_ratio = 1 - non_occluded / np.sum(seg_target > 0).astype(np.float)

        if occluded_ratio < 0.8:
            del context
            break
        else:
            # remove occluders
            for i in range(num_occluder):
                context.scene.remove_node(context.object_nodes.pop())
                context.objects.pop()
            # recover material
            object_node = context.object_nodes[0]
            for primitive in object_node.mesh.primitives:
                primitive.material = primitive.material_old

    image_blob = shapenet.transform_image(image_tensor)
    image_target_blob = shapenet.transform_image(image_tensor_target, rand=False)
    mask_blob = mask.unsqueeze(0).repeat((3, 1, 1)).float()

    # affine transformation
    shift = np.float32([np.random.uniform(shapenet.lb_shift, shapenet.ub_shift), np.random.uniform(shapenet.lb_shift, shapenet.ub_shift)])
    scale = np.random.uniform(shapenet.lb_scale, shapenet.ub_scale)
    affine_matrix = np.float32([[scale, 0, shift[0] / shapenet._width], [0, scale, shift[1] / shapenet._height]])
    affine_blob = torch.from_numpy(affine_matrix)

    sample = {'image_input': image_blob.unsqueeze(0),
              'image_target': image_target_blob.unsqueeze(0),
              'mask': mask_blob.unsqueeze(0),
              'affine': affine_blob.unsqueeze(0)}

    return sample, materials


def process_sample(sample, background_loader):

    # construct input
    image_input = sample['image_input'].cuda()
    image_target = sample['image_target'].cuda()
    mask = sample['mask'].cuda()
    affine_matrix = sample['affine'].cuda()

    # affine transformation
    grids = nn.functional.affine_grid(affine_matrix, image_input.size())
    image_input = nn.functional.grid_sample(image_input , grids, padding_mode='border')
    mask = nn.functional.grid_sample(mask, grids, mode='nearest')

    # load a random background
    try:
        _, background = next(enum_background)
    except:
        enum_background = enumerate(background_loader)
        _, background = next(enum_background)

    num = image_input.size(0)
    if background['background_color'].size(0) < num:
        enum_background = enumerate(background_loader)
        _, background = next(enum_background)

    background_color = background['background_color'].cuda()
    inputs = mask * image_input + (1 - mask) * background_color[:num]
    inputs = torch.clamp(inputs, min=0.0, max=1.0)

    # add truncation
    if np.random.uniform(0, 1) < 0.1:
        affine_mat_trunc = torch.zeros(inputs.size(0), 2, 3).float().cuda()
        affine_mat_trunc[:, 0, 2] = torch.from_numpy(np.random.uniform(-1, 1, size=(inputs.size(0),))).cuda()
        affine_mat_trunc[:, 1, 2] = torch.from_numpy(np.random.uniform(-1, 1, size=(inputs.size(0),))).cuda()
        affine_mat_trunc[:, 0, 0] = 1
        affine_mat_trunc[:, 1, 1] = 1
        aff_grid_trunc = nn.functional.affine_grid(affine_mat_trunc, inputs.size()).cuda()
        inputs = nn.functional.grid_sample(inputs, aff_grid_trunc)

        affine_mat_trunc[:, 0, 2] *= -1
        affine_mat_trunc[:, 1, 2] *= -1
        aff_grid_trunc = nn.functional.affine_grid(affine_mat_trunc, inputs.size()).cuda()
        inputs = nn.functional.grid_sample(inputs, aff_grid_trunc)

    return inputs, image_target


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == '__main__':
    shapenet = ShapeNetEncoder('train')

    cfg.TRAIN.SYN_HEIGHT = 128
    cfg.TRAIN.SYN_WIDTH = 128
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
    is_save = True

    # background loader
    background_dataset = get_dataset('background_texture')
    background_loader = torch.utils.data.DataLoader(background_dataset, batch_size=1,
                                                    shuffle=True, num_workers=0)

    # for each object
    num = len(shapenet.shape_paths)
    index_lists = list(split(range(num), 6))
    choose = 5
    for k in index_lists[choose]:
        object_path = shapenet.shape_paths[k]
        if object_path.stat().st_size > max_size:
            continue

        # load object
        try:
            obj, _ = rendering.load_object(object_path, load_materials=shapenet.use_model_materials)
        except ValueError as e:
            print('exception while loading mesh', object_path)
            continue

        # make dir
        parts = object_path.parts
        ind = parts.index('shapenet')
        folder = os.path.join(root_dir, parts[ind+1], parts[ind+2])
        print('%d/%d: %s' % (k, index_lists[choose][-1], folder))
        if not os.path.exists(folder):
            os.makedirs(folder)

        # render poses
        interval = 45
        count = 0
        materials = None
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

                    # rendering
                    sample, materials = render_item(shapenet, renderer, obj, translation, quaternion, materials)

                    if sample is None:
                        continue

                    # process sample
                    inputs, image_target = process_sample(sample, background_loader)

                    # convert image
                    im = image_target.cpu().numpy()[0]
                    im = im.transpose((1, 2, 0))
                    im = np.clip(im, 0, 1) * 255
                    im = im.astype(np.uint8)

                    im1 = inputs.cpu().numpy()[0]
                    im1 = im1.transpose((1, 2, 0))
                    im1 = np.clip(im1, 0, 1) * 255
                    im1 = im1.astype(np.uint8)

                    # save image in BGR order
                    if is_save:
                        filename = os.path.join(folder, '%06d-syn.jpg' % (count))
                        cv2.imwrite(filename, im)
                        filename = os.path.join(folder, '%06d-real.jpg' % (count))
                        cv2.imwrite(filename, im1)
                        count += 1
                    else:
                        # visualization
                        import matplotlib.pyplot as plt
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 2, 1)
                        plt.imshow(im1[:, :, (2, 1, 0)])
                        ax.set_title('image input')
                        ax = fig.add_subplot(1, 2, 2)
                        plt.imshow(im[:, :, (2, 1, 0)])
                        ax.set_title('image target')
                        plt.show()
