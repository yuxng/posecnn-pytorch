import torch
import torch.utils.data as data
import csv
import os, math
import sys
import time
import random
import imageio
import json
import os.path as osp
from os.path import *
import numpy as np
import numpy.random as npr
import cv2
import scipy.io
import glob
import matplotlib.pyplot as plt
import datasets
import platform
try:
    import cPickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle as cPickle

# these packages only support python3
if platform.python_version().startswith('3'):
    import three
    from datasets import rendering
    from pyrender import MetallicRoughnessMaterial, RenderFlags

from fcn.config import cfg
from utils.blob import pad_im, chromatic_transform, add_noise, add_noise_cuda, add_noise_depth_cuda
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import euler2quat
from utils.se3 import *
from pathlib import Path
from .shapenet_utils import *
from utils import augmentation
from utils import mask as util_
from utils.correspondences import create_non_correspondences, flatten_uv_tensor, create_non_matches


class ShapeNetEncoder(data.Dataset, datasets.imdb):
    def __init__(self, image_set, shapenet_object_path = None):

        self._name = 'shapenet_object_' + image_set
        self._image_set = image_set
        self._shapenet_object_path = self._get_default_path() if shapenet_object_path is None \
                            else shapenet_object_path
        self._classes_all = ('__background__', 'foreground')
        self._classes = self._classes_all

        self._model_path = self._shapenet_object_path
        self.textures_dir_coco = Path(os.path.join(datasets.ROOT_DIR, 'data', 'coco', 'val2014', 'val2014'))
        self.textures_dir_dtd = Path(os.path.join(datasets.ROOT_DIR, 'data', 'textures'))
        self.textures_dir_stratified = Path(os.path.join(datasets.ROOT_DIR, 'data', 'stratified'))

        # paths of tables (used as background)
        self.table_paths = get_shape_paths_categories(Path(self._model_path), categories=['table'])

        # Load blacklist
        self.taxonomy = load_taxonomy()
        blacklist_categories = ['table', 'chair']
        self.blacklist_synsets = set()
        if blacklist_categories is not None:
            for category in blacklist_categories:
                self.blacklist_synsets.update(category_to_synset_ids(self.taxonomy, category))
        print('filtered synsets', self.blacklist_synsets)

        # list all the 3D models
        self.shape_paths = get_shape_paths(Path(self._model_path), self.blacklist_synsets)
        self.model_num = len(self.shape_paths)
        self.model_list = np.random.permutation(self.model_num)
        self.model_index = 0
        print('%d 3D models' % (self.model_num))

        self._width = cfg.TRAIN.SYN_WIDTH
        self._height = cfg.TRAIN.SYN_HEIGHT
        self._intrinsic_matrix = np.array([[250, 0, 64],
                                          [0, 250, 64],
                                          [0, 0, 1]])
        self.Kinv = np.linalg.inv(self._intrinsic_matrix)
        self.canonical_dis = 2.5
        self.light_intensity = 10

        self.INTRINSIC = [[250, 0, 64, 0.0000],
                          [0, 250, 64, 0.0000],
                          [0.0000, 0.0000, 1.0000, 0.0000]]

        self.class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (0, 64, 0), (64, 0, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64)]

        # rendering parameters
        self.use_constrained_cameras = False
        self.disk_sample_cameras = False
        self.x_bound = (-0.1, 0.1)
        self.y_bound = (self.x_bound[0] / self._width * self._height,
                        self.x_bound[1] / self._width * self._height)
        self.z_bound = (0.5, 1.0)
        self.min_lights = 1
        self.max_lights = 8
        self.camera_angle_min = 20.0 * math.pi / 180.0
        self.camera_angle_max = 80.0 * math.pi / 180.0
        self.camera_theta_min = -20.0 * math.pi / 180.0
        self.camera_theta_max = 20.0 * math.pi / 180.0
        self.obj_default_pose = OBJ_DEFAULT_POSE
        self.size_jitter = (0.9, 1.1)
        self.use_model_materials = False
        self.random_materials = True
        self.use_textures = True
        self.roughness_values = load_roughness_values()
        self.texture_paths_coco = index_paths(self.textures_dir_coco, ext='.jpg')
        self.texture_paths_dtd = index_paths(self.textures_dir_dtd, ext='.jpg')
        self.texture_paths_stratified = index_paths(self.textures_dir_stratified, ext='.jpg')

        self._size = cfg.TRAIN.SYNNUM
        self._build_uniform_poses()
        self._log = None
        self.target_object_path = None

        margin = 20
        self.lb_shift = -margin
        self.ub_shift = margin
        self.lb_scale = 0.8
        self.ub_scale = 1.2

        assert os.path.exists(self._shapenet_object_path), \
                'shapenet_object path does not exist: {}'.format(self._shapenet_object_path)


    def worker_init_fn(self, worker_id):
        self._worker_id = worker_id
        self._renderer = rendering.Renderer(width=self._width, height=self._height)


    def load_random_image(self, paths):
        while True:
            image_path = random.choice(paths)
            try:
                image = imageio.imread(image_path)
                if len(image.shape) != 3 or image.shape[2] < 3:
                    continue
                return image[:, :, :3], image_path
            except Exception:
                self._log.warning("failed to read image", path=image_path)


    def get_random_material(self, is_coco=True):
        roughness = random.choice(self.roughness_values)
        metalness = random.uniform(0.0, 1.0)

        if random.random() < 0.8:
            if is_coco:
                image, image_path = self.load_random_image(self.texture_paths_coco)
            else:
                if random.random() < 0.8:
                    image, image_path = self.load_random_image(self.texture_paths_dtd)
                else:
                    image, image_path = self.load_random_image(self.texture_paths_stratified)
            base_color = [1.0, 1.0, 1.0]
        else:
            base_color = np.random.uniform(0.2, 1.0, 3)
            image = None
            image_path = None

        return MetallicRoughnessMaterial(
            alphaMode='BLEND',
            roughnessFactor=roughness,
            metallicFactor=metalness,
            baseColorFactor=base_color,
            baseColorTexture=image,
        )


    def get_color_material(self, base_color=[1.0, 0.0, 0.0]):
        roughness = random.choice(self.roughness_values)
        metalness = random.uniform(0.0, 1.0)

        return MetallicRoughnessMaterial(
            alphaMode='BLEND',
            roughnessFactor=roughness,
            metallicFactor=metalness,
            baseColorFactor=base_color,
            baseColorTexture=None,
        )


    def random_poses(self, n, constrained=False, disk_sample=False, z_bound=None):
        if z_bound is None:
            z_bound = self.z_bound
        translation = three.rigid.random_translation(n, self.x_bound, self.y_bound, z_bound)

        if constrained:
            '''
            rot_quats = three.orientation.sample_segment_quats(
                n=n,
                up=(0.0, 0.0, 1.0),
                min_angle=self.camera_angle_min,
                max_angle=self.camera_angle_max)
            '''
            rot_quats = three.orientation.sample_azimuth_elevation_quats(
                n, self.camera_angle_min, self.camera_angle_max, self.camera_theta_min, self.camera_theta_max)
        else:
            if disk_sample:
                rot_quats = three.orientation.disk_sample_quats(n, min_angle=math.pi/12)
            else:
                rot_quats = three.quaternion.random(n)

        # Rotate to canonical YCB pose (+Z is up)
        canon_quat = (three.quaternion.mat_to_quat(self.obj_default_pose)
                      .unsqueeze(0)
                      .expand_as(rot_quats))
        # Apply sampled rotated.
        rot_quats = three.quaternion.qmul(rot_quats, canon_quat)
        return translation, rot_quats


    def random_model(self):
        intrinsic = torch.tensor(self.INTRINSIC)
        size_jitter = random.uniform(*self.size_jitter)
        max_size = 2e7
        while True:
            object_path = random.choice(self.shape_paths)
            self.target_object_path = object_path
            if object_path.stat().st_size > max_size:
                    continue

            try:
                context = rendering.SceneContext(intrinsic)
                obj, _ = rendering.load_object(object_path, size=size_jitter, load_materials=self.use_model_materials)
                context.add_obj(obj)
                break
            except ValueError as e:
                print('exception while loading mesh', object_path)
                continue

        # assign random materials.
        if self.random_materials:
            for i in range(len(context.object_nodes)):
                object_node = context.object_nodes[i]
                for primitive in object_node.mesh.primitives:
                    if np.random.rand(1) < 0.5:
                        primitive.material = self.get_random_material(is_coco=False)
                    else:
                        primitive.material = self.get_random_material(is_coco=True)
        return context


    # add occluders in the scene
    def add_occluders(self, context, num):

        intrinsic = torch.tensor(self.INTRINSIC)
        max_size = 2e7
        while True:

            # sample objects
            object_path = []
            i = 0
            while True:
                path = random.choice(self.shape_paths)
                if path.stat().st_size > max_size:
                    continue
                elif self.target_object_path is not None and path == self.target_object_path:
                    continue
                else:
                    object_path.append(path)
                    i += 1
                if i >= num:
                    break

            # add objects
            try:
                txyz = np.zeros((num+1, 3), dtype=np.float32)
                bounding_size = np.zeros((num+1, ), dtype=np.float32)
                bounding_size[0] = context.objects[0].bounding_size
                for i in range(num):
                    obj_size = random.uniform(*self.size_jitter)
                    obj, _ = rendering.load_object(object_path[i], size=obj_size, load_materials=self.use_model_materials)
                    bsize = obj.bounding_size

                    # generate a random pose of the object
                    rot_quats = three.quaternion.random(1)

                    # translation, sample an object nearby
                    while True:
                        trans = np.random.uniform(-1.0, 1.0, size=3)
                        flag = 1
                        for j in range(i+1):
                            d = np.linalg.norm(trans - txyz[j, :])
                            if d < (bounding_size[j] + bsize) / 2:
                                flag = 0
                        if flag:
                            break

                    pose = np.eye(4)
                    pose[:3, :3] = quat2mat(rot_quats.squeeze().numpy())
                    pose[0, 3] = trans[0]
                    pose[1, 3] = trans[1]
                    pose[2, 3] = trans[2]
                    context.add_obj(obj, pose)
                    txyz[i+1, :] = trans
                break
            except ValueError as e:
                print('exception while loading mesh', object_path)
                continue

        # assign random materials.
        if self.random_materials:
            for i in range(1, len(context.object_nodes)):
                object_node = context.object_nodes[i]
                for primitive in object_node.mesh.primitives:
                    if np.random.rand(1) < 0.5:
                        primitive.material = self.get_random_material(is_coco=False)
                    else:
                        primitive.material = self.get_random_material(is_coco=True)
        return context


    def transform_image(self, image_tensor, rand=True):
        if rand:
            # RGB to BGR order
            im = image_tensor.numpy()
            im = np.clip(im, 0, 1)
            im = im[:, :, (2, 1, 0)] * 255
            im = im.astype(np.uint8)
            if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
                im = chromatic_transform(im)
            if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
                im = add_noise(im)
            im_tensor = torch.from_numpy(im) / 255.0
        else:
            im_tensor = image_tensor[:, :, (2, 1, 0)]
        im_tensor = torch.clamp(im_tensor, min=0.0, max=1.0)
        im_tensor = im_tensor.permute(2, 0, 1)
        return im_tensor


    # render views
    def _render_item(self):

        # sample and render a target object with constant lighting
        context = self.random_model()
        _, in_quaternions = self.random_poses(1, constrained=self.use_constrained_cameras, disk_sample=self.disk_sample_cameras)
        quaternion = in_quaternions[0]
        translation = torch.tensor([0, 0, self.canonical_dis]).float()
        context.set_pose(translation, quaternion)
        context.set_lighting(intensity=self.light_intensity)
        image_tensor_target, depth_target, mask_target = self._renderer.render(context)
        seg_target = np.around(mask_target.numpy())

        while True:

            # sample occluders
            if np.random.rand(1) < 0.8:
                num_occluder = 3
                context = self.add_occluders(context, num_occluder)
            else:
                num_occluder = 0

            # randomize lighting
            context.randomize_lights(self.min_lights, self.max_lights)
            context.set_pose(translation, quaternion)
            image_tensor, depth, mask = self._renderer.render(context)

            # render segmentation mask
            for i in range(len(context.object_nodes)):
                object_node = context.object_nodes[i]
                for primitive in object_node.mesh.primitives:
                    instance_color = np.array(self.class_colors_all[i]) / 255.0
                    primitive.material_old = primitive.material
                    primitive.material = self.get_color_material(instance_color)
            seg_tensor, _, _ = self._renderer.render(context, self._renderer._render_flags | RenderFlags.FLAT)

            im_label = seg_tensor.numpy() * 255
            im_label = np.round(im_label).astype(np.uint8)
            im_label = np.clip(im_label, 0, 255)
            label = self.process_label_image(im_label, num_occluder+1)

            # compute occlusion percentage
            non_occluded = np.sum(np.logical_and(seg_target > 0, seg_target == label)).astype(np.float)
            occluded_ratio = 1 - non_occluded / np.sum(seg_target > 0).astype(np.float)

            '''
            import matplotlib.pyplot as plt
            fig = plt.figure()
            m = 2
            n = 2
            start = 1
            ax = fig.add_subplot(m, n, start)
            start += 1
            im = image_tensor_target.cpu().numpy()
            im = np.clip(im, 0, 1)
            im = im[:, :, (2, 1, 0)] * 255
            im = im.astype(np.uint8)
            plt.imshow(im[:, :, (2, 1, 0)])
            ax.set_title('target')

            ax = fig.add_subplot(m, n, start)
            start += 1
            im = image_tensor.cpu().numpy()
            im = np.clip(im, 0, 1)
            im = im[:, :, (2, 1, 0)] * 255
            im = im.astype(np.uint8)
            plt.imshow(im[:, :, (2, 1, 0)])
            ax.set_title('input')

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(label)
            ax.set_title('input label')

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(seg_target)
            ax.set_title('occlusion ratio %.4f' % (occluded_ratio))
            plt.show()
            #'''

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

        image_blob = self.transform_image(image_tensor)
        image_target_blob = self.transform_image(image_tensor_target, rand=False)
        mask_blob = mask.unsqueeze(0).repeat((3, 1, 1)).float()

        # affine transformation
        shift = np.float32([np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)])
        scale = np.random.uniform(self.lb_scale, self.ub_scale)
        affine_matrix = np.float32([[scale, 0, shift[0] / self._width], [0, scale, shift[1] / self._height]])
        affine_blob = torch.from_numpy(affine_matrix)

        sample = {'image_input': image_blob,
                  'image_target': image_target_blob,
                  'mask': mask_blob,
                  'affine': affine_blob}

        return sample


    def __getitem__(self, index):

        return self._render_item()


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where shapenet_object is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'shapenet')


    def process_label_image(self, label_image, num):
        """
        change label image to label index
        """
        height = label_image.shape[0]
        width = label_image.shape[1]
        labels = np.zeros((height, width), dtype=np.int32)

        # label image is in RGB order
        index = label_image[:,:,0] + 256*label_image[:,:,1] + 256*256*label_image[:,:,2]
        count = 1
        for i in range(num):
            color = self.class_colors_all[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            if len(I[0]) > 0:
                labels[I[0], I[1]] = count
                count += 1

        return labels
