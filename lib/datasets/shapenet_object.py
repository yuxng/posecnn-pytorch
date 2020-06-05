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


class ShapeNetObject(data.Dataset, datasets.imdb):
    def __init__(self, image_set, shapenet_object_path = None):

        self._name = 'shapenet_object_' + image_set
        self._image_set = image_set
        self._shapenet_object_path = self._get_default_path() if shapenet_object_path is None \
                            else shapenet_object_path
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        self._classes_all = ('__background__', 'foreground')
        self._classes = self._classes_all
        if 'coseg' in image_set or cfg.TRAIN.EMBEDDING_CONTRASTIVE or cfg.TRAIN.EMBEDDING_PROTOTYPE:
            self._num_view = 2
        else:
            self._num_view = 1

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
        self._intrinsic_matrix = np.array([[524.7917885754071, 0, 332.5213232846151],
                                          [0, 489.3563960810721, 281.2339855172282],
                                          [0, 0, 1]])
        self.Kinv = np.linalg.inv(self._intrinsic_matrix)


        self.INTRINSIC = [[524.7917885754071, 0, 332.5213232846151, 0.0000],
                          [0, 489.3563960810721, 281.2339855172282, 0.0000],
                          [0.0000, 0.0000, 1.0000, 0.0000]]

        self.class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (0, 64, 0), (64, 0, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64)]

        # rendering parameters
        self.use_constrained_cameras =True
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

        # learning correspondences
        self.num_masked_non_matches_per_match = 3
        self.num_others_non_matches_per_match = 3
        self.num_background_non_matches_per_match = 3

        self._size = cfg.TRAIN.SYNNUM
        self._build_uniform_poses()
        self._log = None
        self.target_object_path = None

        self.lb_shift = -0.1
        self.ub_shift = 0.1
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


    def get_image_material(self, image_path):
        roughness = random.choice(self.roughness_values)
        metalness = random.uniform(0.0, 1.0)
        image = imageio.imread(image_path)
        base_color = [1.0, 1.0, 1.0]
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


    def random_model(self, num=1):
        intrinsic = torch.tensor(self.INTRINSIC)
        size_jitter = random.uniform(*self.size_jitter)
        max_size = 2e7
        has_table = False
        while True:
            # sample a table
            table_path = random.choice(self.table_paths)
            if table_path.stat().st_size > max_size:
                continue

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

            try:
                # table
                obj_table, _ = rendering.load_object(table_path, size=size_jitter, load_materials=self.use_model_materials)
                bounds = obj_table.bounds
                context = rendering.SceneContext(intrinsic)

                # object on table
                txz = np.zeros((num, 2), dtype=np.float32)
                for i in range(num):
                    obj_size = random.uniform(0.20, 0.25) * size_jitter
                    obj, _ = rendering.load_object(object_path[i], size=obj_size, load_materials=self.use_model_materials)

                    # generate a random pose of the object
                    rot_quats = three.quaternion.random(1)
                    ty = obj.bounding_size / 2
                    if i == 0:
                        translation = three.rigid.random_translation(1, 0.1 * bounds[:, 0], 0.1 * bounds[:, 1], 0.1 * bounds[:, 2])
                        tx = translation[0, 0]
                        tz = translation[0, 2]
                    else:
                        while True:
                            tx = np.random.uniform(-0.2, 0.2)
                            tz = np.random.uniform(-0.2, 0.2)
                            flag = 1
                            for j in range(i):
                                d = np.sqrt((tx - txz[j, 0]) * (tx - txz[j, 0]) + (tz - txz[j, 1]) * (tz - txz[j, 1]))
                                if d < 0.1:
                                    flag = 0
                            if flag:
                                break

                    pose = np.eye(4)
                    pose[:3, :3] = quat2mat(rot_quats.squeeze().numpy())
                    pose[0, 3] = tx
                    pose[1, 3] = ty
                    pose[2, 3] = tz
                    context.add_obj(obj, pose)
                    txz[i, 0] = tx
                    txz[i, 1] = tz

                # add table with some probability
                if np.random.rand(1) < cfg.TRAIN.SYN_TABLE_PROB:
                    pose_table = np.eye(4)
                    pose_table[1, 3] = bounds[0, 1]
                    context.add_obj(obj_table, pose_table)
                    has_table = True
                break
            except ValueError as e:
                print('exception while loading mesh', table_path)

        self.target_object_path = object_path[0]

        # Assign random materials.
        if self.random_materials:
            for i in range(len(context.object_nodes)):
                object_node = context.object_nodes[i]
                if has_table and i == len(context.object_nodes) - 1:
                    is_table = True
                else:
                    is_table = False
                for primitive in object_node.mesh.primitives:
                    if is_table or np.random.rand(1) < 0.5:
                        primitive.material = self.get_random_material(is_coco=False)
                    else:
                        primitive.material = self.get_random_material(is_coco=True)

        return context


    # remove objects in the scene except for the first one
    def remove_model(self, context):

        # remove nodes except for the first one
        for i in range(1, len(context.object_nodes)):
            context.scene.remove_node(context.object_nodes.pop())

        # recover material
        object_node = context.object_nodes[0]
        for primitive in object_node.mesh.primitives:
            primitive.material = primitive.material_old

        intrinsic = torch.tensor(self.INTRINSIC)
        size_jitter = random.uniform(*self.size_jitter)
        max_size = 2e7
        has_table = False
        while True:
            # sample a table
            table_path = random.choice(self.table_paths)
            if table_path.stat().st_size > max_size:
                continue

            try:
                # table
                obj_table, _ = rendering.load_object(table_path, size=size_jitter, load_materials=self.use_model_materials)
                bounds = obj_table.bounds

                # add talbe with some probability
                if np.random.rand(1) < cfg.TRAIN.SYN_TABLE_PROB:
                    has_table = True
                    pose_table = np.eye(4)
                    pose_table[1, 3] = bounds[0, 1]
                    context.add_obj(obj_table, pose_table)
                break
            except ValueError as e:
                print('exception while loading mesh', table_path)

        # Assign random materials.
        if self.random_materials:
            for i in range(1, len(context.object_nodes)):
                object_node = context.object_nodes[i]
                if has_table and i == len(context.object_nodes) - 1:
                    is_table = True
                else:
                    is_table = False
                for primitive in object_node.mesh.primitives:
                    if is_table or np.random.rand(1) < 0.5:
                        primitive.material = self.get_random_material(is_coco=False)
                    else:
                        primitive.material = self.get_random_material(is_coco=True)

        return context


    def pad_crop_resize(self, img, morphed_label, label, mask):
        """ Crop the image around the label mask, then resize to 224x224
        """

        H, W, _ = img.shape

        # Get tight box around label/morphed label
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(label)
        _xmin, _ymin, _xmax, _ymax = util_.mask_to_tight_box(morphed_label)
        x_min = min(x_min, _xmin); y_min = min(y_min, _ymin); x_max = max(x_max, _xmax); y_max = max(y_max, _ymax)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        # Make bbox square
        x_delta = x_max - x_min
        y_delta = y_max - y_min
        if x_delta > y_delta:
            y_min = cy - x_delta / 2
            y_max = cy + x_delta / 2
        else:
            x_min = cx - y_delta / 2
            x_max = cx + y_delta / 2

        sidelength = x_max - x_min
        padding_percentage = np.random.beta(cfg.TRAIN.padding_alpha, cfg.TRAIN.padding_beta)
        padding_percentage = max(padding_percentage, cfg.TRAIN.min_padding_percentage)
        padding = int(round(sidelength * padding_percentage))
        if padding == 0:
            print('Whoa, padding is 0... sidelength: {sidelength}, %: {padding_percentage}')
            padding = 25 # just make it 25 pixels

        # Pad and be careful of boundaries
        x_min = max(int(x_min - padding), 0)
        x_max = min(int(x_max + padding), W-1)
        y_min = max(int(y_min - padding), 0)
        y_max = min(int(y_max + padding), H-1)

        # Crop
        if (y_min == y_max) or (x_min == x_max):
            print('Whoa... something is wrong:', x_min, y_min, x_max, y_max)
            print(morphed_label)
            print(label)
        img_crop = img[y_min:y_max+1, x_min:x_max+1]
        mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
        morphed_label_crop = morphed_label[y_min:y_max+1, x_min:x_max+1]
        label_crop = label[y_min:y_max+1, x_min:x_max+1]
        roi = [x_min, y_min, x_max, y_max]

        # Resize
        s = cfg.TRAIN.SYN_CROP_SIZE
        img_crop = cv2.resize(img_crop, (s, s))
        mask_crop = cv2.resize(mask_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        morphed_label_crop = cv2.resize(morphed_label_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        label_crop = cv2.resize(label_crop, (s, s), interpolation=cv2.INTER_NEAREST)

        return img_crop, morphed_label_crop, label_crop, mask_crop, roi


    def transform(self, img, label, mask):
        """ Data augmentation for RGB image and label
                - RGB
                    - Image standardization
                - Label
                    - Morphological transformation
                    - rotation/translation
                    - adding/cutting
                    - random ellipses
        """

        img = img.astype(np.float32)

        # Data augmentation for mask
        morphed_label = label.copy()
        if not cfg.TRAIN.EMBEDDING_CONTRASTIVE:
            if np.random.rand() < cfg.TRAIN.rate_of_morphological_transform:
                morphed_label = augmentation.random_morphological_transform(morphed_label)

            if np.random.rand() < cfg.TRAIN.rate_of_translation:
                morphed_label = augmentation.random_translation(morphed_label)

            if np.random.rand() < cfg.TRAIN.rate_of_rotation:
                morphed_label = augmentation.random_rotation(morphed_label)

            sample = np.random.rand()
            if sample < cfg.TRAIN.rate_of_label_adding:
                morphed_label = augmentation.random_add(morphed_label)
            elif sample < cfg.TRAIN.rate_of_label_adding + cfg.TRAIN.rate_of_label_cutting:
                morphed_label = augmentation.random_cut(morphed_label)

            if np.random.rand() < cfg.TRAIN.rate_of_ellipses:
                morphed_label = augmentation.random_ellipses(morphed_label)

        # Next, crop the mask with some padding, and resize to 224x224. Make sure to preserve the aspect ratio
        img_crop, morphed_label_crop, label_crop, mask_crop, roi = self.pad_crop_resize(img, morphed_label, label, mask)

        return img_crop, morphed_label_crop, label_crop, mask_crop, roi


    # compute correspondences
    def compute_correspondences(self, pc_tensor_a, depth_a, label_a, quaternion_a, translation_a,
            pc_tensor_b, depth_b, label_b, quaternion_b, translation_b):

        index_a = torch.nonzero((label_a > 0) & (depth_a > 0))
        '''
        # backproject depth
        depth = depth_a[index_a[:,0], index_a[:,1]].numpy()
        num = index_a.shape[0]
        x2d = torch.cat([index_a[:, 1].unsqueeze(1), index_a[:, 0].unsqueeze(1), torch.ones((num, 1), dtype=torch.long)], dim=1).numpy()

        # backprojection
        R = np.dot(self.Kinv, x2d.transpose())

        # compute the 3D points
        X = np.multiply(np.tile(depth.reshape(1, num), (3, 1)), R)
        points = X.transpose()
        '''
        points = pc_tensor_a[index_a[:,0], index_a[:,1]].numpy()

        # transform points to the second camera
        RT1 = np.eye(4)
        RT1[:3, :3] = quat2mat(quaternion_a.numpy())
        RT1[:3, 3] = translation_a.numpy()

        RT2 = np.eye(4)
        RT2[:3, :3] = quat2mat(quaternion_b.numpy())
        RT2[:3, 3] = translation_b.numpy()

        points_new = np.ones((points.shape[0], 4), dtype=np.float32)
        points_new[:, :3] = points
        points_new = RT2 @ np.linalg.inv(RT1) @ points_new.transpose()
        points_new[0, :] /= points_new[3, :]
        points_new[1, :] /= points_new[3, :]
        points_new[2, :] /= points_new[3, :]
        points_new = points_new.transpose()[:, :3]

        # project points
        x2d = np.matmul(self._intrinsic_matrix, points_new.transpose())
        x2d[0, :] /= x2d[2, :]
        x2d[1, :] /= x2d[2, :]

        # compare depth values
        x1 = np.round(x2d[0, :])
        x1 = np.clip(x1, 0, depth_b.shape[1]-1)
        y1 = np.round(x2d[1, :])
        y1 = np.clip(y1, 0, depth_b.shape[0]-1)
        index_b = np.concatenate((np.expand_dims(x1, 1), np.expand_dims(y1, 1)), axis=1)
        depth_image = depth_b[y1, x1].numpy()
        depth_point = points_new[:, 2]

        # visible points
        index_vis = (np.absolute(depth_image - depth_point) < 0.003) & (label_b[y1, x1].numpy() > 0)

        # pixel-wise correspondences
        index_b = torch.from_numpy(index_b[index_vis, :])
        uv_a = [index_a[index_vis, 1].float(), index_a[index_vis, 0].float()]
        uv_b = [index_b[:, 0], index_b[:, 1]]
        return [uv_a, uv_b]


    def _render_item(self):

        # render views
        if cfg.TRAIN.SYN_CROP:
            height = cfg.TRAIN.SYN_CROP_SIZE
            width = cfg.TRAIN.SYN_CROP_SIZE
        else:
            height = cfg.TRAIN.SYN_HEIGHT
            width = cfg.TRAIN.SYN_WIDTH

        image_blob = torch.zeros([self._num_view, 3, height, width], dtype=torch.float32)
        mask_blob = torch.zeros([self._num_view, 3, height, width], dtype=torch.float32)
        label_blob = torch.zeros([self._num_view, 1, height, width], dtype=torch.float32)
        initial_mask_blob = torch.zeros([self._num_view, 1, height, width], dtype=torch.float32)
        affine_blob = torch.zeros([self._num_view, 2, 3], dtype=torch.float32)
        flag_blob = torch.zeros([1], dtype=torch.float32)

        if cfg.TRAIN.EMBEDDING_PIXELWISE:
            length = cfg.TRAIN.SYN_HEIGHT * cfg.TRAIN.SYN_WIDTH
            matches_a_blob = torch.zeros([length], dtype=torch.long)
            matches_b_blob = torch.zeros([length], dtype=torch.long)
            masked_non_matches_a_blob = torch.zeros([length * self.num_masked_non_matches_per_match], dtype=torch.long)
            masked_non_matches_b_blob = torch.zeros([length * self.num_masked_non_matches_per_match], dtype=torch.long)
            others_non_matches_a_blob = torch.zeros([length * self.num_others_non_matches_per_match], dtype=torch.long)
            others_non_matches_b_blob = torch.zeros([length * self.num_others_non_matches_per_match], dtype=torch.long)
            background_non_matches_a_blob = torch.zeros([length * self.num_background_non_matches_per_match], dtype=torch.long)
            background_non_matches_b_blob = torch.zeros([length * self.num_background_non_matches_per_match], dtype=torch.long)

        while True:
            # sample a table and a 3D object model
            num = np.random.randint(cfg.TRAIN.SYN_MIN_OBJECT, cfg.TRAIN.SYN_MAX_OBJECT+1)
            context = self.random_model(num)

            # samples poses
            in_translations, in_quaternions = self.random_poses(1, constrained=self.use_constrained_cameras, disk_sample=self.disk_sample_cameras)
            translation_a = in_translations[0]
            quaternion_a = in_quaternions[0]
            context.randomize_lights(self.min_lights, self.max_lights)
            context.set_pose(translation_a, quaternion_a)
            image_tensor_a, depth_a, mask_a = self._renderer.render(context)

            # render segmentation mask
            for i in range(len(context.object_nodes)):
                object_node = context.object_nodes[i]
                for primitive in object_node.mesh.primitives:
                    instance_color = np.array(self.class_colors_all[i]) / 255.0
                    primitive.material_old = primitive.material
                    primitive.material = self.get_color_material(instance_color)
            seg_tensor_a, _, _ = self._renderer.render(context, self._renderer._render_flags | RenderFlags.FLAT)

            im_label_a = seg_tensor_a.numpy() * 255
            im_label_a = np.round(im_label_a).astype(np.uint8)
            im_label_a = np.clip(im_label_a, 0, 255)
            label_a = self.process_label_image(im_label_a, num)
            label_a = torch.from_numpy(label_a)
            flag = torch.sum(label_a > 0) > 500
            if not flag:
                continue

            # render the second image if necessary
            if self._num_view == 2:

                if np.random.rand(1) < 0.5:
                    flag_positive = 1.0
                    context = self.remove_model(context)
                else:
                    flag_positive = 0.0
                    del context
                    context = self.random_model(num=1)

                quaternion_b = three.quaternion.perturb(quaternion_a, 0.1)
                # rendering
                context.randomize_lights(self.min_lights, self.max_lights)
                context.set_pose(translation_a, quaternion_b)
                image_tensor_b, depth_b, mask_b = self._renderer.render(context)

                # render segmentation mask
                for i in range(len(context.object_nodes)):
                    object_node = context.object_nodes[i]
                    for primitive in object_node.mesh.primitives:
                        instance_color = np.array(self.class_colors_all[i]) / 255.0
                        primitive.material = self.get_color_material(instance_color)
                seg_tensor_b, _, _ = self._renderer.render(context, self._renderer._render_flags | RenderFlags.FLAT)

                im_label_b = seg_tensor_b.numpy() * 255
                im_label_b = np.round(im_label_b).astype(np.uint8)
                im_label_b = np.clip(im_label_b, 0, 255)
                label_b = self.process_label_image(im_label_b, num=1)
                label_b = torch.from_numpy(label_b)
                flag = flag & (torch.sum(label_b > 0) > 500)
                if not flag:
                    continue

            if cfg.TRAIN.EMBEDDING_PIXELWISE:
                # compute correspondences
                uv_a, uv_b = self.compute_correspondences(pc_tensor_a, depth_a, label_a, quaternion_a, translation_a,
                    pc_tensor_b, depth_b, label_b, quaternion_b, translation_b)
                flag = flag & (len(uv_a[0]) > 10)

            del context
            self.target_object_path = None

            if flag:
                break

        '''
        import matplotlib.pyplot as plt
        fig = plt.figure()
        if self._num_view == 2:
            m = 3
            n = 3
        else:
            m = 1
            n = 3
        start = 1
        ax = fig.add_subplot(m, n, start)
        start += 1
        im = image_tensor_a.cpu().numpy()
        im = np.clip(im, 0, 1)
        im = im[:, :, (2, 1, 0)] * 255
        im = im.astype(np.uint8)
        plt.imshow(im[:, :, (2, 1, 0)])
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(label_a)
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(depth_a.numpy())
        if self._num_view == 2:
            ax = fig.add_subplot(m, n, start)
            start += 1
            im = image_tensor_b.cpu().numpy()
            im = np.clip(im, 0, 1)
            im = im[:, :, (2, 1, 0)] * 255
            im = im.astype(np.uint8)
            plt.imshow(im[:, :, (2, 1, 0)])
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(label_b.numpy())
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(depth_b.numpy())
        plt.show()
        #'''

        for i in range(self._num_view):

            if i == 0:
                image_tensor = image_tensor_a
                depth_tensor = depth_a
                mask = mask_a
                label = label_a
            else:
                image_tensor = image_tensor_b
                depth_tensor = depth_b
                mask = mask_b
                label = label_b

            if cfg.TRAIN.SYN_CROP:
                img = image_tensor.numpy()
                label = (label == 1).float().numpy()
                mask = mask.numpy()
                img_crop, morphed_label_crop, label_crop, mask_crop, roi = self.transform(img, label, mask)
                image_tensor = torch.from_numpy(img_crop)
                label = torch.from_numpy(label_crop)
                mask = torch.from_numpy(mask_crop)
                initial_label = torch.from_numpy(morphed_label_crop)

                if cfg.TRAIN.EMBEDDING_PIXELWISE:
                    x1, y1, x2, y2 = roi
                    sx = float(x2 - x1 + 1) / float(cfg.TRAIN.SYN_CROP_SIZE)
                    sy = float(y2 - y1 + 1) / float(cfg.TRAIN.SYN_CROP_SIZE)
                    uv_a[0] = (uv_a[0] - x1) / sx
                    uv_a[1] = (uv_a[1] - y1) / sy
            else:
                initial_label = label.clone()
            w = initial_label.shape[1]
            h = initial_label.shape[0]
            initial_mask_blob[i, 0, :h, :w] = initial_label

            # foreground mask
            mask = mask.unsqueeze(0).repeat((3, 1, 1)).float()
            mask_blob[i, :, :h, :w] = mask

            # label blob
            label_blob[i, 0, :, :] = label

            # RGB to BGR order
            im = image_tensor.cpu().numpy()
            im = np.clip(im, 0, 1)
            im = im[:, :, (2, 1, 0)] * 255
            im = im.astype(np.uint8)

            # chromatic transform
            if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
                im = chromatic_transform(im)

            if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
                im = add_noise(im)
            im_tensor = torch.from_numpy(im) / 255.0
            im_tensor -= self._pixel_mean
            im_tensor = im_tensor.permute(2, 0, 1)
            image_blob[i, :, :h, :w] = im_tensor

            # affine transformation
            shift = np.float32([np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)])
            scale = np.random.uniform(self.lb_scale, self.ub_scale)
            affine_matrix = np.float32([[scale, 0, shift[0]], [0, scale, shift[1]]])
            affine_blob[i] = torch.from_numpy(affine_matrix)

        if self._num_view == 1:
            image_blob = torch.squeeze(image_blob, 0)
            mask_blob = torch.squeeze(mask_blob, 0)
            label_blob = torch.squeeze(label_blob, 0)
            initial_mask_blob = torch.squeeze(initial_mask_blob, 0)
            affine_blob = torch.squeeze(affine_blob, 0)
        else:
            flag_blob[0] = flag_positive

        sample = {'image_color': image_blob,
                  'label': label_blob,
                  'initial_mask': initial_mask_blob,
                  'mask': mask_blob,
                  'affine': affine_blob}

        if self._num_view > 1:
            sample['flag'] = flag_blob

        # compute non-matches
        if cfg.TRAIN.EMBEDDING_PIXELWISE:

            # object non matches
            uv_b_masked_non_matches = create_non_correspondences(uv_b, label.shape, self.num_masked_non_matches_per_match, img_b_mask=label)
            uv_a_masked_long, uv_b_masked_non_matches_long = create_non_matches(uv_a, uv_b_masked_non_matches,
                                                                                self.num_masked_non_matches_per_match)

            # others non matches
            uv_b_others_non_matches = create_non_correspondences(uv_b, label_others_b.shape,
                self.num_others_non_matches_per_match, img_b_mask=label_others_b)
            uv_a_others_long, uv_b_others_non_matches_long = create_non_matches(uv_a, uv_b_others_non_matches,
                                                                                self.num_others_non_matches_per_match)

            # background non matches
            uv_b_background_non_matches = create_non_correspondences(uv_b, label.shape, self.num_background_non_matches_per_match,
                                                                     img_b_mask = 1 - (label + label_others_b))
            uv_a_background_long, uv_b_background_non_matches_long = create_non_matches(uv_a, uv_b_background_non_matches,
                                                                                        self.num_background_non_matches_per_match)

            matches_a = flatten_uv_tensor(uv_a, image_width=width)
            matches_b = flatten_uv_tensor(uv_b, image_width=width)
            matches_a_blob[:len(matches_a)] = matches_a
            matches_b_blob[:len(matches_b)] = matches_b

            masked_non_matches_a = flatten_uv_tensor(uv_a_masked_long, width).squeeze(1)
            masked_non_matches_b = flatten_uv_tensor(uv_b_masked_non_matches_long, width).squeeze(1)
            masked_non_matches_a_blob[:len(masked_non_matches_a)] = masked_non_matches_a
            masked_non_matches_b_blob[:len(masked_non_matches_b)] = masked_non_matches_b

            others_non_matches_a = flatten_uv_tensor(uv_a_others_long, width).squeeze(1)
            others_non_matches_b = flatten_uv_tensor(uv_b_others_non_matches_long, width).squeeze(1)
            others_non_matches_a_blob[:len(others_non_matches_a)] = others_non_matches_a
            others_non_matches_b_blob[:len(others_non_matches_b)] = others_non_matches_b

            background_non_matches_a = flatten_uv_tensor(uv_a_background_long, width).squeeze(1)
            background_non_matches_b = flatten_uv_tensor(uv_b_background_non_matches_long, width).squeeze(1)
            background_non_matches_a_blob[:len(background_non_matches_a)] = background_non_matches_a
            background_non_matches_b_blob[:len(background_non_matches_b)] = background_non_matches_b

            sample['matches_a'] = matches_a_blob
            sample['matches_b'] = matches_b_blob
            sample['masked_non_matches_a'] = masked_non_matches_a_blob
            sample['masked_non_matches_b'] = masked_non_matches_b_blob
            sample['others_non_matches_a'] = others_non_matches_a_blob
            sample['others_non_matches_b'] = others_non_matches_b_blob
            sample['background_non_matches_a'] = background_non_matches_a_blob
            sample['background_non_matches_b'] = background_non_matches_b_blob

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
