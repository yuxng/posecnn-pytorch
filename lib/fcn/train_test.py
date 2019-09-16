# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import time
import sys, os
import numpy as np
import cv2
import scipy

from fcn.config import cfg
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.se3 import *
from utils.nms import *
from utils.pose_error import re, te
from scipy.optimize import minimize
from utils.loose_bounding_boxes import compute_centroids_and_loose_bounding_boxes, mean_shift_and_loose_bounding_boxes
from utils.blob import add_gaussian_noise_cuda

import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def loss_cross_entropy(scores, labels):
    """
    scores: a tensor [batch_size, num_classes, height, width]
    labels: a tensor [batch_size, num_classes, height, width]
    """

    cross_entropy = -torch.sum(labels * scores, dim=1)
    loss = torch.div(torch.sum(cross_entropy), torch.sum(labels)+1e-10)

    return loss


def BootstrapedMSEloss(pred, target, K=20, factor=10):
    assert pred.dim() == target.dim(), "inconsistent dimensions"
    batch_size = pred.size(0)
    diff = torch.sum((target - pred)**2, 1)
    diff = diff.view(batch_size, -1)
    diff = torch.topk(diff, K, dim=1)
    loss = factor * diff[0].mean()
    loss_batch = factor * torch.mean(diff[0], dim=1)
    return loss, loss_batch


def smooth_l1_loss(vertex_pred, vertex_targets, vertex_weights, sigma=1.0):
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = torch.mul(vertex_weights, vertex_diff)
    abs_diff = torch.abs(diff)
    smoothL1_sign = torch.lt(abs_diff, 1. / sigma_2).float().detach()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss = torch.div( torch.sum(in_loss), torch.sum(vertex_weights) + 1e-10 )
    return loss


'''
sample = {'image_color': im_blob,
          'im_depth': im_depth,
          'label': label_blob,
          'mask': mask,
          'meta_data': meta_data_blob,
          'poses': pose_blob,
          'extents': self._extents,
          'points': self._point_blob,
          'symmetry': self._symmetry,
          'gt_boxes': gt_boxes,
          'im_info': im_info,
          'vertex_targets': vertex_targets,
          'vertex_weights': vertex_weights}
'''

def train(train_loader, background_loader, network, optimizer, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()

    epoch_size = len(train_loader)
    enum_background = enumerate(background_loader)
    pixel_mean = torch.from_numpy(cfg.PIXEL_MEANS.transpose(2, 0, 1) / 255.0).float()

    # switch to train mode
    network.train()

    for i, sample in enumerate(train_loader):

        end = time.time()

        if cfg.INPUT == 'COLOR':
            inputs = sample['image_color']
        elif cfg.INPUT == 'RGBD':
            inputs = torch.cat((sample['image_color'], sample['image_depth'], sample['mask_depth']), dim=1)
        im_info = sample['im_info']
        mask = sample['mask']
        labels = sample['label'].cuda()
        meta_data = sample['meta_data'].cuda()
        extents = sample['extents'][0, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1).cuda()
        gt_boxes = sample['gt_boxes'].cuda()
        poses = sample['poses'].cuda()
        points = sample['points'][0, :, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1, 1).cuda()
        symmetry = sample['symmetry'][0, :].repeat(cfg.TRAIN.GPUNUM, 1).cuda()

        if cfg.TRAIN.VERTEX_REG or cfg.TRAIN.VERTEX_REG_DELTA:
            vertex_targets = sample['vertex_targets'].cuda()
            vertex_weights = sample['vertex_weights'].cuda()
        else:
            vertex_targets = []
            vertex_weights = []

        # affine transformation
        if cfg.TRAIN.AFFINE:
            affine_matrix = sample['affine_matrix']
            affine_matrix_coordinate = sample['affine_matrix_coordinate']
            grids = nn.functional.affine_grid(affine_matrix, inputs.size())
            inputs = nn.functional.grid_sample(inputs, grids, padding_mode='border')
            mask = nn.functional.grid_sample(mask, grids, mode='nearest')

            grids_label = nn.functional.affine_grid(affine_matrix, labels.size())
            labels = nn.functional.grid_sample(labels, grids_label, mode='nearest')
            labels_sum = torch.sum(labels, dim=1)
            labels[:, 0, :, :] = 1 - labels_sum + labels[:, 0, :, :]

            box_tensor = torch.cuda.FloatTensor(3, gt_boxes.shape[1]).detach()
            for j in range(gt_boxes.shape[0]):
                index = gt_boxes[j, :, 4] > 0
                box_tensor[0, :] = gt_boxes[j, :, 0]
                box_tensor[1, :] = gt_boxes[j, :, 1]
                box_tensor[2, :] = 1.0
                box_new = torch.mm(affine_matrix_coordinate[j], box_tensor)
                gt_boxes[j, index, 0] = box_new[0, index]
                gt_boxes[j, index, 1] = box_new[1, index]

                box_tensor[0, :] = gt_boxes[j, :, 2]
                box_tensor[1, :] = gt_boxes[j, :, 3]
                box_tensor[2, :] = 1.0
                box_new = torch.mm(affine_matrix_coordinate[j], box_tensor)
                gt_boxes[j, index, 2] = box_new[0, index]
                gt_boxes[j, index, 3] = box_new[1, index]
            sample['gt_boxes'] = gt_boxes.cpu()

            if cfg.TRAIN.VERTEX_REG or cfg.TRAIN.VERTEX_REG_DELTA:
                grids_vertex = nn.functional.affine_grid(affine_matrix, vertex_targets.size())
                vertex_targets = nn.functional.grid_sample(vertex_targets, grids_vertex, mode='nearest')
                vertex_weights = nn.functional.grid_sample(vertex_weights, grids_vertex, mode='nearest')

        # add background
        try:
            _, background = next(enum_background)
        except:
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        if inputs.size(0) != background['background_color'].size(0):
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        if cfg.INPUT == 'COLOR':
            background_color = background['background_color'].cuda()
            for j in range(inputs.size(0)):
                is_syn = im_info[j, -1]
                if is_syn or np.random.rand(1) > 0.5:
                    inputs[j] = mask[j] * inputs[j] + (1 - mask[j]) * background_color[j]

        elif cfg.INPUT == 'RGBD':
            background_color = background['background_color'].cuda()
            background_depth = background['background_depth'].cuda()
            background_mask = background['mask_depth'].cuda()
            for j in range(inputs.size(0)):
                is_syn = im_info[j, -1]
                if is_syn or np.random.rand(1) > 0.5:
                    # color image
                    inputs[j,:3] = mask[j] * inputs[j,:3] + (1 - mask[j]) * background_color[j]
                    # depth image
                    inputs[j,3:6] = mask[j] * inputs[j,3:6] + (1 - mask[j]) * background_depth[j]
                    # depth mask
                    inputs[j,6] = mask[j,0] * inputs[j,6] + (1 - mask[j,0]) * background_mask[j]

        if cfg.TRAIN.VISUALIZE:
            # print(sample['meta_data_path'])
            _vis_minibatch(inputs, background, labels, vertex_targets, sample, train_loader.dataset.class_colors)

        # compute output
        if cfg.TRAIN.VERTEX_REG:
            if cfg.TRAIN.POSE_REG:
                out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, \
                    bbox_labels, bbox_pred, bbox_targets, bbox_inside_weights, loss_pose_tensor, poses_weight \
                    = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)

                loss_label = loss_cross_entropy(out_logsoftmax, out_weight)
                loss_vertex = cfg.TRAIN.VERTEX_W * smooth_l1_loss(out_vertex, vertex_targets, vertex_weights)
                loss_box = loss_cross_entropy(out_logsoftmax_box, bbox_labels)
                loss_location = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights)
                loss_pose = torch.mean(loss_pose_tensor)
                loss = loss_label + loss_vertex + loss_box + loss_location + loss_pose
            else:
                out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, \
                    bbox_labels, bbox_pred, bbox_targets, bbox_inside_weights \
                    = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)

                loss_label = loss_cross_entropy(out_logsoftmax, out_weight)
                loss_vertex = cfg.TRAIN.VERTEX_W * smooth_l1_loss(out_vertex, vertex_targets, vertex_weights)
                loss_box = loss_cross_entropy(out_logsoftmax_box, bbox_labels)
                loss_location = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights)
                loss = loss_label + loss_vertex + loss_box + loss_location

        elif cfg.TRAIN.VERTEX_REG_DELTA:
            out_logsoftmax, out_weight, out_vertex = network(inputs, labels, meta_data, extents, \
                                                             gt_boxes, poses, points, symmetry)

            loss_label = cfg.TRAIN.LABEL_W * loss_cross_entropy(out_logsoftmax, out_weight)
            loss_vertex = cfg.TRAIN.VERTEX_W * smooth_l1_loss(out_vertex, vertex_targets, vertex_weights)
            loss = loss_label + loss_vertex

        else:
            out_logsoftmax, out_weight = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
            loss = loss_cross_entropy(out_logsoftmax, out_weight)

        # record loss
        losses.update(loss.data, inputs.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if cfg.TRAIN.VERTEX_REG:
            if cfg.TRAIN.POSE_REG:
                num_bg = torch.sum(bbox_labels[:, 0])
                num_fg = torch.sum(torch.sum(bbox_labels[:, 1:], dim=1))
                num_fg_pose = torch.sum(torch.sum(poses_weight[:, 4:], dim=1)) / 4
                print('[%d/%d][%d/%d], %.4f, label %.4f, center %.4f, box %.4f (%03d, %03d), loc %.4f, pose %.4f (%03d), lr %.6f, time %.2f' \
                   % (epoch, cfg.epochs, i, epoch_size, loss.data, loss_label.data, loss_vertex.data, loss_box.data, num_fg.data, num_bg.data, \
                      loss_location.data, loss_pose.data, num_fg_pose, optimizer.param_groups[0]['lr'], batch_time.val))
            else:
                num_bg = torch.sum(bbox_labels[:, 0])
                num_fg = torch.sum(torch.sum(bbox_labels[:, 1:], dim=1))
                print('[%d/%d][%d/%d], %.4f, label %.4f, center %.4f, box %.4f (%03d, %03d), loc %.4f, lr %.6f, time %.2f' \
                   % (epoch, cfg.epochs, i, epoch_size, loss.data, loss_label.data, loss_vertex.data, loss_box.data, num_fg.data, num_bg.data, \
                      loss_location.data, optimizer.param_groups[0]['lr'], batch_time.val))
        elif cfg.TRAIN.VERTEX_REG_DELTA:
            print('[%d/%d][%d/%d], %.4f, label %.4f, vertex %.4f,  lr %.6f, time %.2f' \
                  % (epoch, cfg.epochs, i, epoch_size, loss.data, loss_label.data, loss_vertex.data, optimizer.param_groups[0]['lr'], batch_time.val))
        else:
            print('[%d/%d][%d/%d], loss %.4f, lr %.6f, time %.2f' \
               % (epoch, cfg.epochs, i, epoch_size, loss, optimizer.param_groups[0]['lr'], batch_time.val))

        cfg.TRAIN.ITERS += 1


def test_pose_rbpf(pose_rbpf, inputs, rois, poses, meta_data, dataset, im_depth=None, im_label=None):

    n_init_samples = cfg.PF.N_PROCESS
    num = rois.shape[0]
    uv_init = np.zeros((2, ), dtype=np.float32)
    pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).cuda().float()
    rois_return = rois.copy()
    poses_return = poses.copy()
    image = None

    for i in range(num):
        ind = int(rois[i, 0])
        image = inputs[ind].permute(1, 2, 0) + pixel_mean

        cls = int(rois[i, 1])
        if cfg.TRAIN.CLASSES[cls] not in cfg.TEST.CLASSES:
            continue

        intrinsic_matrix = meta_data[ind, :9].cpu().numpy().reshape((3, 3))
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]

        # project the 3D translation to get the center
        uv_init[0] = fx * poses[i, 4] + px
        uv_init[1] = fy * poses[i, 5] + py

        roi_w = rois[i, 4] - rois[i, 2]
        roi_h = rois[i, 5] - rois[i, 3]

        if im_label is not None:
            mask = np.zeros(im_label.shape, dtype=np.float32)
            mask[im_label == cls] = 1.0

        pose = pose_rbpf.initialize(image, uv_init, n_init_samples, cfg.TRAIN.CLASSES[cls], roi_w, roi_h, intrinsic_matrix, im_depth, mask)
        if dataset.classes[cls] == '052_extra_large_clamp' and 'ycb_video' in dataset.name:
            pose_extra = pose
            pose_render = poses_return[i,:].copy()
            pose_render[0] = poses_return[i, 4] * poses_return[i, 6]
            pose_render[1] = poses_return[i, 5] * poses_return[i, 6]
            pose_render[2] = poses_return[i, 6]
            pose_render[3:] = pose_extra[:4]
            im_extra, size_extra = render_one(dataset, cfg.TRAIN.CLASSES[cls], pose_render)

            print('test RBPF for 051_large_clamp')
            pose_large = pose_rbpf.initialize(image, uv_init, n_init_samples, 19, roi_w, roi_h)
            pose_render = poses_return[i,:].copy()
            pose_render[0] = poses_return[i, 4] * poses_return[i, 6]
            pose_render[1] = poses_return[i, 5] * poses_return[i, 6]
            pose_render[2] = poses_return[i, 6]
            pose_render[3:] = pose_large[:4]
            im_large, size_large = render_one(dataset, 19, pose_render)

            roi_s = max(roi_w, roi_h)
            if abs(size_extra - roi_s) < abs(size_large - roi_s):
                print('it is extra large clamp')
            else:
                pose = pose_large
                rois_return[i, 1] = -1  # mask as large clamp
                print('it is large clamp')

        if pose[-1] > 0:
            # only update rotation from codebook matching
            # poses_return[i, :4] = pose[:4]
            poses_return[i, :] = pose
            poses_return[i, 4] /= poses_return[i, 6]
            poses_return[i, 5] /= poses_return[i, 6]

    return rois_return, poses_return, image


def eval_poses(pose_rbpf, poses, rois, im_rgb, im_depth, meta_data):

    num = rois.shape[0]

    sims = np.zeros((rois.shape[0],), dtype=np.float32)
    depth_errors = np.ones((rois.shape[0],), dtype=np.float32)
    vis_ratios = np.zeros((rois.shape[0],), dtype=np.float32)

    pose_scores = np.zeros((rois.shape[0],), dtype=np.float32)

    for i in range(num):
        ind = int(rois[i, 0])
        cls = int(rois[i, 1])

        # todo: fix the problem for large clamp
        if cls == -1:
            cls_id = 19
        else:
            cls_id = cfg.TRAIN.CLASSES[cls]

        if cls_id not in cfg.TEST.CLASSES:
            continue

        intrinsic_matrix = meta_data[ind, :9].cpu().numpy().reshape((3, 3))
        sims[i], depth_errors[i], vis_ratios[i] = pose_rbpf.evaluate_6d_pose(poses[i],
                                                                             cls_id,
                                                                             im_rgb,
                                                                             im_depth,
                                                                             intrinsic_matrix)

        pose_scores[i] = sims[i] / (depth_errors[i] / 0.002 / vis_ratios[i])

    return sims, depth_errors, vis_ratios, pose_scores


def render_one(dataset, cls, pose):

    cls_id = cfg.TEST.CLASSES.index(cls)
    intrinsic_matrix = dataset._intrinsic_matrix
    height = dataset._height
    width = dataset._width

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 6.0
    znear = 0.25

    image_tensor = torch.cuda.FloatTensor(height, width, 4)
    seg_tensor = torch.cuda.FloatTensor(height, width, 4)

    # set renderer
    cfg.renderer.set_light_pos([0, 0, 0])
    cfg.renderer.set_light_color([1, 1, 1])
    cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

    # render images
    cls_indexes = []
    poses_all = []
    cls_indexes.append(cls_id)
    poses_all.append(pose)

    # rendering
    cfg.renderer.set_poses(poses_all)
    cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
    image_tensor = image_tensor.flip(0)
    seg_tensor = seg_tensor.flip(0)
    seg = seg_tensor[:,:,2] + 256*seg_tensor[:,:,1] + 256*256*seg_tensor[:,:,0]
    image_tensor[seg == 0] = 0.5

    im_render = image_tensor.cpu().numpy()
    im_render = np.clip(im_render, 0, 1)
    im_render = im_render[:, :, :3] * 255
    im_render = im_render.astype(np.uint8)

    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(im_render)
    plt.show()
    '''

    mask = seg.cpu().numpy()
    y, x = np.where(mask > 0)
    x1 = np.min(x)
    x2 = np.max(x)
    y1 = np.min(y)
    y2 = np.max(y)
    s = max((y2 - y1), (x2 - x1))

    return im_render, s


def test(test_loader, background_loader, network, pose_rbpf, output_dir):

    batch_time = AverageMeter()
    epoch_size = len(test_loader)
    enum_background = enumerate(background_loader)

    # switch to test mode
    network.eval()

    for i, sample in enumerate(test_loader):

        end = time.time()

        if cfg.INPUT == 'DEPTH':
            inputs = sample['image_depth']
        elif cfg.INPUT == 'COLOR':
            inputs = sample['image_color']
        elif cfg.INPUT == 'RGBD':
            inputs = torch.cat((sample['image_color'], sample['image_depth'], sample['mask_depth']), dim=1)
        im_info = sample['im_info']

        # add background
        mask = sample['mask']
        try:
            _, background = next(enum_background)
        except:
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        if inputs.size(0) != background['background_color'].size(0):
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        if cfg.INPUT == 'COLOR':
            background_color = background['background_color'].cuda()
            for j in range(inputs.size(0)):
                is_syn = im_info[j, -1]
                if is_syn:
                    inputs[j] = mask[j] * inputs[j] + (1 - mask[j]) * background_color[j]

        elif cfg.INPUT == 'RGBD':
            background_color = background['background_color'].cuda()
            background_depth = background['background_depth'].cuda()
            background_mask = background['mask_depth'].cuda()
            for j in range(inputs.size(0)):
                is_syn = im_info[j, -1]
                if is_syn or np.random.rand(1) > 0.5:
                    # color image
                    inputs[j,:3] = mask[j] * inputs[j,:3] + (1 - mask[j]) * background_color[j]
                    # depth image
                    inputs[j,3:6] = mask[j] * inputs[j,3:6] + (1 - mask[j]) * background_depth[j]
                    # depth mask
                    inputs[j,6] = mask[j,0] * inputs[j,6] + (1 - mask[j,0]) * background_mask[j]

        labels = sample['label'].cuda()
        meta_data = sample['meta_data'].cuda()
        extents = sample['extents'][0, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1).cuda()
        gt_boxes = sample['gt_boxes'].cuda()
        poses = sample['poses'].cuda()
        points = sample['points'][0, :, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1, 1).cuda()
        symmetry = sample['symmetry'][0, :].repeat(cfg.TRAIN.GPUNUM, 1).cuda()
        
        # compute output
        if cfg.TRAIN.VERTEX_REG:
            if cfg.TRAIN.POSE_REG:
                out_label, out_vertex, rois, out_pose, out_quaternion = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
                
                # combine poses
                rois = rois.detach().cpu().numpy()
                out_pose = out_pose.detach().cpu().numpy()
                out_quaternion = out_quaternion.detach().cpu().numpy()
                num = rois.shape[0]
                poses = out_pose.copy()
                for j in xrange(num):
                    cls = int(rois[j, 1])
                    if cls >= 0:
                        qt = out_quaternion[j, 4*cls:4*cls+4]
                        qt = qt / np.linalg.norm(qt)
                        # allocentric to egocentric
                        T = poses[j, 4:]
                        poses[j, :4] = allocentric2egocentric(qt, T)

                poses_refined = []
                pose_scores = None

                # optimize depths
                im_depth = sample['im_depth'].numpy()[0]
                if cfg.TEST.POSE_REFINE:
                    labels_out = out_label.detach().cpu().numpy()[0]
                    poses, poses_refined, cls_render_ids = refine_pose(labels_out, im_depth, rois, poses, sample['meta_data'], test_loader.dataset)
                else:
                    num = rois.shape[0]
                    for j in range(num):
                        poses[j, 4] *= poses[j, 6] 
                        poses[j, 5] *= poses[j, 6]
            else:
                out_label, out_vertex, rois, out_pose = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
                rois = rois.detach().cpu().numpy()
                out_pose = out_pose.detach().cpu().numpy()
                poses = out_pose.copy()
                poses_refined = []
                pose_scores = None

                # non-maximum suppression within class
                index = nms(rois, 0.5)
                rois = rois[index, :]
                poses = poses[index, :]

                # run poseRBPF for codebook matching to compute the rotations
                im_depth = sample['im_depth'].numpy()[0]
                labels_out = out_label.detach().cpu().numpy()[0]
                if pose_rbpf is not None:
                    rois, poses, im_rgb = test_pose_rbpf(pose_rbpf, inputs, rois, poses, sample['meta_data'], test_loader.dataset, im_depth, labels_out)

                # optimize depths
                if cfg.TEST.POSE_REFINE:
                    poses, poses_refined, cls_render_ids = refine_pose(labels_out, im_depth, rois, poses, sample['meta_data'], test_loader.dataset)
                    if pose_rbpf is not None and cfg.TEST.VISUALIZE:
                        sims, depth_errors, vis_ratios, pose_scores = eval_poses(pose_rbpf, poses_refined, rois, im_rgb, im_depth, sample['meta_data'])
                else:
                    num = rois.shape[0]
                    for j in range(num):
                        poses[j, 4] *= poses[j, 6] 
                        poses[j, 5] *= poses[j, 6]

        elif cfg.TRAIN.VERTEX_REG_DELTA:

            out_label, out_vertex = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
            if not cfg.TEST.MEAN_SHIFT:
                out_center, rois = compute_centroids_and_loose_bounding_boxes(out_vertex, out_label, extents, \
                    inputs[:, 3:6, :, :], inputs[:, 6, :, :], test_loader.dataset._intrinsic_matrix)
            else:
                out_center, rois = mean_shift_and_loose_bounding_boxes(out_vertex, out_label, extents, \
                    inputs[:, 3:6, :, :], inputs[:, 6, :, :], test_loader.dataset._intrinsic_matrix)

            rois = rois.detach().cpu().numpy()
            out_center_cpu = out_center.detach().cpu().numpy()
            poses_vis = np.zeros((rois.shape[0], 7))

            for c in range(rois.shape[0]):
                poses_vis[c, :4] = np.array([1.0, 0.0, 0.0, 0.0])
                poses_vis[c, 4:] = np.array([out_center_cpu[c, 0], out_center_cpu[c, 1], out_center_cpu[c, 2]])
            poses = poses_vis
            poses_refined = poses_vis

            visualize = False
            if visualize:
                nclasses = labels.shape[1]
                out_center_cpu = out_center.detach().cpu().numpy()
                poses_vis = np.zeros((nclasses, 3))
                for c in range(nclasses):
                    poses_vis[c, :] = np.array([out_center_cpu[0, c * 3 + 0], out_center_cpu[0, c * 3 + 1], out_center_cpu[0, c * 3 + 2]])

                input_ims = inputs[:,3:6].detach().cpu().numpy()
                height = input_ims.shape[2]
                width = input_ims.shape[3]
                num = 3000
                index = np.random.permutation(np.arange(width * height))[:num]
                x_im = input_ims[0, 0, :, :].reshape(width*height)
                y_im = input_ims[0, 1, :, :].reshape(width*height)
                z_im = input_ims[0, 2, :, :].reshape(width*height)
                xyz = np.zeros((num, 3))
                xyz[:, 0] = x_im[index]
                xyz[:, 1] = y_im[index]
                xyz[:, 2] = z_im[index]

                xyz = np.concatenate((xyz, poses_vis[1:, :]), axis=0)
                colors = np.zeros((xyz.shape[0], 4))
                colors[0:num, :] = [0.0, 0.0, 1.0, 0.4]
                colors[num:, :] = [1.0, 0.0, 0.0, 1.0]

                from mpl_toolkits.mplot3d import Axes3D
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors)
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()

        else:
            out_label = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
            out_vertex = []
            rois = []
            poses = []
            poses_refined = []
            pose_scores = None

        if cfg.TEST.VISUALIZE:
            _vis_test(inputs, labels, out_label, out_vertex, rois, poses, poses_refined, sample, \
                test_loader.dataset._points_all, test_loader.dataset._points_clamp, test_loader.dataset.classes, test_loader.dataset.class_colors, \
                      pose_scores)

        # measure elapsed time
        batch_time.update(time.time() - end)

        if not cfg.TEST.VISUALIZE:
            result = {'labels': labels_out, 'rois': rois, 'poses': poses, 'poses_refined': poses_refined}
            if 'video_id' in sample and 'image_id' in sample:
                filename = os.path.join(output_dir, sample['video_id'][0] + '_' + sample['image_id'][0] + '.mat')
            else:
                result['meta_data_path'] = sample['meta_data_path']
                print(result['meta_data_path'])
                filename = os.path.join(output_dir, '%06d.mat' % i)
            print(filename)
            scipy.io.savemat(filename, result, do_compression=True)

        print('[%d/%d], batch time %.2f' % (i, epoch_size, batch_time.val))

    filename = os.path.join(output_dir, 'results_posecnn.mat')
    if os.path.exists(filename):
        os.remove(filename)

def test_image_simple(network, dataset, im_color, im_depth=None):
    """test on a single image"""

    num_classes = dataset.num_classes

    # compute image blob
    im = im_color.astype(np.float32, copy=True)
    im -= cfg.PIXEL_MEANS
    height = im.shape[0]
    width = im.shape[1]
    im = np.transpose(im / 255.0, (2, 0, 1))
    im = im[np.newaxis, :, :, :]

    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        im_xyz = dataset.backproject(im_depth, dataset._intrinsic_matrix, 1.0)
        im_xyz = np.transpose(im_xyz, (2, 0, 1))
        im_xyz = im_xyz[np.newaxis, :, :, :]
        depth_mask = im_depth > 0.0
        depth_mask = depth_mask.astype('float')

    # construct the meta data
    K = dataset._intrinsic_matrix
    Kinv = np.linalg.pinv(K)
    meta_data_blob = np.zeros((1, 18), dtype=np.float32)
    meta_data_blob[0, 0:9] = K.flatten()
    meta_data_blob[0, 9:18] = Kinv.flatten()

    # use fake label blob
    label_blob = np.zeros((1, num_classes, height, width), dtype=np.float32)
    pose_blob = np.zeros((1, num_classes, 9), dtype=np.float32)
    gt_boxes = np.zeros((1, num_classes, 5), dtype=np.float32)

    # transfer to GPU
    if cfg.INPUT == 'DEPTH':
        inputs = torch.from_numpy(im_xyz).cuda().float()
    elif cfg.INPUT == 'COLOR':
        inputs = torch.from_numpy(im).cuda()
    elif cfg.INPUT == 'RGBD':
        im_1 = torch.from_numpy(im).cuda()
        im_2 = torch.from_numpy(im_xyz).cuda().float()
        im_3 = torch.from_numpy(depth_mask).cuda().float()
        im_3.unsqueeze_(0).unsqueeze_(0)
        inputs = torch.cat((im_1, im_2, im_3), dim=1)

    labels = torch.from_numpy(label_blob).cuda()
    meta_data = torch.from_numpy(meta_data_blob).cuda()
    extents = torch.from_numpy(dataset._extents).cuda()
    gt_boxes = torch.from_numpy(gt_boxes).cuda()
    poses = torch.from_numpy(pose_blob).cuda()
    points = torch.from_numpy(dataset._point_blob).cuda()
    symmetry = torch.from_numpy(dataset._symmetry).cuda()

    cfg.TRAIN.POSE_REG = False
    out_label, out_vertex, rois, out_pose = network(inputs, labels, meta_data, extents, gt_boxes, poses, points,
                                                    symmetry)
    labels = out_label.detach().cpu().numpy()[0]

    rois = rois.detach().cpu().numpy()
    out_pose = out_pose.detach().cpu().numpy()
    poses = out_pose.copy()

    # filter out detections
    index = np.where(rois[:, -1] > cfg.TEST.DET_THRESHOLD)[0]
    rois = rois[index, :]
    poses = poses[index, :]

    # non-maximum suppression within class
    index = nms(rois, 0.5)
    rois = rois[index, :]
    poses = poses[index, :]

    num = rois.shape[0]
    for i in range(num):
        poses[i, 4] *= poses[i, 6]
        poses[i, 5] *= poses[i, 6]

    im_label = render_image_detection(dataset, im_color, rois, labels)

    return rois, labels, poses, im_label


def test_image(network, pose_rbpf, dataset, im_color, im_depth=None):
    """test on a single image"""

    # compute image blob
    im = im_color.astype(np.float32, copy=True)
    im -= cfg.PIXEL_MEANS
    height = im.shape[0]
    width = im.shape[1]
    im = np.transpose(im / 255.0, (2, 0, 1))
    im = im[np.newaxis, :, :, :]

    K = dataset._intrinsic_matrix
    K[2, 2] = 1
    Kinv = np.linalg.pinv(K)
    meta_data = np.zeros((1, 18), dtype=np.float32)
    meta_data[0, 0:9] = K.flatten()
    meta_data[0, 9:18] = Kinv.flatten()
    meta_data = torch.from_numpy(meta_data).cuda()

    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        im_xyz = dataset.backproject(im_depth, dataset._intrinsic_matrix, 1.0)
        im_xyz = np.transpose(im_xyz, (2, 0, 1))
        im_xyz = im_xyz[np.newaxis, :, :, :]
        depth_mask = im_depth > 0.0
        depth_mask = depth_mask.astype('float')

    # transfer to GPU
    if cfg.INPUT == 'DEPTH':
        inputs = torch.from_numpy(im_xyz).cuda().float()
    elif cfg.INPUT == 'COLOR':
        inputs = torch.from_numpy(im).cuda()
    elif cfg.INPUT == 'RGBD':
        im_1 = torch.from_numpy(im).cuda()
        im_2 = torch.from_numpy(im_xyz).cuda().float()
        im_3 = torch.from_numpy(depth_mask).cuda().float()
        im_3.unsqueeze_(0).unsqueeze_(0)
        inputs = torch.cat((im_1, im_2, im_3), dim=1)

    if cfg.TRAIN.VERTEX_REG:

        if cfg.TRAIN.POSE_REG:
            out_label, out_vertex, rois, out_pose, out_quaternion = network(inputs, dataset.input_labels, meta_data, \
                dataset.input_extents, dataset.input_gt_boxes, dataset.input_poses, dataset.input_points, dataset.input_symmetry)
            labels = out_label.detach().cpu().numpy()[0]

            # combine poses
            rois = rois.detach().cpu().numpy()
            out_pose = out_pose.detach().cpu().numpy()
            out_quaternion = out_quaternion.detach().cpu().numpy()
            num = rois.shape[0]
            poses = out_pose.copy()
            for j in xrange(num):
                cls = int(rois[j, 1])
                if cls >= 0:
                    qt = out_quaternion[j, 4*cls:4*cls+4]
                    qt = qt / np.linalg.norm(qt)
                    # allocentric to egocentric
                    T = poses[j, 4:]
                    poses[j, :4] = allocentric2egocentric(qt, T)

            # filter out detections
            index = np.where(rois[:, -1] > cfg.TEST.DET_THRESHOLD)[0]
            rois = rois[index, :]
            poses = poses[index, :]

            # non-maximum suppression within class
            index = nms(rois, 0.5)
            rois = rois[index, :]
            poses = poses[index, :]
           
            # optimize depths
            if cfg.TEST.POSE_REFINE and im_depth is not None:
                poses = refine_pose(labels, im_depth, rois, poses, dataset)
            else:
                poses_tmp, poses = optimize_depths(rois, poses, dataset._points_all, dataset._intrinsic_matrix)

        else:
            out_label, out_vertex, rois, out_pose = network(inputs, dataset.input_labels, meta_data, \
                dataset.input_extents, dataset.input_gt_boxes, dataset.input_poses, dataset.input_points, dataset.input_symmetry)

            labels = out_label.detach().cpu().numpy()[0]

            rois = rois.detach().cpu().numpy()
            out_pose = out_pose.detach().cpu().numpy()
            poses = out_pose.copy()

            # filter out detections
            index = np.where(rois[:, -1] > cfg.TEST.DET_THRESHOLD)[0]
            rois = rois[index, :]
            poses = poses[index, :]
            poses_refined = []
            pose_scores = []

            # non-maximum suppression within class
            index = nms(rois, 0.5)
            rois = rois[index, :]
            poses = poses[index, :]

            # run poseRBPF for codebook matching to compute the rotations
            labels_out = out_label.detach().cpu().numpy()[0]
            if pose_rbpf is not None:
                rois, poses, im_rgb = test_pose_rbpf(pose_rbpf, inputs, rois, poses, meta_data, dataset, im_depth, labels_out)

            # optimize depths
            cls_render_ids = None
            if cfg.TEST.POSE_REFINE and im_depth is not None:
                poses, poses_refined, cls_render_ids = refine_pose(labels_out, im_depth, rois, poses, meta_data, dataset)
                if pose_rbpf is not None:
                    sims, depth_errors, vis_ratios, pose_scores = eval_poses(pose_rbpf, poses_refined, rois, im_rgb, im_depth, meta_data)
            else:
                num = rois.shape[0]
                for j in range(num):
                    poses[j, 4] *= poses[j, 6] 
                    poses[j, 5] *= poses[j, 6]

    elif cfg.TRAIN.VERTEX_REG_DELTA:
        out_label, out_vertex = network(inputs, dataset.input_labels, dataset.input_meta_data, \
            dataset.input_extents, dataset.input_gt_boxes, dataset.input_poses, dataset.input_points, dataset.input_symmetry)
        if not cfg.TEST.MEAN_SHIFT:
            out_center, rois = compute_centroids_and_loose_bounding_boxes(out_vertex, out_label, extents,
                                                                          inputs[:, 3:6, :, :], inputs[:, 6, :, :],
                                                                          dataset._intrinsic_matrix)

        elif cfg.TEST.MEAN_SHIFT:
            out_center, rois = mean_shift_and_loose_bounding_boxes(out_vertex, out_label, extents, inputs[:, 3:6, :, :],
                                                                   inputs[:, 6, :, :], dataset._intrinsic_matrix)

        labels = out_label.detach().cpu().numpy()[0]
        rois = rois.detach().cpu().numpy()
        out_center_cpu = out_center.detach().cpu().numpy()
        poses_vis = np.zeros((rois.shape[0], 7))

        for c in range(rois.shape[0]):
            poses_vis[c, :4] = np.array([1.0, 0.0, 0.0, 0.0])
            poses_vis[c, 4:] = np.array([out_center_cpu[c, 0], out_center_cpu[c, 1], out_center_cpu[c, 2]])
        poses = poses_vis

    else:
        out_label = network(inputs, dataset.input_labels, dataset.input_meta_data, \
            dataset.input_extents, dataset.input_gt_boxes, dataset.input_poses, dataset.input_points, dataset.input_symmetry)
        labels = out_label.detach().cpu().numpy()[0]
        rois = np.zeros((0, 7), dtype=np.float32)
        poses = np.zeros((0, 7), dtype=np.float32)

    im_pose, im_pose_refine, im_label = render_image(dataset, im_color, rois, poses, poses_refined, labels, cls_render_ids)
    if cfg.TEST.VISUALIZE:
        vis_test(dataset, im, im_depth, labels, out_vertex, rois, poses, poses_refined, im_pose, im_pose_refine)

    return im_pose, im_label, rois, poses


#************************************#
#    train and test autoencoders     #
#************************************#

def lsgan_loss(input, target):
    loss = (input.squeeze() - target) ** 2
    return loss.mean()

def train_autoencoder(train_loader, background_loader, network, optimizer, optimizer_discriminator, epoch):

    batch_time = AverageMeter()
    epoch_size = len(train_loader)
    enum_background = enumerate(background_loader)
    cls_target = train_loader.dataset.cls_target
    cls = train_loader.dataset.classes[cls_target]

    # switch to train mode
    network.train()

    for i, sample in enumerate(train_loader):

        end = time.time()

        # construct input
        image = sample['image_input']
        mask = sample['mask']
        affine_matrix = sample['affine_matrix']

        # affine transformation
        grids = nn.functional.affine_grid(affine_matrix, image.size())
        image = nn.functional.grid_sample(image, grids, padding_mode='border')
        mask = nn.functional.grid_sample(mask, grids, mode='nearest')

        try:
            _, background = next(enum_background)
        except:
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        num = image.size(0)
        if background['background_color'].size(0) < num:
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        background_color = background['background_color'].cuda()
        inputs = mask * image + (1 - mask) * background_color[:num]
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

        # run generator
        out_images, embeddings = network(inputs)

        ############## train discriminator ##############
        targets = sample['image_target']
        targets_noise = add_gaussian_noise_cuda(targets)
        out_images_noise = add_gaussian_noise_cuda(out_images)
        d_real = network.module.run_discriminator(targets_noise)
        d_fake = network.module.run_discriminator(out_images_noise.detach())
        loss_d_real = lsgan_loss(d_real, 1.0)
        loss_d_fake = lsgan_loss(d_fake, 0.0)
        loss_d = loss_d_real + loss_d_fake

        optimizer_discriminator.zero_grad()
        loss_d.backward()
        optimizer_discriminator.step()

        ############## train generator ##############
        g_fake = network.module.run_discriminator(out_images_noise)
        loss_g_fake = lsgan_loss(g_fake, 1.0)
        loss_recon, losses_euler = BootstrapedMSEloss(out_images, targets, cfg.TRAIN.BOOSTRAP_PIXELS)
        loss = loss_recon + 0.1 * loss_g_fake

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record the losses for each euler pose
        index_euler = torch.flatten(sample['index_euler'])
        losses_euler_numpy = losses_euler.detach().cpu().numpy()
        for j in range(len(index_euler)):
            if index_euler[j] >= 0:
                train_loader.dataset._losses_pose[cls_target, index_euler[j]] = losses_euler_numpy[j]

        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch_autoencoder(inputs, background_color, mask, sample, out_images)

        # measure elapsed time
        batch_time.update(time.time() - end)

        print('%s, [%d/%d][%d/%d], loss_r %.4f, loss_g %.4f, loss_d %.4f, lr %.6f, time %.2f' \
           % (cls, epoch, cfg.epochs, i, epoch_size, loss_recon, loss_g_fake, loss_d, \
              optimizer.param_groups[0]['lr'], batch_time.val))
        cfg.TRAIN.ITERS += 1


def render_images(dataset, poses):

    intrinsic_matrix = dataset._intrinsic_matrix
    height = dataset._height
    width = dataset._width

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 6.0
    znear = 0.25
    num = poses.shape[0]

    im_output = np.zeros((num, height, width, 3), dtype=np.uint8)
    image_tensor = torch.cuda.FloatTensor(height, width, 4)
    seg_tensor = torch.cuda.FloatTensor(height, width, 4)

    # set renderer
    cfg.renderer.set_light_pos([0, 0, 0])
    cfg.renderer.set_light_color([1, 1, 1])
    cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

    # render images
    for i in range(num):

        cls_indexes = []
        poses_all = []

        cls_index = cfg.TRAIN.CLASSES[0] - 1
        cls_indexes.append(cls_index)
        poses_all.append(poses[i,:])

        # rendering
        cfg.renderer.set_poses(poses_all)
        cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)

        im_render = image_tensor.cpu().numpy()
        im_render = np.clip(im_render, 0, 1)
        im_render = im_render[:, :, :3] * 255
        im_render = im_render.astype(np.uint8)
        im_output[i] = im_render

    return im_output


def _vis_minibatch_autoencoder(inputs, background, mask, sample, outputs, im_render=None):

    im_blob = inputs.cpu().numpy()
    background_blob = background.cpu().numpy()
    mask_blob = mask.cpu().numpy()
    targets = sample['image_target'].cpu().numpy()
    im_output = outputs.cpu().detach().numpy()

    for i in range(im_blob.shape[0]):
        fig = plt.figure()
        # show image
        im = im_blob[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im [:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(2, 3, 1)
        plt.imshow(im)
        ax.set_title('input')

        # show target
        im = targets[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im.astype(np.uint8)
        im = im [:, :, (2, 1, 0)]
        ax = fig.add_subplot(2, 3, 2)
        plt.imshow(im)
        ax.set_title('target')

        # show matched codebook
        if im_render is not None:
            im = im_render[i].copy()
            ax = fig.add_subplot(2, 3, 3)
            plt.imshow(im)
            ax.set_title('matched code')
 
        # show background
        im = background_blob[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im [:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(2, 3, 4)
        plt.imshow(im)
        ax.set_title('background')

        # show mask
        im = mask_blob[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im [:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(2, 3, 5)
        plt.imshow(im)
        ax.set_title('mask')

        # show output
        im = im_output[i, :, :, :].copy()
        im = np.clip(im, 0, 1)
        im = im.transpose((1, 2, 0)) * 255.0
        im = im [:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(2, 3, 6)
        plt.imshow(im)
        ax.set_title('reconstruction')

        plt.show()


def test_autoencoder(test_loader, background_loader, network, output_dir):

    batch_time = AverageMeter()
    epoch_size = len(test_loader)
    enum_background = enumerate(background_loader)
    cls = test_loader.dataset.classes[0]

    if cfg.TEST.BUILD_CODEBOOK:
        num = test_loader.dataset._size
        codes = np.zeros((num, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
        poses = np.zeros((num, 7), dtype=np.float32)
        codebook_cpt_poserbpf = torch.zeros(num, cfg.TRAIN.NUM_UNITS).cuda().detach()
        codepose_cpt_poserbpf = torch.zeros(num, 7).cuda().detach()
        codebook_poserbpf = torch.zeros(1, num, cfg.TRAIN.NUM_UNITS).cuda().detach()
        codepose_poserbpf = torch.zeros(1, num,  7).cuda().detach()
        count = 0
    else:
        # check if codebook exists
        filename = os.path.join(output_dir, 'codebook_%s.mat' % (test_loader.dataset.name + '_' + cls))
        if os.path.exists(filename):
            codebook = scipy.io.loadmat(filename)
            codes_gpu = torch.from_numpy(codebook['codes']).cuda()
        else:
            codebook = None

    # switch to test mode
    network.eval()

    for i, sample in enumerate(test_loader):

        end = time.time()

        # construct input
        image = sample['image_input']
        mask = sample['mask']

        if cfg.TEST.BUILD_CODEBOOK == False:
            # affine transformation
            affine_matrix = sample['affine_matrix']
            grids = nn.functional.affine_grid(affine_matrix, image.size())
            image = nn.functional.grid_sample(image, grids, padding_mode='border')
            mask = nn.functional.grid_sample(mask, grids, mode='nearest')

        try:
            _, background = next(enum_background)
        except:
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        num = image.size(0)
        if background['background_color'].size(0) < num:
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        background_color = background['background_color'].cuda()
        inputs = mask * image + (1 - mask) * background_color[:num]
        inputs = torch.clamp(inputs, min=0.0, max=1.0)

        # compute output
        if cfg.TEST.BUILD_CODEBOOK and not cfg.TEST.VISUALIZE:
            embeddings = network.module.encode(inputs)
        else:
            out_images, embeddings = network(inputs)

        im_render = None
        if cfg.TEST.BUILD_CODEBOOK:
            num = embeddings.shape[0]
            codes[count:count+num, :] = embeddings.cpu().detach().numpy()
            poses[count:count+num, :] = sample['pose_target']
            codebook_cpt_poserbpf[count:count+num, :] = embeddings.detach()
            codepose_cpt_poserbpf[count:count+num, :] = sample['pose_target'].cuda()
            count += num
        elif codebook is not None:
            # codebook matching
            distance_matrix = network.module.pairwise_cosine_distances(embeddings, codes_gpu)
            best_match = torch.argmax(distance_matrix, dim=1).cpu().numpy()
            quaternions_match = codebook['quaternions'][best_match, :]

            # render codebook images
            if cfg.TEST.VISUALIZE:
                num = quaternions_match.shape[0]
                poses = np.zeros((num, 7), dtype=np.float32)
                poses[:, 2] = codebook['distance']
                poses[:, 3:] = quaternions_match
                im_render = render_images(test_loader.dataset, poses)

        if cfg.TEST.VISUALIZE:
            _vis_minibatch_autoencoder(inputs, background_color, mask, sample, out_images, im_render)

        # measure elapsed time
        batch_time.update(time.time() - end)
        if cfg.TEST.BUILD_CODEBOOK:
            print('%s, [%d/%d], code index %d, batch time %.2f' % (cls, i, epoch_size, count, batch_time.val))
        else:
            print('%s, [%d/%d], batch time %.2f' % (cls, i, epoch_size, batch_time.val))

    # save codebook
    if cfg.TEST.BUILD_CODEBOOK:
        codebook = {'codes': codes, 'quaternions': poses[:, 3:], 'distance': poses[0, 2], 'intrinsic_matrix': test_loader.dataset._intrinsic_matrix}
        filename = os.path.join(output_dir, 'codebook_%s.mat' % (test_loader.dataset.name + '_' + cls))
        print('save codebook to %s' % (filename))
        scipy.io.savemat(filename, codebook, do_compression=True)

        codebook_poserbpf[0] = codebook_cpt_poserbpf
        codepose_poserbpf[0] = codepose_cpt_poserbpf
        filename = os.path.join(output_dir, 'codebook_%s.pth' % (test_loader.dataset.name + '_' + cls))
        torch.save((codebook_poserbpf, codepose_poserbpf), filename)
        print('code book is saved to {}'.format(filename))


def optimize_depths(rois, poses, points, points_clamp, intrinsic_matrix):

    num = rois.shape[0]
    poses_refined = poses.copy()
    for i in range(num):
        roi = rois[i, 2:6]
        width = roi[2] - roi[0]
        height = roi[3] - roi[1]
        cls = int(rois[i, 1])

        RT = np.zeros((3, 4), dtype=np.float32)
        RT[:3, :3] = quat2mat(poses[i, :4])
        RT[0, 3] = poses[i, 4]
        RT[1, 3] = poses[i, 5]
        RT[2, 3] = poses[i, 6]

        # extract 3D points
        x3d = np.ones((4, points.shape[1]), dtype=np.float32)
        if cls == -1:
            x3d[0, :] = points_clamp[:,0]
            x3d[1, :] = points_clamp[:,1]
            x3d[2, :] = points_clamp[:,2]
        else:
            x3d[0, :] = points[cls,:,0]
            x3d[1, :] = points[cls,:,1]
            x3d[2, :] = points[cls,:,2]

        # optimization
        x0 = poses[i, 6]
        res = minimize(objective_depth, x0, args=(width, height, RT, x3d, intrinsic_matrix), method='nelder-mead', options={'xtol': 1e-3, 'disp': False})
        poses_refined[i, 4] *= res.x 
        poses_refined[i, 5] *= res.x
        poses_refined[i, 6] = res.x

        poses[i, 4] *= poses[i, 6] 
        poses[i, 5] *= poses[i, 6]

    return poses, poses_refined


def objective_depth(x, width, height, RT, x3d, intrinsic_matrix):

    # project points
    RT[0, 3] = RT[0, 3] * x
    RT[1, 3] = RT[1, 3] * x
    RT[2, 3] = x
    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

    roi_pred = np.zeros((4, ), dtype=np.float32)
    roi_pred[0] = np.min(x2d[0, :])
    roi_pred[1] = np.min(x2d[1, :])
    roi_pred[2] = np.max(x2d[0, :])
    roi_pred[3] = np.max(x2d[1, :])
    w = roi_pred[2] - roi_pred[0]
    h = roi_pred[3] - roi_pred[1]
    return np.abs(w * h - width * height)


# backproject pixels into 3D points in camera's coordinate system
def backproject(depth_cv, intrinsic_matrix):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
    return np.array(X).transpose()


def refine_pose(im_label, im_depth, rois, poses, meta_data, dataset):

    # backprojection
    intrinsic_matrix = meta_data[0, :9].cpu().numpy().reshape((3, 3))
    poses_refined = poses.copy()
    dpoints = backproject(im_depth, intrinsic_matrix)
    width = im_depth.shape[1]
    height = im_depth.shape[0]
    im_pcloud = dpoints.reshape((height, width, 3))

    # renderer
    num = rois.shape[0]
    width = im_label.shape[1]
    height = im_label.shape[0]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 6.0
    znear = 0.01
    # rendering
    cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    pcloud_tensor = torch.cuda.FloatTensor(height, width, 4).detach()

    # refine pose
    cls_render_ids = []
    for i in range(num):
        cls_indexes = []
        cls = int(rois[i, 1])

        # todo: fix the problem for large clamp
        if cls == -1:
            cls_label = cfg.TRAIN.CLASSES.index(20)
            cls_id = 19
        else:
            cls_id = cfg.TRAIN.CLASSES[cls]
            cls_label = cls

        if cls_id not in cfg.TEST.CLASSES:
            cls_render_ids.append(-1)
            continue

        cls_render = cfg.TEST.CLASSES.index(cls_id)
        cls_indexes.append(cls_render)

        poses_all = []
        qt = np.zeros((7, ), dtype=np.float32)
        qt[3:] = poses[i, :4]
        qt[0] = poses[i, 4] * poses[i, 6]
        qt[1] = poses[i, 5] * poses[i, 6]
        qt[2] = poses[i, 6]
        poses_all.append(qt)
        cfg.renderer.set_poses(poses_all)

        cfg.renderer.render(cls_indexes, image_tensor, seg_tensor, pc2_tensor=pcloud_tensor)
        pcloud_tensor = pcloud_tensor.flip(0)
        pcloud = pcloud_tensor[:,:,:3].cpu().numpy().reshape((-1, 3))

        x1 = max(int(rois[i, 2]), 0)
        y1 = max(int(rois[i, 3]), 0)
        x2 = min(int(rois[i, 4]), width-1)
        y2 = min(int(rois[i, 5]), height-1)
        labels = np.zeros((height, width), dtype=np.float32)
        labels[y1:y2, x1:x2] = im_label[y1:y2, x1:x2]
        mask_label = np.ma.getmaskarray(np.ma.masked_equal(labels, cls_label))
        labels = labels.reshape((width * height, ))
        diff = np.abs(dpoints[:, 2] - pcloud[:, 2])
        index = np.where((labels == cls_label) & np.isfinite(dpoints[:, 2]) & (dpoints[:, 2] > 0) & (pcloud[:, 2] > 0) & (diff < 0.5))[0]  

        if len(index) > 10:
            T = np.mean(dpoints[index, :] - pcloud[index, :], axis=0)
            poses_refined[i, 6] += T[2]
            poses_refined[i, 4] *= poses_refined[i, 6]
            poses_refined[i, 5] *= poses_refined[i, 6]
        else:
            poses_refined[i, 4] *= poses_refined[i, 6]
            poses_refined[i, 5] *= poses_refined[i, 6]
            print('no pose refinement')

        poses[i, 4] *= poses[i, 6]
        poses[i, 5] *= poses[i, 6]

        # check if object with different size
        threshold = -0.2
        if cfg.TEST.CHECK_SIZE and T[2] < threshold:
           cls_name = dataset._classes_all[cfg.TEST.CLASSES[cls_render]] + '_small'
           for j in range(len(dataset._classes_all)):
               if cls_name == dataset._classes_all[j]:
                   print(j, 'small object ' + cls_name)
                   cls_render = cfg.TEST.CLASSES.index(j)
                   break

        cls_render_ids.append(cls_render)

        # run SDF optimization
        if cfg.TEST.POSE_SDF and len(index) > 10:
            sdf_optim = cfg.sdf_optimizers[cls_render]

            # re-render
            qt[3:] = poses_refined[i, :4]
            qt[:3] = poses_refined[i, 4:]
            poses_all[0] = qt
            cfg.renderer.set_poses(poses_all)
            cfg.renderer.render(cls_indexes, image_tensor, seg_tensor, pc2_tensor=pcloud_tensor)
            pcloud_tensor = pcloud_tensor.flip(0)

            # compare the depth
            delta = 0.05
            depth_meas_roi = im_pcloud[:, :, 2]
            depth_render_roi = pcloud_tensor[:, :, 2].cpu().numpy()
            mask_depth_valid = np.ma.getmaskarray(np.ma.masked_where(np.isfinite(depth_meas_roi), depth_meas_roi))
            mask_depth_meas = np.ma.getmaskarray(np.ma.masked_not_equal(depth_meas_roi, 0))
            mask_depth_render = np.ma.getmaskarray(np.ma.masked_greater(depth_render_roi, 0))
            mask_depth_vis = np.ma.getmaskarray(np.ma.masked_less(np.abs(depth_render_roi - depth_meas_roi), delta))
            mask = mask_label * mask_depth_valid * mask_depth_meas * mask_depth_render * mask_depth_vis
            index_p = mask.flatten().nonzero()[0]

            if len(index_p) > 10:
                points = torch.from_numpy(dpoints[index_p, :]).float()
                points = torch.cat((points, torch.ones((points.size(0), 1), dtype=torch.float32)), dim=1)
                RT = np.zeros((4, 4), dtype=np.float32)
                qt = poses_refined[i, :4]
                T = poses_refined[i, 4:]
                RT[:3, :3] = quat2mat(qt)
                RT[:3, 3] = T
                RT[3, 3] = 1.0
                T_co_init = RT
                T_co_opt, sdf_values = sdf_optim.refine_pose_layer(T_co_init, points.cuda(), steps=cfg.TEST.NUM_SDF_ITERATIONS)
                RT_opt = T_co_opt
                poses_refined[i, :4] = mat2quat(RT_opt[:3, :3])
                poses_refined[i, 4:] = RT_opt[:3, 3]

                # if 0:
                if cfg.TEST.VISUALIZE:
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    ax = fig.add_subplot(3, 3, 1, projection='3d')
                    if cls == -1:
                        points_obj = dataset._points_clamp
                    else:
                        points_obj = dataset._points_all[cls, :, :]

                    points_init = np.matmul(np.linalg.inv(T_co_init), points.numpy().transpose()).transpose()
                    points_opt = np.matmul(np.linalg.inv(T_co_opt), points.numpy().transpose()).transpose()

                    ax.scatter(points_obj[::5, 0], points_obj[::5, 1], points_obj[::5, 2], color='green')
                    ax.scatter(points_init[::10, 0], points_init[::10, 1], points_init[::10, 2], color='red')
                    ax.scatter(points_opt[::10, 0], points_opt[::10, 1], points_opt[::10, 2], color='blue')

                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')

                    ax.set_xlim(sdf_optim.xmin, sdf_optim.xmax)
                    ax.set_ylim(sdf_optim.ymin, sdf_optim.ymax)
                    ax.set_zlim(sdf_optim.zmin, sdf_optim.zmax)

                    ax = fig.add_subplot(3, 3, 2)
                    label_image = dataset.labels_to_image(np.multiply(im_label, mask_label))
                    plt.imshow(label_image)
                    ax.set_title('mask label')

                    ax = fig.add_subplot(3, 3, 3)
                    plt.imshow(mask_depth_meas)
                    ax.set_title('mask_depth_meas')

                    ax = fig.add_subplot(3, 3, 4)
                    plt.imshow(mask_depth_render)
                    ax.set_title('mask_depth_render')

                    ax = fig.add_subplot(3, 3, 5)
                    plt.imshow(mask_depth_vis)
                    ax.set_title('mask_depth_vis')

                    ax = fig.add_subplot(3, 3, 6)
                    plt.imshow(mask)
                    ax.set_title('mask')

                    ax = fig.add_subplot(3, 3, 7)
                    plt.imshow(depth_meas_roi)
                    ax.set_title('depth input')

                    ax = fig.add_subplot(3, 3, 8)
                    plt.imshow(depth_render_roi)
                    ax.set_title('depth render')

                    ax = fig.add_subplot(3, 3, 9)
                    plt.imshow(im_depth)
                    ax.set_title('depth image')

                    plt.show()

    return poses, poses_refined, cls_render_ids


# only render rois and segmentation masks
def render_image_detection(dataset, im, rois, labels):
    # label image
    label_image = dataset.labels_to_image(labels)
    im_label = im[:, :, (2, 1, 0)].copy()
    I = np.where(labels != 0)
    im_label[I[0], I[1], :] = 0.5 * label_image[I[0], I[1], :] + 0.5 * im_label[I[0], I[1], :]

    num = rois.shape[0]
    classes = dataset._classes
    class_colors = dataset._class_colors

    for i in range(num):
        if cfg.MODE == 'TEST':
            cls_index = int(rois[i, 1]) - 1
        else:
            cls_index = cfg.TRAIN.CLASSES[int(rois[i, 1])] - 1

        if cls_index < 0:
            continue

        cls = int(rois[i, 1])
        print(classes[cls], rois[i, -1])
        if cls > 0 and rois[i, -1] > cfg.TEST.DET_THRESHOLD:
            # draw roi
            x1 = rois[i, 2]
            y1 = rois[i, 3]
            x2 = rois[i, 4]
            y2 = rois[i, 5]
            cv2.rectangle(im_label, (x1, y1), (x2, y2), class_colors[cls], 2)

    return im_label


def render_image(dataset, im, rois, poses, poses_refine, labels, cls_render_ids=None):

    # label image
    label_image = dataset.labels_to_image(labels)
    im_label = im[:, :, (2, 1, 0)].copy()
    I = np.where(labels != 0)
    im_label[I[0], I[1], :] = 0.5 * label_image[I[0], I[1], :] + 0.5 * im_label[I[0], I[1], :]

    num = poses.shape[0]
    classes = dataset._classes
    class_colors = dataset._class_colors

    cls_indexes = []
    poses_all = []
    poses_refine_all = []
    for i in range(num):
        if cls_render_ids is not None and len(cls_render_ids) == num:
            cls_index = cls_render_ids[i]
        else:
            if cfg.MODE == 'TEST':
                cls_index = int(rois[i, 1]) - 1
            else:
                cls_index = cfg.TRAIN.CLASSES[int(rois[i, 1])] - 1

        if cls_index < 0:
            continue

        cls_indexes.append(cls_index)
        qt = np.zeros((7, ), dtype=np.float32)
        qt[:3] = poses[i, 4:7]
        qt[3:] = poses[i, :4]
        poses_all.append(qt.copy())

        if cfg.TEST.POSE_REFINE:
            qt[:3] = poses_refine[i, 4:7]
            qt[3:] = poses_refine[i, :4]
            poses_refine_all.append(qt.copy())

        cls = int(rois[i, 1])
        print(classes[cls], rois[i, -1], cls_index)
        if cls > 0 and rois[i, -1] > cfg.TEST.DET_THRESHOLD:
            # draw roi
            x1 = rois[i, 2]
            y1 = rois[i, 3]
            x2 = rois[i, 4]
            y2 = rois[i, 5]
            cv2.rectangle(im_label, (x1, y1), (x2, y2), class_colors[cls], 2)

    # rendering
    if len(cls_indexes) > 0:

        height = im.shape[0]
        width = im.shape[1]
        intrinsic_matrix = dataset._intrinsic_matrix
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.01
        image_tensor = torch.cuda.FloatTensor(height, width, 4)
        seg_tensor = torch.cuda.FloatTensor(height, width, 4)

        # set renderer
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

        # pose
        cfg.renderer.set_poses(poses_all)
        frame = cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        im_render = image_tensor.cpu().numpy()
        im_render = np.clip(im_render, 0, 1)
        im_render = im_render[:, :, :3] * 255
        im_render = im_render.astype(np.uint8)
        im_output = 0.2 * im[:,:,(2, 1, 0)].astype(np.float32) + 0.8 * im_render.astype(np.float32)

        # pose refine
        if cfg.TEST.POSE_REFINE:
             cfg.renderer.set_poses(poses_refine_all)
             frame = cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
             image_tensor = image_tensor.flip(0)
             im_render = image_tensor.cpu().numpy()
             im_render = np.clip(im_render, 0, 1)
             im_render = im_render[:, :, :3] * 255
             im_render = im_render.astype(np.uint8)
             im_output_refine = 0.2 * im[:,:,(2, 1, 0)].astype(np.float32) + 0.8 * im_render.astype(np.float32)
             im_output_refine = im_output_refine.astype(np.uint8)
        else:
             im_output_refine = None
    else:
        im_output = 0.4 * im[:,:,(2, 1, 0)]

    return im_output.astype(np.uint8), im_output_refine, im_label


def overlay_image(dataset, im, rois, poses, labels):

    im = im[:, :, (2, 1, 0)]
    classes = dataset._classes
    class_colors = dataset._class_colors
    points = dataset._points_all
    intrinsic_matrix = dataset._intrinsic_matrix
    height = im.shape[0]
    width = im.shape[1]

    label_image = dataset.labels_to_image(labels)
    im_label = im.copy()
    I = np.where(labels != 0)
    im_label[I[0], I[1], :] = 0.5 * label_image[I[0], I[1], :] + 0.5 * im_label[I[0], I[1], :]

    for j in xrange(rois.shape[0]):
        cls = int(rois[j, 1])
        print classes[cls], rois[j, -1]
        if cls > 0 and rois[j, -1] > cfg.TEST.DET_THRESHOLD:

            # draw roi
            x1 = rois[j, 2]
            y1 = rois[j, 3]
            x2 = rois[j, 4]
            y2 = rois[j, 5]
            cv2.rectangle(im_label, (x1, y1), (x2, y2), class_colors[cls], 2)

            # extract 3D points
            x3d = np.ones((4, points.shape[1]), dtype=np.float32)
            x3d[0, :] = points[cls,:,0]
            x3d[1, :] = points[cls,:,1]
            x3d[2, :] = points[cls,:,2]

            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(poses[j, :4])
            RT[:, 3] = poses[j, 4:7]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x = np.round(np.divide(x2d[0, :], x2d[2, :]))
            y = np.round(np.divide(x2d[1, :], x2d[2, :]))
            index = np.where((x >= 0) & (x < width) & (y >= 0) & (y < height))[0]
            x = x[index].astype(np.int32)
            y = y[index].astype(np.int32)
            im[y, x, 0] = class_colors[cls][0]
            im[y, x, 1] = class_colors[cls][1]
            im[y, x, 2] = class_colors[cls][2]

    return im, im_label


def _get_bb3D(extent):
    bb = np.zeros((3, 8), dtype=np.float32)
    
    xHalf = extent[0] * 0.5
    yHalf = extent[1] * 0.5
    zHalf = extent[2] * 0.5
    
    bb[:, 0] = [xHalf, yHalf, zHalf]
    bb[:, 1] = [-xHalf, yHalf, zHalf]
    bb[:, 2] = [xHalf, -yHalf, zHalf]
    bb[:, 3] = [-xHalf, -yHalf, zHalf]
    bb[:, 4] = [xHalf, yHalf, -zHalf]
    bb[:, 5] = [-xHalf, yHalf, -zHalf]
    bb[:, 6] = [xHalf, -yHalf, -zHalf]
    bb[:, 7] = [-xHalf, -yHalf, -zHalf]
    
    return bb


def convert_to_image(im_blob):
    return np.clip(255 * im_blob, 0, 255).astype(np.uint8)


def _vis_minibatch(inputs, background, labels, vertex_targets, sample, class_colors):

    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    im_blob = inputs.cpu().numpy()
    label_blob = labels.cpu().numpy()
    gt_poses = sample['poses'].numpy()
    meta_data_blob = sample['meta_data'].numpy()
    gt_boxes = sample['gt_boxes'].numpy()
    extents = sample['extents'][0, :, :].numpy()
    background_color = background['background_color'].numpy()

    if cfg.TRAIN.VERTEX_REG:
        vertex_target_blob = vertex_targets.cpu().numpy()

    if cfg.INPUT == 'COLOR':
        m = 3
        n = 3
    else:
        m = 3
        n = 4
    
    for i in range(im_blob.shape[0]):
        fig = plt.figure()
        start = 1

        metadata = meta_data_blob[i, :]
        intrinsic_matrix = metadata[:9].reshape((3,3))

        # show image
        if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
            if cfg.INPUT == 'COLOR':
                im = im_blob[i, :, :, :].copy()
            else:
                im = im_blob[i, :3, :, :].copy()
            im = im.transpose((1, 2, 0)) * 255.0
            im += cfg.PIXEL_MEANS
            im = im[:, :, (2, 1, 0)]
            im = np.clip(im, 0, 255)
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, 1)
            plt.imshow(im)
            ax.set_title('color')
            start += 1

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            if cfg.INPUT == 'DEPTH':
                im_depth = im_blob[i, :, :, :].copy()
            else:
                im_depth = im_blob[i, 3:6, :, :].copy()

            ax = fig.add_subplot(m, n, start)
            plt.imshow(im_depth[0, :, :])
            ax.set_title('depth x')
            start += 1

            ax = fig.add_subplot(m, n, start)
            plt.imshow(im_depth[1, :, :])
            ax.set_title('depth y')
            start += 1

            ax = fig.add_subplot(m, n, start)
            plt.imshow(im_depth[2, :, :])
            ax.set_title('depth z')
            start += 1

            if cfg.INPUT == 'RGBD':
                ax = fig.add_subplot(m, n, start)
                mask = im_blob[i, 6, :, :].copy()
                plt.imshow(mask)
                ax.set_title('depth mask')
                start += 1

        # project the 3D box to image
        pose_blob = gt_poses[i]
        for j in range(pose_blob.shape[0]):
            if pose_blob[j, 0] == 0:
                continue

            class_id = int(pose_blob[j, 1])
            bb3d = _get_bb3D(extents[class_id, :])
            x3d = np.ones((4, 8), dtype=np.float32)
            x3d[0:3, :] = bb3d
            
            # projection
            RT = np.zeros((3, 4), dtype=np.float32)

            # allocentric to egocentric
            T = pose_blob[j, 6:]
            qt = allocentric2egocentric(pose_blob[j, 2:6], T)
            RT[:3, :3] = quat2mat(qt)

            # RT[:3, :3] = quat2mat(pose_blob[j, 2:6])
            RT[:, 3] = pose_blob[j, 6:]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

            x1 = np.min(x2d[0, :])
            x2 = np.max(x2d[0, :])
            y1 = np.min(x2d[1, :])
            y2 = np.max(x2d[1, :])
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

        if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
            im_background = background_color[i]
            im_background = im_background.transpose((1, 2, 0)) * 255.0
            im_background += cfg.PIXEL_MEANS
            im_background = im_background[:, :, (2, 1, 0)]
            im_background = np.clip(im_background, 0, 255)
            im_background = im_background.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            plt.imshow(im_background)
            ax.set_title('background')
            start += 1

        # show gt boxes
        ax = fig.add_subplot(m, n, start)
        start += 1
        if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
            plt.imshow(im)
        else:
            plt.imshow(im_depth[2, :, :])
        ax.set_title('gt boxes')
        boxes = gt_boxes[i]
        for j in range(boxes.shape[0]):
            if boxes[j, 4] == 0:
                continue
            x1 = boxes[j, 0]
            y1 = boxes[j, 1]
            x2 = boxes[j, 2]
            y2 = boxes[j, 3]
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

        # show label
        label = label_blob[i, :, :, :]
        label = label.transpose((1, 2, 0))

        height = label.shape[0]
        width = label.shape[1]
        num_classes = label.shape[2]
        im_label = np.zeros((height, width, 3), dtype=np.uint8)
        for j in range(num_classes):
            I = np.where(label[:, :, j] > 0)
            im_label[I[0], I[1], :] = class_colors[j]

        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im_label)
        ax.set_title('label')

        # show vertex targets
        if cfg.TRAIN.VERTEX_REG:
            vertex_target = vertex_target_blob[i, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, num_classes):
                index = np.where(label[:, :, j] > 0)
                if len(index[0]) > 0:
                    center[0, index[0], index[1]] = vertex_target[3*j, index[0], index[1]]
                    center[1, index[0], index[1]] = vertex_target[3*j+1, index[0], index[1]]
                    center[2, index[0], index[1]] = np.exp(vertex_target[3*j+2, index[0], index[1]])

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[0,:,:])
            ax.set_title('center x') 

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[1,:,:])
            ax.set_title('center y')

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[2,:,:])
            ax.set_title('z')

        plt.show()


def _vis_test(inputs, labels, out_label, out_vertex, rois, poses, poses_refined, sample, points, points_clamp, classes, class_colors, pose_scores):

    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    im_blob = inputs.cpu().numpy()
    label_blob = labels.cpu().numpy()
    label_pred = out_label.cpu().numpy()
    gt_poses = sample['poses'].numpy()
    meta_data_blob = sample['meta_data'].numpy()
    metadata = meta_data_blob[0, :]
    intrinsic_matrix = metadata[:9].reshape((3,3))
    gt_boxes = sample['gt_boxes'].numpy()
    extents = sample['extents'][0, :, :].numpy()

    if cfg.TRAIN.VERTEX_REG or cfg.TRAIN.VERTEX_REG_DELTA:
        vertex_targets = sample['vertex_targets'].numpy()
        vertex_pred = out_vertex.detach().cpu().numpy()

    m = 4
    n = 4
    for i in range(im_blob.shape[0]):
        fig = plt.figure()
        start = 1

        # show image
        if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
            if cfg.INPUT == 'COLOR':
                im = im_blob[i, :, :, :].copy()
            else:
                im = im_blob[i, :3, :, :].copy()
            im = im.transpose((1, 2, 0)) * 255.0
            im += cfg.PIXEL_MEANS
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, 1)
            plt.imshow(im)
            ax.set_title('color')
            start += 1

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            if cfg.INPUT == 'DEPTH':
                im_depth = im_blob[i, :, :, :].copy()
            else:
                im_depth = im_blob[i, 3:, :, :].copy()

            ax = fig.add_subplot(m, n, start)
            plt.imshow(im_depth[0, :, :])
            ax.set_title('depth x')
            start += 1

            ax = fig.add_subplot(m, n, start)
            plt.imshow(im_depth[1, :, :])
            ax.set_title('depth y')
            start += 1

            ax = fig.add_subplot(m, n, start)
            plt.imshow(im_depth[2, :, :])
            ax.set_title('depth z')
            start += 1

        # show gt label
        label_gt = label_blob[i, :, :, :]
        label_gt = label_gt.transpose((1, 2, 0))
        height = label_gt.shape[0]
        width = label_gt.shape[1]
        num_classes = label_gt.shape[2]
        im_label_gt = np.zeros((height, width, 3), dtype=np.uint8)
        for j in range(num_classes):
            I = np.where(label_gt[:, :, j] > 0)
            im_label_gt[I[0], I[1], :] = class_colors[j]

        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im_label_gt)
        ax.set_title('gt labels') 

        # show predicted label
        label = label_pred[i, :, :]
        height = label.shape[0]
        width = label.shape[1]
        im_label = np.zeros((height, width, 3), dtype=np.uint8)
        for j in range(num_classes):
            I = np.where(label == j)
            im_label[I[0], I[1], :] = class_colors[j]

        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im_label)
        ax.set_title('predicted labels')

        # show gt boxes
        ax = fig.add_subplot(m, n, start)
        start += 1
        if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
            plt.imshow(im)
        else:
            plt.imshow(im_depth[2, :, :])
        ax.set_title('gt boxes') 
        boxes = gt_boxes[i]
        for j in range(boxes.shape[0]):
            if boxes[j, 4] == 0:
                continue
            x1 = boxes[j, 0]
            y1 = boxes[j, 1]
            x2 = boxes[j, 2]
            y2 = boxes[j, 3]
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3))

        if cfg.TRAIN.VERTEX_REG or cfg.TRAIN.VERTEX_REG_DELTA:

            # show predicted boxes
            ax = fig.add_subplot(m, n, start)
            start += 1
            if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
                plt.imshow(im)
            else:
                plt.imshow(im_depth[2, :, :])

            ax.set_title('predicted boxes')
            for j in range(rois.shape[0]):
                if rois[j, 0] != i or rois[j, -1] < cfg.TEST.DET_THRESHOLD:
                    continue
                cls = rois[j, 1]
                x1 = rois[j, 2]
                y1 = rois[j, 3]
                x2 = rois[j, 4]
                y2 = rois[j, 5]
                plt.gca().add_patch(
                    plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=np.array(class_colors[int(cls)])/255.0, linewidth=3))

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                plt.plot(cx, cy, 'yo')

            # show gt poses
            ax = fig.add_subplot(m, n, start)
            start += 1
            ax.set_title('gt poses')
            if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
                plt.imshow(im)
            else:
                plt.imshow(im_depth[2, :, :])

            pose_blob = gt_poses[i]
            for j in range(pose_blob.shape[0]):
                if pose_blob[j, 0] == 0:
                    continue

                cls = int(pose_blob[j, 1])
                # extract 3D points
                x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                x3d[0, :] = points[cls,:,0]
                x3d[1, :] = points[cls,:,1]
                x3d[2, :] = points[cls,:,2]
               
                # projection
                RT = np.zeros((3, 4), dtype=np.float32)
                qt = pose_blob[j, 2:6]
                T = pose_blob[j, 6:]
                qt_new = allocentric2egocentric(qt, T)
                RT[:3, :3] = quat2mat(qt_new)
                RT[:, 3] = T
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(class_colors[cls], 255.0), alpha=0.5)                    

            # show predicted poses
            ax = fig.add_subplot(m, n, start)
            start += 1
            ax.set_title('predicted poses')
            if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
                plt.imshow(im)
            else:
                plt.imshow(im_depth[2, :, :])
            for j in xrange(rois.shape[0]):
                if rois[j, 0] != i:
                    continue
                cls = int(rois[j, 1])
                if cls > 0:
                    print(i, classes[cls], rois[j, -1])
                elif cls == -1:
                    print(i, '051_large_clamp', rois[j, -1])
                if rois[j, -1] > cfg.TEST.DET_THRESHOLD:
                    # extract 3D points
                    x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                    if cls == -1:
                        x3d[0, :] = points_clamp[:,0]
                        x3d[1, :] = points_clamp[:,1]
                        x3d[2, :] = points_clamp[:,2]
                    else:
                        x3d[0, :] = points[cls,:,0]
                        x3d[1, :] = points[cls,:,1]
                        x3d[2, :] = points[cls,:,2]

                    # projection
                    RT = np.zeros((3, 4), dtype=np.float32)
                    RT[:3, :3] = quat2mat(poses[j, :4])
                    RT[:, 3] = poses[j, 4:7]
                    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                    plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(class_colors[cls], 255.0), alpha=0.5)

            # show predicted refined poses
            if cfg.TEST.POSE_REFINE:
                ax = fig.add_subplot(m, n, start)
                start += 1
                ax.set_title('predicted refined poses')
                if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
                    plt.imshow(im)
                else:
                    plt.imshow(im_depth[2, :, :])
                for j in xrange(rois.shape[0]):
                    if rois[j, 0] != i:
                        continue
                    cls = int(rois[j, 1])
                    if rois[j, -1] > cfg.TEST.DET_THRESHOLD:
                        # extract 3D points
                        x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                        if cls == -1:
                            x3d[0, :] = points_clamp[:,0]
                            x3d[1, :] = points_clamp[:,1]
                            x3d[2, :] = points_clamp[:,2]
                        else:
                            x3d[0, :] = points[cls,:,0]
                            x3d[1, :] = points[cls,:,1]
                            x3d[2, :] = points[cls,:,2]

                        # projection
                        RT = np.zeros((3, 4), dtype=np.float32)
                        RT[:3, :3] = quat2mat(poses_refined[j, :4])
                        RT[:, 3] = poses_refined[j, 4:7]
                        x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                        plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(class_colors[cls], 255.0), alpha=0.5)

            # show pose estimation quality
            if cfg.TEST.POSE_REFINE and pose_scores is not None:
                ax = fig.add_subplot(m, n, start)
                start += 1
                ax.set_title('pose score')
                if cfg.INPUT == 'COLOR' or cfg.INPUT == 'RGBD':
                    plt.imshow(im)
                else:
                    plt.imshow(im_depth[2, :, :])
                for j in xrange(rois.shape[0]):
                    if rois[j, 0] != i:
                        continue
                    cls = int(rois[j, 1])

                    if cls == -1:
                        cls_id = 19
                    else:
                        cls_id = cfg.TRAIN.CLASSES[cls]
                    if cls_id not in cfg.TEST.CLASSES:
                        continue

                    if rois[j, -1] > cfg.TEST.DET_THRESHOLD:
                        # extract 3D points
                        x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                        if cls == -1:
                            x3d[0, :] = points_clamp[:,0]
                            x3d[1, :] = points_clamp[:,1]
                            x3d[2, :] = points_clamp[:,2]
                        else:
                            x3d[0, :] = points[cls,:,0]
                            x3d[1, :] = points[cls,:,1]
                            x3d[2, :] = points[cls,:,2]

                        # projection
                        RT = np.zeros((3, 4), dtype=np.float32)
                        RT[:3, :3] = quat2mat(poses_refined[j, :4])
                        RT[:, 3] = poses_refined[j, 4:7]
                        x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                        pose_score = pose_scores[j]
                        if pose_score > 1.0:
                            pose_score = 1.0
                        plt.plot(x2d[0, :], x2d[1, :], '.', color=(1.0 - pose_score, pose_score, 0), alpha=0.5)

            # show gt vertex targets
            vertex_target = vertex_targets[i, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, num_classes):
                index = np.where(label_gt[:, :, j] > 0)
                if len(index[0]) > 0:
                    center[:, index[0], index[1]] = vertex_target[3*j:3*j+3, index[0], index[1]]

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[0,:,:])
            ax.set_title('gt center x') 

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[1,:,:])
            ax.set_title('gt center y')

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(np.exp(center[2,:,:]))
            ax.set_title('gt z')

            # show predicted vertex targets
            vertex_target = vertex_pred[i, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, num_classes):
                index = np.where(label == j)
                if len(index[0]) > 0:
                    center[:, index[0], index[1]] = vertex_target[3*j:3*j+3, index[0], index[1]]

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[0,:,:])
            ax.set_title('predicted center x') 

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(center[1,:,:])
            ax.set_title('predicted center y')

            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(np.exp(center[2,:,:]))
            ax.set_title('predicted z')

        plt.show()



def vis_test(dataset, im, im_depth, label, out_vertex, rois, poses, poses_refined, im_pose, im_pose_refine):

    """Visualize a testing results."""
    import matplotlib.pyplot as plt

    num_classes = dataset.num_classes
    classes = dataset._classes
    class_colors = dataset._class_colors
    points = dataset._points_all
    intrinsic_matrix = dataset._intrinsic_matrix
    vertex_pred = out_vertex.detach().cpu().numpy()
    height = label.shape[0]
    width = label.shape[1]

    fig = plt.figure()
    # show image
    im = im[0, :, :, :].copy()
    im = im.transpose((1, 2, 0)) * 255.0
    im += cfg.PIXEL_MEANS
    im = im[:, :, (2, 1, 0)]
    im = im.astype(np.uint8)
    ax = fig.add_subplot(3, 3, 1)
    plt.imshow(im)
    ax.set_title('input image') 

    # show predicted label
    im_label = dataset.labels_to_image(label)
    ax = fig.add_subplot(3, 3, 2)
    plt.imshow(im_label)
    ax.set_title('predicted labels')

    ax = fig.add_subplot(3, 3, 8)
    plt.imshow(im_pose)
    ax.set_title('rendered image')

    if cfg.TEST.POSE_REFINE:
        ax = fig.add_subplot(3, 3, 9)
        plt.imshow(im_pose_refine)
        ax.set_title('rendered image refine')

    if cfg.TRAIN.VERTEX_REG or cfg.TRAIN.VERTEX_REG_DELTA:

        # show predicted boxes
        ax = fig.add_subplot(3, 3, 3)
        plt.imshow(im)
        ax.set_title('predicted boxes')
        for j in range(rois.shape[0]):
            cls = rois[j, 1]
            if cfg.TRAIN.CLASSES[int(cls)] not in cfg.TEST.CLASSES:
                continue

            x1 = rois[j, 2]
            y1 = rois[j, 3]
            x2 = rois[j, 4]
            y2 = rois[j, 5]
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=np.array(class_colors[int(cls)])/255.0, linewidth=3))

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            plt.plot(cx, cy, 'yo')

        # show predicted poses
        if cfg.TRAIN.POSE_REG:
            ax = fig.add_subplot(3, 3, 4)
            ax.set_title('predicted poses')
            plt.imshow(im)
            for j in xrange(rois.shape[0]):
                cls = int(rois[j, 1])
                print classes[cls], rois[j, -1]
                if cls > 0 and rois[j, -1] > cfg.TEST.DET_THRESHOLD:
                    # extract 3D points
                    x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                    x3d[0, :] = points[cls,:,0]
                    x3d[1, :] = points[cls,:,1]
                    x3d[2, :] = points[cls,:,2]

                    # projection
                    RT = np.zeros((3, 4), dtype=np.float32)
                    RT[:3, :3] = quat2mat(poses[j, :4])
                    RT[:, 3] = poses[j, 4:7]
                    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                    plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(class_colors[cls], 255.0), alpha=0.5)

        elif im_depth is not None:
            im = im_depth.copy()
            ax = fig.add_subplot(3, 3, 4)
            plt.imshow(im)
            ax.set_title('input depth') 

        # show predicted vertex targets
        vertex_target = vertex_pred[0, :, :, :]
        center = np.zeros((3, height, width), dtype=np.float32)

        for j in range(1, num_classes):
            index = np.where(label == j)
            if len(index[0]) > 0:
                center[0, index[0], index[1]] = vertex_target[3*j, index[0], index[1]]
                center[1, index[0], index[1]] = vertex_target[3*j+1, index[0], index[1]]
                center[2, index[0], index[1]] = np.exp(vertex_target[3*j+2, index[0], index[1]])

        ax = fig.add_subplot(3, 3, 5)
        plt.imshow(center[0,:,:])
        ax.set_title('predicted center x') 

        ax = fig.add_subplot(3, 3, 6)
        plt.imshow(center[1,:,:])
        ax.set_title('predicted center y')

        ax = fig.add_subplot(3, 3, 7)
        plt.imshow(center[2,:,:])
        ax.set_title('predicted z')

    plt.show()
