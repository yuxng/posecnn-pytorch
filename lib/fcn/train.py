# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import torch
import torch.nn as nn
import time
import sys, os
import numpy as np
import matplotlib.pyplot as plt

from fcn.config import cfg
from fcn.test_common import _vis_minibatch_autoencoder
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.se3 import *
from utils.nms import *
from utils.pose_error import re, te
from utils.blob import add_gaussian_noise_cuda

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

#************************************#
#    train PoseCNN                   #
#************************************#

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


#************************************#
#    train autoencoders              #
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
