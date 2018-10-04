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
import sys
import numpy as np

from fcn.config import cfg
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.se3 import T_inv_transform
from utils.pose_error import re, te

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


def smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights, sigma=1.0):
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = torch.mul(vertex_weights, vertex_diff)
    abs_diff = torch.abs(diff)
    smoothL1_sign = torch.lt(abs_diff, 1. / sigma_2).float().detach()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss = torch.div( torch.sum(in_loss), torch.sum(vertex_weights) + 1e-10 )
    return loss


def train(train_loader, network, optimizer, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()

    epoch_size = len(train_loader)

    # switch to train mode
    network.train()

    for i, sample in enumerate(train_loader):

        end = time.time()

        inputs = sample['image'].cuda()
        labels = sample['label'].cuda()
        input_var = torch.autograd.Variable(inputs)
        label_var = torch.autograd.Variable(labels)

        if cfg.TRAIN.VERTEX_REG:
            vertex_targets = sample['vertex_targets'].cuda()
            vertex_weights = sample['vertex_weights'].cuda()
            vertex_targets_var = torch.autograd.Variable(vertex_targets)
            vertex_weights_var = torch.autograd.Variable(vertex_weights)
        else:
            vertex_targets_var = []
            vertex_weights_var = []

        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch(input_var, label_var, vertex_targets_var, sample, train_loader.dataset.class_colors)

        # compute output
        if cfg.TRAIN.VERTEX_REG:
            out_logsoftmax, out_weight, out_vertex = network(input_var, label_var)
            loss_label = loss_cross_entropy(out_logsoftmax, out_weight)
            loss_vertex = smooth_l1_loss_vertex(out_vertex, vertex_targets_var, vertex_weights_var)
            loss = loss_label + loss_vertex
        else:
            out_logsoftmax, out_weight = network(input_var, label_var)
            loss = loss_cross_entropy(out_logsoftmax, out_weight)

        # record loss
        losses.update(loss.data, input_var.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if cfg.TRAIN.VERTEX_REG:
            print('epoch: [%d/%d][%d/%d], loss %.4f, loss_label %.4f, loss_center %.4f, lr %.6f, batch time %.2f' \
               % (epoch, cfg.epochs, i, epoch_size, loss.data, loss_label.data, loss_vertex.data, optimizer.param_groups[0]['lr'], batch_time.val))
        else:
            print('epoch: [%d/%d][%d/%d], loss %.4f, lr %.6f, batch time %.2f' \
               % (epoch, cfg.epochs, i, epoch_size, loss, optimizer.param_groups[0]['lr'], batch_time.val))

        cfg.TRAIN.ITERS += 1


def test(test_loader, network):

    batch_time = AverageMeter()
    epoch_size = len(test_loader)

    # switch to test mode
    network.eval()

    for i, sample in enumerate(test_loader):

        end = time.time()

        inputs = sample['image'].cuda()
        labels = sample['label'].cuda()
        meta_data = sample['meta_data'].cuda()
        extents = sample['extents'][0, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1).cuda()

        input_var = torch.autograd.Variable(inputs)
        label_var = torch.autograd.Variable(labels)
        meta_data_var = torch.autograd.Variable(meta_data)
        extents_var = torch.autograd.Variable(extents)
        
        # compute output
        if cfg.TRAIN.VERTEX_REG:
            out_label, out_vertex, out_box, out_pose = network(input_var, label_var, meta_data_var, extents_var)
        else:
            out_label = network(input_var, label_var, meta_data_var, extents_var)
            out_vertex = []
            out_box = []
            out_pose = []

        if cfg.TEST.VISUALIZE:
            _vis_test(input_var, label_var, out_label, out_vertex, out_box, out_pose, sample, test_loader.dataset.class_colors)

        # measure elapsed time
        batch_time.update(time.time() - end)

        print('[%d/%d], batch time %.2f' % (i, epoch_size, batch_time.val))


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


def _vis_minibatch(input_var, label_var, vertex_targets_var, sample, class_colors):

    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    im_blob = input_var.cpu().numpy()
    label_blob = label_var.cpu().numpy()
    poses = sample['poses'].numpy()
    meta_data_blob = sample['meta_data'].numpy()
    metadata = meta_data_blob[0, :]
    intrinsic_matrix = metadata[:9].reshape((3,3))
    gt_boxes = sample['gt_boxes'].numpy()
    extents = sample['extents'][0, :, :].numpy()

    if cfg.TRAIN.VERTEX_REG:
        vertex_target_blob = vertex_targets_var.cpu().numpy()
    
    for i in range(im_blob.shape[0]):
        fig = plt.figure()
        # show image
        im = im_blob[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(2, 3, 1)
        plt.imshow(im)
        ax.set_title('color') 

        # project the 3D box to image
        pose_blob = poses[i]
        for j in range(pose_blob.shape[0]):
            if pose_blob[j, 0] == 0:
                continue

            class_id = int(pose_blob[j, 1])
            bb3d = _get_bb3D(extents[class_id, :])
            x3d = np.ones((4, 8), dtype=np.float32)
            x3d[0:3, :] = bb3d
            
            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(pose_blob[j, 2:6])
            RT[:, 3] = pose_blob[j, 6:]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

            x1 = np.min(x2d[0, :])
            x2 = np.max(x2d[0, :])
            y1 = np.min(x2d[1, :])
            y2 = np.max(x2d[1, :])
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3))

        # show gt boxes
        ax = fig.add_subplot(2, 3, 2)
        plt.imshow(im)
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

        ax = fig.add_subplot(2, 3, 3)
        plt.imshow(im_label)
        ax.set_title('label')

        # show vertex targets
        if cfg.TRAIN.VERTEX_REG:
            vertex_target = vertex_target_blob[i, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, num_classes):
                index = np.where(label[:, :, j] > 0)
                if len(index[0]) > 0:
                    center[:, index[0], index[1]] = vertex_target[3*j:3*j+3, index[0], index[1]]

            ax = fig.add_subplot(2, 3, 4)
            plt.imshow(center[0,:,:])
            ax.set_title('center x') 

            ax = fig.add_subplot(2, 3, 5)
            plt.imshow(center[1,:,:])
            ax.set_title('center y')

            ax = fig.add_subplot(2, 3, 6)
            plt.imshow(np.exp(center[2,:,:]))
            ax.set_title('z')

        plt.show()


def _vis_test(input_var, label_var, out_label, out_vertex, out_box, out_pose, sample, class_colors):

    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    im_blob = input_var.cpu().numpy()
    label_blob = label_var.cpu().numpy()
    label_pred = out_label.cpu().numpy()
    poses = sample['poses'].numpy()
    meta_data_blob = sample['meta_data'].numpy()
    metadata = meta_data_blob[0, :]
    intrinsic_matrix = metadata[:9].reshape((3,3))
    gt_boxes = sample['gt_boxes'].numpy()
    extents = sample['extents'][0, :, :].numpy()

    if cfg.TRAIN.VERTEX_REG:
        vertex_targets = sample['vertex_targets'].numpy()
        vertex_pred = out_vertex.detach().cpu().numpy()
        box_pred = out_box.detach().cpu().numpy()
    
    for i in range(im_blob.shape[0]):
        fig = plt.figure()
        # show image
        im = im_blob[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(3, 4, 1)
        plt.imshow(im)
        ax.set_title('color') 

        # project the 3D box to image
        pose_blob = poses[i]
        for j in range(pose_blob.shape[0]):
            if pose_blob[j, 0] == 0:
                continue

            class_id = int(pose_blob[j, 1])
            bb3d = _get_bb3D(extents[class_id, :])
            x3d = np.ones((4, 8), dtype=np.float32)
            x3d[0:3, :] = bb3d
            
            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(pose_blob[j, 2:6])
            RT[:, 3] = pose_blob[j, 6:]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

            x1 = np.min(x2d[0, :])
            x2 = np.max(x2d[0, :])
            y1 = np.min(x2d[1, :])
            y2 = np.max(x2d[1, :])
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3))

        # show gt boxes
        ax = fig.add_subplot(3, 4, 2)
        plt.imshow(im)
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

        ax = fig.add_subplot(3, 4, 3)
        plt.imshow(im_label_gt)
        ax.set_title('gt label') 

        # show predicted label
        label = label_pred[i, :, :]
        height = label.shape[0]
        width = label.shape[1]
        im_label = np.zeros((height, width, 3), dtype=np.uint8)
        for j in range(num_classes):
            I = np.where(label == j)
            im_label[I[0], I[1], :] = class_colors[j]

        ax = fig.add_subplot(3, 4, 4)
        plt.imshow(im_label)
        ax.set_title('predicted label')

        if cfg.TRAIN.VERTEX_REG:

            # show predicted boxes
            ax = fig.add_subplot(3, 4, 5)
            plt.imshow(im)
            ax.set_title('predicted boxes')
            boxes = gt_boxes[i]
            for j in range(box_pred.shape[0]):
                if box_pred[j, 0] != i:
                    continue
                cls = box_pred[j, 1]
                x1 = box_pred[j, 2]
                y1 = box_pred[j, 3]
                x2 = box_pred[j, 4]
                y2 = box_pred[j, 5]
                plt.gca().add_patch(
                    plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=np.array(class_colors[int(cls)])/255.0, linewidth=3))

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                plt.plot(cx, cy, 'yo')

            # show gt vertex targets
            vertex_target = vertex_targets[i, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, num_classes):
                index = np.where(label_gt[:, :, j] > 0)
                if len(index[0]) > 0:
                    center[:, index[0], index[1]] = vertex_target[3*j:3*j+3, index[0], index[1]]

            ax = fig.add_subplot(3, 4, 6)
            plt.imshow(center[0,:,:])
            ax.set_title('gt center x') 

            ax = fig.add_subplot(3, 4, 7)
            plt.imshow(center[1,:,:])
            ax.set_title('gt center y')

            ax = fig.add_subplot(3, 4, 8)
            plt.imshow(np.exp(center[2,:,:]))
            ax.set_title('gt z')

            # show predicted vertex targets
            vertex_target = vertex_pred[i, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, num_classes):
                index = np.where(label == j)
                if len(index[0]) > 0:
                    center[:, index[0], index[1]] = vertex_target[3*j:3*j+3, index[0], index[1]]

            ax = fig.add_subplot(3, 4, 10)
            plt.imshow(center[0,:,:])
            ax.set_title('predicted center x') 

            ax = fig.add_subplot(3, 4, 11)
            plt.imshow(center[1,:,:])
            ax.set_title('predicted center y')

            ax = fig.add_subplot(3, 4, 12)
            plt.imshow(np.exp(center[2,:,:]))
            ax.set_title('predicted z')

        plt.show()
