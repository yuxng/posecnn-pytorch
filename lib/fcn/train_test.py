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
from utils.se3 import *
from utils.pose_error import re, te
from scipy.optimize import minimize

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
        meta_data = sample['meta_data'].cuda()
        extents = sample['extents'][0, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1).cuda()
        gt_boxes = sample['gt_boxes'].cuda()
        poses = sample['poses'].cuda()
        points = sample['points'][0, :, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1, 1).cuda()
        symmetry = sample['symmetry'][0, :].repeat(cfg.TRAIN.GPUNUM, 1).cuda()

        if cfg.TRAIN.VERTEX_REG:
            vertex_targets = sample['vertex_targets'].cuda()
            vertex_weights = sample['vertex_weights'].cuda()
        else:
            vertex_targets = []
            vertex_weights = []

        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch(inputs, labels, vertex_targets, sample, train_loader.dataset.class_colors)

        # compute output
        if cfg.TRAIN.VERTEX_REG:
            out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, \
                bbox_labels, bbox_pred, bbox_targets, bbox_inside_weights, loss_pose_tensor, poses_weight \
                = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)

            loss_label = loss_cross_entropy(out_logsoftmax, out_weight)
            loss_vertex = smooth_l1_loss(out_vertex, vertex_targets, vertex_weights)
            loss_box = loss_cross_entropy(out_logsoftmax_box, bbox_labels)
            loss_location = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights)
            loss_pose = torch.mean(loss_pose_tensor)
            loss = loss_label + loss_vertex + loss_box + loss_location + loss_pose
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
            num_bg = torch.sum(bbox_labels[:, 0])
            num_fg = torch.sum(torch.sum(bbox_labels[:, 1:], dim=1))
            num_fg_pose = torch.sum(torch.sum(poses_weight[:, 4:], dim=1)) / 4
            print('[%d/%d][%d/%d], %.4f, label %.4f, center %.4f, box %.4f (%03d, %03d), loc %.4f, pose %.4f (%03d), lr %.6f, time %.2f' \
               % (epoch, cfg.epochs, i, epoch_size, loss.data, loss_label.data, loss_vertex.data, loss_box.data, num_fg.data, num_bg.data, \
                  loss_location.data, loss_pose.data, num_fg_pose, optimizer.param_groups[0]['lr'], batch_time.val))
        else:
            print('[%d/%d][%d/%d], loss %.4f, lr %.6f, time %.2f' \
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
        gt_boxes = sample['gt_boxes'].cuda()
        poses = sample['poses'].cuda()
        points = sample['points'][0, :, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1, 1).cuda()
        symmetry = sample['symmetry'][0, :].repeat(cfg.TRAIN.GPUNUM, 1).cuda()
        
        # compute output
        if cfg.TRAIN.VERTEX_REG:
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

            # optimize depths
            poses = optimize_depths(rois, poses, test_loader.dataset._points_all, test_loader.dataset._intrinsic_matrix)
        else:
            out_label = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
            out_vertex = []
            rois = []
            poses = []

        if cfg.TEST.VISUALIZE:
            _vis_test(inputs, labels, out_label, out_vertex, rois, poses, sample, \
                test_loader.dataset._points_all, test_loader.dataset.classes, test_loader.dataset.class_colors)

        # measure elapsed time
        batch_time.update(time.time() - end)

        print('[%d/%d], batch time %.2f' % (i, epoch_size, batch_time.val))


def optimize_depths(rois, poses, points, intrinsic_matrix):

    num = rois.shape[0]
    for i in range(num):
        roi = rois[i, 2:6]
        width = roi[2] - roi[0]
        height = roi[3] - roi[1]
        cls = int(rois[i, 1])

        RT = np.zeros((3, 4), dtype=np.float32)
        RT[:3, :3] = quat2mat(poses[i, :4])
        RT[0, 3] = poses[i, 4] * poses[i, 6]
        RT[1, 3] = poses[i, 5] * poses[i, 6]
        RT[2, 3] = poses[i, 6]

        # extract 3D points
        x3d = np.ones((4, points.shape[1]), dtype=np.float32)
        x3d[0, :] = points[cls,:,0]
        x3d[1, :] = points[cls,:,1]
        x3d[2, :] = points[cls,:,2]

        # optimization
        x0 = poses[i, 6]
        res = minimize(objective_depth, x0, args=(width, height, RT, x3d, intrinsic_matrix), method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
        poses[i, 4] *= res.x 
        poses[i, 5] *= res.x
        poses[i, 6] = res.x

    return poses


def objective_depth(x, width, height, RT, x3d, intrinsic_matrix):

    # project points
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


def _vis_minibatch(inputs, labels, vertex_targets, sample, class_colors):

    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    im_blob = inputs.cpu().numpy()
    label_blob = labels.cpu().numpy()
    gt_poses = sample['poses'].numpy()
    meta_data_blob = sample['meta_data'].numpy()
    metadata = meta_data_blob[0, :]
    intrinsic_matrix = metadata[:9].reshape((3,3))
    gt_boxes = sample['gt_boxes'].numpy()
    extents = sample['extents'][0, :, :].numpy()

    if cfg.TRAIN.VERTEX_REG:
        vertex_target_blob = vertex_targets.cpu().numpy()
    
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
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

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
                    center[0, index[0], index[1]] = vertex_target[3*j, index[0], index[1]]
                    center[1, index[0], index[1]] = vertex_target[3*j+1, index[0], index[1]]
                    center[2, index[0], index[1]] = np.exp(vertex_target[3*j+2, index[0], index[1]])

            ax = fig.add_subplot(2, 3, 4)
            plt.imshow(center[0,:,:])
            ax.set_title('center x') 

            ax = fig.add_subplot(2, 3, 5)
            plt.imshow(center[1,:,:])
            ax.set_title('center y')

            ax = fig.add_subplot(2, 3, 6)
            plt.imshow(center[2,:,:])
            ax.set_title('z')

        plt.show()


def _vis_test(inputs, labels, out_label, out_vertex, rois, poses, sample, points, classes, class_colors):

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

    if cfg.TRAIN.VERTEX_REG:
        vertex_targets = sample['vertex_targets'].numpy()
        vertex_pred = out_vertex.detach().cpu().numpy()
    
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
        ax.set_title('gt boxes') 

        # show gt boxes
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

        ax = fig.add_subplot(3, 4, 5)
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

        ax = fig.add_subplot(3, 4, 6)
        plt.imshow(im_label)
        ax.set_title('predicted labels')

        if cfg.TRAIN.VERTEX_REG:

            # show predicted boxes
            ax = fig.add_subplot(3, 4, 2)
            plt.imshow(im)
            ax.set_title('predicted boxes')
            for j in range(rois.shape[0]):
                if rois[j, 0] != i:
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
            ax = fig.add_subplot(3, 4, 9)
            ax.set_title('gt poses')
            plt.imshow(im)
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
            ax = fig.add_subplot(3, 4, 10)
            ax.set_title('predicted poses')
            plt.imshow(im)
            for j in xrange(rois.shape[0]):
                cls = int(rois[j, 1])
                print classes[cls], rois[j, -1]
                if cls > 0 and rois[j, -1] > 0.1:
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

            # show gt vertex targets
            vertex_target = vertex_targets[i, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, num_classes):
                index = np.where(label_gt[:, :, j] > 0)
                if len(index[0]) > 0:
                    center[:, index[0], index[1]] = vertex_target[3*j:3*j+3, index[0], index[1]]

            ax = fig.add_subplot(3, 4, 3)
            plt.imshow(center[0,:,:])
            ax.set_title('gt center x') 

            ax = fig.add_subplot(3, 4, 7)
            plt.imshow(center[1,:,:])
            ax.set_title('gt center y')

            ax = fig.add_subplot(3, 4, 11)
            plt.imshow(np.exp(center[2,:,:]))
            ax.set_title('gt z')

            # show predicted vertex targets
            vertex_target = vertex_pred[i, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, num_classes):
                index = np.where(label == j)
                if len(index[0]) > 0:
                    center[:, index[0], index[1]] = vertex_target[3*j:3*j+3, index[0], index[1]]

            ax = fig.add_subplot(3, 4, 4)
            plt.imshow(center[0,:,:])
            ax.set_title('predicted center x') 

            ax = fig.add_subplot(3, 4, 8)
            plt.imshow(center[1,:,:])
            ax.set_title('predicted center y')

            ax = fig.add_subplot(3, 4, 12)
            plt.imshow(np.exp(center[2,:,:]))
            ax.set_title('predicted z')

        plt.show()
