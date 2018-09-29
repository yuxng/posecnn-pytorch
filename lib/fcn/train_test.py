# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
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

        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch(input_var, label_var, sample, train_loader.dataset.class_colors)

        # compute output
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

        print('epoch: [%d/%d][%d/%d], loss %.4f, lr %.6f, batch time %.2f' \
           % (epoch, cfg.epochs, i, epoch_size, loss, optimizer.param_groups[0]['lr'], batch_time.val))

        cfg.TRAIN.ITERS += 1


def test(test_loader, network):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    epoch_size = len(test_loader)
    num_iterations = cfg.TEST.ITERNUM

    # switch to test mode
    network.eval()

    end = time.time()
    for i, sample in enumerate(test_loader):

        result = []
        vis_data = []
        poses_est = np.zeros((0, 9), dtype=np.float32)
        for j in range(num_iterations):
            inputs, flow, poses_src, poses_tgt, weights_rot, extents, points, affine_matrices, zoom_factor, vdata = process_sample(sample, poses_est)
            vis_data.append(vdata)

            # measure data loading time
            data_time.update(time.time() - end)
            flow = flow.cuda(async=True)
            inputs = [item.cuda() for item in inputs]
            poses_src = poses_src.cuda()
            poses_tgt = poses_tgt.cuda()
            weights_rot = weights_rot.cuda()
            extents = extents.cuda()
            points = points.cuda()
            affine_matrices = affine_matrices.cuda()
            zoom_factor = zoom_factor.cuda()

            input_var = torch.autograd.Variable(torch.cat(inputs, 1))
            flow_var = torch.autograd.Variable(flow)
            poses_src_var = torch.autograd.Variable(poses_src)
            poses_tgt_var = torch.autograd.Variable(poses_tgt)
            extents_var = torch.autograd.Variable(extents)
            points_var = torch.autograd.Variable(points)
            affine_matrices_var = torch.autograd.Variable(affine_matrices)
            zoom_factor_var = torch.autograd.Variable(zoom_factor)

            # zoom in image
            grids = nn.functional.affine_grid(affine_matrices_var, input_var.size())
            input_zoom = nn.functional.grid_sample(input_var, grids)

            # compute output
     	    quaternion_delta_var, translation_var \
                = network(input_zoom, weights_rot, poses_src_var, poses_tgt_var, extents_var, points_var, zoom_factor_var)
            quaternion_delta = quaternion_delta_var.detach().cpu().numpy()
            translation = translation_var.detach().cpu().numpy()
            poses_est = _compute_pose_target(quaternion_delta, translation, vdata['pose_src'])
            result.append(poses_est)

        if cfg.TEST.VISUALIZE:
            _vis_test(result, vis_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('[%d/%d]' % (i, epoch_size))


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


def _vis_test(result, vis_data):

    num_iter = len(result)
    num_obj = vis_data[0]['pose_tgt'].shape[0]
    im_blob = vis_data[0]['image']

    import matplotlib.pyplot as plt
    for j in xrange(num_obj):

        fig = plt.figure()

        # show input image
        im = im_blob[0, :, :, :].copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(3, num_iter, 1)
        plt.imshow(im)
        ax.set_title('input image')

        for i in xrange(num_iter):
            poses_est = result[i]

            intrinsic_matrix = vis_data[i]['intrinsic_matrix']
            poses_src = vis_data[i]['pose_src']
            poses_tgt = vis_data[i]['pose_tgt']
            image_src_blob = vis_data[i]['image_src']
            image_tgt_blob = vis_data[i]['image_tgt']

            height = image_src_blob.shape[1]
            width = image_src_blob.shape[2]

            # images in BGR order
            num = poses_est.shape[0]
            images_est = np.zeros((num, height, width, 3), dtype=np.float32)
            render_one_poses(cfg.synthesizer, height, width, intrinsic_matrix, poses_est, images_est)
            images_est = convert_to_image(images_est)
	
            # compute error
            R_est = quat2mat(poses_est[j, 2:6])
            R_src = quat2mat(poses_src[j, 2:6])
            R_tgt = quat2mat(poses_tgt[j, 2:6])
            error_rot_src = re(R_src, R_tgt)
            error_rot_est = re(R_est, R_tgt)

            T_est = poses_est[j, 6:]
            T_src = poses_src[j, 6:]
            T_tgt = poses_tgt[j, 6:]
            error_tran_src = te(T_src, T_tgt)
            error_tran_est = te(T_est, T_tgt)

            # show rendered images
            im = image_src_blob[j, :, :, :].copy()
            im += cfg.PIXEL_MEANS
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
            ax = fig.add_subplot(3, num_iter, num_iter + 1 + i)
            ax.set_title('source iter %d (rot %.2f, tran %.4f)' % (i+1, error_rot_src, error_tran_src)) 
            plt.imshow(im)

            if i == 0:
                im = image_tgt_blob[j, :, :, :3].copy()
                im += cfg.PIXEL_MEANS
                im = im[:, :, (2, 1, 0)]
                im = im.astype(np.uint8)
                ax = fig.add_subplot(3, num_iter, 2)
                ax.set_title('target image') 
                plt.imshow(im)

            # show estimated image
            im = images_est[j, :, :, :3].copy()
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
            ax = fig.add_subplot(3, num_iter, 2 * num_iter + 1 + i)
            ax.set_title('estimate iter %d (rot %.2f, tran %.4f)' % (i+1, error_rot_est, error_tran_est)) 
            plt.imshow(im)

        plt.show()


def convert_to_image(im_blob):
    return np.clip(255 * im_blob, 0, 255).astype(np.uint8)


def _vis_minibatch(input_var, label_var, sample, class_colors):

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

        plt.show()
