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

        if cfg.INPUT == 'DEPTH':
            inputs = sample['image_depth']
        elif cfg.INPUT == 'COLOR':
            inputs = sample['image_color']
        elif cfg.INPUT == 'RGBD':
            inputs = torch.cat((sample['image_color'], sample['image_depth']), dim=1)
        im_info = sample['im_info']

        # add background
        mask = sample['mask']
        try:
            _, background = next(enum_background)
        except:
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)
        background = background.cuda()

        for j in range(inputs.size(0)):
            is_syn = im_info[j, -1]
            if is_syn or np.random.rand(1) > 0.5:
                inputs[j] = mask[j] * inputs[j] + (1 - mask[j]) * background[j]

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
        else:
            print('[%d/%d][%d/%d], loss %.4f, lr %.6f, time %.2f' \
               % (epoch, cfg.epochs, i, epoch_size, loss, optimizer.param_groups[0]['lr'], batch_time.val))

        cfg.TRAIN.ITERS += 1


def train_autoencoder(train_loader, background_loader, network, optimizer, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()

    epoch_size = len(train_loader)
    enum_background = enumerate(background_loader)

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
        background = background.cuda()
        inputs = image + (1 - mask) * background
        inputs = torch.clamp(inputs, min=0.0, max=1.0)

        # compute output
        out_images, embeddings = network(inputs)

        # reconstruction loss
        targets = sample['image_target']
        loss, losses_euler = BootstrapedMSEloss(out_images, targets, cfg.TRAIN.BOOSTRAP_PIXELS)

        # record the losses for each euler pose
        index_euler = sample['index_euler']
        train_loader.dataset._losses_pose[torch.flatten(index_euler)] = losses_euler

        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch_autoencoder(inputs, sample, out_images)

        # record loss
        losses.update(loss.data, inputs.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        print('[%d/%d][%d/%d], loss %.4f, lr %.6f, boost %d, time %.2f' \
           % (epoch, cfg.epochs, i, epoch_size, loss, optimizer.param_groups[0]['lr'], cfg.TRAIN.BOOSTRAP_PIXELS, batch_time.val))
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


def _vis_minibatch_autoencoder(inputs, sample, outputs, im_render=None):

    im_blob = inputs.cpu().numpy()
    targets = sample['image_target'].cpu().numpy()
    im_output = outputs.cpu().detach().numpy()

    import matplotlib.pyplot as plt
    for i in range(im_blob.shape[0]):
        fig = plt.figure()
        # show image
        im = im_blob[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im [:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(im)
        ax.set_title('input')

        # show target
        im = targets[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im.astype(np.uint8)
        im = im [:, :, (2, 1, 0)]
        ax = fig.add_subplot(2, 2, 2)
        plt.imshow(im)
        ax.set_title('target')

        # show matched codebook
        if im_render is not None:
            im = im_render[i].copy()
            ax = fig.add_subplot(2, 2, 3)
            plt.imshow(im)
            ax.set_title('matched code')

        # show output
        im = im_output[i, :, :, :].copy()
        im = np.clip(im, 0, 1)
        im = im.transpose((1, 2, 0)) * 255.0
        im = im [:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(2, 2, 4)
        plt.imshow(im)
        ax.set_title('reconstruction')

        plt.show()


def test_autoencoder(test_loader, background_loader, network, output_dir):

    batch_time = AverageMeter()
    epoch_size = len(test_loader)
    enum_background = enumerate(background_loader)

    if cfg.TEST.BUILD_CODEBOOK:
        num = len(test_loader.dataset.eulers)
        codes = np.zeros((num, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
        poses = np.zeros((num, 7), dtype=np.float32)
        count = 0
    else:
        # check if codebook exists
        filename = os.path.join(output_dir, 'codebook_%s.mat' % test_loader.dataset.name)
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

        if cfg.TEST.BUILD_CODEBOOK == False:
            mask = sample['mask']
            affine_matrix = sample['affine_matrix']

            # affine transformation
            grids = nn.functional.affine_grid(affine_matrix, image.size())
            image = nn.functional.grid_sample(image, grids, padding_mode='border')
            mask = nn.functional.grid_sample(mask, grids, mode='nearest')

            _, background = next(enum_background)
            if image.size(0) != background.size(0):
                enum_background = enumerate(background_loader)
                _, background = next(enum_background)

            background = background.cuda()
            inputs = image + (1 - mask) * background
            inputs = torch.clamp(inputs, min=0.0, max=1.0)
        else:
            inputs = image
            background = None

        # compute output
        out_images, embeddings = network(inputs)

        im_render = None
        if cfg.TEST.BUILD_CODEBOOK:
            num = embeddings.shape[0]
            codes[count:count+num] = embeddings.cpu().detach().numpy()
            poses[count:count+num] = sample['pose_target']
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
            _vis_minibatch_autoencoder(inputs, sample, out_images, im_render)

        # measure elapsed time
        batch_time.update(time.time() - end)
        if cfg.TEST.BUILD_CODEBOOK:
            print('[%d/%d], code index %d, batch time %.2f' % (i, epoch_size, count, batch_time.val))
        else:
            print('[%d/%d], batch time %.2f' % (i, epoch_size, batch_time.val))

    # save codebook
    if cfg.TEST.BUILD_CODEBOOK:
        codebook = {'codes': codes, 'quaternions': poses[:, 3:], 'distance': poses[0, 2], 'intrinsic_matrix': test_loader.dataset._intrinsic_matrix}
        filename = os.path.join(output_dir, 'codebook_%s.mat' % test_loader.dataset.name)
        print('save codebook to %s' % (filename))
        scipy.io.savemat(filename, codebook, do_compression=True)


def test_pose_rbpf(pose_rbpf, inputs, rois, poses):

    n_init_samples = 300
    num = rois.shape[0]
    uv_init = np.zeros((2, ), dtype=np.float32)
    pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).cuda().float()
    poses_return = poses.copy()

    intrinsic_matrix = pose_rbpf.dataset._intrinsic_matrix
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]

    for i in range(num):
        ind = int(rois[i, 0])
        image = inputs[ind].permute(1, 2, 0) + pixel_mean
        cls = int(rois[i, 1])

        # project the 3D translation to get the center
        uv_init[0] = fx * poses[i, 4] / poses[i, 6] + px
        uv_init[1] = fy * poses[i, 5] / poses[i, 6] + py

        roi_w = rois[i, 4] - rois[i, 2]
        roi_h = rois[i, 5] - rois[i, 3]
        roi_s = max(roi_w, roi_h)

        pose = pose_rbpf.initialize(image, uv_init, n_init_samples, cls, roi_s)
        if pose[-1] > 0:
            poses_return[i, :] = pose
    return poses_return


def test(test_loader, network, pose_rbpf, output_dir):

    batch_time = AverageMeter()
    epoch_size = len(test_loader)

    # switch to test mode
    network.eval()

    for i, sample in enumerate(test_loader):

        end = time.time()

        if cfg.INPUT == 'DEPTH':
            inputs = sample['image_depth']
        elif cfg.INPUT == 'COLOR':
            inputs = sample['image_color']
        elif cfg.INPUT == 'RGBD':
            inputs = torch.cat((sample['image_color'], sample['image_depth']), dim=1)

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

                # optimize depths
                poses, poses_refined = optimize_depths(rois, poses, test_loader.dataset._points_all, test_loader.dataset._intrinsic_matrix)
            else:
                out_label, out_vertex, rois, out_pose = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
                rois = rois.detach().cpu().numpy()
                out_pose = out_pose.detach().cpu().numpy()
                poses = out_pose.copy()
                poses_refined = []

                # non-maximum suppression within class
                index = nms(rois, 0.5)
                rois = rois[index, :]
                poses = poses[index, :]

                num = rois.shape[0]
                for j in range(num):
                    poses[j, 4] *= poses[j, 6] 
                    poses[j, 5] *= poses[j, 6]

                # run poseRBPF for codebook matching
                poses = test_pose_rbpf(pose_rbpf, inputs, rois, poses)
        else:
            out_label = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
            out_vertex = []
            rois = []
            poses = []
            poses_refined = []

        if cfg.TEST.VISUALIZE:
            _vis_test(inputs, labels, out_label, out_vertex, rois, poses, sample, \
                test_loader.dataset._points_all, test_loader.dataset.classes, test_loader.dataset.class_colors)

        # measure elapsed time
        batch_time.update(time.time() - end)

        result = {'labels': out_label, 'rois': rois, 'poses': poses, 'poses_refined': poses_refined}
        if 'video_id' in sample and 'image_id' in sample:
            filename = os.path.join(output_dir, sample['video_id'][0] + '_' + sample['image_id'][0] + '.mat')
        else:
            filename = os.path.join(output_dir, '%06d.mat' % i)
        print filename
        scipy.io.savemat(filename, result, do_compression=True)

        print('[%d/%d], batch time %.2f' % (i, epoch_size, batch_time.val))

    filename = os.path.join(output_dir, 'results_posecnn.mat')
    if os.path.exists(filename):
        os.remove(filename)


def test_image(network, dataset, im_color, im_depth=None):
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
        inputs = torch.cat((im_1, im_2), dim=1)

    labels = torch.from_numpy(label_blob).cuda()
    meta_data = torch.from_numpy(meta_data_blob).cuda()
    extents = torch.from_numpy(dataset._extents).cuda()
    gt_boxes = torch.from_numpy(gt_boxes).cuda()
    poses = torch.from_numpy(pose_blob).cuda()
    points = torch.from_numpy(dataset._point_blob).cuda()
    symmetry = torch.from_numpy(dataset._symmetry).cuda()

    if cfg.TRAIN.VERTEX_REG:

        if cfg.TRAIN.POSE_REG:
            out_label, out_vertex, rois, out_pose, out_quaternion = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
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
                poses = refine_pose(labels, im_depth, rois, poses, dataset._intrinsic_matrix)
            else:
                poses_tmp, poses = optimize_depths(rois, poses, dataset._points_all, dataset._intrinsic_matrix)

        else:

            out_label, out_vertex, rois, out_pose = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
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

    else:
        out_label = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
        labels = out_label.detach().cpu().numpy()[0]
        rois = np.zeros((0, 7), dtype=np.float32)
        poses = np.zeros((0, 7), dtype=np.float32)

    im_pose, im_label = render_image(dataset, im_color, rois, poses, labels)

    if cfg.TEST.VISUALIZE:
        vis_test(dataset, im, labels, out_vertex, rois, poses, im_pose)

    return im_pose, im_label, rois, poses


def optimize_depths(rois, poses, points, intrinsic_matrix):

    num = rois.shape[0]
    poses_refined = poses.copy()
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
        poses_refined[i, 4] *= res.x 
        poses_refined[i, 5] *= res.x
        poses_refined[i, 6] = res.x

        poses[i, 4] *= poses[i, 6] 
        poses[i, 5] *= poses[i, 6]

    return poses, poses_refined


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


def refine_pose(im_label, im_depth, rois, poses, intrinsic_matrix):

    # backprojection
    dpoints = backproject(im_depth, intrinsic_matrix)

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
    cls_indexes = rois[:, 1].astype(np.int32) - 1

    # poses
    poses_all = []
    for i in range(num):
        qt = np.zeros((7, ), dtype=np.float32)
        qt[3:] = poses[i, :4]
        qt[0] = poses[i, 4] * poses[i, 6]
        qt[1] = poses[i, 5] * poses[i, 6]
        qt[2] = poses[i, 6]
        poses_all.append(qt)
    cfg.renderer.set_poses(poses_all)
            
    # rendering
    cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    pcloud_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    cfg.renderer.render(cls_indexes, image_tensor, seg_tensor, pc2_tensor=pcloud_tensor)
    pcloud_tensor = pcloud_tensor.flip(0)
    pcloud = pcloud_tensor[:,:,:3].cpu().numpy().reshape((-1, 3))

    # refine pose
    for i in range(num):
        cls = int(rois[i, 1])
        x1 = max(int(rois[i, 2]), 0)
        y1 = max(int(rois[i, 3]), 0)
        x2 = min(int(rois[i, 4]), width-1)
        y2 = min(int(rois[i, 5]), height-1)
        labels = np.zeros((height, width), dtype=np.float32)
        labels[y1:y2, x1:x2] = im_label[y1:y2, x1:x2]
        labels = labels.reshape((width * height, ))
        index = np.where((labels == cls) & np.isfinite(dpoints[:, 0]) & (pcloud[:, 0] != 0))[0]
        if len(index) > 10:
            T = np.mean(dpoints[index, :] - pcloud[index, :], axis=0)
            poses[i, 6] += T[2]
            poses[i, 4] *= poses[i, 6]
            poses[i, 5] *= poses[i, 6]
        else:
            poses[i, 4] *= poses[i, 6]
            poses[i, 5] *= poses[i, 6]
            print 'no pose refinement'

    return poses


def render_image(dataset, im, rois, poses, labels):

    intrinsic_matrix = dataset._intrinsic_matrix
    height = im.shape[0]
    width = im.shape[1]
    classes = dataset._classes
    class_colors = dataset._class_colors

    label_image = dataset.labels_to_image(labels)
    im_label = im[:, :, (2, 1, 0)].copy()
    I = np.where(labels != 0)
    im_label[I[0], I[1], :] = 0.5 * label_image[I[0], I[1], :] + 0.5 * im_label[I[0], I[1], :]

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 6.0
    znear = 0.01
    num = poses.shape[0]

    image_tensor = torch.cuda.FloatTensor(height, width, 4)
    seg_tensor = torch.cuda.FloatTensor(height, width, 4)

    # set renderer
    cfg.renderer.set_light_pos([0, 0, 0])
    cfg.renderer.set_light_color([1, 1, 1])
    cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

    # render images
    cls_indexes = []
    poses_all = []
    for i in range(num):
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
        poses_all.append(qt)

        cls = int(rois[i, 1])
        print classes[cls], rois[i, -1]
        if cls > 0 and rois[i, -1] > cfg.TEST.DET_THRESHOLD:

            # draw roi
            x1 = rois[i, 2]
            y1 = rois[i, 3]
            x2 = rois[i, 4]
            y2 = rois[i, 5]
            cv2.rectangle(im_label, (x1, y1), (x2, y2), class_colors[cls], 2)

    # rendering
    if len(cls_indexes) > 0 and cfg.TRAIN.POSE_REG:
        cfg.renderer.set_poses(poses_all)
        frame = cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)

        # RGB to BGR order
        im_render = image_tensor.cpu().numpy()
        im_render = np.clip(im_render, 0, 1)
        im_render = im_render[:, :, :3] * 255
        im_render = im_render.astype(np.uint8)
    
        # mask
        seg = torch.sum(seg_tensor[:, :, :3], dim=2)
        mask = (seg != 0).cpu().numpy()

        im_output = 0.4 * im[:,:,(2, 1, 0)].astype(np.float32) + 0.6 * im_render.astype(np.float32)
    else:
        im_output = 0.4 * im[:,:,(2, 1, 0)]

    return im_output.astype(np.uint8), im_label


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


def _vis_minibatch(inputs, labels, vertex_targets, sample, class_colors):

    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    im_blob = inputs.cpu().numpy()
    label_blob = labels.cpu().numpy()
    gt_poses = sample['poses'].numpy()
    meta_data_blob = sample['meta_data'].numpy()
    gt_boxes = sample['gt_boxes'].numpy()
    extents = sample['extents'][0, :, :].numpy()

    if cfg.TRAIN.VERTEX_REG:
        vertex_target_blob = vertex_targets.cpu().numpy()

    if cfg.INPUT == 'COLOR':
        m = 2
        n = 3
    else:
        m = 3
        n = 3
    
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

        if cfg.TRAIN.VERTEX_REG:

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
                if rois[j, 0] != i or rois[j, -1] < cfg.TEST.DET_THRESHOLD:
                    continue
                cls = int(rois[j, 1])
                print classes[cls], rois[j, -1]
                if cls > 0:
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



def vis_test(dataset, im, label, out_vertex, rois, poses, im_pose):

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
    ax = fig.add_subplot(2, 4, 1)
    plt.imshow(im)
    ax.set_title('input image') 

    # show predicted label
    im_label = dataset.labels_to_image(label)
    ax = fig.add_subplot(2, 4, 2)
    plt.imshow(im_label)
    ax.set_title('predicted labels')

    ax = fig.add_subplot(2, 4, 8)
    plt.imshow(im_pose)
    ax.set_title('rendered image')

    if cfg.TRAIN.VERTEX_REG:

        # show predicted boxes
        ax = fig.add_subplot(2, 4, 3)
        plt.imshow(im)
        ax.set_title('predicted boxes')
        for j in range(rois.shape[0]):
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

        # show predicted poses
        if cfg.TRAIN.POSE_REG:
            ax = fig.add_subplot(2, 4, 4)
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

        # show predicted vertex targets
        vertex_target = vertex_pred[0, :, :, :]
        center = np.zeros((3, height, width), dtype=np.float32)

        for j in range(1, num_classes):
            index = np.where(label == j)
            if len(index[0]) > 0:
                center[0, index[0], index[1]] = vertex_target[3*j, index[0], index[1]]
                center[1, index[0], index[1]] = vertex_target[3*j+1, index[0], index[1]]
                center[2, index[0], index[1]] = np.exp(vertex_target[3*j+2, index[0], index[1]])

        ax = fig.add_subplot(2, 4, 5)
        plt.imshow(center[0,:,:])
        ax.set_title('predicted center x') 

        ax = fig.add_subplot(2, 4, 6)
        plt.imshow(center[1,:,:])
        ax.set_title('predicted center y')

        ax = fig.add_subplot(2, 4, 7)
        plt.imshow(center[2,:,:])
        ax.set_title('predicted z')

    plt.show()
