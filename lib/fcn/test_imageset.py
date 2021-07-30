# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import time
import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from fcn.config import cfg
from fcn.particle_filter import particle_filter
from fcn.test_common import test_pose_rbpf, refine_pose, eval_poses
from fcn.render_utils import render_image, render_image_detection
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.se3 import *
from utils.nms import *
from utils.pose_error import re, te
from utils.loose_bounding_boxes import compute_centroids_and_loose_bounding_boxes, mean_shift_and_loose_bounding_boxes


def test_image(network, pose_rbpf, dataset, im_color, im_depth=None, im_index=None):
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

    if im_depth is not None:
        depth_tensor = torch.from_numpy(im_depth).cuda().float()
    else:
        depth_tensor = None

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

            labels = out_label[0]

            rois = rois.detach().cpu().numpy()
            out_pose = out_pose.detach().cpu().numpy()
            poses = out_pose.copy()

            # filter out detections
            index = np.where(rois[:, -1] > cfg.TEST.DET_THRESHOLD)[0]
            rois = rois[index, :]
            poses = poses[index, :]
            poses_refined = None
            pose_scores = []

            # non-maximum suppression within class
            index = nms(rois, 0.2)
            rois = rois[index, :]
            poses = poses[index, :]

            # run poseRBPF for codebook matching to compute the rotations
            rois, poses, im_rgb, index_sdf = test_pose_rbpf(pose_rbpf, inputs, rois, poses, meta_data, dataset, depth_tensor, labels)

            # optimize depths
            cls_render_ids = None
            if cfg.TEST.POSE_REFINE and im_depth is not None:
                poses_refined, cls_render_ids = refine_pose(pose_rbpf, index_sdf, labels, depth_tensor, rois, poses, meta_data, dataset)
                if pose_rbpf is not None:
                    sims, depth_errors, vis_ratios, pose_scores = eval_poses(pose_rbpf, poses_refined, rois, im_rgb, depth_tensor, labels, meta_data)

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

    im_pose, im_pose_refined, im_label = render_image(dataset, im_color, rois, poses, poses_refined, labels.cpu().numpy(), cls_render_ids)
    if cfg.TEST.VISUALIZE or im_index is not None:
        vis_test(dataset, im, im_depth, labels.cpu().numpy(), rois, poses, poses_refined, im_pose, im_pose_refined, out_vertex, im_index)

    return im_pose, im_pose_refined, labels.cpu().numpy(), rois, poses, poses_refined


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

    im_label = render_image_detection(dataset, im_color, rois, labels)

    if cfg.TEST.VISUALIZE:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(im_label)
        plt.subplot(1, 2, 2)
        plt.imshow(labels)
        plt.show()

    return rois, labels, poses, im_label


def test_image_with_mask(pose_rbpf, dataset, im_color, mask, im_depth=None):
    """test on a single image"""

    # compute image blob
    im = im_color.astype(np.float32, copy=True)
    height = im.shape[0]
    width = im.shape[1]
    im /= 255.0

    K = dataset._intrinsic_matrix
    K[2, 2] = 1
    Kinv = np.linalg.pinv(K)
    meta_data = np.zeros((1, 18), dtype=np.float32)
    meta_data[0, 0:9] = K.flatten()
    meta_data[0, 9:18] = Kinv.flatten()
    meta_data = torch.from_numpy(meta_data).cuda()

    # prepare labels
    mask[mask == 255] = 1
    labels = torch.from_numpy(mask).cuda()

    # transfer to GPU
    image_tensor = torch.from_numpy(im).cuda()
    depth_tensor = torch.from_numpy(im_depth).cuda()
    mask_tensor = torch.from_numpy(mask).cuda()

    # extract roi
    I = np.where(mask == 1)
    x1 = np.min(I[1])
    y1 = np.min(I[0])
    x2 = np.max(I[1])
    y2 = np.max(I[0])
    rois = np.zeros((1, 7), dtype=np.float32)
    rois[0, 1] = 1
    rois[0, 2] = x1
    rois[0, 3] = y1
    rois[0, 4] = x2
    rois[0, 5] = y2
    rois[0, 6] = 1.0

    # run poseRBPF
    n_init_samples = cfg.PF.N_PROCESS
    uv_init = np.zeros((2, ), dtype=np.float32)

    if len(pose_rbpf.rbpfs) == 0:
        pose_rbpf.rbpfs.append(particle_filter(cfg.PF, n_particles=cfg.PF.N_PROCESS))

    cls = int(rois[0, 1])
    intrinsic_matrix = dataset._intrinsic_matrix
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    uv_init[0] = (x1 + x2) / 2.0
    uv_init[1] = (y1 + y2) / 2.0

    print(intrinsic_matrix)
    poses = pose_rbpf.initialize(0, image_tensor, uv_init, n_init_samples, cfg.TEST.CLASSES[cls], rois[0], intrinsic_matrix, depth_tensor, mask_tensor)
    poses = np.expand_dims(poses, axis=0)
    poses[0, 4] /= poses[0, 6]
    poses[0, 5] /= poses[0, 6]

    # optimize depths
    cls_render_ids = None
    if cfg.TEST.POSE_REFINE and im_depth is not None:
        poses_refined, cls_render_ids = refine_pose(labels, im_depth, rois, poses, meta_data, dataset)

    im_pose, im_pose_refine, im_label = render_image(dataset, im_color, rois, poses, poses_refined, labels.cpu().numpy(), cls_render_ids)
    if cfg.TEST.VISUALIZE:
        im = im_color.astype(np.float32, copy=True)
        im -= cfg.PIXEL_MEANS
        im = np.transpose(im / 255.0, (2, 0, 1))
        im = im[np.newaxis, :, :, :]
        vis_test(dataset, im, im_depth, labels.cpu().numpy(), rois, poses, poses_refined, im_pose, im_pose_refine)

    return im_pose, im_pose_refine, im_label, rois, poses, poses_refined


def vis_test(dataset, im, im_depth, label, rois, poses, poses_refined, im_pose, im_pose_refine, out_vertex=None, im_index=None):

    """Visualize a testing results."""
    import matplotlib.pyplot as plt

    num_classes = len(dataset._class_colors_test)
    classes = dataset._classes_test
    class_colors = dataset._class_colors_test
    points = dataset._points_all_test
    intrinsic_matrix = dataset._intrinsic_matrix
    height = label.shape[0]
    width = label.shape[1]

    if out_vertex is not None:
        vertex_pred = out_vertex.detach().cpu().numpy()

    fig = plt.figure()
    plot = 1
    m = 2
    n = 3
    # show image
    im = im[0, :, :, :].copy()
    im = im.transpose((1, 2, 0)) * 255.0
    im += cfg.PIXEL_MEANS
    im = im[:, :, (2, 1, 0)]
    im = im.astype(np.uint8)
    ax = fig.add_subplot(m, n, plot)
    plot += 1
    plt.imshow(im)
    ax.set_title('input image') 

    # show predicted label
    im_label = dataset.labels_to_image(label)
    ax = fig.add_subplot(m, n, plot)
    plot += 1
    plt.imshow(im_label)
    ax.set_title('predicted labels')

    ax = fig.add_subplot(m, n, plot)
    plot += 1
    plt.imshow(im_pose)
    ax.set_title('rendered image')

    if cfg.TEST.POSE_REFINE and im_pose_refine is not None and im_depth is not None:
        ax = fig.add_subplot(m, n, plot)
        plot += 1
        plt.imshow(im_pose_refine)
        ax.set_title('rendered image refine')

    if cfg.TRAIN.VERTEX_REG or cfg.TRAIN.VERTEX_REG_DELTA:

        # show predicted boxes
        ax = fig.add_subplot(m, n, plot)
        plot += 1
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
            ax = fig.add_subplot(m, n, plot)
            plot += 1
            ax.set_title('predicted poses')
            plt.imshow(im)
            for j in xrange(rois.shape[0]):
                cls = int(rois[j, 1])
                print(classes[cls], rois[j, -1])
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
            ax = fig.add_subplot(m, n, plot)
            plot += 1
            plt.imshow(im)
            ax.set_title('input depth') 

        '''
        if out_vertex is not None:
            # show predicted vertex targets
            vertex_target = vertex_pred[0, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, dataset._num_classes):
                index = np.where(label == j)
                if len(index[0]) > 0:
                    center[0, index[0], index[1]] = vertex_target[3*j, index[0], index[1]]
                    center[1, index[0], index[1]] = vertex_target[3*j+1, index[0], index[1]]
                    center[2, index[0], index[1]] = np.exp(vertex_target[3*j+2, index[0], index[1]])

            ax = fig.add_subplot(m, n, plot)
            plot += 1
            plt.imshow(center[0,:,:])
            ax.set_title('predicted center x') 

            ax = fig.add_subplot(m, n, plot)
            plot += 1
            plt.imshow(center[1,:,:])
            ax.set_title('predicted center y')

            ax = fig.add_subplot(m, n, plot)
            plot += 1
            plt.imshow(center[2,:,:])
            ax.set_title('predicted z')
        '''

    if im_index is not None:
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show(block=False)
        plt.pause(1)
        filename = 'output/images/%06d.png' % im_index
        fig.savefig(filename)
        plt.close()
    else:
        plt.show()
