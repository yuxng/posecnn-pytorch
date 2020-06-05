# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import torch
import time
import sys, os
import numpy as np
import posecnn_cuda
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from fcn.config import cfg
from fcn.particle_filter import particle_filter

# add poses to pose_rbpf
def add_pose_rbpf(pose_rbpf, rois, poses):
    num = rois.shape[0]
    rind = 0
    index_sdf = []
    for i in range(num):
        cls = int(rois[i, 1])
        if cls == 0:
            continue
        if cfg.TRAIN.CLASSES[cls] not in cfg.TEST.CLASSES:
            continue
        rind += 1

        roi = rois[i, :].copy()
        roi[1] = cfg.TEST.CLASSES.index(cfg.TRAIN.CLASSES[cls])

        if len(pose_rbpf.rbpfs) < rind:
            pose_rbpf.rbpfs.append(particle_filter(cfg.PF, n_particles=cfg.PF.N_PROCESS))
        pose = poses[i, :].copy()
        pose_rbpf.rbpfs[rind-1].roi = roi.copy()
        pose_rbpf.rbpfs[rind-1].pose = pose.copy()
        index_sdf.append(rind-1)
    return index_sdf


def test_pose_rbpf(pose_rbpf, inputs, rois, poses, meta_data, dataset, depth_tensor=None, label_tensor=None):

    n_init_samples = cfg.PF.N_PROCESS
    num = rois.shape[0]
    uv_init = np.zeros((2, ), dtype=np.float32)
    pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).cuda().float()
    rois_return = np.zeros((0, rois.shape[1]), dtype=np.float32)
    poses_return = np.zeros((0, poses.shape[1]), dtype=np.float32)
    image = None

    rind = 0
    index_sdf = []
    for i in range(num):
        ind = int(rois[i, 0])
        image = inputs[ind].permute(1, 2, 0) + pixel_mean

        cls = int(rois[i, 1])
        if cfg.TRAIN.CLASSES[cls] not in cfg.TEST.CLASSES:
            continue
        rind += 1

        intrinsic_matrix = meta_data[ind, :9].cpu().numpy().reshape((3, 3))
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]

        # project the 3D translation to get the center
        uv_init[0] = fx * poses[i, 4] + px
        uv_init[1] = fy * poses[i, 5] + py

        if label_tensor is not None:
            mask = torch.zeros_like(label_tensor)
            mask[label_tensor == cls] = 1.0

        roi = rois[i, :].copy()
        roi[1] = cfg.TEST.CLASSES.index(cfg.TRAIN.CLASSES[cls])
        rois_return = np.concatenate((rois_return,  np.expand_dims(roi, axis=0)), axis=0)

        # run poserbpf initialization
        if len(pose_rbpf.rbpfs) < rind:
            pose_rbpf.rbpfs.append(particle_filter(cfg.PF, n_particles=cfg.PF.N_PROCESS))
        pose = pose_rbpf.initialize(rind-1, image, uv_init, n_init_samples, cfg.TRAIN.CLASSES[cls], roi, intrinsic_matrix, depth_tensor, mask)
        pose_rbpf.rbpfs[rind-1].roi = roi.copy()
        pose_rbpf.rbpfs[rind-1].pose = pose.copy()
        index_sdf.append(rind-1)
        poses_return = np.concatenate((poses_return,  np.expand_dims(pose, axis=0)), axis=0)

    return rois_return, poses_return, image, index_sdf


def eval_poses(pose_rbpf, poses, rois, im_rgb, im_depth, im_label, meta_data):

    sims = np.zeros((rois.shape[0],), dtype=np.float32)
    depth_errors = np.ones((rois.shape[0],), dtype=np.float32)
    vis_ratios = np.zeros((rois.shape[0],), dtype=np.float32)
    pose_scores = np.zeros((rois.shape[0],), dtype=np.float32)

    intrinsic_matrix = meta_data[0, :9].cpu().numpy().reshape((3, 3))
    image_tensor, pcloud_tensor = pose_rbpf.render_poses_all(poses, rois, intrinsic_matrix)
    num = rois.shape[0]
    for i in range(num):
        cls = int(rois[i, 1])
        # todo: fix the problem for large clamp
        if cls == -1:
            cls_id = 19
        else:
            cls_id = cfg.TRAIN.CLASSES[cls]

        if cls_id not in cfg.TEST.CLASSES:
            continue

        sims[i], depth_errors[i], vis_ratios[i] = pose_rbpf.evaluate_6d_pose(rois[i], poses[i], cls_id, im_rgb, \
            image_tensor, pcloud_tensor, im_depth, intrinsic_matrix, im_label)
        pose_scores[i] = sims[i] / (depth_errors[i] / 0.002 / vis_ratios[i])

    return sims, depth_errors, vis_ratios, pose_scores


def refine_pose(pose_rbpf, index_sdf, im_label, im_depth, rois, poses, meta_data, dataset):

    # backproject depth
    num = rois.shape[0]
    intrinsic_matrix = meta_data[0, :9].cpu().numpy().reshape((3, 3))
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    im_pcloud = posecnn_cuda.backproject_forward(fx, fy, px, py, im_depth)[0]

    # SDF refinement
    start_time = time.time()
    pose_rbpf.refine_pose(index_sdf, im_depth, im_pcloud, intrinsic_matrix, dataset, im_label, steps=cfg.TEST.NUM_SDF_ITERATIONS_TRACKING)
    print('pose refine time %.6f' % (time.time() - start_time))

    # collect poses
    poses_refined = poses.copy()
    cls_render_ids = []
    for i in range(num):
        cls = int(rois[i, 1])
        cls_render = cls - 1
        cls_render_ids.append(cls_render)
        poses_refined[i, :4] = pose_rbpf.rbpfs[i].pose[:4]
        poses_refined[i, 4:] = pose_rbpf.rbpfs[i].pose[4:]

    return poses_refined, cls_render_ids


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


def _vis_minibatch_correspondences(input_a, input_b, label_a, label_b, features_a, features_b, background, matches_a, matches_b,
        masked_non_matches_a, masked_non_matches_b, others_non_matches_a, others_non_matches_b,
        background_non_matches_a, background_non_matches_b, uv_a_predict=None, uv_b_predict=None):

    im_blob_a = input_a.cpu().numpy()
    im_blob_b = input_b.cpu().numpy()
    label_blob_a = label_a.cpu().numpy()
    label_blob_b = label_b.cpu().numpy()
    background_blob = background.cpu().numpy()
    matches_a = matches_a.cpu().numpy()
    matches_b = matches_b.cpu().numpy()
    masked_non_matches_a = masked_non_matches_a.cpu().numpy()
    masked_non_matches_b = masked_non_matches_b.cpu().numpy()
    others_non_matches_a = others_non_matches_a.cpu().numpy()
    others_non_matches_b = others_non_matches_b.cpu().numpy()
    background_non_matches_a = background_non_matches_a.cpu().numpy()
    background_non_matches_b = background_non_matches_b.cpu().numpy()
    height_a = im_blob_a.shape[2]
    width_a = im_blob_a.shape[3]
    height_b = im_blob_b.shape[2]
    width_b = im_blob_b.shape[3]
    m = 3
    n = 3

    for i in range(im_blob_a.shape[0]):
        fig = plt.figure()
        start = 1

        # show correspondences
        index = np.nonzero(matches_a[i, :])
        ma = matches_a[i, index].squeeze(0)
        index = np.nonzero(matches_b[i, :])
        mb = matches_b[i, index].squeeze(0)
        ind = np.random.randint(0, len(ma))
        uv_a = (ma[ind] % width_a, ma[ind] / width_a)
        uv_b = (mb[ind] % width_b, mb[ind] / width_b)

        index = np.nonzero(masked_non_matches_a[i, :])
        ma = masked_non_matches_a[i, index].squeeze(0)
        index = np.nonzero(masked_non_matches_b[i, :])
        mb = masked_non_matches_b[i, index].squeeze(0)
        ind = np.random.randint(0, len(ma))
        uv_a_masked_non = (ma[ind] % width_a, ma[ind] / width_a)
        uv_b_masked_non = (mb[ind] % width_b, mb[ind] / width_b)

        index = np.nonzero(others_non_matches_a[i, :])
        ma = others_non_matches_a[i, index].squeeze(0)
        index = np.nonzero(others_non_matches_b[i, :])
        mb = others_non_matches_b[i, index].squeeze(0)
        ind = np.random.randint(0, len(ma))
        uv_a_others_non = (ma[ind] % width_a, ma[ind] / width_a)
        uv_b_others_non = (mb[ind] % width_b, mb[ind] / width_b)

        index = np.nonzero(background_non_matches_a[i, :])
        ma = background_non_matches_a[i, index].squeeze(0)
        index = np.nonzero(background_non_matches_b[i, :])
        mb = background_non_matches_b[i, index].squeeze(0)
        ind = np.random.randint(0, len(ma))
        uv_a_background_non = (ma[ind] % width_a, ma[ind] / width_a)
        uv_b_background_non = (mb[ind] % width_b, mb[ind] / width_b)

        # show image a
        im = im_blob_a[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        im_a = im.copy()
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('image a')
        circ = Circle((uv_a[0], uv_a[1]), radius=5, facecolor='r', edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
        ax.add_patch(circ)
        circ = Circle((uv_a_masked_non[0], uv_a_masked_non[1]), radius=5, facecolor='g', edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
        ax.add_patch(circ)
        circ = Circle((uv_a_others_non[0], uv_a_others_non[1]), radius=5, facecolor='y', edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
        ax.add_patch(circ)
        circ = Circle((uv_a_background_non[0], uv_a_background_non[1]), radius=5, facecolor='b', 
                      edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
        ax.add_patch(circ)

        # show image b
        im = im_blob_b[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        im_b = im.copy()
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('image b')
        circ = Circle((uv_b[0], uv_b[1]), radius=10, facecolor='r', edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
        ax.add_patch(circ)
        circ = Circle((uv_b_masked_non[0], uv_b_masked_non[1]), radius=10, facecolor='g',
                      edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
        ax.add_patch(circ)
        circ = Circle((uv_b_others_non[0], uv_b_others_non[1]), radius=10, facecolor='y',
                      edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
        ax.add_patch(circ)
        circ = Circle((uv_b_background_non[0], uv_b_background_non[1]), radius=10, facecolor='b', 
                      edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
        ax.add_patch(circ)

        # show background
        im = background_blob[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im [:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('background')

        # show feature a
        im = torch.cuda.FloatTensor(features_a.shape[2], features_a.shape[3], 3)
        for j in range(3):
            im[:, :, j] = torch.sum(features_a[i, j::3, :, :], dim=0)
        im = normalize_descriptor(im.detach().cpu().numpy())
        im *= 255
        im = im.astype(np.uint8)
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('features a')

        # show feature b
        im = torch.cuda.FloatTensor(features_b.shape[2], features_b.shape[3], 3)
        for j in range(3):
            im[:, :, j] = torch.sum(features_b[i, j::3, :, :], dim=0)
        im = normalize_descriptor(im.detach().cpu().numpy())
        im *= 255
        im = im.astype(np.uint8)
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('features b')

        # show label a
        label = label_blob_a[i, 1, :, :]
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(label)
        ax.set_title('label a')

        # show label b
        label = label_blob_b[i, 1, :, :]
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(label)
        ax.set_title('label b')

        # show computed correspondences
        if uv_a_predict is not None and uv_b_predict is not None:
            # show image a
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im_a)
            ax.set_title('image a')
            for j in range(len(uv_a_predict[0])):
                x = uv_a_predict[0][j]
                y = uv_a_predict[1][j]
                circ = Circle((x, y), radius=5, facecolor='g', edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
                ax.add_patch(circ)
                plt.text(x, y, str(j+1), fontsize=12, color='r')

            # show image b
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im_b)
            ax.set_title('image b')
            for j in range(len(uv_b_predict[0])):
                x = uv_b_predict[0][j]
                y = uv_b_predict[1][j]
                circ = Circle((x, y), radius=5, facecolor='g', edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
                ax.add_patch(circ)
                plt.text(x, y, str(j+1), fontsize=12, color='r')

        plt.show()


def _vis_minibatch_prototype(input_a, input_b, label_a, label_b, features_a, features_b, background, distances=None, flag=None):

    im_blob_a = input_a.cpu().numpy()
    im_blob_b = input_b.cpu().numpy()
    label_blob_a = label_a.cpu().numpy()
    label_blob_b = label_b.cpu().numpy()
    background_blob = background.cpu().numpy()
    height = im_blob_a.shape[2]
    width = im_blob_a.shape[3]
    m = 2
    n = 3

    for i in range(im_blob_a.shape[0]):
        fig = plt.figure()
        start = 1

        # show image a
        im = im_blob_a[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        im_a = im.copy()
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('image a')

        # show image b
        im = im_blob_b[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        im_b = im.copy()
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('image b')

        # show background
        im = background_blob[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im [:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('background')

        # show feature a
        if len(features_a.shape) == 4:
            im = torch.cuda.FloatTensor(height, width, 3)
            for j in range(3):
                im[:, :, j] = torch.sum(features_a[i, j::3, :, :], dim=0)
            im = normalize_descriptor(im.detach().cpu().numpy())
            im *= 255
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)
            ax.set_title('features a')

            # show feature b
            im = torch.cuda.FloatTensor(height, width, 3)
            for j in range(3):
                im[:, :, j] = torch.sum(features_b[i, j::3, :, :], dim=0)
            im = normalize_descriptor(im.detach().cpu().numpy())
            im *= 255
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)
            ax.set_title('features b')

        # show label a
        label = label_blob_a[i, 0, :, :]
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(label)
        ax.set_title('label a')

        # show label b
        label = label_blob_b[i, 0, :, :]
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(label)
        ax.set_title('label b')

        if flag is not None:
            im = np.zeros(im_a.shape, dtype=np.uint8)
            if flag[i] == 1:
                im[:, :, 1] = 255
            else:
                im[:, :, 0] = 255
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)

        if distances is not None:
            label = distances[i, :, :].cpu().detach().numpy()
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(label)
            ax.set_title('distances')

            label = (distances[i, :, :] < cfg.TRAIN.EMBEDDING_DELTA / 2).float().cpu().numpy()
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(label)
            ax.set_title('distance thresholded')

        plt.show()


def _vis_minibatch_triplet(input_a, input_b, input_c, dp=None, dn=None, features_anchor=None, features_positive=None, features_negative=None):

    im_blob_a = input_a.cpu().numpy()
    im_blob_b = input_b.cpu().numpy()
    im_blob_c = input_c.cpu().numpy()
    height = im_blob_a.shape[2]
    width = im_blob_a.shape[3]
    if features_anchor is not None:
        m = 2
    else:
        m = 1
    n = 3

    for i in range(im_blob_a.shape[0]):
        fig = plt.figure()
        start = 1

        # show image a
        im = im_blob_a[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        im_a = im.copy()
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('image anchor')

        # show image b
        im = im_blob_b[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        im_b = im.copy()
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        if dp:
            tit = 'positive %.4f' % dp
        else:
            tit = 'positive'
        ax.set_title(tit)

        # show image c
        im = im_blob_c[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        im_b = im.copy()
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        if dn:
            tit = 'negative %.4f' % dn
        else:
            tit = 'negative'
        ax.set_title(tit)

        if features_anchor is not None:
            f = features_anchor[i].detach().cpu().numpy()
            length = int(np.sqrt(len(f)))
            f = f.reshape((length, length))
            im = normalize_descriptor(f)
            im *= 255
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)
            ax.set_title('feature anchor')

        if features_positive is not None:
            f = features_positive[i].detach().cpu().numpy()
            length = int(np.sqrt(len(f)))
            f = f.reshape((length, length))
            im = normalize_descriptor(f)
            im *= 255
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)
            ax.set_title('feature positive')

        if features_negative is not None:
            f = features_negative[i].detach().cpu().numpy()
            length = int(np.sqrt(len(f)))
            f = f.reshape((length, length))
            im = normalize_descriptor(f)
            im *= 255
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)
            ax.set_title('feature negative')

        plt.show()


def _vis_minibatch_mapping(input_a, input_b, dp=None):

    im_blob_a = input_a.cpu().numpy()
    im_blob_b = input_b.cpu().numpy()
    height = im_blob_a.shape[2]
    width = im_blob_a.shape[3]
    m = 1
    n = 2

    for i in range(im_blob_a.shape[0]):
        fig = plt.figure()
        start = 1

        # show image a
        im = im_blob_a[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        im_a = im.copy()
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('input')

        # show image b
        im = im_blob_b[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        im_b = im.copy()
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        if dp:
            tit = 'target %.4f' % dp
        else:
            tit = 'target'
        ax.set_title(tit)
        plt.show()


def _vis_minibatch_cosegmentation(input_a, input_b, label_a, label_b, background, out_label_a, out_label_b):

    im_blob_a = input_a.cpu().numpy()
    im_blob_b = input_b.cpu().numpy()
    label_blob_a = label_a.cpu().numpy()
    label_blob_b = label_b.cpu().numpy()
    background_blob = background.cpu().numpy()
    out_label_blob_a = out_label_a.cpu().numpy()
    out_label_blob_b = out_label_b.cpu().numpy()

    for i in range(im_blob_a.shape[0]):
        fig = plt.figure()
        # show image a
        im = im_blob_a[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        ax = fig.add_subplot(3, 3, 1)
        plt.imshow(im)
        ax.set_title('image a')

        # show image b
        im = im_blob_b[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        ax = fig.add_subplot(3, 3, 2)
        plt.imshow(im)
        ax.set_title('image b')
 
        # show background
        im = background_blob[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im [:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        ax = fig.add_subplot(3, 3, 3)
        plt.imshow(im)
        ax.set_title('background')

        # show label a
        label = label_blob_a[i, 1, :, :]
        ax = fig.add_subplot(3, 3, 4)
        plt.imshow(label)
        ax.set_title('label a')

        # show label b
        label = label_blob_b[i, 1, :, :]
        ax = fig.add_subplot(3, 3, 5)
        plt.imshow(label)
        ax.set_title('label b')

        # show out label a
        label = out_label_blob_a[i, :, :]
        ax = fig.add_subplot(3, 3, 7)
        plt.imshow(label)
        ax.set_title('out label a')

        # show out label b
        label = out_label_blob_b[i, :, :]
        ax = fig.add_subplot(3, 3, 8)
        plt.imshow(label)
        ax.set_title('out label b')

        plt.show()


def normalize_descriptor(res, stats=None):
    """
    Normalizes the descriptor into RGB color space
    :param res: numpy.array [H,W,D]
        Output of the network, per-pixel dense descriptor
    :param stats: dict, with fields ['min', 'max', 'mean'], which are used to normalize descriptor
    :return: numpy.array
        normalized descriptor
    """

    if stats is None:
        res_min = res.min()
        res_max = res.max()
    else:
        res_min = np.array(stats['min'])
        res_max = np.array(stats['max'])

    normed_res = np.clip(res, res_min, res_max)
    eps = 1e-10
    scale = (res_max - res_min) + eps
    normed_res = (normed_res - res_min) / scale
    return normed_res


def _vis_minibatch_segmentation(inputs, label, background, out_label=None, features=None, ind=None):

    im_blob = inputs.cpu().numpy()
    m = 2
    n = 3
    label_blob = label.cpu().numpy()
    background_blob = background.cpu().numpy()
    height = im_blob.shape[2]
    width = im_blob.shape[3]
    if out_label is not None:
        out_label_blob = out_label.cpu().numpy()

    for i in range(im_blob.shape[0]):
        fig = plt.figure()
        start = 1
        # show image
        im = im_blob[i, :3, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('image')

        # show background
        im = background_blob[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im [:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('background')

        # show label
        label = label_blob[i, 0, :, :]
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(label)
        ax.set_title('gt label')

        if im_blob.shape[1] == 4:
            label = im_blob[i, 3, :, :]
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(label)
            ax.set_title('initial label')

        # show out label
        if out_label is not None:
            label = out_label_blob[i, :, :]
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(label)
            ax.set_title('out label')

        if features is not None:
            im = torch.cuda.FloatTensor(height, width, 3)
            for j in range(3):
                im[:, :, j] = torch.sum(features[i, j::3, :, :], dim=0)
            im = normalize_descriptor(im.detach().cpu().numpy())
            im *= 255
            im = im.astype(np.uint8)
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im)
            ax.set_title('features')

        if ind is not None:
            mng = plt.get_current_fig_manager()
            plt.show()
            filename = 'output/images/%06d.png' % ind
            fig.savefig(filename)

        plt.show()
