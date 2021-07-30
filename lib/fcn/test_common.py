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
import cv2
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


def _save_images_autoencoder(output_dir, save_count, inputs, background, mask, sample, outputs, im_render=None):

    # input image dir
    outdir_input = os.path.join(output_dir, 'input')
    if not os.path.exists(outdir_input):
        os.makedirs(outdir_input)

    # recon image dir
    outdir_recon = os.path.join(output_dir, 'recon')
    if not os.path.exists(outdir_recon):
        os.makedirs(outdir_recon)

    # target image dir
    outdir_target = os.path.join(output_dir, 'target')
    if not os.path.exists(outdir_target):
        os.makedirs(outdir_target)

    # process data
    im_blob = inputs.cpu().numpy()
    background_blob = background.cpu().numpy()
    mask_blob = mask.cpu().numpy()
    targets = sample['image_target'].cpu().numpy()
    im_output = outputs.cpu().detach().numpy()

    for i in range(im_blob.shape[0]):

        # input image
        im = im_blob[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im.astype(np.uint8)
        filename = os.path.join(outdir_input, '%06d.jpg' % save_count)
        cv2.imwrite(filename, im)

        # target image
        im = targets[i, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im = im.astype(np.uint8)
        filename = os.path.join(outdir_target, '%06d.jpg' % save_count)
        cv2.imwrite(filename, im)

        # reconstruction
        im = im_output[i, :, :, :].copy()
        im = np.clip(im, 0, 1)
        im = im.transpose((1, 2, 0)) * 255.0
        im = im.astype(np.uint8)
        filename = os.path.join(outdir_recon, '%06d.jpg' % save_count)
        cv2.imwrite(filename, im)

        save_count += 1
    return save_count
