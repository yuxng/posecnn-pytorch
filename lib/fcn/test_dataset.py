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
import cv2
import scipy
import matplotlib.pyplot as plt

from fcn.config import cfg
from fcn.particle_filter import particle_filter
from fcn.test_common import test_pose_rbpf, add_pose_rbpf, refine_pose, eval_poses
from fcn.test_common import _vis_minibatch_autoencoder, _vis_minibatch_cosegmentation, _vis_minibatch_segmentation
from fcn.test_common import _vis_minibatch_correspondences, _vis_minibatch_prototype, _vis_minibatch_triplet
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.se3 import *
from utils.nms import *
from utils.pose_error import re, te
from utils.loose_bounding_boxes import compute_centroids_and_loose_bounding_boxes, mean_shift_and_loose_bounding_boxes
from utils.mean_shift import mean_shift_smart_init
from utils.correspondences import find_correspondences, compute_prototype_distances

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
                        poses[j, 4] *= poses[j, 6] 
                        poses[j, 5] *= poses[j, 6]

                poses_refined = []
                pose_scores = None

                # refine pose
                if cfg.TEST.POSE_REFINE:
                    im_depth = sample['im_depth'].numpy()[0]
                    depth_tensor = torch.from_numpy(im_depth).cuda().float()
                    labels_out = out_label[0]
                    index_sdf = add_pose_rbpf(pose_rbpf, rois, poses)
                    poses_refined, cls_render_ids = refine_pose(pose_rbpf, index_sdf,
                        labels_out, depth_tensor, rois, poses, sample['meta_data'], test_loader.dataset)
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
                depth_tensor = torch.from_numpy(im_depth).cuda().float()
                labels_out = out_label[0]
                if pose_rbpf is not None:
                    rois, poses, im_rgb, index_sdf = test_pose_rbpf(pose_rbpf, inputs, rois, poses, 
                        sample['meta_data'], test_loader.dataset, depth_tensor, labels_out)

                # optimize depths
                if cfg.TEST.POSE_REFINE and im_depth is not None:
                    poses_refined, cls_render_ids = refine_pose(pose_rbpf, index_sdf, 
                        labels_out, depth_tensor, rois, poses, meta_data, test_loader.dataset)
                    if pose_rbpf is not None:
                        sims, depth_errors, vis_ratios, pose_scores = eval_poses(pose_rbpf, poses_refined, 
                            rois, im_rgb, depth_tensor, labels_out, meta_data)

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
            result = {'labels': labels_out.detach().cpu().numpy(), 'rois': rois, 'poses': poses, 'poses_refined': poses_refined}
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


def test_docsnet(test_loader, background_loader, network, output_dir, contrastive=False, prototype=False):

    batch_time = AverageMeter()
    epoch_size = len(test_loader)
    enum_background = enumerate(background_loader)

    # switch to test mode
    network.eval()

    for i, sample in enumerate(test_loader):

        end = time.time()

        # construct input
        image = sample['image_color']
        mask = sample['mask']
        label = sample['label']

        # separate two images
        h = cfg.TRAIN.SYN_CROP_SIZE
        w = cfg.TRAIN.SYN_CROP_SIZE
        image_a = image[:, 0, :, :h, :w].contiguous().cuda()
        image_b = image[:, 1].contiguous().cuda()
        mask_a = mask[:, 0, :, :h, :w].contiguous().cuda()
        mask_b = mask[:, 1].contiguous().cuda()
        label_a = label[:, 0, :, :h, :w].contiguous().cuda()
        label_b = label[:, 1].contiguous().cuda()

        if contrastive:
            matches_a = sample['matches_a'].contiguous().cuda()
            matches_b = sample['matches_b'].contiguous().cuda()
            masked_non_matches_a = sample['masked_non_matches_a'].contiguous().cuda()
            masked_non_matches_b = sample['masked_non_matches_b'].contiguous().cuda()
            others_non_matches_a = sample['others_non_matches_a'].contiguous().cuda()
            others_non_matches_b = sample['others_non_matches_b'].contiguous().cuda()
            background_non_matches_a = sample['background_non_matches_a'].contiguous().cuda()
            background_non_matches_b = sample['background_non_matches_b'].contiguous().cuda()
        else:
            matches_a = None
            matches_b = None
            masked_non_matches_a = None
            masked_non_matches_b = None
            others_non_matches_a = None
            others_non_matches_b = None
            background_non_matches_a = None
            background_non_matches_b = None

        # add background
        for j in range(2):
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

            if j == 0:
                input_a = mask_a * image_a + (1 - mask_a) * background_color[:num, :, :h, :w]
            else:
                input_b = mask_b * image_b + (1 - mask_b) * background_color[:num]

        # run network
        out_label_a, out_label_b = network(input_a, input_b, label_a, label_b, \
            matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, others_non_matches_a, others_non_matches_b, \
            background_non_matches_a, background_non_matches_b)

        if contrastive:
            uv_a, uv_b = find_correspondences(out_label_a, out_label_b, label_a, label_b, num_samples=5)

        if prototype:
            distances = compute_prototype_distances(out_label_a, out_label_b, label_a)

        if cfg.TEST.VISUALIZE:
            if contrastive:
                _vis_minibatch_correspondences(input_a, input_b, label_a, label_b, out_label_a, out_label_b, background_color, matches_a, matches_b,
                    masked_non_matches_a, masked_non_matches_b, others_non_matches_a, others_non_matches_b,
                    background_non_matches_a, background_non_matches_b, uv_a, uv_b)
            elif prototype:
                _vis_minibatch_prototype(input_a, input_b, label_a, label_b, out_label_a, out_label_b, background_color, distances)
            else:
                _vis_minibatch_cosegmentation(input_a, input_b, label_a, label_b, background_color, out_label_a, out_label_b)

        # measure elapsed time
        batch_time.update(time.time() - end)
        print('[%d/%d], batch time %.2f' % (i, epoch_size, batch_time.val))


def test_segnet(test_loader, background_loader, network, output_dir, rrn=False):

    batch_time = AverageMeter()
    epoch_size = len(test_loader)
    enum_background = enumerate(background_loader)

    # switch to test mode
    network.eval()

    for i, sample in enumerate(test_loader):

        end = time.time()

        # construct input
        image = sample['image_color'].contiguous().cuda()
        mask = sample['mask'].contiguous().cuda()
        label = sample['label'].contiguous().cuda()
        if rrn:
            initial_mask = sample['initial_mask'].contiguous().cuda()

        # add background
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

        # run network
        if not rrn and network.module.embedding:
            features = network(inputs, label)
            height = features.shape[2]
            width = features.shape[3]
            out_label = torch.zeros((features.shape[0], height, width))

            # mean shift clustering
            num_seeds = 10
            kappa = 20
            for i in range(features.shape[0]):
                X = features[i].view(features.shape[1], -1)
                X = torch.transpose(X, 0, 1)
                cluster_labels, selected_indices = mean_shift_smart_init(X, kappa=kappa, num_seeds=num_seeds, max_iters=10, metric='cosine')
                out_label[i] = cluster_labels.view(height, width)
        else:
            if rrn:
                inputs = torch.cat([inputs, initial_mask], dim=1) # Shape: [N x 4 x H x W]

            out_label = network(inputs, label)
            features = None

        if cfg.TEST.VISUALIZE:
            _vis_minibatch_segmentation(inputs, label, background_color, out_label, features, i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        print('[%d/%d], batch time %.2f' % (i, epoch_size, batch_time.val))


def test_triplet_net(test_loader, background_loader, network, output_dir):

    batch_time = AverageMeter()
    epoch_size = len(test_loader)
    enum_background = enumerate(background_loader)

    # switch to test mode
    network.eval()

    for i, sample in enumerate(test_loader):

        end = time.time()

        # construct input
        image_anchor = sample['image_anchor'].cuda()
        image_positive = sample['image_positive'].cuda()
        image_negative = sample['image_negative'].cuda()

        features_anchor, features_positive, features_negative \
            = network(image_anchor, image_positive, image_negative)

        if cfg.TRAIN.EMBEDDING_METRIC == 'euclidean':
            norm_degree = 2
            dp = (features_anchor - features_positive).norm(norm_degree, 1)
            dn = (features_anchor - features_negative).norm(norm_degree, 1)
        elif cfg.TRAIN.EMBEDDING_METRIC == 'cosine':
            dp = 0.5 * (1 - torch.sum(features_anchor * features_positive, dim=1))
            dn = 0.5 * (1 - torch.sum(features_anchor * features_negative, dim=1))

        if cfg.TEST.VISUALIZE:
            print('positive distance %.4f, negative distance %.4f' % (dp, dn))
            _vis_minibatch_triplet(image_anchor, image_positive, image_negative, dp, dn, features_anchor, features_positive, features_negative)

        # measure elapsed time
        batch_time.update(time.time() - end)
        print('[%d/%d], batch time %.2f' % (i, epoch_size, batch_time.val))


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
