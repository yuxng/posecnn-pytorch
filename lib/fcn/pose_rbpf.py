import os, sys
import copy
import torch
import scipy.io
import random
import networks
import numpy as np
from layers.roi_align import ROIAlign
from fcn.config import cfg
from fcn.particle_filter import particle_filter
from utils.blob import add_noise_cuda
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.prbpf_utils import *

class PoseRBPF:

    def __init__(self, dataset, path_pretrained, path_codebook):

        # prepare autoencoder and codebook
        autoencoders = [[] for i in range(len(cfg.TEST.CLASSES))]
        codebooks = [[] for i in range(len(cfg.TEST.CLASSES))]
        codes_gpu = [[] for i in range(len(cfg.TEST.CLASSES))]
        poses_cpu = [[] for i in range(len(cfg.TEST.CLASSES))]
        codebook_names = [[] for i in range(len(cfg.TEST.CLASSES))]

        for i in range(len(cfg.TEST.CLASSES)):
            ind = cfg.TEST.CLASSES[i]
            cls = dataset._classes_all[ind]

            # load autoencoder
            filename = path_pretrained.replace('cls', cls)
            if os.path.exists(filename):
                autoencoder_data = torch.load(filename)
                autoencoders[i] = networks.__dict__['autoencoder'](1, 128, autoencoder_data).cuda(device=cfg.device)
                autoencoders[i] = torch.nn.DataParallel(autoencoders[i], device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
                print(filename)
            else:
                print('file not exists %s' % (filename))

            filename = path_codebook.replace('cls', cls)
            filename_mat = filename + '.mat'
            filename_pth = filename + '.pth'
            if os.path.exists(filename_mat) and os.path.exists(filename_pth):
                codebook_names[i] = filename_mat[:-4]
                codebooks[i] = scipy.io.loadmat(filename_mat)
                data = torch.load(filename_pth)
                codes_gpu[i] = data[0][0, :, :]
                poses_cpu[i] = data[1][0, :, :].cpu().numpy()
                print(filename_mat, filename_pth)
            else:
                print('file not exists %s or %s' % (filename_mat, filename_pth))

        self.autoencoders = autoencoders
        self.codebooks = codebooks
        self.codebook_names = codebook_names
        self.codes_gpu = codes_gpu
        self.poses_cpu = poses_cpu
        self.dataset = dataset
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        # motion model
        self.T_c1c0 = np.eye(4, dtype=np.float32)
        self.T_o0o1 = np.eye(4, dtype=np.float32)
        self.T_c0o = np.eye(4, dtype=np.float32)
        self.T_c1o = np.eye(4, dtype=np.float32)
        self.Tbr1 = np.eye(4, dtype=np.float32)
        self.Tbr0 = np.eye(4, dtype=np.float32)
        self.Trc = np.eye(4, dtype=np.float32)

        # initialize the particle filters
        self.rbpf = None
        self.rbpfs = []


    @property
    def num_rbpfs(self):
        return len(self.rbpfs)


    # pose estimation pipeline
    # roi: object detection from posecnn, shape (1, 6)
    # image_bgr: input bgr image, range (0, 1)
    def Pose_Estimation_PRBPF(self, roi, name, intrinsic_matrix, image_bgr, im_depth, dpoints, im_label=None):

        n_init_samples = cfg.PF.N_PROCESS
        uv_init = np.zeros((2, ), dtype=np.float32)
        image = torch.from_numpy(image_bgr)
        roi = roi.flatten()

        cls = int(roi[1])
        if cfg.TRAIN.CLASSES[cls] not in cfg.TEST.CLASSES:
            return np.zeros((7,), dtype=np.float32)

        # use bounding box center
        uv_init[0] = (roi[4] + roi[2]) / 2
        uv_init[1] = (roi[5] + roi[3]) / 2

        if im_label is not None:
            mask = np.zeros(im_label.shape, dtype=np.float32)
            mask[im_label == cls] = 1.0

        self.rbpfs.append(particle_filter(cfg.PF, n_particles=cfg.PF.N_PROCESS))
        self.rbpfs[-1].name = name
        pose = self.initialize(self.num_rbpfs-1, image, uv_init, n_init_samples, cfg.TRAIN.CLASSES[cls], roi, intrinsic_matrix, im_depth, mask)

        # SDF refine
        pose_refined, cls_render = self.refine_pose(im_label, im_depth, dpoints, roi, pose, intrinsic_matrix, self.dataset)
        self.rbpfs[-1].pose = pose_refined.flatten()
        im_render_refine, box_refine = self.render_image(self.dataset, intrinsic_matrix, cls_render, pose_refined.flatten())
        self.rbpfs[-1].roi[2:] = box_refine

        if cfg.TEST.VISUALIZE:
            im_render, box = self.render_image(self.dataset, intrinsic_matrix, cls_render, pose.flatten())    
            # show image
            import matplotlib.pyplot as plt
            fig = plt.figure()
            im = image_bgr
            im = im * 255.0
            im = im [:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
            ax = fig.add_subplot(2, 2, 1)
            plt.imshow(im)
            ax.set_title('input image')

            # show output
            ax = fig.add_subplot(2, 2, 2)
            plt.imshow(im_render)
            ax.set_title('rendering')
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

            ax = fig.add_subplot(2, 2, 3)
            plt.imshow(im_render_refine)
            ax.set_title('rendering refined')
            x1 = box_refine[0]
            y1 = box_refine[1]
            x2 = box_refine[2]
            y2 = box_refine[3]
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

            ax = fig.add_subplot(2, 2, 4)
            plt.imshow(mask)
            ax.set_title('mask')
            plt.show()

        # success initialization
        if pose[-1] > 0:
            return pose_refined.flatten()
        else:
            return np.zeros((7,), dtype=np.float32)


    # pose estimation pipeline
    # roi: object detection from posecnn, shape (1, 6)
    # image_bgr: input bgr image, range (0, 1)
    def Filtering_PRBPF(self, intrinsics, image_bgr, im_depth, dpoints, im_label=None):

        n_init_samples = cfg.PF.N_PROCESS
        image = torch.from_numpy(image_bgr)

        # for each particle filter
        for i in range(self.num_rbpfs):
            cls_id = self.rbpfs[i].cls_id
            autoencoder = self.autoencoders[cls_id]
            codebook = self.codebooks[cls_id]
            codes_gpu = self.codes_gpu[cls_id]
            poses_cpu = self.poses_cpu[cls_id]

            render_dist = codebook['distance']
            intrinsic_matrix = codebook['intrinsic_matrix']
            cfg.PF.FU = intrinsic_matrix[0, 0]
            cfg.PF.FV = intrinsic_matrix[1, 1]

            if im_label is not None:
                mask = np.zeros(im_label.shape, dtype=np.float32)
                cls = int(self.rbpfs[i].roi[1])
                mask[im_label == cls] = 1.0
            else:
                mask = None

            out_image, in_image = self.process_poserbpf(i, cls_id, autoencoder, codes_gpu, poses_cpu, image,
                                  intrinsics, render_dist, im_depth, mask, apply_motion_prior=False, init_mode=False)

            # box and poses
            box_center = self.rbpfs[i].uv_bar[:2]
            box_size = 128 * render_dist / self.rbpfs[i].z_bar * intrinsics[0, 0] / cfg.PF.FU
            pose = np.zeros((7,), dtype=np.float32)
            pose[4:] = self.rbpfs[i].trans_bar
            pose[:4] = mat2quat(self.rbpfs[i].rot_bar)

            # SDF refine
            pose_refined, cls_render = self.refine_pose(im_label, im_depth, dpoints, self.rbpfs[i].roi, pose, intrinsics, self.dataset)
            self.rbpfs[i].pose = pose_refined.flatten()
            #self.rbpfs[i].pose = pose

            if cfg.TEST.SYNTHESIZE:
                cls_render = cls - 1
            else:
                cls_render = cls_id
            im_render, box = self.render_image(self.dataset, intrinsics, cls_render, self.rbpfs[i].pose)
            self.rbpfs[i].roi[2:] = box

            if cfg.TEST.VISUALIZE:
                self.visualize(image, im_render, out_image, in_image, box_center, box_size, box)


    # initialize PoseRBPF
    '''
    image (height, width, 3) with values (0, 1)
    '''
    def initialize(self, rind, image, uv_init, n_init_samples, cls, roi, intrinsics, depth=None, mask=None):

        cls_id = cfg.TEST.CLASSES.index(cls)
        roi_w = roi[4] - roi[2]
        roi_h = roi[5] - roi[3]

        # network and codebook of the class
        autoencoder = self.autoencoders[cls_id]
        codebook = self.codebooks[cls_id]
        codes_gpu = self.codes_gpu[cls_id]
        poses_cpu = self.poses_cpu[cls_id]
        pose = np.zeros((7,), dtype=np.float32)
        if not autoencoder or not codebook:
            return pose

        render_dist = codebook['distance']
        intrinsic_matrix = codebook['intrinsic_matrix']
        cfg.PF.FU = intrinsic_matrix[0, 0]
        cfg.PF.FV = intrinsic_matrix[1, 1]

        # sample around the center of bounding box
        uv_h = np.array([uv_init[0], uv_init[1], 1])
        uv_h = np.repeat(np.expand_dims(uv_h, axis=0), n_init_samples, axis=0)

        bound = roi_w * 0.1
        uv_h[:, 0] += np.random.uniform(-bound, bound, (n_init_samples, ))

        bound = roi_h * 0.1
        uv_h[:, 1] += np.random.uniform(-bound, bound, (n_init_samples, ))

        uv_h[:, 0] = np.clip(uv_h[:, 0], 0, image.shape[1])
        uv_h[:, 1] = np.clip(uv_h[:, 1], 0, image.shape[0])

        # sample around z
        roi_size = max(roi_w, roi_h)
        z_init = (128 - 40) * render_dist / roi_size * intrinsics[0, 0] / cfg.PF.FU
        z_init = z_init[0, 0]

        # sampling with depth
        
        if depth is not None:
            uv_h_int = uv_h.astype(int)
            uv_h_int[:, 0] = np.clip(uv_h_int[:, 0], 0, image.shape[1] - 1)
            uv_h_int[:, 1] = np.clip(uv_h_int[:, 1], 0, image.shape[0] - 1)
            z = depth[uv_h_int[:, 1], uv_h_int[:, 0]]
            z = np.expand_dims(z, axis=1)
            extent = np.mean(self.dataset._extents[int(roi[1]), :]) / 2
            z[z > 0] += np.random.uniform(-extent, extent, z[z > 0].shape)
            z[z == 0 | ~np.isfinite(z)] = np.random.uniform(0.9 * z_init, 1.1 * z_init, z[z == 0 | ~np.isfinite(z)].shape)
        else:
            z = np.random.uniform(0.9 * z_init, 1.1 * z_init, (n_init_samples, 1))

        # evaluate
        distribution, max_sim_all, out_images, in_images = self.evaluate_particles(self.rbpfs[rind], cls_id, autoencoder, codes_gpu, \
            poses_cpu, image, intrinsics, \
            uv_h, z, render_dist, cfg.PF.WT_RESHAPE_VAR, depth, mask, init_mode=True)

        # find the max pdf from the distribution matrix
        index_star = self.arg_max_func(distribution)
        uv_star = uv_h[index_star[0], :]  # .copy()
        z_star = z[index_star[0], :]  # .copy()

        # update particle filter
        self.rbpfs[rind].update_trans_star_uvz(uv_star, z_star, intrinsics)
        distribution[index_star[0], :] /= torch.sum(distribution[index_star[0], :])
        self.rbpfs[rind].rot = distribution[index_star[0], :].view(1, 1, 37, 72, 72).repeat(self.rbpfs[rind].n_particles, 1, 1, 1, 1)
        self.rbpfs[rind].update_rot_star_R(quat2mat(poses_cpu[index_star[1], 3:]))
        self.rbpfs[rind].rot_bar = self.rbpfs[rind].rot_star
        self.rbpfs[rind].uv_bar = uv_star
        self.rbpfs[rind].z_bar = z_star
        self.rbpf_init_max_sim = max_sim_all
        self.rbpfs[rind].trans_bar = back_project(self.rbpfs[rind].uv_bar, intrinsics, self.rbpfs[rind].z_bar)
        self.rbpfs[rind].pose = pose

        # filtering on the same image
        for i in range(cfg.PF.N_INIT_FILTERING):
            self.process_poserbpf(rind, cls_id, autoencoder, codes_gpu, poses_cpu, image,
                                  intrinsics, render_dist, depth, mask, apply_motion_prior=False, init_mode=True)


        # box and poses
        box_center = self.rbpfs[rind].uv_bar[:2]
        box_size = 128 * render_dist / self.rbpfs[rind].z_bar * intrinsics[0, 0] / cfg.PF.FU
        pose[4:] = self.rbpfs[rind].trans_bar
        pose[:4] = mat2quat(self.rbpfs[rind].rot_bar)

        if cfg.TEST.SYNTHESIZE:
            cls_render = cls - 1
        else:
            cls_render = cls_id
        im_render, box = self.render_image(self.dataset, intrinsics, cls_render, pose)
        self.rbpfs[rind].roi = roi.copy()
        self.rbpfs[rind].roi[2:] = box
        self.rbpfs[rind].cls_id = cls_id

        if cfg.TEST.VISUALIZE:
            self.visualize(image, im_render, out_images[index_star[0]], in_images[index_star[0]], box_center, box_size, box)

        return pose


    # filtering
    def process_poserbpf(self, rind, cls_id, autoencoder, codes_gpu, poses_cpu, image, intrinsics, render_dist,
                         depth=None, mask=None, apply_motion_prior=True, init_mode=False):

        # propagation
        if apply_motion_prior:
            self.rbpfs[rind].propagate_particles(self.T_c1c0, self.T_o0o1, 0, 0, intrinsics)
            uv_noise = cfg.PF.UV_NOISE
            z_noise = cfg.PF.Z_NOISE
            self.rbpfs[rind].add_noise_r3(uv_noise, z_noise)
            self.rbpfs[rind].add_noise_rot()
        else:
            uv_noise = cfg.PF.UV_NOISE
            z_noise = cfg.PF.Z_NOISE
            self.rbpfs[rind].add_noise_r3(uv_noise, z_noise)
            self.rbpfs[rind].add_noise_rot()

        # add particles from estimated pose
        if self.rbpfs[rind].pose[-1] > 0:
            n_gt_particles = int(cfg.PF.N_PROCESS / 2)
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            px = intrinsics[0, 2]
            py = intrinsics[1, 2]
            pose = self.rbpfs[rind].pose
            uv_init = np.zeros((2,), dtype=np.float32)
            uv_init[0] = fx * pose[4] / pose[6] + px
            uv_init[1] = fy * pose[5] / pose[6] + py
            uv_h = np.array([uv_init[0], uv_init[1], 1])
            self.rbpfs[rind].uv[-n_gt_particles:] = np.repeat(np.expand_dims(uv_h, axis=0), n_gt_particles, axis=0)
            self.rbpfs[rind].uv[-n_gt_particles:, :2] += np.random.randn(n_gt_particles, 2) * cfg.PF.UV_NOISE_PRIOR
            self.rbpfs[rind].z[-n_gt_particles:] = np.ones((n_gt_particles, 1), dtype=np.float32) * pose[-1]
            self.rbpfs[rind].z[-n_gt_particles:] += np.random.randn(n_gt_particles, 1) * cfg.PF.Z_NOISE_PRIOR

        # compute pdf matrix for each particle
        est_pdf_matrix, max_sim_all, out_images, in_images = self.evaluate_particles(self.rbpfs[rind], cls_id, autoencoder, codes_gpu, \
            poses_cpu, image, intrinsics, self.rbpfs[rind].uv, self.rbpfs[rind].z, render_dist, cfg.PF.WT_RESHAPE_VAR, depth, mask, init_mode)

        # most likely particle
        index_star = self.arg_max_func(est_pdf_matrix)
        uv_star = self.rbpfs[rind].uv[index_star[0], :].copy()
        z_star = self.rbpfs[rind].z[index_star[0], :].copy()
        self.rbpfs[rind].update_trans_star(uv_star, z_star, intrinsics)
        self.rbpfs[rind].update_rot_star_R(quat2mat(poses_cpu[index_star[1], 3:]))

        # match rotation distribution
        self.rbpfs[rind].rot = torch.clamp(self.rbpfs[rind].rot, 1e-5, 1)
        rot_dist = torch.exp(torch.add(torch.log(est_pdf_matrix), torch.log(self.rbpfs[rind].rot.view(self.rbpfs[rind].n_particles, -1))))
        normalizers = torch.sum(rot_dist, dim=1)

        normalizers_cpu = normalizers.detach().cpu().numpy()
        self.rbpfs[rind].weights = normalizers_cpu / np.sum(normalizers_cpu)

        rot_dist /= normalizers.unsqueeze(1).repeat(1, codes_gpu.size(0))

        # matched distributions
        self.rbpfs[rind].rot = rot_dist.view(self.rbpfs[rind].n_particles, 1, 37, 72, 72)

        # resample
        self.rbpfs[rind].resample_ddpf(poses_cpu, intrinsics, cfg.PF)

        if cfg.TEST.VISUALIZE:
            return out_images[index_star[0]], in_images[index_star[0]]
        else:
            return None, None


    def arg_max_func(self, input):
        index = (input == torch.max(input)).nonzero().detach()
        return index[0]


    # evaluate particles according to the RGB(D) images
    def evaluate_particles(self, rbpf, cls_id, autoencoder, codes_gpu, codepose, image, intrinsics, uv, z, render_dist, gaussian_std,
                           depth=None, mask=None, init_mode=False):

        # crop the rois from input image
        fu = intrinsics[0, 0]
        fv = intrinsics[1, 1]
        images_roi_cuda, scale_roi = trans_zoom_uvz_cuda(image.detach(), uv, z, fu, fv, render_dist)

        # forward passing
        if cfg.TEST.VISUALIZE:
            out_images, embeddings = autoencoder(images_roi_cuda)
        else:
            embeddings = autoencoder.module.encode(images_roi_cuda)
            out_images = None

        # compute the similarity between particles' codes and the codebook
        cosine_distance_matrix = autoencoder.module.pairwise_cosine_distances(embeddings, codes_gpu)

        # get the maximum similarity for each particle
        v_sims, i_sims = torch.max(cosine_distance_matrix, dim=1)

        # evaluate particles with depth images
        depth_scores = np.ones_like(z)
        if depth is not None:
            depth_gpu = torch.from_numpy(depth).unsqueeze(2).cuda()
            if mask is not None:
                mask_gpu = torch.from_numpy(mask).unsqueeze(2).cuda()
            else:
                mask_gpu = None
            if init_mode:
                depth_scores = self.Evaluate_Depths_Init(cls_id,
                                                         depth=depth_gpu, uv=uv, z=z,
                                                         q_idx=i_sims.cpu().numpy(), intrinsics=intrinsics,
                                                         render_dist=render_dist, codepose=codepose,
                                                         delta=cfg.PF.DEPTH_DELTA,
                                                         tau=cfg.PF.DEPTH_DELTA,
                                                         mask=mask_gpu)
            else:
                depth_scores = self.Evaluate_Depths_Tracking(rbpf, cls_id,
                                                             depth=depth_gpu, uv=uv, z=z,
                                                             q_idx=i_sims.cpu().numpy(), intrinsics=intrinsics,
                                                             render_dist=render_dist, codepose=codepose,
                                                             delta=cfg.PF.DEPTH_DELTA,
                                                             tau=cfg.PF.DEPTH_DELTA)


            # reshape the depth score
            if np.max(depth_scores) > 0:
                depth_scores = depth_scores / np.max(depth_scores)
                depth_scores = mat2pdf_np(depth_scores, 1.0, cfg.PF.DEPTH_STD)
            else:
                depth_scores = np.ones_like(depth_scores)
                depth_scores /= np.sum(depth_scores)

        # compute distribution from similarity
        max_sim_all = torch.max(v_sims)
        # cosine_distance_matrix[cosine_distance_matrix > 0.95 * max_sim_all] = max_sim_all
        pdf_matrix = mat2pdf(cosine_distance_matrix/max_sim_all, 1, gaussian_std)

        # combine RGB and D
        depth_score_torch = torch.from_numpy(depth_scores).float().cuda()
        pdf_matrix = torch.mul(pdf_matrix, depth_score_torch)

        return pdf_matrix, max_sim_all, out_images, images_roi_cuda


    # evaluate particles according to depth measurements
    def Evaluate_Depths_Init(self, cls_id, depth, uv, z, q_idx, intrinsics, render_dist, codepose, delta=0.03, tau=0.05, mask=None):

        score = np.zeros_like(z)
        height = self.dataset._height
        width = self.dataset._width
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        px = intrinsics[0, 2]
        py = intrinsics[1, 2]
        zfar = 6.0
        znear = 0.01
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

        # crop rois
        depth_roi_cuda, _ = trans_zoom_uvz_cuda(depth.detach(), uv, z, intrinsics[0, 0], intrinsics[1, 1], render_dist)
        depth_roi_np = depth_roi_cuda.cpu().numpy()

        if mask is not None:
            mask_roi_cuda, _ = trans_zoom_uvz_cuda(mask.detach(), uv, z, intrinsics[0, 0], intrinsics[1, 1], render_dist)
            mask_roi_np = mask_roi_cuda.cpu().numpy()

        # render
        pose_v = np.zeros((7,))
        frame_cuda = torch.cuda.FloatTensor(height, width, 4)
        seg_cuda = torch.cuda.FloatTensor(height, width, 4)
        pc_cuda = torch.cuda.FloatTensor(height, width, 4)

        q_idx_unique, idx_inv = np.unique(q_idx, return_inverse=True)
        pc_render_all = np.zeros((q_idx_unique.shape[0], 128, 128, 3), dtype=np.float32)
        q_render = codepose[q_idx_unique, 3:]
        for i in range(q_render.shape[0]):
            pose_v[:3] = [0, 0, render_dist]
            pose_v[3:] = q_render[i]
            cfg.renderer.set_poses([pose_v])
            cfg.renderer.render([cls_id], frame_cuda, seg_cuda, pc2_tensor=pc_cuda)
            render_roi_cuda, _ = trans_zoom_uvz_cuda(pc_cuda.flip(0),
                                                     np.array([[intrinsics[0, 2], intrinsics[1, 2], 1]]),
                                                     np.array([[render_dist]]),
                                                     intrinsics[0, 0], intrinsics[1, 1],
                                                     render_dist)
            pc_render_all[i] = render_roi_cuda[0, :3, :, :].permute(1, 2, 0).cpu().numpy()

        # evaluate every particle
        for i in range(uv.shape[0]):

            pc_render = pc_render_all[idx_inv[i]].copy()
            depth_mask = pc_render[:, :, 2] > 0
            pc_render[depth_mask, 2] = pc_render[depth_mask, 2] - render_dist + z[i]

            depth_render = pc_render[:, :, [2]]
            depth_meas_np = depth_roi_np[i, 0, :, :]
            depth_render_np = depth_render[:, :, 0]

            # compute visibility mask
            if mask is None:
                visibility_mask = estimate_visib_mask_numba(depth_meas_np, depth_render_np, delta=delta)
            else:
                visibility_mask = np.logical_and((mask_roi_np[i, 0, :, :] > 0), np.logical_and(depth_meas_np>0, depth_render_np>0))

            if np.sum(visibility_mask) == 0:
                continue

            # compute depth error
            depth_error = np.abs(depth_meas_np[visibility_mask] - depth_render_np[visibility_mask])
            depth_error /= tau
            depth_error[depth_error > 1] = 1

            # score computation
            total_pixels = np.sum((depth_render_np > 0).astype(np.float32))
            if total_pixels is not 0:
                vis_ratio = np.sum(visibility_mask.astype(np.float32)) / total_pixels
                score[i] = (1 - np.mean(depth_error)) * vis_ratio
            else:
                score[i] = 0

        return score

    # evaluate particles according to depth measurements
    def Evaluate_Depths_Tracking(self, rbpf, cls_id, depth, uv, z, q_idx, intrinsics, render_dist, codepose, delta=0.03, tau=0.05):
        score = np.zeros_like(z)
        height = self.dataset._height
        width = self.dataset._width
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        px = intrinsics[0, 2]
        py = intrinsics[1, 2]
        zfar = 6.0
        znear = 0.01
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

        # crop rois
        depth_roi_cuda, _ = trans_zoom_uvz_cuda(depth.detach(), uv, z, intrinsics[0, 0], intrinsics[1, 1], render_dist)
        depth_roi_np = depth_roi_cuda.cpu().numpy()

        # render
        pose_v = np.zeros((7,))
        frame_cuda = torch.cuda.FloatTensor(height, width, 4)
        seg_cuda = torch.cuda.FloatTensor(height, width, 4)
        pc_cuda = torch.cuda.FloatTensor(height, width, 4)

        fast_rendering = False
        if np.linalg.norm(rbpf.trans_bar) > 0:
            pose_v[:3] = rbpf.trans_bar
            pose_v[3:] = mat2quat(rbpf.rot_bar)
            cfg.renderer.set_poses([pose_v])
            cfg.renderer.render([cls_id], frame_cuda, seg_cuda, pc2_tensor=pc_cuda)
            uv_crop = project(rbpf.trans_bar, intrinsics)
            uv_crop = np.repeat(np.expand_dims(uv_crop, axis=0), uv.shape[0], axis=0)
            z_crop = np.ones_like(z) * rbpf.trans_bar[2]
            render_roi_cuda, _ = trans_zoom_uvz_cuda(pc_cuda.flip(0), uv_crop, z_crop, intrinsics[0, 0], intrinsics[1, 1], render_dist)
            pc_render_all = render_roi_cuda[:, :3, :, :].permute(0, 2, 3, 1).cpu().numpy()
            fast_rendering = True
        else:
            q_idx_unique, idx_inv = np.unique(q_idx, return_inverse=True)
            pc_render_all = np.zeros((q_idx_unique.shape[0], 128, 128, 3), dtype=np.float32)
            q_render = codepose[q_idx_unique][:, 3:]
            for i in range(q_render.shape[0]):
                pose_v[:3] = [0, 0, render_dist]
                pose_v[3:] = q_render[i]
                cfg.renderer.set_poses([pose_v])
                cfg.renderer.render([cls_id], frame_cuda, seg_cuda, pc_cuda)
                render_roi_cuda, _ = trans_zoom_uvz_cuda(pc_cuda.flip(0),
                                                         np.array([[intrinsics[0, 2], intrinsics[1, 2], 1]]),
                                                         np.array([[render_dist]]),
                                                         intrinsics[0, 0], intrinsics[1, 1],
                                                         render_dist)
                pc_render_all[i] = render_roi_cuda[0, :3, :, :].permute(1, 2, 0).cpu().numpy()

        # evaluate every particle
        for i in range(uv.shape[0]):

            if fast_rendering:
                pc_render = pc_render_all[i].copy()
                depth_mask = pc_render[:, :, 2] > 0
                pc_render[:, :, 2][depth_mask] = pc_render[:, :, 2][depth_mask] - rbpf.trans_bar[2] + z[i]
            else:
                pc_render = pc_render_all[idx_inv[i]].copy()
                depth_mask = pc_render[:, :, 2] > 0
                pc_render[:, :, 2][depth_mask] = pc_render[:, :, 2][depth_mask] - render_dist + z[i]

            depth_render = pc_render[:, :, [2]]
            depth_meas_np = depth_roi_np[i, 0, :, :]
            depth_render_np = depth_render[:, :, 0]

            # compute visibility mask
            visibility_mask = estimate_visib_mask_numba(depth_meas_np, depth_render_np, delta=delta)

            if np.sum(visibility_mask) == 0:
                continue

            # compute depth error
            depth_error = np.abs(depth_meas_np[visibility_mask] - depth_render_np[visibility_mask])
            depth_error /= tau
            depth_error[depth_error > 1] = 1

            # score computation
            total_pixels = np.sum((depth_render_np > 0).astype(np.float32))
            if total_pixels is not 0:
                vis_ratio = np.sum(visibility_mask.astype(np.float32)) / total_pixels
                score[i] = (1 - np.mean(depth_error)) * vis_ratio
            else:
                score[i] = 0

        return score


    # run SDF pose refine
    def refine_pose(self, im_label, im_depth, dpoints, roi, pose, intrinsics, dataset):

        cls = int(roi[1])
        cls_id = cfg.TRAIN.CLASSES[cls]
        cls_render = cfg.TEST.CLASSES.index(cls_id)
        sdf_optim = cfg.sdf_optimizers[cls_render]
        width = im_label.shape[1]
        height = im_label.shape[0]
        im_pcloud = dpoints.reshape((height, width, 3))

        # setup render
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        px = intrinsics[0, 2]
        py = intrinsics[1, 2]
        zfar = 6.0
        znear = 0.01
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
        image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        pcloud_tensor = torch.cuda.FloatTensor(height, width, 4).detach()

        # render
        poses_all = []
        cls_indexes = []
        cls_indexes.append(cls_render)
        qt = np.zeros((7, ), dtype=np.float32)
        qt[3:] = pose[:4]
        qt[:3] = pose[4:]
        poses_all.append(qt.copy())
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

        # mask label
        x1 = max(int(roi[2]), 0)
        y1 = max(int(roi[3]), 0)
        x2 = min(int(roi[4]), width-1)
        y2 = min(int(roi[5]), height-1)
        labels = np.zeros((height, width), dtype=np.float32)
        labels[y1:y2, x1:x2] = im_label[y1:y2, x1:x2]
        mask_label = np.ma.getmaskarray(np.ma.masked_equal(labels, cls))
        mask = mask_label * mask_depth_valid * mask_depth_meas * mask_depth_render * mask_depth_vis
        index_p = mask.flatten().nonzero()[0]

        pose_refined = pose.copy()
        if len(index_p) > 10:
            points = torch.from_numpy(dpoints[index_p, :]).float()
            points = torch.cat((points, torch.ones((points.size(0), 1), dtype=torch.float32)), dim=1)
            RT = np.zeros((4, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(pose[:4])
            RT[:3, 3] = pose[4:]
            RT[3, 3] = 1.0
            T_co_init = RT
            T_co_opt, sdf_values = sdf_optim.refine_pose_layer(T_co_init, points.cuda(), steps=cfg.TEST.NUM_SDF_ITERATIONS)
            RT_opt = T_co_opt
            pose_refined[:4] = mat2quat(RT_opt[:3, :3])
            pose_refined[4:] = RT_opt[:3, 3]

            if cfg.TEST.VISUALIZE:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(3, 3, 1, projection='3d')
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

        return pose_refined, cls_render


    # evaluate a single 6D pose of a certain object
    def evaluate_6d_pose(self, pose, cls, image_bgr, image_depth, intrinsics, mask=None):

        sim = 0
        depth_error = 1
        vis_ratio = 0

        if cls in cfg.TEST.CLASSES:

            cls_id = cfg.TEST.CLASSES.index(cls)
            render_dist = self.codebooks[cls_id]['distance']

            t = pose[4:]
            uv = project(np.expand_dims(t, axis=1), intrinsics).transpose()

            z = np.array([t[2]], dtype=np.float32)
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            px = intrinsics[0, 2]
            py = intrinsics[1, 2]

            # get roi
            rois, scale_roi = trans_zoom_uvz_cuda(image_bgr.detach(), uv, z, fx, fy, render_dist)

            # render object
            zfar = 6.0
            znear = 0.01
            width = image_bgr.shape[1]
            height = image_bgr.shape[0]
            # rendering
            cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
            image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
            seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
            pcloud_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
            poses_all = []
            qt = np.zeros((7,), dtype=np.float32)
            qt[3:] = pose[:4]
            qt[0] = pose[4]
            qt[1] = pose[5]
            qt[2] = pose[6]
            poses_all.append(qt)
            cfg.renderer.set_poses(poses_all)

            cfg.renderer.render([cls_id], image_tensor, seg_tensor, pc2_tensor=pcloud_tensor)
            pcloud_tensor = pcloud_tensor[:, :, :3].flip(0)

            render_bgr = image_tensor[:, :, :3].flip(0)[:, :, (2,1,0)]
            rois_render, scale_roi_render = trans_zoom_uvz_cuda(render_bgr.detach(), uv, z, fx, fy, render_dist)

            # forward passing
            out_img, embeddings = self.autoencoders[cls_id](torch.cat((rois, rois_render), dim=0))
            embeddings = embeddings.detach()
            sim = self.cos_sim(embeddings[[0], :], embeddings[[1], :])[0].detach().cpu().numpy()

            # evaluate depth error
            depth_render = pcloud_tensor[:, :, 2].cpu().numpy()

            # compute visibility mask
            if not mask is None:
                cls_id_train = cfg.TRAIN.CLASSES.index(cls)
                visibility_mask = np.logical_and(np.logical_and(mask == cls_id_train, depth_render > 0),
                                                 estimate_visib_mask_numba(image_depth, depth_render, 0.05))
            else:
                visibility_mask = estimate_visib_mask_numba(image_depth, depth_render, 0.02)

            vis_ratio = np.sum(visibility_mask.astype(np.float32)) * 1.0 / np.sum(depth_render != 0)
            depth_error = np.mean(np.abs(depth_render[visibility_mask] - image_depth[visibility_mask]))

        return sim, depth_error, vis_ratio


    def visualize(self, image, im_render, im_output, im_input, box_center, box_size, roi):

        import matplotlib.pyplot as plt
        fig = plt.figure()
        # show image
        im = image.cpu().numpy()
        im = im * 255.0
        im = im [:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(im)
        ax.set_title('input image')

        plt.plot(box_center[0], box_center[1], 'ro', markersize=5)
        x1 = box_center[0] - box_size / 2
        x2 = box_center[0] + box_size / 2
        y1 = box_center[1] - box_size / 2
        y2 = box_center[1] + box_size / 2
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

        # show output
        im = im_input.cpu().detach().numpy()
        im = np.clip(im, 0, 1)
        im = im.transpose((1, 2, 0)) * 255.0
        im = im [:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(2, 2, 3)
        plt.imshow(im)
        ax.set_title('input roi')

        # show output
        im = im_output.cpu().detach().numpy()
        im = np.clip(im, 0, 1)
        im = im.transpose((1, 2, 0)) * 255.0
        im = im [:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(2, 2, 4)
        plt.imshow(im)
        ax.set_title('reconstruction')

        # show output
        ax = fig.add_subplot(2, 2, 2)
        plt.imshow(im_render)
        plt.plot(box_center[0], box_center[1], 'ro', markersize=5)
        ax.set_title('rendering')

        x1 = roi[0]
        y1 = roi[1]
        x2 = roi[2]
        y2 = roi[3]
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

        plt.show()


    def render_image(self, dataset, intrinsic_matrix, cls, pose):

        height = dataset._height
        width = dataset._width
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.25

        im_output = np.zeros((height, width, 3), dtype=np.uint8)
        image_tensor = torch.cuda.FloatTensor(height, width, 4)
        seg_tensor = torch.cuda.FloatTensor(height, width, 4)

        # set renderer
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

        cls_indexes = []
        poses_all = []

        # todo: fix the problem for large clamp
        if cls == -1:
            cls_index = 18
        else:
            cls_index = cls

        cls_indexes.append(cls_index)
        pose_render = pose.copy()
        pose_render[:3] = pose[4:]
        pose_render[3:] = pose[:4]
        poses_all.append(pose_render)

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
        im_output = im_render

        # compute box
        I = np.where(seg.cpu().numpy() > 0)
        if len(I[0]) > 0 and len(I[1]) > 0:
            x1 = np.min(I[1])
            y1 = np.min(I[0])
            x2 = np.max(I[1])
            y2 = np.max(I[0])
            box = np.array([x1, y1, x2, y2])
        else:
            box = np.array([0, 0, 0, 0])

        return im_output, box



    def render_image_all(self, intrinsic_matrix):

        height = self.dataset._height
        width = self.dataset._width
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.25

        im_output = np.zeros((height, width, 3), dtype=np.uint8)
        image_tensor = torch.cuda.FloatTensor(height, width, 4)
        seg_tensor = torch.cuda.FloatTensor(height, width, 4)

        # set renderer
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

        cls_indexes = []
        poses_all = []

        for i in range(self.num_rbpfs):

            rbpf = self.rbpfs[i]
            cls_index = rbpf.cls_id
            cls_indexes.append(cls_index)

            pose = rbpf.pose
            pose_render = np.zeros((7,), dtype=np.float32)
            pose_render[:3] = pose[4:]
            pose_render[3:] = pose[:4]
            poses_all.append(pose_render)

        # rendering
        cfg.renderer.set_poses(poses_all)
        cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)
        seg = seg_tensor[:,:,2] + 256*seg_tensor[:,:,1] + 256*256*seg_tensor[:,:,0]

        im_render = image_tensor.cpu().numpy()
        im_render = np.clip(im_render, 0, 1)
        im_render = im_render[:, :, :3] * 255
        im_render = im_render.astype(np.uint8)
        im_output = im_render
        return im_output
