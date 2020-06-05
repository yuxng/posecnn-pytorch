import os, sys
import copy
import torch
import scipy.io
import random
import networks
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn as nn

from layers.roi_align import ROIAlign
from fcn.config import cfg
from fcn.particle_filter import particle_filter
from utils.blob import add_noise_cuda
from utils.se3 import se3_inverse
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.prbpf_utils import trans_zoom_uvz_cuda, mat2pdf_np, mat2pdf, back_project, project, block_poses_dolly

class PoseRBPF:

    def __init__(self, dataset, path_pretrained=None, path_codebook=None):

        # prepare autoencoder and codebook
        autoencoders = [[] for i in range(len(cfg.TEST.CLASSES))]
        codebooks = [[] for i in range(len(cfg.TEST.CLASSES))]
        codes_gpu = [[] for i in range(len(cfg.TEST.CLASSES))]
        poses_cpu = [[] for i in range(len(cfg.TEST.CLASSES))]
        codebook_names = [[] for i in range(len(cfg.TEST.CLASSES))]
        block_indexes = [None for i in range(len(cfg.TEST.CLASSES))]

        if path_pretrained is not None and path_codebook is not None:
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

                    if cls == 'industrial_dolly':
                        print(cls)
                        block_index = block_poses_dolly(poses_cpu[i])
                        block_indexes[i] = block_index
                else:
                    print('file not exists %s or %s' % (filename_mat, filename_pth))

        self.autoencoders = autoencoders
        self.codebooks = codebooks
        self.codebook_names = codebook_names
        self.codes_gpu = codes_gpu
        self.poses_cpu = poses_cpu
        self.block_indexes = block_indexes
        self.dataset = dataset
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.num_objects_per_class = np.zeros((len(cfg.TEST.CLASSES), 10), dtype=np.int32)
        self.prefix = '%02d_' % (cfg.instance_id)

        # motion model
        self.forward_kinematics = True
        self.T_c1c0 = np.eye(4, dtype=np.float32)
        self.T_o0o1 = np.eye(4, dtype=np.float32)
        self.Tbc = np.eye(4, dtype=np.float32)

        # initialize the particle filters
        self.reset = False
        self.rbpf = None
        self.rbpfs = []


    @property
    def num_rbpfs(self):
        return len(self.rbpfs)


    # pose estimation pipeline
    # roi: object detection from posecnn, shape (1, 6)
    # image_bgr: input bgr image, range (0, 1)
    def estimation_poserbpf(self, roi, intrinsic_matrix, image, im_depth, im_pcloud, im_label=None, grasp_mode=False, grasp_cls=-1):

        n_init_samples = cfg.PF.N_PROCESS
        uv_init = np.zeros((2, ), dtype=np.float32)
        roi = roi.flatten()
        cls = int(roi[1])

        # use bounding box center
        uv_init[0] = (roi[4] + roi[2]) / 2
        uv_init[1] = (roi[5] + roi[3]) / 2

        if im_label is not None:
            mask = torch.zeros_like(im_label)
            cls_train = cfg.TRAIN.CLASSES.index(cfg.TEST.CLASSES[cls])
            mask[im_label == cls_train] = 1.0

        index = np.where(self.num_objects_per_class[cls, :] == 0)[0]
        object_id = index[0]
        self.num_objects_per_class[cls, object_id] = 1
        ind = cfg.TEST.CLASSES[cls]
        name = self.prefix + self.dataset._classes_all[ind] + '_%02d' % (object_id)
        self.rbpfs.append(particle_filter(cfg.PF, n_particles=cfg.PF.N_PROCESS))
        self.rbpfs[-1].object_id = object_id   
        self.rbpfs[-1].name = name
        self.rbpfs[-1].Tbc = self.Tbc.copy()

        end = time.time()
        pose = self.initialize(self.num_rbpfs-1, image, uv_init, n_init_samples, cfg.TEST.CLASSES[cls], roi, intrinsic_matrix, im_depth, mask, grasp_mode=False, grasp_cls=-1)
        self.rbpfs[-1].roi = roi
        self.rbpfs[-1].pose = pose
        print('estimation initialize time %.2f' % (time.time() - end))

        # SDF refine
        end = time.time()
        index_sdf = [self.num_rbpfs-1]
        self.refine_pose(index_sdf, im_depth, im_pcloud, intrinsic_matrix, self.dataset, im_label, steps=cfg.TEST.NUM_SDF_ITERATIONS_INIT)
        print('estimation sdf time %.2f' % (time.time() - end))

        if cfg.TEST.VISUALIZE:

            if cfg.TEST.SYNTHESIZE:
                cls_render = int(cfg.TEST.CLASSES[int(roi[1])]) - 1
            else:
                cls_render = int(roi[1]) - 1

            im_render_refine = self.render_image(self.dataset, intrinsic_matrix, cls_render, self.rbpfs[-1].pose.flatten())
            box = self.compute_box(self.dataset, intrinsic_matrix, int(roi[1]), pose.flatten())
            im_render = self.render_image(self.dataset, intrinsic_matrix, cls_render, pose.flatten())    
            # show image
            import matplotlib.pyplot as plt
            fig = plt.figure()
            im = image.cpu().numpy()
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
            box_refine = self.rbpfs[-1].roi[2:6]
            x1 = box_refine[0]
            y1 = box_refine[1]
            x2 = box_refine[2]
            y2 = box_refine[3]
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

            ax = fig.add_subplot(2, 2, 4)
            plt.imshow(mask.cpu().numpy())
            ax.set_title('mask')
            plt.show()


    # pose estimation pipeline
    # roi: object detection from posecnn, shape (1, 6)
    # image_bgr: input bgr image, range (0, 1)
    def filtering_poserbpf(self, intrinsics, image, im_depth, im_pcloud, im_label=None, grasp_mode=False, grasp_cls=-1):

        n_init_samples = cfg.PF.N_PROCESS
        save = True

        # render pcloud of all the tracked objects
        _, pcloud_tensor = self.render_image_all(intrinsics, grasp_mode, grasp_cls)

        # for each particle filter
        end = time.time()
        index_sdf = []
        print('{} pose rbpfs'.format(self.num_rbpfs))
        for i in range(self.num_rbpfs):
            if self.reset:
                print('reset return')
                return False, None

            if grasp_mode and self.rbpfs[i].cls_id != grasp_cls:
                print('grasp mode not class return')
                continue

            # the current pose is good for grasping
            if grasp_mode and self.rbpfs[i].graspable and self.rbpfs[i].status:
                print('grasp mode ready return')
                continue

            # no detection for the target, do nothing
            if self.rbpfs[i].roi_assign is None:
                print('no roi assign return')
                continue

            if self.rbpfs[i].pose_prev is not None and self.rbpfs[i].num_tracked % 10 == 0:
                self.rbpfs[i].pose_prev = self.rbpfs[i].pose.copy()
            if self.rbpfs[i].need_filter:
                cls_id = self.rbpfs[i].cls_id
                autoencoder = self.autoencoders[cls_id]
                codebook = self.codebooks[cls_id]
                codes_gpu = self.codes_gpu[cls_id]
                poses_cpu = self.poses_cpu[cls_id]
                block_index = self.block_indexes[cls_id]

                render_dist = codebook['distance'][0, 0]
                intrinsic_matrix = codebook['intrinsic_matrix']
                cfg.PF.FU = intrinsic_matrix[0, 0]
                cfg.PF.FV = intrinsic_matrix[1, 1]

                if im_label is not None:
                    mask = torch.zeros_like(im_label)
                    cls = int(self.rbpfs[i].roi[1])
                    cls_train = cfg.TRAIN.CLASSES.index(cfg.TEST.CLASSES[cls])
                    mask[im_label == cls_train] = 1.0
                else:
                    mask = None

                out_image, in_image = self.process_poserbpf(i, cls_id, autoencoder, codes_gpu, poses_cpu, block_index, image, \
                                      intrinsics, render_dist, im_depth, mask, apply_motion_prior=self.forward_kinematics, \
                                      init_mode=False, pcloud_tensor=pcloud_tensor)

                # box and poses
                box_center = self.rbpfs[i].uv_bar[:2]
                box_size = 128 * render_dist / self.rbpfs[i].z_bar * intrinsics[0, 0] / cfg.PF.FU
                pose = np.zeros((7,), dtype=np.float32)
                pose[4:] = self.rbpfs[i].trans_bar
                pose[:4] = mat2quat(self.rbpfs[i].rot_bar)
                self.rbpfs[i].pose = pose
                self.rbpfs[i].Tbc = self.Tbc.copy()
                print('filtering:', self.rbpfs[i].name)

                if cfg.TEST.VISUALIZE:
                    cls = int(self.rbpfs[i].roi[1])
                    if cfg.TEST.SYNTHESIZE:
                        cls_render = int(cfg.TEST.CLASSES[cls]) - 1
                    else:
                        cls_render = cls - 1
                    box = self.compute_box(self.dataset, intrinsics, cls, pose)
                    im_render = self.render_image(self.dataset, intrinsics, cls_render, self.rbpfs[i].pose)
                    self.visualize(image, im_render, out_image, in_image, box_center, box_size, box)

            index_sdf.append(i)
        print('filtering poserbpf time %.6f' % (time.time() - end))

        # SDF refine for multiple objects
        if len(index_sdf) > 0:
            start_time = time.time()
            self.refine_pose(index_sdf, im_depth, im_pcloud, intrinsics, self.dataset, im_label, steps=cfg.TEST.NUM_SDF_ITERATIONS_TRACKING)
            print('pose refine time %.6f' % (time.time() - start_time)) 

        # pose evaluation
        end = time.time()
        image_tensor, pcloud_tensor = self.render_image_all(intrinsics, grasp_mode, grasp_cls)
        sims, depth_errors, vis_ratios = self.evaluate_6d_pose_all(image, image_tensor, pcloud_tensor, im_depth, intrinsics, im_label)
        for i in range(self.num_rbpfs):
            if self.reset:
                save = False
                return False, None

            if grasp_mode and self.rbpfs[i].cls_id != grasp_cls:
                save = False
                continue

            # no detection for the target, do nothing
            if self.rbpfs[i].roi_assign is None:
                self.rbpfs[i].num_lost += 1
                self.rbpfs[i].num_tracked = 0
                save = False
                continue

            sim = sims[i]
            depth_error = depth_errors[i]
            vis_ratio = vis_ratios[i]

            if grasp_mode:
               threshold_sim = cfg.PF.THRESHOLD_SIM_GRASPING
               threshold_depth = cfg.PF.THRESHOLD_DEPTH_GRASPING
               threshold_ratio = cfg.PF.THRESHOLD_RATIO_GRASPING
            else:
               threshold_sim = cfg.PF.THRESHOLD_SIM
               threshold_depth = cfg.PF.THRESHOLD_DEPTH
               threshold_ratio = cfg.PF.THRESHOLD_RATIO

            if sim < threshold_sim or np.isnan(depth_error) or depth_error > threshold_depth or vis_ratio < threshold_ratio:
                save = False
                self.rbpfs[i].num_lost += 1
                self.rbpfs[i].num_tracked = 0                
                self.rbpfs[i].status = False
            else:
                self.rbpfs[i].num_lost = 0
                self.rbpfs[i].num_tracked += 1
                self.rbpfs[i].status = True
            self.rbpfs[i].num_frame += 1

            # more strict threshold
            if sim < cfg.PF.THRESHOLD_SIM_GRASPING or np.isnan(depth_error) or depth_error > cfg.PF.THRESHOLD_DEPTH_GRASPING or vis_ratio < cfg.PF.THRESHOLD_RATIO_GRASPING:
                self.rbpfs[i].graspable = False
                self.rbpfs[i].need_filter = True
            else:
                self.rbpfs[i].graspable = True
                self.rbpfs[i].need_filter = False

            print('Tracking {}, Sim obs: {}, Depth Err: {:.3}, Vis Ratio: {:.2}, lost: {}, tracked {}, graspable {}, status {}'.format(self.rbpfs[i].name, \
                sim, depth_error, vis_ratio, self.rbpfs[i].num_lost, self.rbpfs[i].num_tracked, self.rbpfs[i].graspable, self.rbpfs[i].status))

        print('pose evaluation evaluate time %.6f' % (time.time() - end))

        # check pose difference
        if save and cfg.TEST.GEN_DATA:
            same_pose = True
            for i in range(self.num_rbpfs):
                if self.rbpfs[i].pose_prev is None:
                    self.rbpfs[i].pose_prev = self.rbpfs[i].pose.copy()
                    same_pose = False
                    break
                q1 = self.rbpfs[i].pose_prev[:4]
                q2 = self.rbpfs[i].pose[:4]
                theta = np.arccos(2 * np.dot(q1,q2)**2 - 1) * 180 / np.pi
                t1 = self.rbpfs[i].pose_prev[4:]
                t2 = self.rbpfs[i].pose[4:]
                d = np.linalg.norm(t1 - t2)
                print(i, theta, d)
                if theta > 5.0 or d > 0.01:
                    same_pose = False
                    break
            if same_pose:
                save = False
        return save, image_tensor



    # pose estimation globally, search all image locations for objects
    # image_bgr: input bgr image, range (0, 1)
    def estimation_poserbpf_globally(self, intrinsic_matrix, image, im_depth, im_pcloud, im_label=None, grasp_mode=False, grasp_cls=-1):

        n_init_samples = cfg.PF.N_PROCESS
        cls = 1  # only search for one class

        if im_label is not None:
            mask = torch.zeros_like(im_label)
            cls_train = cfg.TRAIN.CLASSES.index(cfg.TEST.CLASSES[cls])
            mask[im_label == cls_train] = 1.0
        else:
            mask = None

        # add particle filter
        index = np.where(self.num_objects_per_class[cls, :] == 0)[0]
        object_id = index[0]
        self.num_objects_per_class[cls, object_id] = 1
        ind = cfg.TEST.CLASSES[cls]
        name = self.prefix + self.dataset._classes_all[ind] + '_%02d' % (object_id)
        self.rbpfs.append(particle_filter(cfg.PF, n_particles=cfg.PF.N_PROCESS))
        self.rbpfs[-1].object_id = object_id   
        self.rbpfs[-1].name = name
        self.rbpfs[-1].Tbc = self.Tbc.copy()

        end = time.time()
        pose = self.initialize_globally(self.num_rbpfs-1, image, n_init_samples, cfg.TEST.CLASSES[cls], 
            intrinsic_matrix, im_depth, mask, grasp_mode=False, grasp_cls=-1)
        self.rbpfs[-1].pose = pose
        print('estimation initialize time %.2f' % (time.time() - end))

        # SDF refine
        end = time.time()
        index_sdf = [self.num_rbpfs-1]
        self.refine_pose(index_sdf, im_depth, im_pcloud, intrinsic_matrix, self.dataset, im_label, steps=cfg.TEST.NUM_SDF_ITERATIONS_INIT)
        print('estimation sdf time %.2f' % (time.time() - end))

        if cfg.TEST.VISUALIZE:

            if cfg.TEST.SYNTHESIZE:
                cls_render = int(cfg.TEST.CLASSES[int(self.rbpfs[-1].roi[1])]) - 1
            else:
                cls_render = int(self.rbpfs[-1].roi[1]) - 1

            im_render_refine = self.render_image(self.dataset, intrinsic_matrix, cls_render, self.rbpfs[-1].pose.flatten())
            box = self.compute_box(self.dataset, intrinsic_matrix, int(self.rbpfs[-1].roi[1]), pose.flatten())
            im_render = self.render_image(self.dataset, intrinsic_matrix, cls_render, pose.flatten())    
            # show image
            import matplotlib.pyplot as plt
            fig = plt.figure()
            im = image.cpu().numpy()
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
            box_refine = self.rbpfs[-1].roi[2:6]
            x1 = box_refine[0]
            y1 = box_refine[1]
            x2 = box_refine[2]
            y2 = box_refine[3]
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))
            plt.show()


    # initialize PoseRBPF
    '''
    image (height, width, 3) with values (0, 1)
    '''
    def initialize(self, rind, image, uv_init, n_init_samples, cls, roi, intrinsics, depth=None, mask=None, grasp_mode=False, grasp_cls=-1):

        cls_id = cfg.TEST.CLASSES.index(cls)
        roi_w = roi[4] - roi[2]
        roi_h = roi[5] - roi[3]

        # network and codebook of the class
        autoencoder = self.autoencoders[cls_id]
        codebook = self.codebooks[cls_id]
        codes_gpu = self.codes_gpu[cls_id]
        poses_cpu = self.poses_cpu[cls_id]
        block_index = self.block_indexes[cls_id]
        pose = np.zeros((7,), dtype=np.float32)
        if not autoencoder or not codebook:
            print('no codebooks or checkpoint')
            return pose

        render_dist = codebook['distance'][0, 0]
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

        # sampling with depth        
        if depth is not None:
            uv_h_int = uv_h.astype(int)
            uv_h_int[:, 0] = np.clip(uv_h_int[:, 0], 0, image.shape[1] - 1)
            uv_h_int[:, 1] = np.clip(uv_h_int[:, 1], 0, image.shape[0] - 1)

            if mask is not None:
                x1 = max(int(uv_init[0] - roi_w * 0.2), 0)
                y1 = max(int(uv_init[1] - roi_h * 0.2), 0)
                x2 = min(int(uv_init[0] + roi_w * 0.2), image.shape[1]-1)
                y2 = min(int(uv_init[1] + roi_h * 0.2), image.shape[0]-1)
                mask_sample = torch.zeros_like(mask)
                mask_sample[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                index_nonzero = torch.nonzero(mask_sample).cpu().numpy().astype(np.float32)
                num = index_nonzero.shape[0]
                if num > 0:
                    index = np.random.randint(num, size=n_init_samples)
                    uv_h_int[:, 0] = index_nonzero[index, 1]
                    uv_h_int[:, 1] = index_nonzero[index, 0]

            z = depth[uv_h_int[:, 1], uv_h_int[:, 0]]
            z[torch.isnan(z)] = 0
            z = z.cpu().numpy()
            z = np.expand_dims(z, axis=1)
            extent = np.mean(self.dataset._extents_test[int(roi[1]), :]) / 2
            z[z > 0] += np.random.uniform(-extent, extent, z[z > 0].shape)
            z[z == 0 | ~np.isfinite(z)] = np.random.uniform(0.9 * z_init, 1.1 * z_init, z[z == 0 | ~np.isfinite(z)].shape)
        else:
            z = np.random.uniform(0.9 * z_init, 1.1 * z_init, (n_init_samples, 1))

        # evaluate
        end = time.time()
        distribution, max_sim_all, out_images, in_images = self.evaluate_particles(self.rbpfs[rind], cls_id, autoencoder, codes_gpu, \
            poses_cpu, block_index, image, intrinsics, uv_h, z, render_dist, cfg.PF.WT_RESHAPE_VAR, depth, mask, init_mode=True, pcloud_tensor=None)
        print('initialization evaluate partiles time %.2f' % (time.time() - end))

        # find the max pdf from the distribution matrix
        index_star = self.arg_max_func(distribution)
        uv_star = uv_h[index_star[0], :]
        z_star = z[index_star[0], :]

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
        end = time.time()
        self.rbpfs[rind].cls_id = cls_id
        _, pcloud_tensor = self.render_image_all(intrinsics, grasp_mode, grasp_cls)
        for i in range(cfg.PF.N_INIT_FILTERING):
            self.process_poserbpf(rind, cls_id, autoencoder, codes_gpu, poses_cpu, block_index, image,
                                  intrinsics, render_dist, depth, mask, apply_motion_prior=False, init_mode=False, pcloud_tensor=pcloud_tensor)
        print('initialization process poserbpf time %.2f' % (time.time() - end))

        # box and poses
        box_center = self.rbpfs[rind].uv_bar[:2]
        box_size = 128 * render_dist / self.rbpfs[rind].z_bar * intrinsics[0, 0] / cfg.PF.FU
        pose[4:] = self.rbpfs[rind].trans_bar
        pose[:4] = mat2quat(self.rbpfs[rind].rot_bar)

        box = self.compute_box(self.dataset, intrinsics, int(roi[1]), pose)
        self.rbpfs[rind].roi = roi.copy()
        self.rbpfs[rind].roi[2:6] = box

        if cfg.TEST.VISUALIZE:
            if cfg.TEST.SYNTHESIZE:
                cls_render = int(cfg.TEST.CLASSES[cls_id]) - 1
            else:
                cls_render = cls_id - 1
            im_render = self.render_image(self.dataset, intrinsics, cls_render, pose)
            self.visualize(image, im_render, out_images[index_star[0]], in_images[index_star[0]], box_center, box_size, box)

        return pose



    # initialize PoseRBPF globally
    '''
    image (height, width, 3) with values (0, 1)
    '''
    def initialize_globally(self, rind, image, n_init_samples, cls, intrinsics, depth=None, mask=None, grasp_mode=False, grasp_cls=-1):

        width = image.shape[1]
        height = image.shape[0]
        print('%d samples' % (n_init_samples))

        # network and codebook of the class
        cls_id = cfg.TEST.CLASSES.index(cls)
        autoencoder = self.autoencoders[cls_id]
        codebook = self.codebooks[cls_id]
        codes_gpu = self.codes_gpu[cls_id]
        poses_cpu = self.poses_cpu[cls_id]
        block_index = self.block_indexes[cls_id]
        pose = np.zeros((7,), dtype=np.float32)
        if not autoencoder or not codebook:
            print('no codebooks or checkpoint')
            return pose

        render_dist = codebook['distance'][0, 0]
        intrinsic_matrix = codebook['intrinsic_matrix']
        cfg.PF.FU = intrinsic_matrix[0, 0]
        cfg.PF.FV = intrinsic_matrix[1, 1]

        # sample around the center of the image
        uv_init = np.zeros((2, ), dtype=np.float32)
        uv_init[0] = width / 2.0
        uv_init[1] = height / 2.0
        uv_h = np.array([uv_init[0], uv_init[1], 1])
        uv_h = np.repeat(np.expand_dims(uv_h, axis=0), n_init_samples, axis=0)
        bound = width * 0.05
        uv_h[:, 0] += np.random.uniform(-bound, bound, (n_init_samples, ))
        bound = height * 0.05
        uv_h[:, 1] += np.random.uniform(-bound, bound, (n_init_samples, ))
        uv_h[:, 0] = np.clip(uv_h[:, 0], 0, width)
        uv_h[:, 1] = np.clip(uv_h[:, 1], 0, height)

        # sampling with depth        
        if depth is not None:
            uv_h_int = uv_h.astype(int)
            uv_h_int[:, 0] = np.clip(uv_h_int[:, 0], 0, width - 1)
            uv_h_int[:, 1] = np.clip(uv_h_int[:, 1], 0, height - 1)
            z = depth[uv_h_int[:, 1], uv_h_int[:, 0]]
            z[torch.isnan(z)] = 0
            z = z.cpu().numpy()
            z = np.expand_dims(z, axis=1)
            extent = np.mean(self.dataset._extents_test[cls_id, :]) / 2
            z[z > 0] += np.random.uniform(-extent, extent, z[z > 0].shape)
            z[z == 0 | ~np.isfinite(z)] = np.random.uniform(cfg.TRAIN.SYN_TNEAR, cfg.TRAIN.SYN_TFAR, z[z == 0 | ~np.isfinite(z)].shape)
        else:
            z = np.random.uniform(cfg.TRAIN.SYN_TNEAR, cfg.TRAIN.SYN_TFAR, (n_init_samples, 1))

        # evaluate
        end = time.time()
        distribution, max_sim_all, out_images, in_images = self.evaluate_particles(self.rbpfs[rind], cls_id, autoencoder, codes_gpu, \
            poses_cpu, block_index, image, intrinsics, uv_h, z, render_dist, cfg.PF.WT_RESHAPE_VAR, depth, mask, init_mode=True, pcloud_tensor=None)
        print('initialization evaluate partiles time %.2f' % (time.time() - end))

        # find the max pdf from the distribution matrix
        index_star = self.arg_max_func(distribution)
        uv_star = uv_h[index_star[0], :]
        z_star = z[index_star[0], :]

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
        end = time.time()
        self.rbpfs[rind].cls_id = cls_id
        _, pcloud_tensor = self.render_image_all(intrinsics, grasp_mode, grasp_cls)
        for i in range(cfg.PF.N_INIT_FILTERING):
            self.process_poserbpf(rind, cls_id, autoencoder, codes_gpu, poses_cpu, block_index, image,
                                  intrinsics, render_dist, depth, mask, apply_motion_prior=False, init_mode=False, pcloud_tensor=pcloud_tensor)
        print('initialization process poserbpf time %.2f' % (time.time() - end))

        # box and poses
        box_center = self.rbpfs[rind].uv_bar[:2]
        box_size = 128 * render_dist / self.rbpfs[rind].z_bar * intrinsics[0, 0] / cfg.PF.FU
        pose[4:] = self.rbpfs[rind].trans_bar
        pose[:4] = mat2quat(self.rbpfs[rind].rot_bar)

        box = self.compute_box(self.dataset, intrinsics, cls_id, pose)
        self.rbpfs[rind].roi = np.zeros((7, ), dtype=np.float32)
        self.rbpfs[rind].roi[1] = cls_id
        self.rbpfs[rind].roi[2:6] = box
        self.rbpfs[rind].roi[6] = 1.0

        if cfg.TEST.VISUALIZE:
            if cfg.TEST.SYNTHESIZE:
                cls_render = int(cfg.TEST.CLASSES[cls_id]) - 1
            else:
                cls_render = cls_id - 1
            im_render = self.render_image(self.dataset, intrinsics, cls_render, pose)
            self.visualize(image, im_render, out_images[index_star[0]], in_images[index_star[0]], box_center, box_size, box)

        return pose


    # filtering
    def process_poserbpf(self, rind, cls_id, autoencoder, codes_gpu, poses_cpu, block_index, image, intrinsics, render_dist,
                         depth=None, mask=None, apply_motion_prior=True, init_mode=False, pcloud_tensor=None):

        # propagation
        if apply_motion_prior:
            self.T_c1c0 = np.matmul(np.linalg.inv(self.Tbc), self.rbpfs[rind].Tbc)
            self.rbpfs[rind].propagate_particles(self.T_c1c0, self.T_o0o1, 0, 0, intrinsics)
            uv_noise = cfg.PF.UV_NOISE
            z_noise = cfg.PF.Z_NOISE
            self.rbpfs[rind].add_noise_r3(uv_noise, z_noise)
            # 3D convolution which can be slow
            # self.rbpfs[rind].add_noise_rot()
        else:
            uv_noise = cfg.PF.UV_NOISE
            z_noise = cfg.PF.Z_NOISE
            self.rbpfs[rind].add_noise_r3(uv_noise, z_noise)
            # self.rbpfs[rind].add_noise_rot()

        # add particles from detection
        if self.rbpfs[rind].roi_assign is not None:
            n_gt_particles = int(cfg.PF.N_PROCESS / 2)
            roi = self.rbpfs[rind].roi_assign
            uv_init = np.zeros((2,), dtype=np.float32)
            uv_init[0] = (roi[2] + roi[4]) / 2
            uv_init[1] = (roi[3] + roi[5]) / 2
            uv_h = np.array([uv_init[0], uv_init[1], 1])
            uv_h = np.repeat(np.expand_dims(uv_h, axis=0), n_gt_particles, axis=0)
            self.rbpfs[rind].uv[-n_gt_particles:] = uv_h
            self.rbpfs[rind].uv[-n_gt_particles:, :2] += np.random.randn(n_gt_particles, 2) * cfg.PF.UV_NOISE_PRIOR

            uv_h_int = uv_h.astype(int)
            uv_h_int[:, 0] = np.clip(uv_h_int[:, 0], 0, image.shape[1] - 1)
            uv_h_int[:, 1] = np.clip(uv_h_int[:, 1], 0, image.shape[0] - 1)
            roi_w = roi[4] - roi[2]
            roi_h = roi[5] - roi[3]
            roi_size = max(roi_w, roi_h)
            z_init = (128 - 40) * render_dist / roi_size * intrinsics[0, 0] / cfg.PF.FU
            z = depth[uv_h_int[:, 1], uv_h_int[:, 0]]
            z[torch.isnan(z)] = z_init
            z = z.cpu().numpy()
            z = np.expand_dims(z, axis=1)
            extent = np.mean(self.dataset._extents_test[int(roi[1]), :]) / 2
            z[z > 0] += np.random.uniform(0, extent, z[z > 0].shape)
            z[z == 0 | ~np.isfinite(z)] = np.random.uniform(0.9 * z_init, 1.1 * z_init, z[z == 0 | ~np.isfinite(z)].shape)
            self.rbpfs[rind].z[-n_gt_particles:] = z
            self.rbpfs[rind].z[-n_gt_particles:] += np.random.randn(n_gt_particles, 1) * cfg.PF.Z_NOISE_PRIOR

        # compute pdf matrix for each particle
        est_pdf_matrix, max_sim_all, out_images, in_images = self.evaluate_particles(self.rbpfs[rind], cls_id, autoencoder, codes_gpu,
            poses_cpu, block_index, image, intrinsics, self.rbpfs[rind].uv, self.rbpfs[rind].z, 
            render_dist, cfg.PF.WT_RESHAPE_VAR, depth, mask, init_mode, pcloud_tensor)

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
    def evaluate_particles(self, rbpf, cls_id, autoencoder, codes_gpu, codepose, block_index,
                           image, intrinsics, uv, z, render_dist, gaussian_std,
                           depth=None, mask=None, init_mode=False, pcloud_tensor=None):

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
        cosine_distance_matrix = autoencoder.module.pairwise_cosine_distances(embeddings.detach(), codes_gpu)

        if block_index is not None:
            index = np.where(block_index == 1)[0]
            cosine_distance_matrix[:, index] = 1e-8

        # get the maximum similarity for each particle
        v_sims, i_sims = torch.max(cosine_distance_matrix, dim=1)

        # evaluate particles with depth images
        if depth is not None:
            depth_gpu = depth.unsqueeze(2)
            if mask is not None:
                mask_gpu = mask.unsqueeze(2)
            else:
                mask_gpu = None
            if init_mode:
                depth_scores = self.evaluate_depths_init(cls_id,
                                                         depth=depth_gpu, uv=uv, z=z,
                                                         q_idx=i_sims.cpu().numpy(), intrinsics=intrinsics,
                                                         render_dist=render_dist, codepose=codepose,
                                                         delta=cfg.PF.DEPTH_DELTA,
                                                         tau=cfg.PF.DEPTH_DELTA,
                                                         mask=mask_gpu)
            else:
                depth_scores = self.evaluate_depths_tracking(rbpf,
                                                             depth=depth_gpu, uv=uv, z=z,
                                                             intrinsics=intrinsics,
                                                             render_dist=render_dist, codepose=codepose,
                                                             delta=cfg.PF.DEPTH_DELTA,
                                                             tau=cfg.PF.DEPTH_DELTA, 
                                                             mask=mask_gpu, pcloud_tensor=pcloud_tensor)

            # reshape the depth score
            if torch.max(depth_scores) > 0:
                depth_scores = depth_scores / torch.max(depth_scores)
                depth_scores = mat2pdf(depth_scores, 1.0, cfg.PF.DEPTH_STD)
            else:
                depth_scores = torch.ones_like(depth_scores)
                depth_scores /= torch.sum(depth_scores)

        # compute distribution from similarity
        max_sim_all = torch.max(v_sims)
        # cosine_distance_matrix[cosine_distance_matrix > 0.95 * max_sim_all] = max_sim_all
        pdf_matrix = mat2pdf(cosine_distance_matrix/max_sim_all, 1, gaussian_std)

        # combine RGB and D
        if depth is not None:
            pdf_matrix = torch.mul(pdf_matrix, depth_scores)

        return pdf_matrix, max_sim_all, out_images, images_roi_cuda


    # evaluate particles according to depth measurements
    def evaluate_depths_init(self, cls_id, depth, uv, z, q_idx, intrinsics, render_dist, codepose, delta=0.03, tau=0.05, mask=None):

        score = np.zeros_like(z)
        height = cfg.TRAIN.SYN_HEIGHT
        width = cfg.TRAIN.SYN_WIDTH
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        px = intrinsics[0, 2]
        py = intrinsics[1, 2]
        zfar = 10.0
        znear = 0.01
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

        # crop rois
        depth_roi_cuda, _ = trans_zoom_uvz_cuda(depth.detach(), uv, z, intrinsics[0, 0], intrinsics[1, 1], render_dist)
        depth_meas_all = depth_roi_cuda[:, 0, :, :]

        if mask is not None:
            mask_roi_cuda, _ = trans_zoom_uvz_cuda(mask.detach(), uv, z, intrinsics[0, 0], intrinsics[1, 1], render_dist)

        # render
        pose_v = np.zeros((7,))
        frame_cuda = torch.cuda.FloatTensor(height, width, 4)
        seg_cuda = torch.cuda.FloatTensor(height, width, 4)
        pc_cuda = torch.cuda.FloatTensor(height, width, 4)

        q_idx_unique, idx_inv = np.unique(q_idx, return_inverse=True)
        depth_render_all = torch.cuda.FloatTensor(q_idx.shape[0], 128, 128)
        depth_render_all_unique = torch.cuda.FloatTensor(q_idx_unique.shape[0], 128, 128)
        q_render = codepose[q_idx_unique, 3:]
        for i in range(q_render.shape[0]):
            pose_v[:3] = [0, 0, render_dist]
            pose_v[3:] = q_render[i]
            cfg.renderer.set_poses([pose_v])
            if cfg.TEST.SYNTHESIZE:
                cls_render = int(cfg.TEST.CLASSES[cls_id]) - 1
            else:
                cls_render = cls_id - 1
            cfg.renderer.render([cls_render], frame_cuda, seg_cuda, pc2_tensor=pc_cuda)
            render_roi_cuda, _ = trans_zoom_uvz_cuda(pc_cuda.flip(0),
                                                     np.array([[intrinsics[0, 2], intrinsics[1, 2], 1]]),
                                                     np.array([[render_dist]]),
                                                     intrinsics[0, 0], intrinsics[1, 1],
                                                     render_dist)
            depth_render_all_unique[i] = render_roi_cuda[0, 2, :, :]

        for i in range(q_idx.shape[0]):
            depth_render_all[i] = depth_render_all_unique[idx_inv[i]]

        # eval particles
        # shift the rendered image
        depth_invalid_mask = depth_render_all == 0
        depth_shift_np = z[:, 0] - render_dist
        depth_shift = torch.from_numpy(depth_shift_np).cuda().float().repeat(depth_render_all.size(1),
                                                                             depth_render_all.size(2),
                                                                             1).permute(2, 0, 1)
        depth_render_all += depth_shift
        depth_render_all[depth_invalid_mask] = 0

        # compute visibility mask
        if mask is None:
            mask_depth_meas = depth_meas_all > 0
            mask_depth_render = depth_render_all > 0
            mask_depth_vis = torch.abs(depth_render_all - depth_meas_all) < delta
            visibility_mask = mask_depth_meas * mask_depth_render * mask_depth_vis
        else:
            mask_label = mask_roi_cuda[:, 0, :, :] > 0
            mask_depth_meas = depth_meas_all > 0
            mask_depth_render = depth_render_all > 0
            visibility_mask = mask_depth_meas * mask_depth_render * mask_label

        # compute scores
        # depth errors
        depth_error = torch.ones_like(depth_render_all).cuda()
        depth_error[visibility_mask] = torch.abs(depth_meas_all[visibility_mask] -
                                                 depth_render_all[visibility_mask]) / tau
        depth_error = torch.clamp(depth_error, 0, 1.0)
        depth_error_mean = torch.mean(depth_error, (2, 1))

        # visible ratio
        total_pixels = torch.sum(depth_render_all > 0, (2, 1)).float()
        total_pixels[total_pixels == 0] = 10000
        vis_ratio = torch.sum(visibility_mask, (2, 1)).float() / total_pixels

        # scores
        score = (torch.ones_like(depth_error_mean) - depth_error_mean) * vis_ratio
        return score.unsqueeze(1)


    # evaluate particles according to depth measurements
    def evaluate_depths_tracking(self, rbpf, depth, uv, z, intrinsics, render_dist, codepose, delta=0.03, tau=0.05, mask=None, pcloud_tensor=None):

        # crop rois
        depth_roi_cuda, _ = trans_zoom_uvz_cuda(depth.detach(), uv, z, intrinsics[0, 0], intrinsics[1, 1], render_dist)
        depth_meas_all = depth_roi_cuda[:, 0, :, :]

        if mask is not None:
            mask_roi_cuda, _ = trans_zoom_uvz_cuda(mask.detach(), uv, z, intrinsics[0, 0], intrinsics[1, 1], render_dist)

        # crop render
        pose_v = np.zeros((7,))
        pose_v[:3] = rbpf.trans_bar
        pose_v[3:] = mat2quat(rbpf.rot_bar)

        uv_crop = project(rbpf.trans_bar, intrinsics)
        uv_crop = np.repeat(np.expand_dims(uv_crop, axis=0), uv.shape[0], axis=0)
        z_crop = np.ones_like(z) * rbpf.trans_bar[2]
        render_roi_cuda, _ = trans_zoom_uvz_cuda(pcloud_tensor, uv_crop, z_crop, intrinsics[0, 0], intrinsics[1, 1], render_dist)
        depth_render_all = render_roi_cuda[:, 2, :, :]

        # eval particles
        # shift the rendered image
        depth_invalid_mask = depth_render_all == 0
        depth_shift_np = z[:, 0] - rbpf.trans_bar[2]
        depth_shift = torch.from_numpy(depth_shift_np).cuda().float().repeat(depth_render_all.size(1),
                                                                             depth_render_all.size(2),
                                                                             1).permute(2, 0, 1)
        depth_render_all += depth_shift
        depth_render_all[depth_invalid_mask] = 0

        # compute visibility mask
        if mask is None:
            mask_depth_meas = depth_meas_all > 0
            mask_depth_render = depth_render_all > 0
            mask_depth_vis = torch.abs(depth_render_all - depth_meas_all) < delta
            visibility_mask = mask_depth_meas * mask_depth_render * mask_depth_vis
        else:
            mask_label = mask_roi_cuda[:, 0, :, :] > 0
            mask_depth_meas = depth_meas_all > 0
            mask_depth_render = depth_render_all > 0
            visibility_mask = mask_depth_meas * mask_depth_render * mask_label

        # compute scores
        # depth errors
        depth_error = torch.ones_like(depth_render_all).cuda()
        depth_error[visibility_mask] = torch.abs(depth_meas_all[visibility_mask] -
                                                 depth_render_all[visibility_mask]) / tau
        depth_error = torch.clamp(depth_error, 0, 1.0)
        depth_error_mean = torch.mean(depth_error, (2, 1))

        # visible ratio
        total_pixels = torch.sum(depth_render_all > 0, (2, 1)).float()
        total_pixels[total_pixels == 0] = 10000
        vis_ratio = torch.sum(visibility_mask, (2, 1)).float() / total_pixels

        # scores
        score = (torch.ones_like(depth_error_mean) - depth_error_mean) * vis_ratio
        return score.unsqueeze(1)


    # run SDF pose refine
    def refine_pose(self, index_sdf, im_depth, im_pcloud, intrinsics, dataset, im_label=None, steps=200):

        width = im_depth.shape[1]
        height = im_depth.shape[0]
        sdf_optim = cfg.sdf_optimizer
        # compare the depth
        depth_meas_roi = im_pcloud[:, :, 2]
        mask_depth_meas = depth_meas_roi > 0
        mask_depth_valid = torch.isfinite(depth_meas_roi)

        # prepare data
        num = len(index_sdf)
        T_oc_init = np.zeros((num, 4, 4), dtype=np.float32)
        cls_index = torch.cuda.FloatTensor(0, 1)
        obj_index = torch.cuda.FloatTensor(0, 1)
        pix_index = torch.cuda.LongTensor(0, 2)
        for i in range(num):

            # pose
            ind = index_sdf[i]
            pose = self.rbpfs[ind].pose
            T_co = np.eye(4, dtype=np.float32)
            T_co[:3, :3] = quat2mat(pose[:4])
            T_co[:3, 3] = pose[4:]
            T_oc_init[i] = np.linalg.inv(T_co)

            # filter out points far away
            z = float(pose[6])
            roi = self.rbpfs[ind].roi
            extent = 1.2 * np.mean(dataset._extents_test[int(roi[1]), :]) / 2
            mask_distance = torch.abs(depth_meas_roi - z) < extent
            
            # mask label
            roi = self.rbpfs[ind].roi
            cls = int(roi[1])
            cls_render = cls - 1
            w = roi[4] - roi[2]
            h = roi[5] - roi[3]
            x1 = max(int(roi[2] - w / 2), 0)
            y1 = max(int(roi[3] - h / 2), 0)
            x2 = min(int(roi[4] + w / 2), width - 1)
            y2 = min(int(roi[5] + h / 2), height - 1)
            if im_label is not None:
                labels = torch.zeros_like(im_label)
                labels[y1:y2, x1:x2] = im_label[y1:y2, x1:x2]
                cls_train = cfg.TRAIN.CLASSES.index(cfg.TEST.CLASSES[cls])
                mask_label = labels == cls_train
            else:
                mask_label = torch.zeros_like(mask_depth_meas)
                mask_label[y1:y2, x1:x2] = 1

            mask = mask_label * mask_depth_meas * mask_depth_valid * mask_distance
            index_p = torch.nonzero(mask)
            n = index_p.shape[0]

            if n > 100:
                pix_index = torch.cat((pix_index, index_p), dim=0)
                index = cls_render * torch.ones((n, 1), dtype=torch.float32, device=0)
                cls_index = torch.cat((cls_index, index), dim=0)
                index = i * torch.ones((n, 1), dtype=torch.float32, device=0)
                obj_index = torch.cat((obj_index, index), dim=0)
                print('sdf {} points for object {}, class {} {}'.format(n, i, cls_render, dataset._classes_test[cls_render+1]))
            else:
                print('sdf {} points for object {}, class {} {}, no refinement'.format(n, i, cls_render, dataset._classes_test[cls_render+1]))

            if cfg.TEST.VISUALIZE and n <= 100:
                fig = plt.figure()
                ax = fig.add_subplot(2, 3, 1)
                plt.imshow(mask_label.cpu().numpy())
                ax.set_title('mask label')
                ax = fig.add_subplot(2, 3, 2)
                plt.imshow(mask_depth_meas.cpu().numpy())
                ax.set_title('mask_depth_meas')
                ax = fig.add_subplot(2, 3, 3)
                plt.imshow(mask_depth_valid.cpu().numpy())
                ax.set_title('mask_depth_valid')
                ax = fig.add_subplot(2, 3, 4)
                plt.imshow(mask_distance.cpu().numpy())
                ax.set_title('mask_distance')
                print(extent, z)
                ax = fig.add_subplot(2, 3, 5)
                plt.imshow(depth_meas_roi.cpu().numpy())
                ax.set_title('depth')
                plt.show()

        # data
        n = pix_index.shape[0]
        print('sdf with {} points'.format(n))
        if n == 0:
            return
        points = im_pcloud[pix_index[:, 0], pix_index[:, 1], :]
        points = torch.cat((points, cls_index, obj_index), dim=1)
        T_oc_opt = sdf_optim.refine_pose_layer(T_oc_init, points, steps=steps)

        # update poses and bounding boxes
        for i in range(num):
            RT_opt = T_oc_opt[i]
            if RT_opt[3, 3] > 0:
                RT_opt = np.linalg.inv(RT_opt)
                ind = index_sdf[i]
                self.rbpfs[ind].pose[:4] = mat2quat(RT_opt[:3, :3])
                self.rbpfs[ind].pose[4:] = RT_opt[:3, 3]
                self.rbpfs[ind].roi[2:6] = self.compute_box(dataset, intrinsics, int(self.rbpfs[ind].roi[1]), self.rbpfs[ind].pose)
                self.rbpfs[ind].T_in_base = self.Tbc.dot(RT_opt)

        if cfg.TEST.VISUALIZE:

            points = points.cpu().numpy()
            for i in range(num):

                ind = index_sdf[i]
                roi = self.rbpfs[ind].roi
                cls = int(roi[1])
                T_co_init = np.linalg.inv(T_oc_init[i])

                pose = self.rbpfs[ind].pose
                T_co_opt = np.eye(4, dtype=np.float32)
                T_co_opt[:3, :3] = quat2mat(pose[:4])
                T_co_opt[:3, 3] = pose[4:]

                index = np.where(points[:, 4] == i)[0]
                if len(index) == 0:
                    continue
                pts = points[index, :4].copy()
                pts[:, 3] = 1.0

                # show points
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                points_obj = dataset._points_all_test[cls, :, :]
                points_init = np.matmul(np.linalg.inv(T_co_init), pts.transpose()).transpose()
                points_opt = np.matmul(np.linalg.inv(T_co_opt), pts.transpose()).transpose()

                ax.scatter(points_obj[::5, 0], points_obj[::5, 1], points_obj[::5, 2], color='yellow')
                ax.scatter(points_init[::5, 0], points_init[::5, 1], points_init[::5, 2], color='red')
                ax.scatter(points_opt[::5, 0], points_opt[::5, 1], points_opt[::5, 2], color='blue')

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax.set_xlim(sdf_optim.xmins[cls-1], sdf_optim.xmaxs[cls-1])
                ax.set_ylim(sdf_optim.ymins[cls-1], sdf_optim.ymaxs[cls-1])
                ax.set_zlim(sdf_optim.zmins[cls-1], sdf_optim.zmaxs[cls-1])
                ax.set_title(dataset._classes_test[cls])
                plt.show()


    # run SDF pose refine
    def refine_pose_one_by_one(self, index_sdf, im_depth, im_pcloud, intrinsics, dataset, im_label=None, steps=200):

        width = im_label.shape[1]
        height = im_label.shape[0]
        sdf_optim = cfg.sdf_optimizer
        # compare the depth
        depth_meas_roi = im_pcloud[:, :, 2]
        mask_depth_meas = depth_meas_roi > 0
        mask_depth_valid = torch.isfinite(depth_meas_roi)

        num = len(index_sdf)
        for i in range(num):

            # prepare data
            T_oc_init = np.zeros((1, 4, 4), dtype=np.float32)
            cls_index = torch.cuda.FloatTensor(0, 1)
            obj_index = torch.cuda.FloatTensor(0, 1)
            pix_index = torch.cuda.LongTensor(0, 2)

            # pose
            ind = index_sdf[i]
            pose = self.rbpfs[ind].pose
            T_co = np.eye(4, dtype=np.float32)
            T_co[:3, :3] = quat2mat(pose[:4])
            T_co[:3, 3] = pose[4:]
            T_oc_init[0] = np.linalg.inv(T_co)

            # filter out points far away
            z = float(pose[6])
            roi = self.rbpfs[ind].roi
            extent = 1.2 * np.mean(dataset._extents_test[int(roi[1]), :]) / 2
            mask_distance = torch.abs(depth_meas_roi - z) < extent
            
            # mask label
            if im_label is not None:
                roi = self.rbpfs[ind].roi
                cls = int(roi[1])
                cls_render = cls - 1
                w = roi[4] - roi[2]
                h = roi[5] - roi[3]
                x1 = max(int(roi[2] - w / 2), 0)
                y1 = max(int(roi[3] - h / 2), 0)
                x2 = min(int(roi[4] + w / 2), width - 1)
                y2 = min(int(roi[5] + h / 2), height - 1)
                labels = torch.zeros_like(im_label)
                labels[y1:y2, x1:x2] = im_label[y1:y2, x1:x2]
                cls_train = cfg.TRAIN.CLASSES.index(cfg.TEST.CLASSES[cls])
                mask_label = labels == cls_train
                mask = mask_label * mask_depth_meas * mask_depth_valid * mask_distance
            else:
                mask = mask_depth_meas * mask_depth_valid * mask_distance
            index_p = torch.nonzero(mask)
            n = index_p.shape[0]

            if n > 100:
                pix_index = torch.cat((pix_index, index_p), dim=0)
                index = cls_render * torch.ones((n, 1), dtype=torch.float32, device=0)
                cls_index = torch.cat((cls_index, index), dim=0)
                index = torch.zeros((n, 1), dtype=torch.float32, device=0)
                obj_index = torch.cat((obj_index, index), dim=0)
                print('sdf {} points for object {}, class {}'.format(n, i, cls_render))
            else:
                print('sdf {} points for object {}, class {}, no refinement'.format(n, i, cls_render))
                continue

            # data
            n = pix_index.shape[0]
            print('sdf with {} points'.format(n))
            points = im_pcloud[pix_index[:, 0], pix_index[:, 1], :]
            points = torch.cat((points, cls_index, obj_index), dim=1)
            T_oc_opt = sdf_optim.refine_pose_layer(T_oc_init, points, steps=steps)

            # update poses and bounding boxes
            RT_opt = T_oc_opt[0]
            if RT_opt[3, 3] > 0:
                RT_opt = np.linalg.inv(RT_opt)
                ind = index_sdf[i]
                self.rbpfs[ind].pose[:4] = mat2quat(RT_opt[:3, :3])
                self.rbpfs[ind].pose[4:] = RT_opt[:3, 3]
                self.rbpfs[ind].roi[2:6] = self.compute_box(dataset, intrinsics, int(self.rbpfs[ind].roi[1]), self.rbpfs[ind].pose)
                self.rbpfs[ind].T_in_base = self.Tbc.dot(RT_opt)

            if cfg.TEST.VISUALIZE:

                points = points.cpu().numpy()
                ind = index_sdf[i]
                roi = self.rbpfs[ind].roi
                cls = int(roi[1])
                T_co_init = np.linalg.inv(T_oc_init[0])

                pose = self.rbpfs[ind].pose
                T_co_opt = np.eye(4, dtype=np.float32)
                T_co_opt[:3, :3] = quat2mat(pose[:4])
                T_co_opt[:3, 3] = pose[4:]

                index = np.where(points[:, 4] == 0)[0]
                if len(index) == 0:
                    continue
                pts = points[index, :4].copy()
                pts[:, 3] = 1.0

                # show points
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                points_obj = dataset._points_all_test[cls, :, :]
                points_init = np.matmul(np.linalg.inv(T_co_init), pts.transpose()).transpose()
                points_opt = np.matmul(np.linalg.inv(T_co_opt), pts.transpose()).transpose()

                ax.scatter(points_obj[::5, 0], points_obj[::5, 1], points_obj[::5, 2], color='yellow')
                ax.scatter(points_init[::5, 0], points_init[::5, 1], points_init[::5, 2], color='red')
                ax.scatter(points_opt[::5, 0], points_opt[::5, 1], points_opt[::5, 2], color='blue')

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax.set_xlim(sdf_optim.xmins[cls-1], sdf_optim.xmaxs[cls-1])
                ax.set_ylim(sdf_optim.ymins[cls-1], sdf_optim.ymaxs[cls-1])
                ax.set_zlim(sdf_optim.zmins[cls-1], sdf_optim.zmaxs[cls-1])
                ax.set_title(dataset._classes_test[cls])
                plt.show()


    # evaluate a single 6D pose of a certain object
    def evaluate_6d_pose(self, roi, pose, cls, image_bgr, image_tensor, pcloud_tensor, image_depth, intrinsics, im_label=None, fast_mode=True):

        if fast_mode:
            sim = 1
        else:
            sim = 0
        depth_error = 1
        vis_ratio = 0
        height = image_depth.shape[0]
        width = image_depth.shape[1]

        if cls in cfg.TEST.CLASSES:

            cls_id = cfg.TEST.CLASSES.index(cls)

            if not fast_mode:
                autoencoder = self.autoencoders[cls_id]
                render_dist = self.codebooks[cls_id]['distance'][0, 0]
                t = pose[4:]
                uv = project(np.expand_dims(t, axis=1), intrinsics).transpose()
                z = np.array([t[2]], dtype=np.float32).reshape((1, 1))
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]

                # get roi
                rois, scale_roi = trans_zoom_uvz_cuda(image_bgr.detach(), uv, z, fx, fy, render_dist)

                # render object
                render_bgr = image_tensor[:, :, (2,1,0)]
                rois_render, scale_roi_render = trans_zoom_uvz_cuda(render_bgr.detach(), uv, z, fx, fy, render_dist)

                # forward passing
                embeddings = autoencoder.module.encode(torch.cat((rois, rois_render), dim=0)).detach()
                sim = self.cos_sim(embeddings[[0], :], embeddings[[1], :])[0].detach().cpu().numpy()

            # evaluate depth error
            depth_render = pcloud_tensor[:, :, 2]

            # compute visibility mask
            mask_depth_render = depth_render > 0
            mask_depth_valid = torch.isfinite(image_depth)
            mask_depth_meas = image_depth > 0
            mask_depth_vis = torch.abs(image_depth - depth_render) < 0.05

            # mask
            x1 = max(int(roi[2]), 0)
            y1 = max(int(roi[3]), 0)
            x2 = min(int(roi[4]), width-1)
            y2 = min(int(roi[5]), height-1)
            if im_label is not None:
                mask = torch.zeros_like(im_label)
                mask[y1:y2, x1:x2] = im_label[y1:y2, x1:x2]
                cls_train = cfg.TRAIN.CLASSES.index(cfg.TEST.CLASSES[cls_id])
                mask_label = mask == cls_train
            else:
                mask_label = torch.zeros_like(mask_depth_render)
                mask_label[y1:y2, x1:x2] = 1
            visibility_mask = mask_label * mask_depth_valid * mask_depth_meas * mask_depth_render * mask_depth_vis
            visibility_label = mask_label * mask_depth_valid * mask_depth_meas

            # print('cls_train', cls_train)
            # print('mask_label', torch.sum(mask_label).float())
            # print('visibility_mask', torch.sum(visibility_mask).float())
            # print('visibility_label', torch.sum(visibility_label).float())

            vis_ratio = torch.sum(visibility_mask).float() / torch.sum(visibility_label).float()
            depth_error = torch.mean(torch.abs(depth_render[visibility_mask] - image_depth[visibility_mask]))

        return sim, depth_error, vis_ratio



    # evaluate all the 6D poses
    def evaluate_6d_pose_all(self, image_bgr, image_tensor, pcloud_tensor, image_depth, intrinsics, im_label=None, fast_mode=True):

        height = image_depth.shape[0]
        width = image_depth.shape[1]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]

        # masks
        depth_render = pcloud_tensor[:, :, 2]
        mask_depth_render = depth_render > 0
        mask_depth_valid = torch.isfinite(image_depth)
        mask_depth_meas = image_depth > 0
        mask_depth_vis = torch.abs(image_depth - depth_render) < 0.05

        num = self.num_rbpfs
        if fast_mode:
            sim = np.ones((num, ), dtype=np.float32)
        else:
            render_bgr = image_tensor[:, :, (2,1,0)]
            sim = np.zeros((num, ), dtype=np.float32)
        depth_error = np.ones((num, ), dtype=np.float32)
        vis_ratio = np.zeros((num, ), dtype=np.float32)

        for i in range(num):

            roi = self.rbpfs[i].roi
            pose = self.rbpfs[i].pose
            cls = cfg.TEST.CLASSES[int(roi[1])]

            if cls in cfg.TEST.CLASSES:

                cls_id = cfg.TEST.CLASSES.index(cls)

                if not fast_mode:
                    autoencoder = self.autoencoders[cls_id]
                    render_dist = self.codebooks[cls_id]['distance'][0, 0]
                    t = pose[4:]
                    uv = project(np.expand_dims(t, axis=1), intrinsics).transpose()
                    z = np.array([t[2]], dtype=np.float32).reshape((1, 1))

                    # get roi
                    rois, scale_roi = trans_zoom_uvz_cuda(image_bgr.detach(), uv, z, fx, fy, render_dist)

                    # render object
                    rois_render, scale_roi_render = trans_zoom_uvz_cuda(render_bgr.detach(), uv, z, fx, fy, render_dist)

                    # forward passing
                    embeddings = autoencoder.module.encode(torch.cat((rois, rois_render), dim=0)).detach()
                    sim[i] = self.cos_sim(embeddings[[0], :], embeddings[[1], :])[0].detach().cpu().numpy()

                # mask
                x1 = max(int(roi[2]), 0)
                y1 = max(int(roi[3]), 0)
                x2 = min(int(roi[4]), width-1)
                y2 = min(int(roi[5]), height-1)
                if im_label is not None:
                    mask = torch.zeros_like(im_label)
                    mask[y1:y2, x1:x2] = im_label[y1:y2, x1:x2]
                    cls_train = cfg.TRAIN.CLASSES.index(cfg.TEST.CLASSES[cls_id])
                    mask_label = mask == cls_train
                else:
                    mask_label = torch.zeros_like(mask_depth_render)
                    mask_label[y1:y2, x1:x2] = 1

                # compute visibility mask
                visibility_mask = mask_label * mask_depth_valid * mask_depth_meas * mask_depth_render * mask_depth_vis
                visibility_label = mask_label * mask_depth_valid * mask_depth_meas

                vis_ratio[i] = torch.sum(visibility_mask).float() / torch.sum(visibility_label).float()
                depth_error[i] = torch.mean(torch.abs(depth_render[visibility_mask] - image_depth[visibility_mask]))

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


    def render_image(self, dataset, intrinsic_matrix, cls_render, pose):

        height = cfg.TRAIN.SYN_HEIGHT
        width = cfg.TRAIN.SYN_WIDTH
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        zfar = 10.0
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
 
        cls_indexes.append(cls_render)
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
        return im_output


    # compute bounding box by projection
    def compute_box(self, dataset, intrinsic_matrix, cls, pose):
        x3d = np.ones((4, dataset._points_all_test.shape[1]), dtype=np.float32)
        x3d[0, :] = dataset._points_all_test[cls,:,0]
        x3d[1, :] = dataset._points_all_test[cls,:,1]
        x3d[2, :] = dataset._points_all_test[cls,:,2]
        RT = np.zeros((3, 4), dtype=np.float32)
        RT[:3, :3] = quat2mat(pose[:4])
        RT[:, 3] = pose[4:]
        x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

        x1 = np.min(x2d[0, :])
        y1 = np.min(x2d[1, :])
        x2 = np.max(x2d[0, :])
        y2 = np.max(x2d[1, :])
        box = np.array([x1, y1, x2, y2])
        return box


    # render all the tracked objects
    def render_image_all(self, intrinsic_matrix, grasp_mode=False, grasp_cls=-1):

        height = cfg.TRAIN.SYN_HEIGHT
        width = cfg.TRAIN.SYN_WIDTH
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        zfar = 10.0
        znear = 0.01

        image_tensor = torch.cuda.FloatTensor(height, width, 4)
        seg_tensor = torch.cuda.FloatTensor(height, width, 4)
        pcloud_tensor = torch.cuda.FloatTensor(height, width, 4)

        # set renderer
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

        cls_indexes = []
        poses_all = []
        for i in range(self.num_rbpfs):
            rbpf = self.rbpfs[i]

            if grasp_mode and rbpf.cls_id != grasp_cls:
                continue

            if cfg.TEST.SYNTHESIZE:
                cls_index = int(cfg.TEST.CLASSES[rbpf.cls_id]) - 1
            else:
                cls_index = rbpf.cls_id - 1
            cls_indexes.append(cls_index)
            pose = rbpf.pose
            pose_render = np.zeros((7,), dtype=np.float32)
            pose_render[:3] = pose[4:]
            pose_render[3:] = pose[:4]
            poses_all.append(pose_render)

        # rendering
        cfg.renderer.set_poses(poses_all)
        cfg.renderer.render(cls_indexes, image_tensor, seg_tensor, pc2_tensor=pcloud_tensor)
        image_tensor = image_tensor[:, :, :3].flip(0)
        pcloud_tensor = pcloud_tensor[:, :, :3].flip(0)
        return image_tensor, pcloud_tensor


    # render all the tracked objects
    def render_poses_all(self, poses, rois, intrinsic_matrix):

        height = cfg.TRAIN.SYN_HEIGHT
        width = cfg.TRAIN.SYN_WIDTH
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        zfar = 10.0
        znear = 0.01

        im_output = np.zeros((height, width, 3), dtype=np.uint8)
        image_tensor = torch.cuda.FloatTensor(height, width, 4)
        seg_tensor = torch.cuda.FloatTensor(height, width, 4)
        pcloud_tensor = torch.cuda.FloatTensor(height, width, 4)

        # set renderer
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

        cls_indexes = []
        poses_all = []
        num = rois.shape[0]
        for i in range(num):
            cls = int(rois[i, 1])
            if cfg.TEST.SYNTHESIZE:
                cls_index = int(cfg.TEST.CLASSES[cls]) - 1
            else:
                cls_index = cls - 1
            cls_indexes.append(cls_index)
            pose = poses[i, :]
            pose_render = np.zeros((7,), dtype=np.float32)
            pose_render[:3] = pose[4:]
            pose_render[3:] = pose[:4]
            poses_all.append(pose_render)

        # rendering
        cfg.renderer.set_poses(poses_all)
        cfg.renderer.render(cls_indexes, image_tensor, seg_tensor, pc2_tensor=pcloud_tensor)
        image_tensor = image_tensor[:, :, :3].flip(0)
        pcloud_tensor = pcloud_tensor[:, :, :3].flip(0)
        return image_tensor, pcloud_tensor
