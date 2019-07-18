import os, sys
import torch
import scipy.io
import random
import networks
import numpy as np
from layers.roi_align import ROIAlign
from fcn.config import cfg
from utils.blob import add_noise_cuda
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.prbpf_utils import *

class PoseRBPF:

    def __init__(self, dataset):

        # prepare autoencoder and codebook
        autoencoders = [[] for i in range(len(cfg.TEST.CLASSES))]
        codebooks = [[] for i in range(len(cfg.TEST.CLASSES))]
        codes_gpu = [[] for i in range(len(cfg.TEST.CLASSES))]
        codebook_names = [[] for i in range(len(cfg.TEST.CLASSES))]

        for i in range(len(cfg.TEST.CLASSES)):
            ind = cfg.TEST.CLASSES[i]
            cls = dataset._classes_all[ind]

            filename = os.path.join('data', 'checkpoints', 'encoder_ycb_object_' + cls + '_epoch_200.checkpoint.pth')
            if os.path.exists(filename):
                autoencoder_data = torch.load(filename)
                autoencoders[i] = networks.__dict__['autoencoder'](1, 128, autoencoder_data).cuda(device=cfg.device)
                autoencoders[i] = torch.nn.DataParallel(autoencoders[i], device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
                print(filename)

                # load codebook
                filename = os.path.join('data', 'codebooks', 'codebook_ycb_encoder_test_' + cls + '.mat')
                codebook_names[i] = filename
                codebooks[i] = scipy.io.loadmat(filename)
                codes_gpu[i] = torch.from_numpy(codebooks[i]['codes']).cuda()
                print(filename)

        self.autoencoders = autoencoders
        self.codebooks = codebooks
        self.codebook_names = codebook_names
        self.codes_gpu = codes_gpu
        self.dataset = dataset
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    # initialize PoseRBPF
    '''
    image (height, width, 3) with values (0, 1)
    '''
    def initialize(self, image, uv_init, n_init_samples, cls, roi_w, roi_h):

        cls_id = cfg.TEST.CLASSES.index(cls)

        # network and codebook of the class
        autoencoder = self.autoencoders[cls_id]
        codebook = self.codebooks[cls_id]
        codes_gpu = self.codes_gpu[cls_id]
        pose = np.zeros((7,), dtype=np.float32)
        if not autoencoder or not codebook:
            return pose

        render_dist = codebook['distance']
        intrinsic_matrix = codebook['intrinsic_matrix']
        cfg.TRAIN.FU = intrinsic_matrix[0, 0]
        cfg.TRAIN.FV = intrinsic_matrix[1, 1]
        intrinsics = self.dataset._intrinsic_matrix

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
        z_init = (128 - 40) * render_dist / roi_size * intrinsics[0, 0] / cfg.TRAIN.FU
        z = np.random.uniform(0.9 * z_init, 1.1 * z_init, (n_init_samples, 1))

        # evaluate translation
        distribution, out_images, in_images = self.evaluate_particles(autoencoder, codebook, codes_gpu, image, intrinsics, uv_h, z, render_dist, 0.1)

        # find the max pdf from the distribution matrix
        index_star = self.arg_max_func(distribution)
        uv_star = uv_h[index_star[0], :]  # .copy()
        z_star = z[index_star[0], :]  # .copy()

        box_center = uv_star[:2]
        box_size = 128 * render_dist / z_star * intrinsics[0, 0] / cfg.TRAIN.FU

        pose[4:] = self.back_project(uv_star, intrinsics, z_star)
        pose[:4] = codebook['quaternions'][index_star[1], :]

        if cfg.TEST.VISUALIZE:
            if cfg.TEST.SYNTHESIZE:
                cls_render = cls - 1
            else:
                cls_render = cls_id
            im_render = self.render_image(self.dataset, cls_render, pose)
            self.visualize(image, im_render, out_images[index_star[0]], in_images[index_star[0]], box_center, box_size)

        return pose

    def project(self, ts, intrinsics):
        # input: ts: nx3,
        #        intrinsics: 3x3
        # output: uvs: nx3

        uvs = np.matmul(intrinsics, ts.transpose()).transpose()

        uvs /= np.repeat(uvs[:, [2]], 3, axis=1)

        return uvs

    def back_project(self, uv, intrinsics, z):
        # here uv is the homogeneous coordinates todo: maybe make this generic if time permits
        xyz = np.matmul(np.linalg.inv(intrinsics), np.transpose(uv))
        xyz = np.multiply(np.transpose(xyz), z)
        return xyz

    def mat2pdf(self, distance_matrix, mean, std):
        coeff = torch.ones_like(distance_matrix).cuda() * (1/(np.sqrt(2*np.pi) * std))
        mean = torch.ones_like(distance_matrix).cuda() * mean
        std = torch.ones_like(distance_matrix).cuda() * std
        pdf = coeff * torch.exp(- (distance_matrix - mean)**2 / (2 * std**2))
        return pdf

    def arg_max_func(self, input):
        index = (input == torch.max(input)).nonzero().detach()
        return index[0]

    def CropAndResizeFunction(self, image, rois):
        return ROIAlign((128, 128), 1.0, 0)(image, rois)

    def trans_zoom_uvz_cuda(self, image, uvs, zs, pf_fu, pf_fv, target_distance=2.5, out_size=128):
        image = image.permute(2, 0, 1).float().unsqueeze(0).cuda()

        bbox_u = target_distance * (1 / zs) / cfg.TRAIN.FU * pf_fu * out_size / image.size(3)
        bbox_u = torch.from_numpy(bbox_u).cuda().float().squeeze(1)
        bbox_v = target_distance * (1 / zs) / cfg.TRAIN.FV * pf_fv * out_size / image.size(2)
        bbox_v = torch.from_numpy(bbox_v).cuda().float().squeeze(1)

        center_uvs = torch.from_numpy(uvs).cuda().float()
        center_uvs[:, 0] /= image.size(3)
        center_uvs[:, 1] /= image.size(2)

        boxes = torch.zeros(center_uvs.size(0), 5).cuda()
        boxes[:, 1] = (center_uvs[:, 0] - bbox_u/2) * float(image.size(3))
        boxes[:, 2] = (center_uvs[:, 1] - bbox_v/2) * float(image.size(2))
        boxes[:, 3] = (center_uvs[:, 0] + bbox_u/2) * float(image.size(3))
        boxes[:, 4] = (center_uvs[:, 1] + bbox_v/2) * float(image.size(2))

        out = self.CropAndResizeFunction(image, boxes)
        uv_scale = target_distance * (1 / zs) / cfg.TRAIN.FU * pf_fu

        '''
        for i in range(out.shape[0]):
            roi = out[i].permute(1, 2, 0).cpu().numpy()
            roi = np.clip(roi, 0, 1)
            im = roi * 255
            im = im.astype(np.uint8)
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.imshow(im[:, :, (2, 1, 0)])
            plt.show()
        '''
        return out, uv_scale


    # evaluate particles according to the RGB images
    def evaluate_particles(self, autoencoder, codebook, codes_gpu, image, intrinsics, uv, z, render_dist, gaussian_std):

        # crop the rois from input image
        fu = intrinsics[0, 0]
        fv = intrinsics[1, 1]
        images_roi_cuda, scale_roi = self.trans_zoom_uvz_cuda(image.detach(), uv, z, fu, fv, render_dist)

        # forward passing
        out_images, embeddings = autoencoder(images_roi_cuda)

        # compute the similarity between particles' codes and the codebook
        cosine_distance_matrix = autoencoder.module.pairwise_cosine_distances(embeddings, codes_gpu)

        # get the maximum similarity for each particle
        v_sims, i_sims = torch.max(cosine_distance_matrix, dim=1)

        # compute distribution from similarity
        max_sim_all = torch.max(v_sims)

        # cosine_distance_matrix[cosine_distance_matrix > 0.95 * max_sim_all] = max_sim_all
        pdf_matrix = self.mat2pdf(cosine_distance_matrix/max_sim_all, 1, gaussian_std)

        return pdf_matrix, out_images, images_roi_cuda

    # evaluate a single 6D pose of a certain object
    def evaluate_6d_pose(self, pose, cls, image_rgb, image_depth, intrinsics):

        sim = 0
        depth_error = 1
        vis_ratio = 0

        if cls in cfg.TEST.CLASSES:

            cls_id = cfg.TEST.CLASSES.index(cls)

            render_dist = self.codebooks[cls_id]['distance']

            t = pose[4:]
            uv = self.project(np.expand_dims(t, axis=0), intrinsics)

            z = np.array([t[2]], dtype=np.float32)
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            px = intrinsics[0, 2]
            py = intrinsics[1, 2]

            # get roi
            rois, scale_roi = self.trans_zoom_uvz_cuda(image_rgb.detach(), uv, z, fx, fy, render_dist)

            # render object
            zfar = 6.0
            znear = 0.01
            width = image_rgb.shape[1]
            height = image_rgb.shape[0]
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
            rois_render, scale_roi_render = self.trans_zoom_uvz_cuda(render_bgr.detach(), uv, z, fx, fy, render_dist)

            # forward passing
            out_img, embeddings = self.autoencoders[cls_id](torch.cat((rois,rois_render), dim=0))
            embeddings = embeddings.detach()
            sim = self.cos_sim(embeddings[[0], :], embeddings[[1], :])[0].detach().cpu().numpy()

            # evaluate depth error
            depth_render = pcloud_tensor[:, :, 2].cpu().numpy()

            # compute visibility mask
            visibility_mask = estimate_visib_mask_numba(image_depth, depth_render, 0.02)

            vis_ratio = np.sum(visibility_mask) * 1.0 / np.sum(depth_render!=0)

            depth_error = np.mean(np.abs(depth_render[visibility_mask] - image_depth[visibility_mask]))

        return sim, depth_error, vis_ratio


    def visualize(self, image, im_render, im_output, im_input, box_center, box_size):

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

        plt.show()


    def render_image(self, dataset, cls, pose):

        intrinsic_matrix = dataset._intrinsic_matrix
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

        return im_output
