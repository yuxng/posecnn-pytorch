import os, sys
import torch
import scipy.io
import networks
import numpy as np
from layers.roi_align import ROIAlign
from fcn.config import cfg

class PoseRBPF:

    def __init__(self, dataset):

        # prepare autoencoder and codebook
        autoencoders = [[] for i in range(len(cfg.TRAIN.CLASSES))]
        codebooks = [[] for i in range(len(cfg.TRAIN.CLASSES))]
        codes_gpu = [[] for i in range(len(cfg.TRAIN.CLASSES))]
        for i in range(len(cfg.TRAIN.CLASSES)):
            ind = cfg.TRAIN.CLASSES[i]
            cls = dataset._classes_all[ind]

            filename = os.path.join('data', 'checkpoints', 'encoder_' + cls + '_epoch_200.checkpoint.pth')
            if os.path.exists(filename):
                autoencoder_data = torch.load(filename)
                autoencoders[i] = networks.__dict__['autoencoder'](1, 128, autoencoder_data).cuda(device=cfg.device)
                autoencoders[i] = torch.nn.DataParallel(autoencoders[i], device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
                print(filename)

                # load codebook
                filename = os.path.join('data', 'codebooks', 'codebook_' + cls + '.mat')
                codebooks[i] = scipy.io.loadmat(filename)
                codes_gpu[i] = torch.from_numpy(codebooks[i]['codes']).cuda()
                print(filename)

        self.autoencoders = autoencoders
        self.codebooks = codebooks
        self.codes_gpu = codes_gpu
        self.dataset = dataset

    # initialize PoseRBPF
    '''
    image (height, width, 3) with values (0, 1)
    '''
    def initialize(self, image, uv_init, n_init_samples, cls, roi_size):

        # network and codebook of the class
        autoencoder = self.autoencoders[cls]
        codebook = self.codebooks[cls]
        codes_gpu = self.codes_gpu[cls]
        pose = np.zeros((7,), dtype=np.float32)
        if not autoencoder or not codebook:
            return pose

        render_dist = codebook['distance']
        intrinsic_matrix = codebook['intrinsic_matrix']
        cfg.TRAIN.FU = intrinsic_matrix[0, 0]
        cfg.TRAIN.FV = intrinsic_matrix[1, 1]
        intrinsics = self.dataset._intrinsic_matrix

        # sample around the center of bounding box
        bound = 5
        uv_h = np.array([uv_init[0], uv_init[1], 1])
        uv_h = np.repeat(np.expand_dims(uv_h, axis=0), n_init_samples, axis=0)
        uv_h[:, :2] += np.random.uniform(-bound, bound, (n_init_samples, 2))
        uv_h[:, 0] = np.clip(uv_h[:, 0], 0, image.shape[1])
        uv_h[:, 1] = np.clip(uv_h[:, 1], 0, image.shape[0])

        # sample around z
        z_init = (128 - 40) * render_dist / roi_size * intrinsics[0, 0] / cfg.TRAIN.FU
        z = np.random.uniform(z_init - 0.2, z_init + 0.2, (n_init_samples, 1))

        # evaluate translation
        distribution, out_images = self.evaluate_particles(autoencoder, codebook, codes_gpu, image, intrinsics, uv_h, z, render_dist, 0.1)

        # find the max pdf from the distribution matrix
        index_star = self.arg_max_func(distribution)
        uv_star = uv_h[index_star[0], :]  # .copy()
        z_star = z[index_star[0], :]  # .copy()

        box_center = uv_star[:2]
        box_size = 128 * render_dist / z_star * intrinsics[0, 0] / cfg.TRAIN.FU

        pose[4:] = self.back_project(uv_star, intrinsics, z_star)
        pose[:4] = codebook['quaternions'][index_star[1], :]
        print(pose)

        im_render = self.render_image(self.dataset, cls, pose)
        self.visualize(image, im_render, out_images[index_star[0]], box_center, box_size)

        return pose


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

        return pdf_matrix, out_images


    def visualize(self, image, im_render, im_output, box_center, box_size):

        import matplotlib.pyplot as plt
        fig = plt.figure()
        # show image
        im = image.cpu().numpy()
        im = im * 255.0
        im = im [:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(im)
        ax.set_title('input')

        plt.plot(box_center[0], box_center[1], 'ro', markersize=5)
        x1 = box_center[0] - box_size / 2
        x2 = box_center[0] + box_size / 2
        y1 = box_center[1] - box_size / 2
        y2 = box_center[1] + box_size / 2
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

        # show output
        im = im_output.cpu().detach().numpy()
        im = np.clip(im, 0, 1)
        im = im.transpose((1, 2, 0)) * 255.0
        im = im [:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(1, 3, 2)
        plt.imshow(im)
        ax.set_title('reconstruction')

        # show output
        ax = fig.add_subplot(1, 3, 3)
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

        cls_index = cfg.TRAIN.CLASSES[cls] - 1
        cls_indexes.append(cls_index)
        pose_render = pose.copy()
        pose_render[:3] = pose[4:]
        pose_render[3:] = pose[:4]
        poses_all.append(pose_render)

        # rendering
        cfg.renderer.set_poses(poses_all)
        cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)

        im_render = image_tensor.cpu().numpy()
        im_render = np.clip(im_render, 0, 1)
        im_render = im_render[:, :, :3] * 255
        im_render = im_render.astype(np.uint8)
        im_output = im_render

        return im_output
