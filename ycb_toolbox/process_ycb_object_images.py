import os
import torch
import cv2
import numpy as np
import glob
from transforms3d.quaternions import mat2quat, quat2mat
import _init_paths
from utils.se3 import *

classes = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
           '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
           '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
           '052_extra_large_clamp', '061_foam_brick')
is_show = 0


if __name__ == '__main__':

    model_path = '/capri/YCB_Object_Dataset'
    width = 640
    height = 480
    ratio = float(height) / float(width)

    # for each class
    for i in range(len(classes)):
        cls = classes[i]
        print(cls)

        outdir = os.path.join(model_path, 'data_processed', cls)
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        for k in range(2):
            if k == 0:
                data_dir = 'data_rgb'
                suffix = '_rgb'
            else:
                data_dir = 'data_rgbd'
                suffix = '_rgbd'

            # list images
            dirname = os.path.join(model_path, data_dir, cls, '*.jpg')
            files = glob.glob(dirname)

            # for each image
            for j in range(len(files)):
                filename = files[j]

                # read image
                im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                print(filename)

                # read mask
                head, name = os.path.split(filename)
                filename = os.path.join(model_path, data_dir, cls, 'masks', name[:-4] + '_mask.pbm')
                mask = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                print(filename)

                # fix maker mask errors
                if cls == '040_large_marker':                    
                    N, labels = cv2.connectedComponents(255 - mask)
                    num = np.zeros(N, dtype=np.int32)
                    for n in range(N):
                        index = np.where(labels == n)
                        num[n] = len(index[0])
                    num = num[1:]
                    ind = np.argmax(num) + 1
                    index = np.where(labels != ind)
                    mask[index[0], index[1]] = 255

                # process mask
                nz_y, nz_x = np.where(mask == 0)
                start_x = np.min(nz_x)
                end_x = np.max(nz_x)
                start_y = np.min(nz_y) - 10
                end_y = np.max(nz_y) + 10
                c_x = (start_x + end_x) * 0.5
                c_y = (start_y + end_y) * 0.5

                # mask region
                left_dist = c_x - start_x
                right_dist = end_x - c_x
                up_dist = c_y - start_y
                down_dist = end_y - c_y
                crop_height = np.max([ratio * right_dist, ratio * left_dist, up_dist, down_dist]) * 2
                crop_width = crop_height / ratio

                # affine transformation
                x1 = c_x - crop_width / 2
                x2 = c_x + crop_width / 2
                y1 = c_y - crop_height / 2
                y2 = c_y + crop_height / 2

                pts1 = np.float32([[x1, y1], [x1, y2], [x2, y1]])
                pts2 = np.float32([[0, 0], [0, height], [width, 0]])
                affine_matrix = cv2.getAffineTransform(pts1, pts2)
                im_final = cv2.warpAffine(im, affine_matrix, (width, height))
                mask_final = cv2.warpAffine(mask, affine_matrix, (width, height))

                # save image and mask
                if is_show == 0:
                    filename = os.path.join(outdir, name[:-4] + suffix + '.jpg')
                    cv2.imwrite(filename, im_final)
                    filename = os.path.join(outdir, name[:-4] + suffix + '_mask.pbm')
                    cv2.imwrite(filename, mask_final)
                else:
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    im = im.astype(np.uint8)
                    ax = fig.add_subplot(2, 2, 1)
                    plt.imshow(im[:, :, (2, 1, 0)])
                    ax.set_title('color')
                    ax = fig.add_subplot(2, 2, 2)
                    plt.imshow(mask)
                    ax.set_title('mask')
                    ax = fig.add_subplot(2, 2, 3)
                    plt.imshow(im_final[:, :, (2, 1, 0)])
                    ax.set_title('color final')
                    ax = fig.add_subplot(2, 2, 4)
                    plt.imshow(mask_final)
                    ax.set_title('mask final')
                    plt.show()
                    #break
