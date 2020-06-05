import numpy as np
from scipy.io import loadmat
import glob
import numpy.ma as ma
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

def depth2pc(depth, xmap, ymap, intrinsics):
    pt2 = depth
    pt0 = (xmap - intrinsics[0, 2]) * pt2 / intrinsics[0, 0]
    pt1 = (ymap - intrinsics[1, 2]) * pt2 / intrinsics[1, 1]

    mask_depth = ma.getmaskarray(ma.masked_greater(pt2, 0))
    mask = mask_depth

    choose = mask.flatten().nonzero()[0]

    pt2_valid = pt2.flatten()[choose][:, np.newaxis].astype(np.float32)
    pt0_valid = pt0.flatten()[choose][:, np.newaxis].astype(np.float32)
    pt1_valid = pt1.flatten()[choose][:, np.newaxis].astype(np.float32)

    ps_c = np.concatenate((pt0_valid, pt1_valid, pt2_valid, np.ones_like(pt0_valid)), axis=1)
    return ps_c

def load_depth(fn, factor_depth):
    depth = np.array(Image.open(fn), dtype=np.float32)
    cam_scale = factor_depth.astype(np.float32)
    depth = depth / cam_scale

    return depth

def transform_pts(Transformation, points):
    return Transformation.dot(points.T).T

def point_match_viz(ps_init, ps_refined, ps_target, n_plot=500):

    # downsample the points
    ds_ratio_init = int(ps_init.shape[0] * 1.0 / n_plot)
    ds_ratio_refined = int(ps_refined.shape[0] * 1.0 / n_plot)
    ds_ratio_target = int(ps_target.shape[0] * 1.0 / n_plot)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(ps_target[::ds_ratio_target, 0], ps_target[::ds_ratio_target, 1], ps_target[::ds_ratio_target, 2], color='green')
    ax.scatter(ps_init[::ds_ratio_init, 0], ps_init[::ds_ratio_init, 1], ps_init[::ds_ratio_init, 2], color='red')
    ax.scatter(ps_refined[::ds_ratio_refined, 0], ps_refined[::ds_ratio_refined, 1], ps_refined[::ds_ratio_refined, 2], color='blue')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    min_coor = -1
    max_coor = 1

    ax.set_xlim(min_coor, max_coor)
    ax.set_ylim(min_coor, max_coor)
    ax.set_zlim(min_coor, max_coor)

    plt.show()

if __name__ == '__main__':

    # data
    data_dir = './data/'
    depth_files = sorted(glob.glob(data_dir + '*depth.png'))
    meta_files = sorted(glob.glob(data_dir + '*mat'))

    idx_image_0 = 0
    idx_image_1 = 2

    ymap = np.array([[j for i in range(640)] for j in range(480)])
    xmap = np.array([[i for i in range(640)] for j in range(480)])

    info_0 = loadmat(meta_files[idx_image_0])
    info_1 = loadmat(meta_files[idx_image_1])

    intrinsics = info_0['intrinsic_matrix']
    factor_depth = info_0['factor_depth']

    depth_0 = load_depth(depth_files[idx_image_0], factor_depth)
    pc_0 = depth2pc(depth_0, xmap, ymap, intrinsics)
    T_b_c0 = info_0['poses']

    print(T_b_c0)

    depth_1 = load_depth(depth_files[idx_image_1], factor_depth)
    pc_1 = depth2pc(depth_1, xmap, ymap, intrinsics)
    T_b_c1 = info_1['poses']
    pc_0_in1 = transform_pts(np.linalg.inv(T_b_c1).dot(T_b_c0), pc_0)

    print(T_b_c1)

    center = np.mean(pc_1, axis=0)
    print(center)
    pc_0_in1_vis = pc_0_in1 - center
    pc_1_vis = pc_1 - center

    point_match_viz(pc_0_in1_vis, pc_0_in1_vis, pc_1_vis)
