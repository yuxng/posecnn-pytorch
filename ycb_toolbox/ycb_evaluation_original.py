import sys
import os.path as osp
import numpy as np
import scipy.io
import json
from transforms3d.quaternions import mat2quat, quat2mat
from ycb_globals import ycb_video
from se3 import se3_mul
from pose_error import *

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', 'data', 'YCB_Video')
opt = ycb_video()

# your results
res_dir_posecnn = 'results_PoseCNN_RSS2018'
num_results = 2

# read keyframe index
filename = osp.join(root_path, 'keyframe.txt')
keyframes = []
with open(filename) as f:
    for x in f.readlines():
        index = x.rstrip('\n')
        keyframes.append(index)

# load model points
points = [[] for _ in range(len(opt.classes))]
for i in range(len(opt.classes)):
    point_file = osp.join(root_path, 'models', opt.classes[i], 'points.xyz')
    print point_file
    assert osp.exists(point_file), 'Path does not exist: {}'.format(point_file)
    points[i] = np.loadtxt(point_file)

# save results
num_max = 100000
distances_sys = np.zeros((num_max, num_results), dtype=np.float32)
distances_non = np.zeros((num_max, num_results), dtype=np.float32)
errors_rotation = np.zeros((num_max, num_results), dtype=np.float32)
errors_translation = np.zeros((num_max, num_results), dtype=np.float32)
results_seq_id = np.zeros((num_max, ), dtype=np.float32)
results_frame_id = np.zeros((num_max, ), dtype=np.float32)
results_object_id = np.zeros((num_max, ), dtype=np.float32)
results_cls_id = np.zeros((num_max, ), dtype=np.float32)

# for each image
count = -1
for i in range(len(keyframes)):
    
    # parse keyframe name
    name = keyframes[i]
    pos = name.find('/')
    seq_id = int(name[:pos])
    frame_id = int(name[pos+1:])

    # load PoseCNN result
    filename = osp.join(res_dir_posecnn, '%06d.mat' % i)
    print(filename)
    result_posecnn = scipy.io.loadmat(filename)

    # load gt poses
    filename = osp.join(root_path, 'data', '%04d/%06d-meta.mat' % (seq_id, frame_id))
    print(filename)
    gt = scipy.io.loadmat(filename)

    # for each gt poses
    cls_indexes = gt['cls_indexes'].flatten()
    for j in range(len(cls_indexes)):
        count += 1
        cls_index = cls_indexes[j]
        RT_gt = gt['poses'][:, :, j]

        results_seq_id[count] = seq_id
        results_frame_id[count] = frame_id
        results_object_id[count] = j
        results_cls_id[count] = cls_index

        # network result
        result = result_posecnn
        if len(result['rois']) > 0:
            roi_index = np.where(result['rois'][:, 1] == cls_index)[0]
        else:
            roi_index = []

        if len(roi_index) > 0:
            RT = np.zeros((3, 4), dtype=np.float32)

            # pose from network
            RT[:3, :3] = quat2mat(result['poses'][roi_index, :4].flatten())
            RT[:, 3] = result['poses'][roi_index, 4:]
            distances_sys[count, 0] = adi(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points[cls_index-1])
            distances_non[count, 0] = add(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points[cls_index-1])
            errors_rotation[count, 0] = re(RT[:3, :3], RT_gt[:3, :3])
            errors_translation[count, 0] = te(RT[:, 3], RT_gt[:, 3])

            # pose after icp
            RT[:3, :3] = quat2mat(result['poses_icp'][roi_index, :4].flatten())
            RT[:, 3] = result['poses_icp'][roi_index, 4:]
            distances_sys[count, 1] = adi(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points[cls_index-1])
            distances_non[count, 1] = add(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points[cls_index-1])
            errors_rotation[count, 1] = re(RT[:3, :3], RT_gt[:3, :3])
            errors_translation[count, 1] = te(RT[:, 3], RT_gt[:, 3])                   

        else:
            distances_sys[count, :] = np.inf
            distances_non[count, :] = np.inf
            errors_rotation[count, :] = np.inf
            errors_translation[count, :] = np.inf

distances_sys = distances_sys[:count+1, :]
distances_non = distances_non[:count+1, :]
errors_rotation = errors_rotation[:count+1, :]
errors_translation = errors_translation[:count+1, :]
results_seq_id = results_seq_id[:count+1]
results_frame_id = results_frame_id[:count+1]
results_object_id = results_object_id[:count+1]
results_cls_id = results_cls_id[:count+1]

results_all = {'distances_sys': distances_sys,
               'distances_non': distances_non,
               'errors_rotation': errors_rotation,
               'errors_translation': errors_translation,
               'results_seq_id': results_seq_id,
               'results_frame_id': results_frame_id,
               'results_object_id': results_object_id,
               'results_cls_id': results_cls_id }

scipy.io.savemat('results_posecnn.mat', results_all)
