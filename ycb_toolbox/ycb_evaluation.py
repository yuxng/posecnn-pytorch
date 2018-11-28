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
results_dirs = ['results_iser', 'drfat_yu_single', 'drfat_yu_single_30', 'dr_yu_single']
results_prefix = ['iser', 'lov', 'lov', 'lov']
num_results = len(results_dirs)
res_dir_posecnn = 'results_posecnn_drfat'

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

# load transforms
mdata = scipy.io.loadmat('transforms.mat')
transforms = mdata['transforms']

# save results
num_max = 100000
distances_sys = np.zeros((num_max, num_results + 1), dtype=np.float32)
distances_non = np.zeros((num_max, num_results + 1), dtype=np.float32)
errors_rotation = np.zeros((num_max, num_results + 1), dtype=np.float32)
errors_translation = np.zeros((num_max, num_results + 1), dtype=np.float32)
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
    
    results = []
    for j in range(num_results):
            
        # load result
        filename = osp.join(results_dirs[j], '%s_%04d/%06d-color.json' % (results_prefix[j], seq_id, frame_id))
        print(filename)

        with open(filename) as f:
            data = json.load(f)
        results.append(data)

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

        # for each result
        for jj in range(num_results):
            result = results[jj]
            index_obj = -1
            for k in range(len(result['objects'])):
                cls = result['objects'][k]['class']
                if '16k' in cls or '16K' in cls:
                    cls = cls[:-4]

                cls_index_obj = [ind for ind, item in enumerate(opt.classes) if item == cls]
                if len(cls_index_obj) > 0 and cls_index_obj[0] == cls_index-1:
                    index_obj = k
                    break

            if index_obj >= 0:
                q = np.zeros((4, ), dtype=np.float32)
                q[0] = result['objects'][index_obj]['quaternion_xyzw'][3]
                q[1] = result['objects'][index_obj]['quaternion_xyzw'][0]
                q[2] = result['objects'][index_obj]['quaternion_xyzw'][1]
                q[3] = result['objects'][index_obj]['quaternion_xyzw'][2]

                t = np.zeros((3, ), dtype=np.float32)
                t[0] = result['objects'][index_obj]['location'][0]
                t[1] = result['objects'][index_obj]['location'][1]
                t[2] = result['objects'][index_obj]['location'][2]

                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = quat2mat(q)
                RT[:, 3] = t / 100
                RT = se3_mul(RT, transforms[cls_index-1][0] / 100)

                distances_sys[count, jj] = adi(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points[cls_index-1])
                distances_non[count, jj] = add(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points[cls_index-1])
                errors_rotation[count, jj] = re(RT[:3, :3], RT_gt[:3, :3])
                errors_translation[count, jj] = te(RT[:, 3], RT_gt[:, 3])
            else:
                distances_sys[count, jj] = np.inf
                distances_non[count, jj] = np.inf
                errors_rotation[count, jj] = np.inf
                errors_translation[count, jj] = np.inf

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
            distances_sys[count, num_results] = adi(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points[cls_index-1])
            distances_non[count, num_results] = add(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points[cls_index-1])
            errors_rotation[count, num_results] = re(RT[:3, :3], RT_gt[:3, :3])
            errors_translation[count, num_results] = te(RT[:, 3], RT_gt[:, 3])                   
        else:
            distances_sys[count, num_results] = np.inf
            distances_non[count, num_results] = np.inf
            errors_rotation[count, num_results] = np.inf
            errors_translation[count, num_results] = np.inf

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

scipy.io.savemat('results_comparison_all.mat', results_all)
