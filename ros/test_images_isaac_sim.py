#!/usr/bin/env python

# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a PoseCNN on images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import tf
import rosnode
import message_filters
import cv2
import torch.nn as nn
import threading

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import networks
import rospy
import _init_paths

from cv_bridge import CvBridge, CvBridgeError
from fcn.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
from datasets.factory import get_dataset
from ycb_renderer import YCBRenderer
from fcn.test_imageset import test_image, test_image_simple
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from scipy.optimize import minimize
from utils.blob import pad_im, chromatic_transform, add_noise
from geometry_msgs.msg import PoseStamped
from utils.se3 import *
from utils.nms import *
from Queue import Queue
from sdf.sdf_optimizer import *
from fcn.pose_rbpf import *
import matplotlib.pyplot as plt

lock = threading.Lock()

def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T

def get_relative_pose_from_tf(listener, source_frame, target_frame):
    first_time = True
    while True:
        try:
            init_trans, init_rot = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            break
        except Exception as e:
            if first_time:
                print(str(e))
                # first_time = False
            continue

    # print('got relative pose between {} and {}'.format(source_frame, target_frame))

    return ros_qt_to_rt(init_rot, init_trans)

class ImageListener:

    def __init__(self, network, dataset):

        self.net = network
        self.dataset = dataset
        self.cv_bridge = CvBridge()
        self.renders = dict()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.q_br = None
        self.t_br = None

        suffix = '_%02d' % (cfg.instance_id)
        prefix = '%02d_' % (cfg.instance_id)
        self.suffix = suffix
        self.prefix = prefix

        fusion_type = '_rgb_'
        if cfg.TRAIN.VERTEX_REG_DELTA:
            fusion_type = '_rgbd_'

        # initialize a node
        self.listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()
        rospy.init_node("posecnn_rgb")
        self.label_pub = rospy.Publisher('posecnn_label' + fusion_type + suffix, Image, queue_size=10)
        self.rgb_pub = rospy.Publisher('posecnn_rgb' + fusion_type + suffix, Image, queue_size=10)
        self.depth_pub = rospy.Publisher('posecnn_depth' + fusion_type + suffix, Image, queue_size=10)
        self.fk_pub = rospy.Publisher('posecnn/T_br', PoseStamped, queue_size=10)  # gripper in base

        # create pose publisher for each known object class
        self.pubs = []
        for i in range(1, self.dataset.num_classes):
            if self.dataset.classes[i][3] == '_':
                cls = prefix + self.dataset.classes[i][4:]
            else:
                cls = prefix + self.dataset.classes[i]
            cls = cls + fusion_type
            self.pubs.append(rospy.Publisher('/objects/prior_pose/' + cls, PoseStamped, queue_size=10))

        if cfg.TEST.ROS_CAMERA == 'ISAAC_SIM':
            # use RealSense D435
            rgb_sub = message_filters.Subscriber('/sim/left_color_camera/image', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/sim/left_depth_camera/image', Image, queue_size=10)

            # update camera intrinsics
            msg = rospy.wait_for_message('/sim/left_color_camera/camera_info', CameraInfo)

            K = np.array(msg.K).reshape(3, 3)
            dataset._intrinsic_matrix = K
            print(dataset._intrinsic_matrix)

            queue_size = 1
            slop_seconds = 0.1
            ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
            ts.registerCallback(self.callback_rgbd)

    def callback_rgbd(self, rgb, depth):

        Tbr = get_relative_pose_from_tf(self.listener, 'measured/camera_link', 'base_link')

        self.q_br = mat2quat(Tbr[:3, :3])
        self.t_br = Tbr[:3, 3]

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        if cfg.TEST.ROS_CAMERA == 'ISAAC_SIM':
            depth_cv = depth_cv

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        with lock:
            #print 'writing objects'
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp

    def run_network(self):

        with lock:
            if listener.im is None:
              return
            im = self.im.copy()
            depth_cv = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        fusion_type = '_rgb_'
        if cfg.TRAIN.VERTEX_REG_DELTA:
            fusion_type = '_rgbd_'

        rois, seg_im, poses = test_image_simple(self.net, self.dataset, im, depth_cv)

        # publish
        # publish segmentation mask
        label_msg = self.cv_bridge.cv2_to_imgmsg(seg_im.astype(np.uint8))
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'mono8'
        self.label_pub.publish(label_msg)
        # publish rgb
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(im, 'bgr8')
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.rgb_pub.publish(rgb_msg)
        # publish depth
        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_cv, '32FC1')
        depth_msg.header.stamp = rgb_frame_stamp
        depth_msg.header.frame_id = rgb_frame_id
        self.depth_pub.publish(depth_msg)
        # forward kinematics
        self.br.sendTransform(self.t_br, [self.q_br[1], self.q_br[2], self.q_br[3], self.q_br[0]],
                              rgb_frame_stamp, 'posecnn_camera_link', 'base_link')

        indexes = np.zeros((self.dataset.num_classes, ), dtype=np.int32)

        if not rois.shape[0]:
            return

        index = np.argsort(rois[:, 2])
        rois = rois[index, :]
        poses = poses[index, :]
        for i in range(rois.shape[0]):
            cls = int(rois[i, 1])
            if cls > 0 and rois[i, -1] > cfg.TEST.DET_THRESHOLD:
                if not np.any(poses[i, 4:]):
                    continue

                quat = [poses[i, 1], poses[i, 2], poses[i, 3], poses[i, 0]]
                if self.dataset.classes[cls][3] == '_':
                    name = self.prefix + self.dataset.classes[cls][4:]
                else:
                    name = self.prefix + self.dataset.classes[cls]
                name = name + fusion_type
                indexes[cls] += 1

                name = name + '_%02d' % (indexes[cls])
                tf_name = os.path.join("posecnn", name)

                # send another transformation as bounding box (mis-used)
                n = np.linalg.norm(rois[i, 2:6])
                x1 = rois[i, 2] / n
                y1 = rois[i, 3] / n
                x2 = rois[i, 4] / n
                y2 = rois[i, 5] / n
                self.br.sendTransform([n, rgb_frame_stamp.secs, 0], [x1, y1, x2, y2], rgb_frame_stamp, tf_name + '_roi', 'base_link')


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--instance', dest='instance_id', help='PoseCNN instance id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--cad', dest='cad_name',
                        help='name of the CAD file',
                        default=None, type=str)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.device = torch.device('cuda:{:d}'.format(0))
    print 'GPU device {:d}'.format(args.gpu_id)
    cfg.gpu_id = args.gpu_id
    cfg.instance_id = args.instance_id

    # dataset
    cfg.MODE = 'TEST'
    dataset = get_dataset(args.dataset_name)
    dataset._intrinsic_matrix = np.array([[554.2562584220408, 0.0, 320.0],
                                          [0.0, 554.2562584220408, 240.0],
                                          [0.0, 0.0, 1.0]],
                                         dtype=np.float32)

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[0]).cuda(device=cfg.device)
    cudnn.benchmark = True

    # image listener
    network.eval()
    listener = ImageListener(network, dataset)

    while not rospy.is_shutdown():
       listener.run_network()
