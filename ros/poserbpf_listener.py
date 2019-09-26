import rospy
import tf
import message_filters
import cv2
import numpy as np
import torch
import torch.nn as nn
import threading
import sys
import scipy.io
import random
import time
import tf.transformations as tra
import matplotlib.pyplot as plt

from Queue import Queue
from random import shuffle
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial import distance_matrix as scipy_distance_matrix
from rospy_tutorials.srv import *
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROS_Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import JointState
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from scipy.optimize import minimize
from geometry_msgs.msg import PoseStamped, PoseArray

from fcn.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
from fcn.train_test import backproject
from video_recorder import *
from utils.cython_bbox import bbox_overlaps

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

class ImageListener:

    def __init__(self, pose_rbpf):

        print(' *** Initializing PoseRBPF ROS Node ... ')

        # variables
        self.cv_bridge = CvBridge()
        self.count = 0
        self.objects = []
        self.frame_names = []
        self.frame_lost = []
        self.num_lost = 50
        self.queue_size = 10
        self.scene = 1

        self.pose_rbpf = pose_rbpf
        self.dataset = pose_rbpf.dataset
        self.camera_type = cfg.TEST.ROS_CAMERA
        self.suffix = '_%02d' % (cfg.instance_id)
        self.prefix = '%02d_' % (cfg.instance_id)

        self.init_failure_steps = 0
        self.input_rgb = None
        self.input_depth = None
        self.input_seg = None
        self.input_rois = None
        self.input_stamp = None
        self.input_frame_id = None
        self.input_joint_states = None
        self.input_robot_joint_states = None
        self.main_thread_free = True
        self.kf_time_stamp = None

        # initialize a node
        rospy.init_node('poserbpf_image_listener')
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        rospy.sleep(3.0)
        self.pose_pub = rospy.Publisher('poserbpf_image', ROS_Image, queue_size=1)

        # publish poserbpf states
        self.status_pub = rospy.Publisher('poserbpf_status', numpy_msg(Floats), queue_size=10)
        self.poserbpf_ok_status = False

        # target detection
        self.flag_detected = False

        # subscriber for camera information
        if cfg.TEST.ROS_CAMERA == 'D415':
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=1)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.target_frame = 'measured/base_link'
            self.forward_kinematics = True
        elif cfg.TEST.ROS_CAMERA == 'Azure':
            rgb_sub = message_filters.Subscriber('/rgb/image_raw', Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/depth_to_rgb/image_raw', Image, queue_size=1)
            msg = rospy.wait_for_message('/rgb/camera_info', CameraInfo)
            self.target_frame = 'rgb_camera_link'
            self.forward_kinematics = False
        elif cfg.TEST.ROS_CAMERA == 'ISAAC_SIM':
            rgb_sub = message_filters.Subscriber('/sim/left_color_camera/image', Image, queue_size=2)
            depth_sub = message_filters.Subscriber('/sim/left_depth_camera/image', Image, queue_size=2)
            msg = rospy.wait_for_message('/sim/left_color_camera/camera_info', CameraInfo)
            self.target_frame = 'measured/base_link'
            self.forward_kinematics = True
        else:
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (cfg.TEST.ROS_CAMERA), Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (cfg.TEST.ROS_CAMERA), Image, queue_size=1)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (cfg.TEST.ROS_CAMERA), CameraInfo)
            self.forward_kinematics = False

        if self.forward_kinematics:
            # forward kinematics (base to camera link transformation)
            self.Tbr_now = np.eye(4, dtype=np.float32)
            self.Tbr_prev = np.eye(4, dtype=np.float32)
            self.Trc = np.load('./data/cameras/extrinsics_{}.npy'.format(self.camera_type))
            self.Tbc_now = np.eye(4, dtype=np.float32)

        K = np.array(msg.K).reshape(3, 3)
        self.intrinsic_matrix = K
        print('Intrinsics matrix : ')
        print(self.intrinsic_matrix)

        # set up ros service
        print(' PoseRBPF ROS Node is Initialized ! *** ')
        self.is_keyframe = False

        # subscriber for posecnn label
        label_sub = message_filters.Subscriber('/posecnn_label' + self.suffix, ROS_Image, queue_size=2)
        queue_size = 1
        slop_seconds = 0.5
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, label_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)

        # data saving directory
        self.Tbr_kf = np.eye(4, dtype=np.float32)  # keyframe which is used for refine the object pose
        self.Tbr_kf_list = []
        self.Tco_kf_list = []
        self.record = False
        self.Tbr_save = np.eye(4, dtype=np.float32)


    # callback function to get images
    def callback(self, rgb, depth, label):

        # decode image
        if depth is not None:
            if depth.encoding == '32FC1':
                depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
            elif depth.encoding == '16UC1':
                depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
                depth_cv /= 1000.0
            else:
                rospy.logerr_throttle(1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(depth.encoding))
                return
        else:
            depth_cv = None

        with lock:
            self.input_depth = depth_cv
            # rgb image used for posecnn detection
            self.input_rgb = self.cv_bridge.imgmsg_to_cv2(rgb, 'rgb8')
            # segmentation information from posecnn
            self.input_seg = self.cv_bridge.imgmsg_to_cv2(label, 'mono8')
            # other information
            self.input_stamp = rgb.header.stamp
            self.input_frame_id = rgb.header.frame_id


    def process_data(self):
        # callback data
        with lock:
            input_stamp = self.input_stamp
            input_rgb = self.input_rgb.copy()
            input_depth = self.input_depth.copy()
            input_seg = self.input_seg.copy()

        # subscribe the transformation
        if self.forward_kinematics:
            try:
                source_frame = 'measured/camera_color_optical_frame'
                target_frame = 'measured/base_link'
                trans, rot = self.listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
                Tbc = ros_qt_to_rt(rot, trans)
                if self.camera_type == 'D415':
                    T_delta = np.array([[0.99911077, 0.04145749, -0.00767817, -0.003222],
                                        [-0.04163608, 0.99882554, -0.02477858, -0.00289],
                                        [0.0066419, 0.02507623, 0.99966348, 0.003118],
                                        [0., 0., 0., 1.]], dtype=np.float32)
                    Tbc = Tbc.dot(T_delta)
                self.Tbc_now = Tbc.copy()
                self.Tbr_now = Tbc.dot(np.linalg.inv(self.Trc))
                if np.linalg.norm(self.Tbr_prev[:3, 3]) == 0:
                    self.pose_rbpf.T_c1c0 = np.eye(4, dtype=np.float32)
                else:
                    Tbc0 = np.matmul(self.Tbr_prev, self.Trc)
                    Tbc1 = np.matmul(self.Tbr_now, self.Trc)
                    self.pose_rbpf.T_c1c0 = np.matmul(np.linalg.inv(Tbc1), Tbc0)
                self.Tbr_prev = self.Tbr_now.copy()
            except:
                print('missing forward kinematics info')
                return

        # detection information of the target object
        rois_est = np.zeros((0, 6), dtype=np.float32)
        # TODO look for multiple object instances
        max_objects = 3
        for i in range(len(cfg.TEST.CLASSES)):

            for object_id in range(max_objects):
                suffix_frame = '_%02d_roi' % (object_id)

                # check posecnn frame
                ind = cfg.TEST.CLASSES[i]
                if self.dataset._classes_all[ind][3] == '_':
                    source_frame = 'posecnn/' + self.prefix + self.dataset._classes_all[ind][4:] + suffix_frame
                else:
                    source_frame = 'posecnn/' + self.prefix + self.dataset._classes_all[ind] + suffix_frame

                try:
                    # print('look for posecnn detection ' + source_frame)
                    trans, rot = self.listener.lookupTransform(self.target_frame, source_frame, rospy.Time(0))
                    n = trans[0]
                    secs = trans[1]
                    now = rospy.Time.now()
                    if abs(now.secs - secs) > 2.0:
                        print 'posecnn pose for %s time out %f %f' % (source_frame, now.secs, secs)
                        continue
                    roi = np.zeros((1, 6), dtype=np.float32)
                    roi[0, 0] = 0
                    roi[0, 1] = cfg.TRAIN.CLASSES.index(ind)
                    roi[0, 2] = rot[0] * n
                    roi[0, 3] = rot[1] * n
                    roi[0, 4] = rot[2] * n
                    roi[0, 5] = rot[3] * n
                    rois_est = np.concatenate((rois_est, roi), axis=0)
                    print('find posecnn detection ' + source_frame)
                except:
                    continue

        # call pose estimation function
        save = self.process_image_multi_obj(input_rgb, input_depth, input_seg, rois_est)

        # publish pose
        for i in range(self.pose_rbpf.num_rbpfs):
            Tco = np.eye(4, dtype=np.float32)
            Tco[:3, :3] = quat2mat(self.pose_rbpf.rbpfs[i].pose[:4])
            Tco[:3, 3] = self.pose_rbpf.rbpfs[i].pose[4:]
            if self.forward_kinematics:
                Tbo = self.Tbc_now.dot(Tco)
            else:
                Tbo = Tco.copy()
            # publish tf
            t_bo = Tbo[:3, 3]
            q_bo = mat2quat(Tbo[:3, :3])
            name = 'poserbpf/' + self.pose_rbpf.rbpfs[i].name
            self.br.sendTransform(t_bo, [q_bo[1], q_bo[2], q_bo[3], q_bo[0]], rospy.Time.now(), name, self.target_frame)

        # visualization
        image_disp = self.pose_rbpf.render_image_all(self.intrinsic_matrix)
        image_disp = image_disp.astype(np.uint8)
        image_disp = np.clip(image_disp, 0, 255)
        pose_msg = self.cv_bridge.cv2_to_imgmsg(image_disp)
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.input_frame_id
        pose_msg.encoding = 'rgb8'
        self.pose_pub.publish(pose_msg)


    # function for pose etimation and tracking
    def process_image_multi_obj(self, rgb, depth, mask, rois):
        image_rgb = rgb.astype(np.float32) / 255.0
        image_bgr = image_rgb[:, :, (2, 1, 0)]

        # data association based on bounding box overlap
        num_rbpfs = self.pose_rbpf.num_rbpfs
        num_rois = rois.shape[0]
        if num_rbpfs > 0 and num_rois > 0:
            rois_rbpf = np.zeros((num_rbpfs, 6), dtype=np.float32)
            for i in range(num_rbpfs):
                rois_rbpf[i, :] = self.pose_rbpf.rbpfs[i].roi.copy()

            # overlaps: (rois x gt_boxes) (batch_id, x1, y1, x2, y2)
            overlaps = bbox_overlaps(np.ascontiguousarray(rois_rbpf[:, (0, 2, 3, 4, 5)], dtype=np.float),
                np.ascontiguousarray(rois[:, (0, 2, 3, 4, 5)], dtype=np.float))

            # assignment
            assignment = overlaps.argmax(axis=1)
            overlaps_rbpf = overlaps.max(axis=1)
            overlaps_rois = overlaps.max(axis=0)

            for i in range(num_rbpfs):
                if overlaps_rbpf[i] > 0.5:
                    self.pose_rbpf.rbpfs[i].roi_assign = rois[assignment[i]]
                else:
                    self.pose_rbpf.rbpfs[i].roi_assign = None

        elif num_rbpfs == 0 and num_rois == 0:
            return False
        elif num_rois > 0:
            overlaps_rois = np.zeros((num_rois, ), dtype=np.float32)

        # backproject depth
        dpoints = backproject(depth, self.intrinsic_matrix)

        # initialize new object
        for i in range(num_rois):
            if overlaps_rois[i] > 0.2:
                continue
            roi = rois[i]
            print('Initializing detection {} ... '.format(i))
            print(roi)
            pose = self.pose_rbpf.Pose_Estimation_PRBPF(roi, self.intrinsic_matrix, image_bgr, depth, dpoints, mask)

            # pose evaluation
            cls = cfg.TRAIN.CLASSES[int(roi[1])]
            sim, depth_error, vis_ratio = self.pose_rbpf.evaluate_6d_pose(pose, cls, torch.from_numpy(image_bgr), depth, self.intrinsic_matrix, mask)
            print('Initialization : Object: {}, Sim obs: {:.2}, Depth Err: {:.3}, Vis Ratio: {:.2}'.format(i, sim, depth_error, vis_ratio))

            if sim < 0.4 or depth_error > 0.02 or vis_ratio < 0.3:
                print('is NOT initialized!')
                self.pose_rbpf.num_objects_per_class[self.pose_rbpf.rbpfs[-1].cls_test] -= 1
                del self.pose_rbpf.rbpfs[-1]
            else:
                print('is initialized!')
                self.pose_rbpf.rbpfs[-1].roi_assign = roi

        # filter all the objects
        print('Filtering objects')
        self.pose_rbpf.Filtering_PRBPF(self.intrinsic_matrix, image_bgr, depth, dpoints, mask)
        save = True

        return save
