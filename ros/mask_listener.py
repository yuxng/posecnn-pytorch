import time
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
import datetime
import tf.transformations as tra
import matplotlib.pyplot as plt
import posecnn_cuda

from Queue import Queue
from random import shuffle
from cv_bridge import CvBridge, CvBridgeError
from rospy_tutorials.srv import AddTwoInts
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray, Marker
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from scipy.optimize import minimize
from geometry_msgs.msg import PoseStamped, PoseArray

from fcn.config import cfg, cfg_from_file
from utils.nms import *

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
            stamp = rospy.Time.now()
            init_trans, init_rot = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            break
        except Exception as e:
            if first_time:
                print(str(e))
                first_time = False
            continue
    return ros_qt_to_rt(init_rot, init_trans), stamp


class ImageListener:

    def __init__(self, dataset):

        print(' *** Initializing depth mask ROS Node ... ')

        # variables
        self.cv_bridge = CvBridge()
        self.dataset = dataset

        self.camera_type = cfg.TEST.ROS_CAMERA
        self.suffix = '_%02d' % (cfg.instance_id)
        self.prefix = '%02d_' % (cfg.instance_id)

        self.input_depth = None
        self.input_seg = None
        self.input_rois = None
        self.input_stamp = None
        self.input_frame_id = None

        # initialize a node
        rospy.init_node('mask_image_listener' + self.suffix)
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        rospy.sleep(2.0)

        # subscriber for camera information
        self.base_frame = 'measured/base_link'
        if cfg.TEST.ROS_CAMERA == 'D415':
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=1)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.target_frame = self.base_frame
            self.camera_frame = 'measured/camera_color_optical_frame'

            '''
            self.T_delta = np.array([[0.99911077, 0.04145749, -0.00767817, 0.003222],  # -0.003222, -0.013222 (left plus and right minus)
                                     [-0.04163608, 0.99882554, -0.02477858, 0.01589],  # -0.00289, 0.01089 (close plus and far minus)
                                     [0.0066419, 0.02507623, 0.99966348, 0.003118],
                                     [0., 0., 0., 1.]], dtype=np.float32)
            '''
            self.T_delta = np.eye(4, dtype=np.float32)
            self.viz_pub = rospy.Publisher('/obj/mask_estimates/realsense', MarkerArray, queue_size=1)

        elif cfg.TEST.ROS_CAMERA == 'Azure':
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', Image, queue_size=1)
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)
            self.target_frame = self.base_frame
            self.camera_frame = 'rgb_camera_link'
            self.viz_pub = rospy.Publisher('/obj/mask_estimates/azure', MarkerArray, queue_size=1)
        elif cfg.TEST.ROS_CAMERA == 'ISAAC_SIM':
            depth_sub = message_filters.Subscriber('/sim/left_depth_camera/image', Image, queue_size=2)
            msg = rospy.wait_for_message('/sim/left_color_camera/camera_info', CameraInfo)
            self.target_frame = self.base_frame
        else:
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (cfg.TEST.ROS_CAMERA), Image, queue_size=1)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (cfg.TEST.ROS_CAMERA), CameraInfo)

        # camera to base transformation
        self.Tbc_now = np.eye(4, dtype=np.float32)

        K = np.array(msg.K).reshape(3, 3)
        self.intrinsic_matrix = K
        print('Intrinsics matrix : ')
        print(self.intrinsic_matrix)

        # set up ros service
        print(' Depth mask ROS Node is Initialized ! *** ')

        # subscriber for posecnn label
        label_sub = message_filters.Subscriber('/posecnn_label' + self.suffix, Image, queue_size=2)
        queue_size = 1
        slop_seconds = 0.5
        ts = message_filters.ApproximateTimeSynchronizer([depth_sub, label_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)


    # callback function to get images
    def callback(self, depth, label):

        self.Tbc_now, self.Tbc_stamp = get_relative_pose_from_tf(self.listener, self.camera_frame, self.base_frame)

        # decode image
        if depth is not None:
            if depth.encoding == '32FC1':
                depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
            elif depth.encoding == '16UC1':
                depth = self.cv_bridge.imgmsg_to_cv2(depth)
                depth_cv = depth.copy().astype(np.float32)
                depth_cv /= 1000.0
            else:
                rospy.logerr_throttle(1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(depth.encoding))
                return
        else:
            depth_cv = None

        with lock:
            self.input_depth = depth_cv
            self.input_seg = self.cv_bridge.imgmsg_to_cv2(label, 'mono8')
            self.input_stamp = label.header.stamp
            self.input_frame_id = label.header.frame_id


    def process_data(self):

        start_time = rospy.Time.now()

        # callback data
        with lock:
            input_stamp = self.input_stamp
            input_depth = self.input_depth.copy()
            input_seg = self.input_seg.copy()
            input_Tbc = self.Tbc_now.copy()
            input_Tbc_stamp = self.Tbc_stamp

        # detection information of the target object
        rois_est = np.zeros((0, 7), dtype=np.float32)
        # TODO look for multiple object instances
        max_objects = 5
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
                    if abs(now.secs - secs) > 1.0:
                        print 'posecnn pose for %s time out %f %f' % (source_frame, now.secs, secs)
                        continue
                    roi = np.zeros((1, 7), dtype=np.float32)
                    roi[0, 0] = 0
                    roi[0, 1] = cfg.TRAIN.CLASSES.index(ind)
                    roi[0, 2] = rot[0] * n
                    roi[0, 3] = rot[1] * n
                    roi[0, 4] = rot[2] * n
                    roi[0, 5] = rot[3] * n
                    roi[0, 6] = trans[2]
                    rois_est = np.concatenate((rois_est, roi), axis=0)
                    print('find posecnn detection ' + source_frame)
                except:
                    continue

        if rois_est.shape[0] > 0:
            # non-maximum suppression within class
            index = nms(rois_est, 0.2)
            rois_est = rois_est[index, :]

        # call mask computation function
        self.process_image_multi_obj(input_depth, input_seg, input_Tbc, rois_est)
        print('computing mask time %.6f' % (rospy.Time.now() - start_time).to_sec())


    # function for pose etimation and tracking
    def process_image_multi_obj(self, depth, im_label, Tbc, rois):

        # tranform to gpu
        Tbc = torch.from_numpy(Tbc).cuda().float()
        im_label = torch.from_numpy(im_label).cuda()
        width = im_label.shape[1]
        height = im_label.shape[0]

        # backproject depth
        depth = torch.from_numpy(depth).cuda()
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        px = self.intrinsic_matrix[0, 2]
        py = self.intrinsic_matrix[1, 2]
        im_pcloud = posecnn_cuda.backproject_forward(fx, fy, px, py, depth)[0]

        # compare the depth
        depth_meas_roi = im_pcloud[:, :, 2]
        mask_depth_meas = depth_meas_roi > 0
        mask_depth_valid = torch.isfinite(depth_meas_roi)

        # for each detection
        num_rois = rois.shape[0]
        now = rospy.Time.now()
        markers = []
        for i in range(num_rois):
            roi = rois[i]
            cls = int(roi[1])
            cls_name = self.dataset._classes_test[cls]
            w = roi[4] - roi[2]
            h = roi[5] - roi[3]
            x1 = max(int(roi[2]), 0)
            y1 = max(int(roi[3]), 0)
            x2 = min(int(roi[4]), width - 1)
            y2 = min(int(roi[5]), height - 1)
            labels = torch.zeros_like(im_label)
            labels[y1:y2, x1:x2] = im_label[y1:y2, x1:x2]
            mask_label = labels == cls
            mask = mask_label * mask_depth_meas * mask_depth_valid
            pix_index = torch.nonzero(mask)
            n = pix_index.shape[0]
            print('{} points for object {}'.format(n, cls_name))
            if n == 0:
                '''
                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(depth.cpu().numpy())
                ax.set_title('depth')

                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(im_label.cpu().numpy())
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))
                ax.set_title('label')
                plt.show()
                '''
                continue
            points = im_pcloud[pix_index[:, 0], pix_index[:, 1], :]

            # filter points
            m = torch.mean(points, dim=0, keepdim=True)
            mpoints = m.repeat(n, 1)
            distance = torch.norm(points - mpoints, dim=1)
            extent = np.mean(self.dataset._extents_test[cls, :])
            points = points[distance < 1.5 * extent, :]
            if points.shape[0] == 0:
                continue
            
            # transform points to base
            ones = torch.ones((points.shape[0], 1), dtype=torch.float32, device=0)
            points = torch.cat((points, ones), dim=1)
            points = torch.mm(Tbc, points.t())
            location = torch.mean(points[:3, :], dim=1).cpu().numpy()
            print('[%6s] mean:' % cls_name, location)

            # extend the location away from camera a bit
            c = Tbc[:3, 3].cpu().numpy()
            d = location - c
            d = d / np.linalg.norm(d)
            location = location + (extent / 2) * d

            # publish tf raw
            self.br.sendTransform(location, [0, 0, 0, 1], now, self.prefix + cls_name + '_raw', self.target_frame)

            # project location to base plane
            location[2] = extent / 2
            print('[%6s] mean on table:' % cls_name, location)

            # publish tf
            self.br.sendTransform(location, [0, 0, 0, 1], now, self.prefix + cls_name, self.target_frame)

            # publish marker
            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = now
            marker.id = cls
            marker.type = Marker.SPHERE;
            marker.action = Marker.ADD;
            marker.pose.position.x = location[0]
            marker.pose.position.y = location[1]
            marker.pose.position.z = location[2]
            marker.pose.orientation.x = 0.
            marker.pose.orientation.y = 0.
            marker.pose.orientation.z = 0.
            marker.pose.orientation.w = 1.
            marker.scale.x = .05
            marker.scale.y = .05
            marker.scale.z = .05

            if cfg.TEST.ROS_CAMERA == 'Azure':
                marker.color.a = .3
            elif cfg.TEST.ROS_CAMERA == 'D415':
                marker.color.a = 1.
            marker.color.r = self.dataset._class_colors_test[cls][0] / 255.0
            marker.color.g = self.dataset._class_colors_test[cls][1] / 255.0
            marker.color.b = self.dataset._class_colors_test[cls][2] / 255.0
            markers.append(marker)
        self.viz_pub.publish(MarkerArray(markers))
