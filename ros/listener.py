import rospy
import tf
import rosnode
import message_filters
import cv2
import numpy as np
import torch
import torch.nn as nn
import threading
import sys
import os

from fcn.config import cfg
from fcn.test_imageset import test_image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from scipy.optimize import minimize
from utils.blob import pad_im, chromatic_transform, add_noise
from geometry_msgs.msg import PoseStamped
from ycb_renderer import YCBRenderer
from utils.se3 import *
from utils.nms import *

from posecnn_pytorch.msg import DetectionList, Detection, BBox

class ImageListener:

    def __init__(self, network, dataset):

        self.net = network
        self.dataset = dataset
        self.cv_bridge = CvBridge()
        self.count = 0
        self.renders = dict()

        suffix = '_%02d' % (cfg.instance_id)
        prefix = '%02d_' % (cfg.instance_id)
        self.suffix = suffix
        self.prefix = prefix

        # initialize a node
        rospy.init_node('posecnn_image_listener' + suffix)
        self.br = tf.TransformBroadcaster()
        self.label_pub = rospy.Publisher('posecnn_label' + suffix, Image, queue_size=1)
        self.pose_pub = rospy.Publisher('posecnn_pose' + suffix, Image, queue_size=1)
        self.detection_pub = rospy.Publisher('posecnn/%02d/info' % cfg.instance_id, DetectionList, queue_size=1)

        # create pose publisher for each known object class
        self.pubs = []
        for i in range(1, self.dataset.num_classes):
            if self.dataset.classes[i][3] == '_':
                cls = prefix + self.dataset.classes[i][4:]
            else:
                cls = prefix + self.dataset.classes[i]
            self.pubs.append(rospy.Publisher('/objects/prior_pose/' + cls, PoseStamped, queue_size=1))

        if cfg.TEST.ROS_CAMERA == 'logitech':

            # subscribe rgb image
            rospy.Subscriber('/logitech/image_color', Image, self.callback_rgb)

        elif cfg.TEST.ROS_CAMERA == 'D435':
            # use RealSense D435
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=2)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=2)

            # update camera intrinsics
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)

            K = np.array(msg.K).reshape(3, 3)
            dataset._intrinsic_matrix = K
            print(dataset._intrinsic_matrix)

            queue_size = 1
            slop_seconds = 0.1
            ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
            ts.registerCallback(self.callback_rgbd)

        if cfg.TEST.POSE_REFINE or cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':

            # use kinect
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (cfg.TEST.ROS_CAMERA), Image, queue_size=2)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (cfg.TEST.ROS_CAMERA), Image, queue_size=2)

            # update camera intrinsics
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (cfg.TEST.ROS_CAMERA), CameraInfo)

            K = np.array(msg.K).reshape(3, 3)
            dataset._intrinsic_matrix = K
            print(dataset._intrinsic_matrix)

            queue_size = 1
            slop_seconds = 0.1
            ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
            ts.registerCallback(self.callback_rgbd)

        else:

            rospy.Subscriber('/%s/rgb/image_color' % (cfg.TEST.ROS_CAMERA), Image, self.callback_rgb)

            # update camera intrinsics
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (cfg.TEST.ROS_CAMERA), CameraInfo)

            K = np.array(msg.K).reshape(3, 3)
            dataset._intrinsic_matrix = K
            print(dataset._intrinsic_matrix)


    def callback_rgb(self, rgb):

        thread_name = threading.current_thread().name
        if not thread_name in self.renders:
            print(thread_name)
            self.renders[thread_name] = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, gpu_id=cfg.gpu_id, render_marker=True)
            self.renders[thread_name].load_objects(self.dataset.model_mesh_paths_target,
                                                   self.dataset.model_texture_paths_target,
                                                   self.dataset.model_colors_target)
            self.renders[thread_name].set_camera_default()
            self.renders[thread_name].set_light_pos([0, 0, 0])
            self.renders[thread_name].set_light_color([1, 1, 1])
            print self.dataset.model_mesh_paths_target
        cfg.renderer = self.renders[thread_name]

        # run network
        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        # resize image if necessary
        height = im.shape[0]
        width = im.shape[1]
        if height != cfg.TRAIN.SYN_HEIGHT or width != cfg.TRAIN.SYN_WIDTH:
            im = cv2.resize(im, (cfg.TRAIN.SYN_WIDTH, cfg.TRAIN.SYN_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # run the network
        im_pose, im_label, rois, poses = test_image(self.net, self.dataset, im)

        # publish
        label_msg = self.cv_bridge.cv2_to_imgmsg(im_label)
        label_msg.header.stamp = rospy.Time.now()
        label_msg.header.frame_id = rgb.header.frame_id
        label_msg.encoding = 'rgb8'
        self.label_pub.publish(label_msg)

        pose_msg = self.cv_bridge.cv2_to_imgmsg(im_pose)
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = rgb.header.frame_id
        pose_msg.encoding = 'rgb8'
        self.pose_pub.publish(pose_msg)

        # poses
        if cfg.TEST.ROS_CAMERA == 'D435':
            frame = 'camera_depth_optical_frame'
        else:
            frame = '%s_depth_optical_frame' % (cfg.TEST.ROS_CAMERA)

        indexes = np.zeros((self.dataset.num_classes, ), dtype=np.int32)
        index = np.argsort(rois[:, 2])
        rois = rois[index, :]
        poses = poses[index, :]

        detections = DetectionList()

        for i in range(rois.shape[0]):
            cls = int(rois[i, 1])
            if cls > 0 and rois[i, -1] > cfg.TEST.DET_THRESHOLD:
                quat = [poses[i, 1], poses[i, 2], poses[i, 3], poses[i, 0]]
                name = self.dataset.classes[cls] + '_%02d' % (indexes[cls])
                indexes[cls] += 1
                tf_name = os.path.join("posecnn", name)
                self.br.sendTransform(poses[i, 4:7], quat, rospy.Time.now(), tf_name, frame)

                # send another transformation as bounding box (mis-used)
                n = np.linalg.norm(rois[i, 2:6])
                x1 = rois[i, 2] / n
                y1 = rois[i, 3] / n
                x2 = rois[i, 4] / n
                y2 = rois[i, 5] / n
                now = rospy.Time.now()
                self.br.sendTransform([n, now.secs, 0], [x1, y1, x2, y2], now, tf_name + '_roi', frame)
		
                # create pose msg
                msg = PoseStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = frame
                msg.pose.orientation.x = poses[i, 1]
                msg.pose.orientation.y = poses[i, 2]
                msg.pose.orientation.z = poses[i, 3]
                msg.pose.orientation.w = poses[i, 0]
                msg.pose.position.x = poses[i, 4]
                msg.pose.position.y = poses[i, 5]
                msg.pose.position.z = poses[i, 6]
                pub = self.pubs[cls - 1]
                pub.publish(msg)

                detection = Detection()
                detection.name = name
                detection.score = rois[i, 6]
                detection.roi.x1 = x1
                detection.roi.y1 = y1
                detection.roi.x2 = x2
                detection.roi.y2 = y2
                detection.pose = msg
                detections.detections.append(detection)

        self.detection_pub.publish(detections)


    def callback_rgbd(self, rgb, depth):

        thread_name = threading.current_thread().name
        if not thread_name in self.renders:
            print(thread_name)
            self.renders[thread_name] = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, gpu_id=cfg.gpu_id, render_marker=True)
            self.renders[thread_name].load_objects(self.dataset.model_mesh_paths_target,
                                                   self.dataset.model_texture_paths_target,
                                                   self.dataset.model_colors_target)
            self.renders[thread_name].set_camera_default()
            self.renders[thread_name].set_light_pos([0, 0, 0])
            self.renders[thread_name].set_light_color([1, 1, 1])
            print self.dataset.model_mesh_paths_target
        cfg.renderer = self.renders[thread_name]

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        if cfg.TEST.ROS_CAMERA == 'D435':
            depth_cv = depth_cv/1000.0

        # run network
        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        im_pose, im_label, rois, poses = test_image(self.net, self.dataset, im, depth_cv)

        # publish
        label_msg = self.cv_bridge.cv2_to_imgmsg(im_label)
        label_msg.header.stamp = rospy.Time.now()
        label_msg.header.frame_id = rgb.header.frame_id
        label_msg.encoding = 'rgb8'
        self.label_pub.publish(label_msg)

        pose_msg = self.cv_bridge.cv2_to_imgmsg(im_pose)
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = rgb.header.frame_id
        pose_msg.encoding = 'rgb8'
        self.pose_pub.publish(pose_msg)

        # poses
        if cfg.TEST.ROS_CAMERA == 'D435':
            frame = 'camera_depth_optical_frame'
        else:
            frame = '%s_depth_optical_frame' % (cfg.TEST.ROS_CAMERA)

        detections = DetectionList()

        indexes = np.zeros((self.dataset.num_classes, ), dtype=np.int32)
        index = np.argsort(rois[:, 2])
        rois = rois[index, :]
        poses = poses[index, :]
        for i in range(rois.shape[0]):
            cls = int(rois[i, 1])
            if cls > 0 and rois[i, -1] > cfg.TEST.DET_THRESHOLD:
                quat = [poses[i, 1], poses[i, 2], poses[i, 3], poses[i, 0]]
                # name = self.dataset.classes[cls] + '_%02d' % (indexes[cls])
                if self.dataset.classes[cls][3] == '_':
                    name = self.prefix + self.dataset.classes[cls][4:]
                else:
                    name = self.prefix + self.dataset.classes[cls]

                indexes[cls] += 1

                tf_name = os.path.join("posecnn", name)
                self.br.sendTransform(poses[i, 4:7], quat, rospy.Time.now(), tf_name, frame)

                # send another transformation as bounding box (mis-used)
                n = np.linalg.norm(rois[i, 2:6])
                x1 = rois[i, 2] / n
                y1 = rois[i, 3] / n
                x2 = rois[i, 4] / n
                y2 = rois[i, 5] / n
                now = rospy.Time.now()
                self.br.sendTransform([n, now.secs, 0], [x1, y1, x2, y2], now, tf_name + '_roi', frame)

                # create pose msg
                msg = PoseStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = frame
                msg.pose.orientation.x = poses[i, 1]
                msg.pose.orientation.y = poses[i, 2]
                msg.pose.orientation.z = poses[i, 3]
                msg.pose.orientation.w = poses[i, 0]
                msg.pose.position.x = poses[i, 4]
                msg.pose.position.y = poses[i, 5]
                msg.pose.position.z = poses[i, 6]
                pub = self.pubs[cls - 1]
                pub.publish(msg)

                detection = Detection()
                detection.name = name
                detection.score = rois[i, 6]
                detection.roi.x1 = x1
                detection.roi.y1 = y1
                detection.roi.x2 = x2
                detection.roi.y2 = y2
                detection.pose = msg
                detections.detections.append(detection)

        self.detection_pub.publish(detections)
