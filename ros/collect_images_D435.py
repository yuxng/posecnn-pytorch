#!/usr/bin/env python

# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""collect images from Intel RealSense D435"""

import rospy
import message_filters
import cv2
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import datetime
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class ImageListener:

    def __init__(self):

        self.cv_bridge = CvBridge()
        self.count = 0

        # output dir
        this_dir = osp.dirname(__file__)
        self.outdir = osp.join(this_dir, '..', 'data', 'D435')
        if not osp.exists(self.outdir):
            os.mkdir(self.outdir)

        now = datetime.datetime.now()
        seq_name = "{:%m%dT%H%M%S}/".format(now)
        self.save_dir = osp.join(self.outdir, seq_name)
        if not osp.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # initialize a node
        rospy.init_node("image_listener")
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=2)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=2)

        queue_size = 1
        slop_seconds = 0.025
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)

    def callback(self, rgb, depth):
        if depth.encoding == '32FC1':
            depth_32 = self.cv_bridge.imgmsg_to_cv2(depth) * 1000
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        # write images
        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        filename = self.save_dir + '/%06d-color.png' % self.count
        if self.count % 10 == 0:
            cv2.imwrite(filename, im)
            print(filename)

        # filename = self.save_dir + '/%06d-depth.png' % self.count
        # cv2.imwrite(filename, depth_cv)
        # print(filename)

        self.count += 1


if __name__ == '__main__':

    # image listener
    listener = ImageListener()
    try:  
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
