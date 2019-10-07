from __future__ import print_function

import sys, os
import json
from time import time
import glob

import numpy as np
import cv2

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import message_filters

from Queue import Queue

import sys


def aggregate_depth(q, min_size=30, min_count=30):
    images = []
    masks = []
    list_q = list(q.queue)
    # import pdb; pdb.set_trace()
    if len(list_q) < min_size:
        return None, None

    for image in list_q:
        images.append(np.expand_dims(image, 0))
        masks.append((np.expand_dims(~np.isnan(image), 0)).astype(np.float32))

    masks = np.asarray(masks) #10 x height x width
    images = np.asarray(images)
    stds = np.squeeze(np.max(images, 0) - np.min(images, 0), 0)
    masks = np.sum(masks, 0) # 1 x heigth x width
    images = np.sum(images, 0)
    masks = np.squeeze(masks, 0) # height x width
    images = np.squeeze(images, 0) # height x width
    selection = (masks >= min_count) & (stds < 0.03)
    #print(np.unique(images[selection]))
    images[selection] = images[selection] / masks[selection]
    images[~selection] = np.nan
    ratio_valid = np.sum(selection).astype(np.float32) / (selection.shape[0]*selection.shape[1])
    rospy.loginfo('valid percentage =======> {}'.format(ratio_valid))
    if ratio_valid < 0.4:
        q.queue.clear()
        return None, None

    return images, stds


class DepthFilter:
    def __init__(self, queue_size=10, min_count=10, ns=''):
        self.last_stamp = None
        self.queue_size = queue_size
        self.min_count = min_count

        self.depth_buffer = Queue()
        self._cv_bridge = CvBridge()

        # camera info
        msg = rospy.wait_for_message(ns + '/rgb/camera_info', CameraInfo)
        rospy.loginfo('got camera info')
        rospy.loginfo(msg.K)
        self.intrinsics_matrix = np.asarray(msg.K).reshape(3, 3)

        # depth
        depth_sub = message_filters.Subscriber(ns + '/depth_to_rgb/image_raw', Image, queue_size=10)
        color_sub = message_filters.Subscriber(ns + '/rgb/image_raw', Image, queue_size=10)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self._decode_image)


        self.depth_pub = rospy.Publisher(ns + '/depth_to_rgb_filtered/image_raw', Image, queue_size = 2)
        self.color_pub = rospy.Publisher(ns + '/rgb_filtered/image_raw', Image, queue_size = 2)



    def _create_image_message(self, np_arr, stamp=None, encoding='32FC1'):
        msg = self._cv_bridge.cv2_to_imgmsg(np_arr)
        msg.header.stamp = stamp if stamp is not None else stamp
        msg.encoding = encoding

        return msg


    def _decode_image(self, color_data, depth_data):
        if self.depth_buffer.qsize() == self.queue_size:
            depth_filtered, stds = aggregate_depth(self.depth_buffer, min_size=self.queue_size, min_count=self.min_count)
            if depth_filtered is not None:
                stamp = rospy.Time.now()
                depth_filtered_msg = self._create_image_message(depth_filtered, stamp=stamp)
                rgb = self._cv_bridge.imgmsg_to_cv2(color_data)
                rgb_filtered_msg = self._create_image_message(rgb, stamp=stamp, encoding=color_data.encoding)
                self.depth_pub.publish(depth_filtered_msg)
                self.color_pub.publish(rgb_filtered_msg)

            self.depth_buffer.get()

        # self.last_color = self._cv_bridge.imgmsg_to_cv2(color_data).copy()
        # print(self.last_color.shape, self.last_color.dtype)
        self.last_stamp = depth_data.header.stamp
        depth = np.nan_to_num(self._cv_bridge.imgmsg_to_cv2(depth_data))
        self.depth_buffer.put(depth)

        # import pdb; pdb.set_trace()
        # print(depth.shape)
        # self.last_depth = depth.copy()
        #
        # float_depth = depth.copy()
        # float_depth = float_depth.astype(np.float32)
        # float_depth[float_depth < 4] = np.nan
        # float_depth /= 1000


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='depth_filter', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--queue_size', type=int, default=10)
    parser.add_argument('--min_count', type=int, default=10)
    parser.add_argument('--ns', type=str, default='')
    args = parser.parse_args(sys.argv[1:])

    rospy.init_node('depth_filter_azure')

    depth_filter = DepthFilter(queue_size=args.queue_size, min_count=args.min_count, ns=args.ns)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
