import sys, os
import json
from time import time
import glob

import numpy as np
import scipy
import scipy.io
import cv2

import mayavi.mlab as mlab

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import message_filters
import threading
import cv2

from queue import Queue
import matplotlib.pyplot as plt


class VideoRecorder:
    def __init__(self, sync_multi_view=False, sim=False):
        self._folder_name = None
        self._name = None
        self._fourcc = None
        self._out = None
        self._record_mode = 'stop'
        self._lock = threading.Lock()
        self._cv_bridge = CvBridge()
        self._sync_multi_view = sync_multi_view

        # todo: add pose estimation here
        if not sim:
            self._topic_rgb = '/camera/color/image_raw'
            self._topic_rgb2 = '/cam_2/color/image_raw'
        else:
            self._topic_rgb2 = '/sim/left_color_camera/image'
            self._topic_rgb = '/sim/right_color_camera/image'

        if sync_multi_view:
            rgb1_sub = message_filters.Subscriber(self._topic_rgb, Image, queue_size=10)
            rgb2_sub = message_filters.Subscriber(self._topic_rgb2, Image, queue_size=10)
            ts = message_filters.ApproximateTimeSynchronizer([rgb1_sub, rgb2_sub], 5, 0.1)
            ts.registerCallback(self.ros_callback_multi)
        else:
            rospy.Subscriber(self._topic_rgb, Image, self.ros_callback)
            print('subscribing to {}'.format(self._topic_rgb))

    def create_new_capture(self, folder_name, fps=30, size=(640, 480)):
        print('create new capture size {}'.format(size))
        self._folder_name = folder_name
        if not os.path.isdir(self._folder_name):
            os.makedirs(self._folder_name)

        self._fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        if not self._sync_multi_view:
            print('video capture ', os.path.join(self._folder_name, 'video_robot.mp4'))
            raw_input('press any key')
            self._out = cv2.VideoWriter(
                os.path.join(self._folder_name, 'video_robot.mp4'), self._fourcc, fps, size
            )
        else:
            self._out = []
            for s in ['robot', 'third_person']:
                video_path = os.path.join(self._folder_name, 'video_{}.mp4'.format(s))
                print('creating videowriter for {}'.format(video_path))
                self._out.append(
                    cv2.VideoWriter(video_path, self._fourcc, fps, size)
                )

        self._record_mode = 'ready'

    def stop_capture(self):
        with self._lock:
            print('............. stoping capture ................')
            self._record_mode = 'stop'
            if isinstance(self._out, list):
                self._out[0].release()
                self._out[1].release()
            else:
                self._out.release()

    def start_capture(self):

        with self._lock:
            if self._record_mode != 'ready':
                raise ValueError('capture is not ready yet!!!')

            self._record_mode = 'recording'
            print('..................... Recording video started ==> {}'.format(self._folder_name))

    def get_image_from_topic(self, topic_name, cv_bridge):
        rospy.loginfo('video recoder: waiting for message from {}'.format(topic_name))
        msg = rospy.wait_for_message(topic_name, Image)
        return cv_bridge.imgmsg_to_cv2(msg).copy()

    def ros_callback(self, rgb):
        with self._lock:
            if self._record_mode != 'recording':
                return
            #
            # print('callback')

            img = self._cv_bridge.imgmsg_to_cv2(rgb).copy()
            self._out.write(img[:, :, ::-1])

    def ros_callback_multi(self, rgb1_data, rgb2_data):
        # print('multi callback')
        with self._lock:
            if self._record_mode != 'recording':
                return
            img1 = self._cv_bridge.imgmsg_to_cv2(rgb1_data).copy()
            img2 = self._cv_bridge.imgmsg_to_cv2(rgb2_data).copy()
            self._out[0].write(img1[:, :, ::-1])
            self._out[1].write(img2[:, :, ::-1])
            # print('multi')


if __name__ == '__main__':
    rospy.init_node('video_recorder')
    recorder = VideoRecorder(sync_multi_view=True, sim=False)
    recorder.create_new_capture('videos/seq_test', size=(640, 480))
    recorder.start_capture()

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        rate.sleep()

    recorder.stop_capture()

