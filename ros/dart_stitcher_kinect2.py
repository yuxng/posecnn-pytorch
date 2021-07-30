#!/usr/bin/env python
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import rospy
import tf
import tf2_ros

if __name__ == '__main__':
    rospy.init_node('stitcher')
    listener = tf.TransformListener()
    broadcaster = tf.TransformBroadcaster()

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        try:
            trans, rot = listener.lookupTransform(
                "depth_camera_2",
                "01_base_link",
                rospy.Time(0))
            broadcaster.sendTransform(trans, rot, rospy.Time.now(),
                                      "00_base_link",
                                      "kinect2_rgb_optical_frame",)

            trans, rot = listener.lookupTransform(
                "depth_camera_2",
                "world",
                rospy.Time(0))
            broadcaster.sendTransform(trans, rot, rospy.Time.now(),
                                      "world",
                                      "kinect2_rgb_optical_frame",)

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Update failed... " + str(e))
        finally:
            rate.sleep()
