#!/usr/bin/env python

import rospy
import tf
import tf2_ros
import _init_paths
import argparse
import signal
import pprint
import numpy as np
import os
import sys

from pose_listener import PoseListener
from joint_listener import JointListener
from moveit import MoveitBridge
from kitchen_domain import KitchenDomain
from lula_franka.franka import Franka
from omg_planner import OMGPlanner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tabletop', help='table top grasping', action="store_true")
    parser.add_argument('-i', '--index', dest='index_target', help='index target', default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tabletop = args.tabletop
    index_target = args.index_target
    rospy.init_node('kitchen_tasks_tamp')

    # create robot
    franka = Franka(is_physical_robot=True)
    moveit = MoveitBridge(franka)
    moveit.open_gripper()
    moveit.retract()

    joint_listener = JointListener()
    pose_listener = PoseListener('poserbpf/00/info')
    rospy.sleep(1.0)

    # omg planner
    start_conf = np.array([-0.05929, -1.6124, -0.197, -2.53, -0.0952, 1.678, 0.587, 0.0398, 0.0398])
    planner = OMGPlanner()
    np.random.seed()

    # query object pose
    model_names, object_poses, object_names = pose_listener.get_object_poses()
    flag_compute_grasp = [False] * len(model_names)

    # query kitchen pose
    kitchen_model_names, kitchen_object_poses, kitchen_object_names = pose_listener.get_kitchen_pose(translate_z=-0.03)
    model_names += kitchen_model_names
    object_poses += kitchen_object_poses
    object_names += kitchen_object_names
    flag_compute_grasp += [False] * len(kitchen_model_names)
    
    # create planner interface
    planner.interface(start_conf, object_names, model_names, object_poses, flag_compute_grasp)

    # create domain
    domain = KitchenDomain(pose_listener, planner, moveit, joint_listener)

    # open left door
    handle_name = 'chewie_door_left_handle'
    domain.operator_look_at_a_handle(handle_name)
    domain.operator_grasp_a_handle(handle_name)
    domain.operator_open_a_handle(handle_name)
    domain.operator_go_home()

    # open right door
    handle_name = 'chewie_door_right_handle'
    domain.operator_look_at_a_handle(handle_name)
    domain.operator_grasp_a_handle(handle_name)
    domain.operator_open_a_handle(handle_name)
    domain.operator_go_home()

    # grasp an object in the cabinet
    object_name = '004_sugar_box'
    domain.operator_look_inside_cabinet()
    domain.operator_grasp_an_object(object_name)

    # place object
    domain.operator_place_an_object_on_counter(object_name)
    domain.operator_go_home()

    # close left door
    handle_name = 'chewie_door_left_handle'
    domain.operator_look_at_a_handle(handle_name)
    domain.operator_grasp_a_handle(handle_name)
    domain.operator_close_a_handle(handle_name)
    domain.operator_go_home()

    # close right door
    handle_name = 'chewie_door_right_handle'
    domain.operator_look_at_a_handle(handle_name)
    domain.operator_grasp_a_handle(handle_name)
    domain.operator_close_a_handle(handle_name)
    domain.operator_go_home()

    # Kill lula
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)
