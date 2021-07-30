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
import tf.transformations as tra
from pose_listener import PoseListener
from joint_listener import JointListener
from moveit import MoveitBridge
from transforms3d.quaternions import mat2quat, quat2mat
from lula_franka.franka import Franka
from actions import open_cabinet, close_cabinet, pickup_object, place_object, compute_look_at_pose, move_to
from actions import turn_camera_towards_object
from omg_planner import OMGPlanner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tabletop', help='table top grasping', action="store_true")
    parser.add_argument('-i', '--index', dest='index_target', help='index target', default=0, type=int)
    parser.add_argument('-v', '--vis', help='visualization', action="store_true")
    parser.add_argument('-vc', '--vis_collision', help='visualization', action="store_true")
    args = parser.parse_args()
    return args


def retract(planner, moveit, traj, start_conf,  disable_list=[]):
    while 1:
        traj, flag_execute = planner.plan_to_conf(traj[-1, :], start_conf, disable_list=disable_list)
        if flag_execute:
            moveit.execute(traj)
            break
        else:
            print('plan failed, replaning')


if __name__ == '__main__':
    args = parse_args()
    tabletop = args.tabletop
    index_target = args.index_target
    rospy.init_node('kitchen_tasks')

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

    # save poses
    planner.save_data(joint_listener)

    # open cabinet
    handle_axis_distance = 0.34
    handle_turn_angle = 90
    Rx_angle = 40

    # close cabinet
    # close_cabinet('chewie_door_left_handle', handle_axis_distance, moveit, planner, pose_listener, joint_listener)
    # close_cabinet('chewie_door_right_handle', handle_axis_distance, moveit, planner, pose_listener, joint_listener)
    # sys.exit(1)

    if not tabletop:
        open_cabinet('chewie_door_left_handle', handle_axis_distance, handle_turn_angle, moveit, planner, pose_listener, joint_listener, Rx_angle)
        open_cabinet('chewie_door_right_handle', handle_axis_distance, handle_turn_angle-15, moveit, planner, pose_listener, joint_listener, Rx_angle)

        # look inside the cabinet
        pose_drawer = pose_listener.get_tf_pose('hitman_drawer_top')
        pose_drawer[2] += 1.0
        lookat_angle = 15
        lookat_distance = 0.5
        T_lookat = compute_look_at_pose(pose_listener, pose_drawer, angle=lookat_angle, distance=lookat_distance)
        move_to(T_lookat, planner, moveit, joint_listener, debug=False)
    else:
        lookat_angle = 45
        lookat_distance = 0.5

    # turn camera towards object
    model_names, object_poses, object_names = pose_listener.get_object_poses(block=True)
    target_name = object_names[index_target]
    psi = turn_camera_towards_object(object_poses[index_target], moveit, joint_listener)
    T_lookat = compute_look_at_pose(pose_listener, object_poses[index_target], angle=lookat_angle+5, distance=lookat_distance, psi=psi)

    # look at and pick up object
    while 1:
        move_to(T_lookat, planner, moveit, joint_listener, debug=False)
        model_names, object_poses, object_names = pose_listener.get_object_poses(block=True)
        if target_name not in object_names:
            continue
        print('openning gripper')
        moveit.open_gripper()

        # detect object
        print('detecting objects')
        rospy.sleep(0.5)
        model_names, object_poses, object_names = pose_listener.get_object_poses(block=True)

        # update planner
        planner.update_objects(object_names, model_names, object_poses)
        planner.save_data(joint_listener)

        # pick up object
        print('pick up target', target_name)
        traj, success = pickup_object(target_name, planner, moveit, joint_listener, debug=True)

        if success:
            break

    if not tabletop:
        # compute the place location on the counter
        pose_drawer = pose_listener.get_tf_pose('hitman_drawer_top')
        table_top = pose_drawer[:3].copy()
        table_top[2] += 0.21

        # move back from cabinet
        # cabinet to table top distance is 57cm
        translation = table_top.copy()
        translation[0] -= 0.4
        translation[2] += 0.7
        traj = place_object(target_name, translation, planner, moveit, joint_listener, is_delta=False, debug=True)

        # place on table
        translation = table_top.copy()
        translation[0] -= 0.1
        translation[2] += 0.2
        traj = place_object(target_name, translation, planner, moveit, joint_listener, is_delta=False, apply_standoff=True, debug=True)
        rospy.sleep(1.0)
        moveit.open_gripper()

        # retract
        retract(planner, moveit, traj, start_conf)

        # close cabinet
        close_cabinet('chewie_door_left_handle', handle_axis_distance, moveit, planner, pose_listener, joint_listener)
        close_cabinet('chewie_door_right_handle', handle_axis_distance, moveit, planner, pose_listener, joint_listener)
    else:
        # put object down
        T = moveit.forward_kinematics(traj[-1, :-2])
        T_put = T.copy()
        T_put[2, 3] += 0.01
        moveit.go_local(T_put, wait=True)
        moveit.open_gripper()
        T_lift = T.copy()
        T_lift[2, 3] += 0.2
        moveit.go_local(T_lift, wait=True)

        # retract
        joint_position = joint_listener.joint_position
        traj, flag_execute = planner.plan_to_conf(joint_position, start_conf)
        if flag_execute:
            moveit.execute(traj)
        else:
            print('plan failed, replaning')


    # Kill lula
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)
