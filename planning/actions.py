#!/usr/bin/env python
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function
import rospy
import sys
import numpy as np
import _init_paths
import tf
import tf2_ros
from utils.se3 import rotation_z, rotation_x, rotation_y
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import mat2euler, quat2euler


# for debuging
def send_transform(T, name, base_frame='measured/base_link'):
    broadcaster = tf.TransformBroadcaster()
    while 1:
        qt = mat2quat(T[:3, :3])
        broadcaster.sendTransform(T[:3, 3], [qt[1], qt[2], qt[3], qt[0]], rospy.Time.now(), name, base_frame)
        rospy.sleep(0.1)


def move_to(T, planner, moveit, joint_listener, base_obstacle_weight=1.0, disable_list=[], start_conf=None, debug=False):
    if start_conf is None:
        start_conf = joint_listener.joint_position

    target_conf = moveit.ik(T, start_conf[:-2])
    if target_conf is None:
        print('No IK for the target location')
        print(T)
        return None

    target_conf = np.append(target_conf, start_conf[7:])
    traj, flag_execute = planner.plan_to_conf(start_conf, target_conf, base_obstacle_weight, disable_list)
    if debug:
        print('start conf', start_conf)
        print('target conf', target_conf)
        planner.scene.fast_debug_vis(collision_pt=False)

    if traj is not None:
        moveit.execute(traj)

    return traj


# query pose with name and update the object in the planner
def update_planner(planner, pose_listener, name):
    pose = pose_listener.get_tf_pose(name)
    if pose is None:
        print('cannot find tf:', name)
        return
    planner.scene.env.update_pose(name, pose)


def detect_cabinet_handle(pose_listener, pose_handle, handle_name):

    # query cabinent handle pose
    # assume at maximum 5 handles visible
    while 1:
        object_names = []
        object_poses = []
        distances = []
        for i in range(5):
            link_name = 'poserbpf/00_cabinet_handle_%02d' % (i)
            pose = pose_listener.get_tf_pose(link_name)
            if pose is not None and pose[2] > 0:
                object_poses.append(pose)
                object_names.append(link_name)
                distances.append(np.linalg.norm(pose[:3] - pose_handle[:3]))
            else:
                continue

        if len(object_names) == 0:
            print('no handle pose detected from hand cameara:', handle_name)
            rospy.sleep(1.0)
        else:
            break

    index = np.argmin(distances)
    pose_handle = object_poses[index]
    print(object_names[index])
    print(object_poses[index])
    print(distances)
    return pose_handle


def turn_camera_towards_object(object_pose, moveit, joint_listener):

    joint_position = joint_listener.joint_position
    T = moveit.forward_kinematics(joint_position[:-2])
    delta = object_pose[:3] - T[:3, 3]
    psi = np.arctan(delta[1] / delta[0]) * 180 / np.pi
    return psi


# compute look at pose according to object pose
def compute_look_at_pose(pose_listener, object_pose, angle, distance, psi=0):

    # find the hand camera to hand transformation
    pose_camera = pose_listener.get_tf_pose('measured/camera_color_optical_frame',
        base_frame='measured/panda_rightfingertip', is_matrix=True)
    if pose_camera is not None:
        pose_camera[:3, :3] = np.eye(3)
    else:
        print('cannot find camera to hand transformation')

    psi /= 57.3
    theta = angle / 57.3
    r = distance
    center_object = object_pose[:3]
    position_robot = center_object + np.array([-r * np.cos(theta) * np.cos(psi),
                                               -r * np.cos(theta) * np.sin(psi),
                                                r * np.sin(theta)], dtype=np.float32)
    Z_BG = center_object - position_robot
    Z_BG /= np.linalg.norm(Z_BG)
    Y_BG = np.array([-np.sin(psi), np.cos(psi), 0], dtype=np.float32)
    X_BG = np.cross(Y_BG, Z_BG)
    R_BG = np.zeros((3, 3), dtype=np.float32)
    R_BG[:, 0] = X_BG
    R_BG[:, 1] = Y_BG
    R_BG[:, 2] = Z_BG

    pose_robot = np.eye(4, dtype=np.float32)
    pose_robot[:3, 3] = position_robot
    pose_robot[:3, :3] = R_BG[:3, :3]

    # adjust for camera offset
    if pose_camera is not None:
        pose_robot = np.dot(pose_camera, pose_robot)
    return pose_robot


def compute_cabinet_target_pose(handle_name, T_handle, axis_point, handle_axis_distance, turn_angle, x_angle, z_angle, action='open'):

    if 'left' in handle_name:
        x = -handle_axis_distance * np.sin(turn_angle * np.pi / 180)
        y = -handle_axis_distance * np.cos(turn_angle * np.pi / 180)
        Rx = rotation_x(-x_angle)
        Rz = rotation_z(-z_angle)
    elif 'right' in handle_name:
        x = -handle_axis_distance * np.sin(turn_angle * np.pi / 180)
        y = handle_axis_distance * np.cos(turn_angle * np.pi / 180)
        Rx = rotation_x(x_angle)
        Rz = rotation_z(z_angle)

    target_point = np.zeros((3, ), dtype=np.float32)
    target_point[0] = axis_point[0] + x
    target_point[1] = axis_point[1] + y
    target_point[2] = axis_point[2]

    # target transformation
    T = T_handle.copy()
    T[:3, 3] = target_point
    if action == 'open':
        T[:3, :3] = np.matmul(Rx, np.matmul(Rz, T[:3, :3]))
    else:
        T[:3, :3] = np.matmul(Rz, np.matmul(Rx, T[:3, :3]))
    return T


# open cabinet give handle pose
def open_cabinet(handle_name, handle_axis_distance, handle_turn_angle, moveit, planner, pose_listener, joint_listener, Rx_angle=15):

    if 'left' in handle_name:
        door_name = 'chewie_door_left_link'
    elif 'right' in handle_name:
        door_name = 'chewie_door_right_link'
    else:
        print('unknown handle name', handle_name)
        return

    # grasp cabinet handle
    while 1:
        print('open cabinet with handle', handle_name)
        pose_handle = pose_listener.get_tf_pose(handle_name)
        if pose_handle is None:
            print('no handle pose detected for DART:', handle_name)
            return

        # check cabinet status
        euler = quat2euler(pose_handle[3:])
        if 'left' in handle_name:
            psi = -1 * (180 - euler[2] * 180 / np.pi)
        elif 'right' in handle_name:
            psi = 180 + euler[2] * 180 / np.pi
        open_degree = np.absolute(psi)
        if open_degree > 180:
            open_degree = 360 - open_degree
        print(handle_name, 'open degree', open_degree)

        if handle_turn_angle - open_degree < 20:
            print('cabinet %s is already open' % (handle_name))
            return

        # compute look at pose
        lookat_angle = open_degree * 0.5
        T_lookat = compute_look_at_pose(pose_listener, pose_handle, angle=-lookat_angle , distance=0.25, psi=psi)
        if 'left' in handle_name:
            T_lookat[0, 3] -= 0.05
        else:
            T_lookat[0, 3] += 0.05
        T_lookat[2, 3] -= 0.05

        # move to look at pose
        move_to(T_lookat, planner, moveit, joint_listener, debug=False)
        moveit.open_gripper()

        # wait for pose estimation
        rospy.sleep(1.0)

        # query cabinent handle pose
        pose_handle = detect_cabinet_handle(pose_listener, pose_handle, handle_name)

        # raw_input('grap handle')
        T_handle = T_lookat.copy()
        T_handle[:3, 3] = pose_handle[:3]
        disable_list = ['chewie_door_right_handle', 'chewie_door_right_link',
                        'chewie_door_left_handle', 'chewie_door_left_link', 'sektion']
        move_to(T_handle, planner, moveit, joint_listener, disable_list=disable_list, debug=False)
        rospy.sleep(0.5)
        moveit.close_gripper(force=60)

        if joint_listener.joint_position[-1] < 0.001:
            print('grasp handle failed, try again')
        else:
            break

    # compute the target location for opening door
    axis_point = pose_handle[:3].copy()
    if 'left' in handle_name:
        axis_point[0] += handle_axis_distance * np.sin(open_degree * np.pi / 180)
        axis_point[1] += handle_axis_distance * np.cos(open_degree * np.pi / 180)
    elif 'right' in handle_name:
        axis_point[0] += handle_axis_distance * np.sin(open_degree * np.pi / 180)
        axis_point[1] -= handle_axis_distance * np.cos(open_degree * np.pi / 180)
    else:
        print('no idea where the rotation axis is for ', handle_name)
        return
    print('axis point', axis_point)

    steps = 10
    for i in range(steps):
        turn_angle = (i + 1) * handle_turn_angle / steps
        if turn_angle < open_degree:
            continue
        x_angle = max(0, (i + 1) * Rx_angle / steps - lookat_angle)
        z_angle = (i + 1) * handle_turn_angle / steps - open_degree

        T = compute_cabinet_target_pose(handle_name, T_handle, axis_point, handle_axis_distance, turn_angle, x_angle, z_angle)
        print('step', i, 'target pose', T)
        moveit.go_local(T, wait=True)

    # raw_input('open gripper')
    moveit.open_gripper()
    rospy.sleep(1.0)

    # update cabinet door pose
    update_planner(planner, pose_listener, door_name)
    update_planner(planner, pose_listener, handle_name)

    # go back more
    delta_turn_anlge = 30
    delta_Rx_angle = 10
    delta_x = 0.2
    delta_y = 0.2
    delta_z = 0.1
    for i in range(2):
        turn_angle = handle_turn_angle - (i+1) * delta_turn_anlge
        x_angle = Rx_angle - (i+1) * delta_Rx_angle
        z_angle = handle_turn_angle - (i+1) * delta_turn_anlge
        T = compute_cabinet_target_pose(handle_name, T_handle, axis_point, handle_axis_distance, turn_angle, x_angle, z_angle)
        T[0, 3] -= (i+1) * delta_x
        T[2, 3] -= (i+1) * delta_z
        if 'left' in handle_name:
            T[1, 3] += (i+1) * delta_y
        else:
            T[1, 3] -= (i+1) * delta_y
        moveit.go_local(T, wait=True)

    # retract to start conf
    disable_list = []
    joint_position = joint_listener.joint_position
    start_conf = np.append(moveit.home_q, joint_position[7:]) 
    traj, flag_execute = planner.plan_to_conf(joint_position, start_conf, disable_list=disable_list)
    if flag_execute:
        moveit.execute(traj)
    else:
        print('plan failed in open cabinet, replaning')


# close cabinet give handle pose
def close_cabinet(handle_name, handle_axis_distance, moveit, planner, pose_listener, joint_listener):

    if 'left' in handle_name:
        door_name = 'chewie_door_left_link'
    elif 'right' in handle_name:
        door_name = 'chewie_door_right_link'
    else:
        print('unknown handle name', handle_name)
        return

    # grasp cabinet handle
    while 1:
        print('close cabinet with handle', handle_name)
        pose_handle = pose_listener.get_tf_pose(handle_name)
        if pose_handle is None:
            print('no handle pose detected for DART:', handle_name)
            return

        # compute look at pose
        euler = quat2euler(pose_handle[3:])
        if 'left' in handle_name:
            psi = -1 * (180 - euler[2] * 180 / np.pi)
        elif 'right' in handle_name:
            psi = 180 + euler[2] * 180 / np.pi
        open_degree = np.absolute(psi)
        if open_degree > 180:
            open_degree = 360 - open_degree
        print(handle_name, 'open degree', open_degree)

        if open_degree < 5:
            print('cabinet %s is already closed' % (handle_name))
            return

        handle_turn_angle = open_degree
        Rx_angle = handle_turn_angle * 0.6

        T_lookat = compute_look_at_pose(pose_listener, pose_handle, angle=-Rx_angle, distance=0.25, psi=psi)
        if 'left' in handle_name:
            T_lookat[0, 3] -= 0.05
        else:
            T_lookat[0, 3] += 0.05
        T_lookat[2, 3] -= 0.05

        # move to look at pose
        move_to(T_lookat, planner, moveit, joint_listener, base_obstacle_weight=3.0, debug=False)
        moveit.open_gripper()

        # wait for pose estimation
        rospy.sleep(1.0)

        # query cabinent handle pose from hand camera
        pose_handle = detect_cabinet_handle(pose_listener, pose_handle, handle_name)

        # raw_input('grap handle')
        T_handle = T_lookat.copy()
        T_handle[:3, 3] = pose_handle[:3]

        # push back more
        distance = 0.01
        if 'left' in handle_name:
            T_handle[0, 3] += distance * np.sin(handle_axis_distance * np.pi / 180)
            T_handle[1, 3] -= distance * np.cos(handle_axis_distance * np.pi / 180)
        elif 'right' in handle_name:
            T_handle[0, 3] += distance * np.sin(handle_axis_distance * np.pi / 180)
            T_handle[1, 3] += distance * np.cos(handle_axis_distance * np.pi / 180)

        moveit.go_local(T_handle, wait=True)
        rospy.sleep(0.5)
        moveit.close_gripper(force=60)

        if joint_listener.joint_position[-1] < 0.001:
            print('grasp handle failed, try again')
        else:
            break

    # compute the target location for opening door
    axis_point = pose_handle[:3].copy()
    if 'left' in handle_name:
        axis_point[0] += handle_axis_distance * np.sin(handle_turn_angle * np.pi / 180)
        axis_point[1] += handle_axis_distance * np.cos(handle_turn_angle * np.pi / 180)
    elif 'right' in handle_name:
        axis_point[0] += handle_axis_distance * np.sin(handle_turn_angle * np.pi / 180)
        axis_point[1] -= handle_axis_distance * np.cos(handle_turn_angle * np.pi / 180)
    else:
        print('no idea where the rotation axis is for ', handle_name)
        return
    print('axis point', axis_point)

    steps = 10
    for i in range(steps-1):
        turn_angle = handle_turn_angle - (i + 1) * handle_turn_angle / steps
        x_angle = -(i + 1) * Rx_angle / steps
        z_angle = -(i + 1) * handle_turn_angle / steps

        T = compute_cabinet_target_pose(handle_name, T_handle, axis_point, handle_axis_distance, turn_angle, x_angle, z_angle, action='close')
        print('step', i, 'target pose', T)
        moveit.go_local(T, wait=True)

    # raw_input('open gripper')
    moveit.open_gripper(wait=False)
    rospy.sleep(1.0)

    # update cabinet door pose
    update_planner(planner, pose_listener, door_name)
    update_planner(planner, pose_listener, handle_name)

    # retract to start conf
    joint_position = joint_listener.joint_position
    start_conf = np.append(moveit.home_q, joint_position[7:]) 
    traj, flag_execute = planner.plan_to_conf(joint_position, start_conf, disable_list=[door_name, handle_name])
    if flag_execute:
        moveit.execute(traj)
    else:
        print('plan failed in closing cabinet, replaning')


def pickup_object(target_name, planner, moveit, joint_listener, start_conf=None, force=40, debug=False):

    if start_conf is None:
        start_conf = joint_listener.joint_position

    traj, flag_execute, standoff_idx = planner.plan_to_target(start_conf, target_name)

    # visualization
    if debug:
        # planner.scene.draw_obj_grasp(target_name, interact=2)
        planner.scene.fast_debug_vis(collision_pt=False, nonstop=False, write_video=True)

    if not flag_execute:
        print('plan failed')
        return traj, False

    # execuate plan
    # raw_input('enter to execuate plan')
    moveit.execute(traj[:standoff_idx])
    moveit.execute(traj[standoff_idx:])
    rospy.sleep(1.0)

    force_before = joint_listener.robot_force
    print('force before grasping', force_before)

    # close the gripper
    moveit.close_gripper(force=force)
    print('joints before lifting', joint_listener.joint_position)

    # lift object
    delta = 0.1
    T = moveit.forward_kinematics(traj[-1, :-2])
    T_lift = T.copy()
    T_lift[2, 3] += delta
    moveit.go_local(T_lift, wait=True)
    print('joints after lifting', joint_listener.joint_position)

    # update pose
    obj_idx = planner.scene.env.names.index(target_name)
    target = planner.scene.env.objects[obj_idx]
    pose = target.pose.copy()
    print('object pose', pose)
    pose[2] += delta
    planner.scene.env.update_pose(target_name, pose)

    force_after = joint_listener.robot_force
    print('force after grasping', force_after)
    force_diff = np.linalg.norm(force_before - force_after)
    print('force diff norm', force_diff)

    joint_position = joint_listener.joint_position
    if joint_position[-1] > 0.002 or force_diff > 0.5 or force_diff == 0:
        success = True
        print('grasp success')
    else:
        success = False
        print('grasp fail')

    return traj, success


def place_object(target_name, place_translation, planner, moveit, joint_listener, start_conf=None, is_delta=True, apply_standoff=False, debug=False):
    if start_conf is None:
        start_conf = joint_listener.joint_position
        print('query start configuration', start_conf)

    while 1:
        traj, flag_execute, grasp_pose, standoff_idx = planner.plan_to_place_target(start_conf, target_name, place_translation, is_delta, apply_standoff, debug)
        if flag_execute:
            if apply_standoff:
                moveit.execute(traj[:standoff_idx])
                rospy.sleep(0.5)
                moveit.open_gripper()
                moveit.execute(traj[standoff_idx:])
            else:
                moveit.execute(traj)
            break
        else:
            print('plan failed, replaning')
            planner.scene.env.update_pose(target_name, grasp_pose)
    return traj
