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
from transforms3d.euler import mat2euler, quat2euler
from actions import compute_look_at_pose, move_to, turn_camera_towards_object, pickup_object, place_object, update_planner
from actions import compute_cabinet_target_pose

### define operators for the kitchen tasks ###

OBJECTS = ['004_sugar_box', 'chewie_door_left_handle', 'chewie_door_right_handle']

class KitchenDomain:

    def __init__(self, pose_listener, planner, moveit, joint_listener):
        self.pose_listener = pose_listener
        self.planner = planner
        self.moveit = moveit
        self.joint_listener = joint_listener

        # manipulatable objects
        self.objects = OBJECTS
        self.num_objects = len(self.objects)

        # dictionary of object status
        self.is_pose_ready = {}
        self.object_poses = {}
        self.is_attached = {}
        self.is_on_counter = {}
        for name in self.objects:
            self.object_poses[name] = None
            self.is_pose_ready[name] = 0
            self.is_attached[name] = 0
            self.is_on_counter[name] = 0

        # axis point of handles and open degrees
        self.handle_axis_distance = 0.34
        self.axis_points = {}
        self.open_degrees = {}
        for name in self.objects:
            if 'handle' in name:
                self.axis_points[name] = None
                self.open_degrees[name] = None


    def reset_pose_flags(self):
        for name in self.objects:
            self.is_pose_ready[name] = 0


    def query_object_poses(self):
        model_names, object_poses, object_names = self.pose_listener.get_object_poses(block=False)
        # set flags and poses
        self.reset_pose_flags()
        for k in range(len(object_names)):
            name = object_names[k]
            pose = object_poses[k]
            for i in range(self.num_objects):
                if name == self.objects[i]:
                    self.is_pose_ready[name] = 1
                    self.object_poses[name] = pose
                    break


    def query_handle_pose(self, handle_name, pose_from_dart):

        # query cabinent handle pose
        # assume at maximum 5 handles visible
        object_names = []
        object_poses = []
        distances = []
        for i in range(5):
            link_name = 'poserbpf/00_cabinet_handle_%02d' % (i)
            pose = self.pose_listener.get_tf_pose(link_name)
            if pose is not None and pose[2] > 0:
                object_poses.append(pose)
                object_names.append(link_name)
                distances.append(np.linalg.norm(pose[:3] - pose_from_dart[:3]))

        self.reset_pose_flags()
        if len(object_names) == 0:
            print('no handle pose detected from hand cameara:', handle_name)
            pose_handle = None
        else:
            index = np.argmin(distances)
            pose_handle = object_poses[index]
            print(object_names[index])
            print(object_poses[index])
            print(distances)

            # set flag and pose
            for i in range(self.num_objects):
                if handle_name == self.objects[i]:
                    self.is_pose_ready[name] = 1
                    self.object_poses[handle_name] = pose_handle
                    break
        return pose_handle


    def query_open_degree(self, handle_name):
        pose_handle = self.pose_listener.get_tf_pose(handle_name)
        if pose_handle is None:
            print('no handle pose detected for DART:', handle_name)
            return None

        euler = quat2euler(pose_handle[3:])
        if 'left' in handle_name:
            psi = -1 * (180 - euler[2] * 180 / np.pi)
        elif 'right' in handle_name:
            psi = 180 + euler[2] * 180 / np.pi
        open_degree = np.absolute(psi)
        if open_degree > 180:
            open_degree = 360 - open_degree
        return open_degree


    # is an object in the gripper
    def predicate_is_object_in_gripper(self, object_name):
        joint_position = self.joint_listener.joint_position
        return self.is_attached[object_name] > 0 and joint_position[-1] > 0.002


    # is an object on the counter
    def predicate_is_object_on_counter(self, object_name):
        return self.is_on_counter[object_name] > 0


    # is an object pose ready
    def predicate_is_object_pose_ready(self, object_name):
        return self.is_pose_ready[object_name] > 0


    # is a door openning
    def predicate_is_cabinet_door_openning(self, handle_name):
        open_degree = self.query_open_degree(handle_name)
        return open_degree > 75


    # is the robot at home
    def predicate_is_robot_at_home(self):
        joint_position = self.joint_listener.joint_position
        return np.all(joint_position[:7] - self.moveit.home_q[:7]) < 0.1)


    # preconditions: no object in the gripper, cabinet left door is open, cabinet right door is open, robot at home
    # effects: set some object poses ready
    def operator_look_inside_cabinet(self):

        # look inside the cabinet
        pose_drawer = self.pose_listener.get_tf_pose('hitman_drawer_top')
        pose_drawer[2] += 1.0
        lookat_angle = 15
        lookat_distance = 0.5
        T_lookat = compute_look_at_pose(self.pose_listener, pose_drawer, angle=lookat_angle, distance=lookat_distance)
        move_to(T_lookat, self.planner, self.moveit, self.joint_listener, debug=False)
        rospy.sleep(0.5)
        self.query_object_poses()


    # preconditions: no object in the gripper, robot at home
    # effects: set some object poses ready
    def operator_look_at_counter(self):

        # look at the counter
        pose_drawer = self.pose_listener.get_tf_pose('hitman_drawer_top')
        pose_drawer[2] += 0.2
        lookat_angle = 15
        lookat_distance = 0.5
        T_lookat = compute_look_at_pose(self.pose_listener, pose_drawer, angle=lookat_angle, distance=lookat_distance)
        move_to(T_lookat, self.planner, self.moveit, self.joint_listener, debug=False)
        rospy.sleep(0.5)
        self.query_object_poses()


    # preconditions: no object in the gripper, robot at home
    # effects: set handle pose ready
    def operator_look_at_a_handle(self, handle_name):

        # look at a handle
        open_degree = self.query_open_degree(handle_name)
        print(handle_name, 'open degree', open_degree)
        self.open_degrees[handle_name] = open_degree

        # compute look at pose
        lookat_angle = open_degree * 0.5
        T_lookat = compute_look_at_pose(self.pose_listener, pose_handle, angle=-lookat_angle , distance=0.25, psi=psi)
        if 'left' in handle_name:
            T_lookat[0, 3] -= 0.05
        else:
            T_lookat[0, 3] += 0.05
        T_lookat[2, 3] -= 0.05

        # move robot
        move_to(T_lookat, self.planner, self.moveit, self.joint_listener, base_obstacle_weight=3.0, debug=False)
        rospy.sleep(1.0)
        pose_handle = self.query_handle_pose(handle_name, pose_from_dart=pose_handle)

        # save the axis point for future use
        if pose_handle is not None:
            
            axis_point = pose_handle[:3].copy()
            if 'left' in handle_name:
                axis_point[0] += self.handle_axis_distance * np.sin(open_degree * np.pi / 180)
                axis_point[1] += self.handle_axis_distance * np.cos(open_degree * np.pi / 180)
            elif 'right' in handle_name:
                axis_point[0] += self.handle_axis_distance * np.sin(open_degree * np.pi / 180)
                axis_point[1] -= self.handle_axis_distance * np.cos(open_degree * np.pi / 180)
            self.axis_points[handle_name] = axis_point


    # grasp a cabinet handle
    # preconditions: no object in the gripper, handle pose is ready
    # effects: handle in gripper, set all poses not ready
    def operator_grasp_a_handle(self, handle_name):

        # query cabinent handle pose
        pose_handle = self.object_poses[handle_name]

        # current gripper pose
        positions = self.joint_listener.joint_position[:-2]
        ee_frame = self.moveit.forward_kinematics(positions)

        # grasp handle
        T_handle = ee_frame.copy()
        T_handle[:3, 3] = pose_handle[:3]

        disable_list = ['chewie_door_right_handle', 'chewie_door_right_link',
                        'chewie_door_left_handle', 'chewie_door_left_link', 'sektion']
        move_to(T_handle, self.planner, self.moveit, self.joint_listener, disable_list=disable_list, debug=False)
        self.reset_pose_flags()
        rospy.sleep(0.5)
        self.moveit.close_gripper(force=60)
        self.is_attached[handle_name] = 1


    # preconditions: handle in gripper, cabinet is closed
    # effects: no object in the gripper, cabinet is open
    # handle name: chewie_door_left_handle, chewie_door_right_handle
    def operator_open_a_handle(self, handle_name):

        # query axis point
        axis_point = self.axis_points[handle_name]
        open_degree = self.open_degrees[handle_name]
        lookat_angle = open_degree * 0.5
        Rx_angle = 40

        if 'left' in handle_name:
            handle_turn_angle = 90
            door_name = 'chewie_door_left_link'
        elif 'right' in handle_name:
            handle_turn_angle = 75
            door_name = 'chewie_door_right_link'
        else:
            print('unknown handle name', handle_name)
            return

        # current gripper pose
        positions = self.joint_listener.joint_position[:-2]
        T_handle = self.moveit.forward_kinematics(positions)

        # openning
        steps = 10
        for i in range(steps):
            turn_angle = (i + 1) * handle_turn_angle / steps
            if turn_angle < open_degree:
                continue
            x_angle = max(0, (i + 1) * Rx_angle / steps - lookat_angle)
            z_angle = (i + 1) * handle_turn_angle / steps - open_degree

            T = compute_cabinet_target_pose(handle_name, T_handle, axis_point, self.handle_axis_distance, turn_angle, x_angle, z_angle)
            print('step', i, 'target pose', T)
            self.moveit.go_local(T, wait=True)

        # raw_input('open gripper')
        self.moveit.open_gripper()
        self.is_attached[handle_name] = 0
        rospy.sleep(1.0)

        # update cabinet door pose
        update_planner(self.planner, self.pose_listener, door_name)
        update_planner(self.planner, self.pose_listener, handle_name)

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
            T = compute_cabinet_target_pose(handle_name, T_handle, axis_point, self.handle_axis_distance, turn_angle, x_angle, z_angle)
            T[0, 3] -= (i+1) * delta_x
            T[2, 3] -= (i+1) * delta_z
            if 'left' in handle_name:
                T[1, 3] += (i+1) * delta_y
            else:
                T[1, 3] -= (i+1) * delta_y
            self.moveit.go_local(T, wait=True)


    # preconditions: handle in gripper, cabinet is open
    # effects: no object in the gripper, cabinet is closed
    # handle name: chewie_door_left_handle, chewie_door_right_handle
    def operator_close_a_handle(self, handle_name):

        # query axis point
        axis_point = self.axis_points[handle_name]
        open_degree = self.open_degrees[handle_name]
        handle_turn_angle = open_degree
        Rx_angle = handle_turn_angle * 0.6

        if 'left' in handle_name:
            door_name = 'chewie_door_left_link'
        elif 'right' in handle_name:
            door_name = 'chewie_door_right_link'
        else:
            print('unknown handle name', handle_name)
            return

        # current gripper pose
        positions = self.joint_listener.joint_position[:-2]
        T_handle = self.moveit.forward_kinematics(positions)

        steps = 10
        for i in range(steps-1):
            turn_angle = handle_turn_angle - (i + 1) * handle_turn_angle / steps
            x_angle = -(i + 1) * Rx_angle / steps
            z_angle = -(i + 1) * handle_turn_angle / steps

            T = compute_cabinet_target_pose(handle_name, T_handle, axis_point, self.handle_axis_distance, turn_angle, x_angle, z_angle, action='close')
            print('step', i, 'target pose', T)
            self.moveit.go_local(T, wait=True)

        # raw_input('open gripper')
        self.moveit.open_gripper(wait=False)
        self.is_attached[handle_name] = 0
        rospy.sleep(1.0)

        # update cabinet door pose
        update_planner(self.planner, self.pose_listener, door_name)
        update_planner(self.planner, self.pose_listener, handle_name)


    # preconditions: no object in the gripper, object pose is ready
    # effects: object in gripper, set all poses not ready
    # object name: sugar box
    def operator_grasp_an_object(self, object_name):

        # look at object first
        object_pose = self.object_poses[object_name]
        psi = turn_camera_towards_object(object_pose, self.moveit, self.joint_listener)
        lookat_angle = 15
        lookat_distance = 0.5
        T_lookat = compute_look_at_pose(self.pose_listener, object_pose, angle=lookat_angle+5, distance=lookat_distance, psi=psi)
        move_to(T_lookat, self.planner, self.moveit, self.joint_listener, debug=False)
        self.reset_pose_flags()

        # detect object
        rospy.sleep(0.5)
        model_names, object_poses, object_names = self.pose_listener.get_object_poses(block=True)
        if object_name not in object_names:
            print('Can not get pose for object %s' % (object_name))
            return

        print('openning gripper')
        self.moveit.open_gripper()

        # update planner
        self.planner.update_objects(object_names, model_names, object_poses)

        # pick up object
        print('pick up object', object_name)
        traj, success = pickup_object(object_name, self.planner, self.moveit, self.joint_listener, debug=True)
        self.is_attached[object_name] = 1


    # preconditions: object in gripper
    # effects: no object in the gripper, object on counter
    # object name: sugar box
    def operator_place_an_object_on_counter(self, object_name):

        # compute the place location on the counter
        pose_drawer = self.pose_listener.get_tf_pose('hitman_drawer_top')
        table_top = pose_drawer[:3].copy()
        table_top[2] += 0.21

        # move back from cabinet
        # cabinet to table top distance is 57cm
        translation = table_top.copy()
        translation[0] -= 0.4
        translation[2] += 0.7
        traj = place_object(object_name, translation, self.planner, self.moveit, self.joint_listener, is_delta=False, debug=True)

        # place on table
        translation = table_top.copy()
        translation[0] -= 0.1
        translation[2] += 0.2
        traj = place_object(object_name, translation, self.planner, self.moveit, self.joint_listener, is_delta=False, apply_standoff=True, debug=True)
        rospy.sleep(1.0)
        self.moveit.open_gripper()
        self.is_attached[object_name] = 0


    # preconditions: no object in the gripper
    # effects: robot at home
    def operator_go_home(self):
        joint_position = self.joint_listener.joint_position
        start_conf = np.append(self.moveit.home_q, joint_position[7:]) 
        traj, flag_execute = self.planner.plan_to_conf(joint_position, start_conf, disable_list=[])
        self.moveit.execute(traj)
        if not flag_execute:
            print('plan should not be executed in operator go home')



