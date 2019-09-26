#!/usr/bin/env python

import math
import signal
import rospy
import tf
import sys
import argparse
import copy
import numpy as np
import numpy.linalg as la
import pickle
import roslib
import json
import tf.transformations as tra
import cv2
import threading
import yaml
import time, os, sys
import os.path as osp
import message_filters
import datetime
import glob

from lula_franka.franka import Franka
from lula_control.object import ControllableObject
from lula_control.frame_commander import RobotConfigModulator
from lula_ros_util_internal.msg import RosVector
from lula_ros_util_internal.srv import (RosVectorService, RosVectorServiceRequest, RosVectorServiceResponse)
from brain_ros.moveit import MoveitBridge
from lula_control.world import make_basic_world
from lula_control.object import ControllableObject
from lula_control.world import make_basic_world
from lula_control.frame_commander import FrameConvergenceCriteria, TimeoutMonitor
from isaac_bridge.manager import SimulationManager
from lula_controller_msgs.msg import TaskSpaceStatus

from transforms3d.quaternions import mat2quat, quat2mat
from easydict import EasyDict as edict
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as RosImage
from tf2_msgs.msg import TFMessage
from rospy_tutorials.srv import *
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from ros_utils import *
#from utils.visualization_utils import *

def myhook():
  print('finish recording!')

class Robot:
    def __init__(self, is_physical_robot, object_yaml_config="world.yaml"):

        rospy.init_node('grasp_poserbpf_node')
        self._franka = Franka(is_physical_robot=is_physical_robot)

        self._bridge = MoveitBridge(
            group_name='panda_arm',
            robot_interface=self._franka,
            dilation=.8,
            verbose=1.,
            ee_link='right_gripper',
            lula_world_objects=None,
        )
        self.open_gripper()
        self.retract()

        rospy.Subscriber("poserbpf_status", numpy_msg(Floats), self.prbpf_callback)
        self.prbpf_ok = False
        self.prbpf_grasp = False
        self.is_physical_robot = is_physical_robot
        self._grasp_frame_offset = tra.quaternion_matrix([0, 0, -0.707, 0.707])           
        self._grasp_frame_offset[:3, 3] = [0, 0, 0.1]
        self._objects_info = self._load_object_info(object_yaml_config)
        self.listener = tf.TransformListener()

        rospy.wait_for_service('reset_poserbpf')
        self.poserbpf_client = rospy.ServiceProxy('reset_poserbpf', AddTwoInts)

    def set_speed(self, speed):
        self._franka.set_speed(speed_level=speed)

    def prbpf_callback(self, data):
        self.prbpf_ok = bool(data.data[0])
        self.prbpf_grasp = bool(data.data[1])

    def _load_object_info(self, path, prefix="00"):
        if path is None:
            return None
        
        output = {}
        d = yaml.load(open(path))['objects']
        for obj_name in d:
            if not obj_name.startswith(prefix):
                continue

            obj_type = d[obj_name]['model']['type']
            obj_dims = d[obj_name]['model']['scale']

            if obj_type == 'cube':
                dims = (obj_dims['x'], obj_dims['y'], obj_dims['z'])
            elif obj_type == 'cylinder':
                dims = (obj_dims['z'], 0.5 * obj_dims['x'], 0)
            else:
                raise ValueError('obj_type {} not supported'.format(obj_type))
            
            output[obj_name] = (obj_type, dims)
            print('loading obj {} with type {} and dims {}'.format(obj_name, obj_type, dims))
        
        return output
    
    def populate_moveit_scene_real(self, object_list):
        # if self._objects_info is None:
        #     rospy.loginfo('No objects has loaded')
        #     return

        dims = np.array([0.8, 2.8, 0.75])
        trans = np.array([0.5, 0, -0.4])
        quat = np.array([0, 0, 0, 1])
        frame = "00_base_link"
        self._bridge.add_box(
                    'table',
                    dims,
                    trans,
                    quat,
                    frame,
                )

        for obj in object_list:
            dims = self.moveit_data[obj]
            trans = np.zeros(3)
            quat = np.array([0, 0, 0, 1])
            frame = 'poserbpf/'+obj
            self._bridge.add_box(
                obj,
                dims,
                trans,
                quat,
                frame,
            )

        # raw_input('check movit!')
        return

    def populate_moveit_scene_isaac_sim(self, object_list):
        if self._objects_info is None:
            rospy.loginfo('No objects has loaded')
            return
        
        self._bridge.remove()
        frame = "00_base_link"
        for obj_name, obj_info in self._objects_info.items():
            if obj_name in self._bridge.tracked_objs:
                raise ValueError('duplicate obj_name {}'.format(obj_name))
            
            obj_type, dims = obj_info
            rt = get_relative_pose_from_tf(self.listener, obj_name, frame)
            quat = tra.quaternion_from_matrix(rt)
            trans = rt[:3, 3]

            if obj_type == 'cube':
                print("robot: ... adding box {} with dims {}".format(obj_name, dims))
                self._bridge.add_box(
                    obj_name,
                    dims,
                    trans, 
                    quat,
                    frame,
                )
            elif obj_type == 'cylinder':
                print("robot: ... adding cylinder {} with dims {}".format(obj_name, dims))
                self._bridge.add_cylinder(
                    obj_name,
                    dims,
                    trans,
                    quat,
                    frame,
                )
            else:
                raise ValueError('not supported {}'.format(obj_type))

        for obj in object_list:
            dims = self.moveit_data[obj]
            trans = np.zeros(3)
            quat = np.array([0, 0, 0, 1])
            frame = 'poserbpf/'+obj
            self._bridge.add_box(
                obj,
                dims,
                trans,
                quat,
                frame,
            )

    
    def retract(self):
        self._bridge.retract(lula=True)
    
    @property
    def grasp_frame_offset(self):
        return self._grasp_frame_offset

    
    def open_gripper(self):
        self._bridge.open_gripper()
    
    
    def close_gripper(self, *args, **kwargs):
        self._bridge.close_gripper(*args, **kwargs)

    
    def grasp_object(
        self,
        object_name,
        standoff_pose, 
        grasp_pose,
        force=20.,
    ):
        controllable_object = None
        if object_name is not None:
            try:
                controllable_object = ControllableObject(
                    self._world.get_object(object_name),
                    robot=self._franka,
                )
            except Exception as e:
                print(str(e))
            

        if not self.goto(standoff_pose.dot(self._grasp_frame_offset), timeout=20,):
            return False
        #raw_input()
        raw_input('check standoff!')

        if controllable_object is not None:
            controllable_object.suppress()
        self.interpolate_go_local(
            standoff_pose.dot(self._grasp_frame_offset),
            grasp_pose.dot(self._grasp_frame_offset),
            rospy.Duration(3.0),
            timeout=4.0,
        )

        self.close_gripper(controllable_object, force=force)

        return True
    

    def goto(self, rt, timeout=7.0):
        quat = tra.quaternion_from_matrix(rt)
        trans = rt[:3, 3]
        
        return self._bridge.goto(trans, quat, timeout=timeout)

    def plan(self, rt):
        quat = tra.quaternion_from_matrix(rt)
        trans = rt[:3, 3]
        
        return self._bridge.plan(trans, quat) is not None



    def interpolate_go_local(self, start, goal, duration, timeout):
        self._bridge.interpolate_go_local(start, goal, duration, timeout)
        return


        
def marker_thread_func(publisher, grasps, base_frame_id, stop_event):
    while not stop_event.is_set() and not rospy.is_shutdown():
        # Publish the grasp with respect to the camera.
        publish_grasps(
            publisher,
            base_frame_id,
            grasps,
            scores=None,
            color_alpha=0.5
        )

        stop_event.wait(timeout=0.1)


def push_object(robot, object_pose, center_clutter, radius_max_push_start=0.2, max_length=0.1):
    # get start position of push
    downward_orientation = tra.rotation_matrix(np.pi, [0, 1, 0])
    ninety_degree_rotation = tra.rotation_matrix(np.pi*0.5, [0, 0, 1])
    start_push = np.eye(4)
    start_push[:3, 3] = object_pose[:3, 3]

    # this is in base_link coordinates
    # push slightly above table surface
    start_push[2, 3] *= 0.5
    if start_push[2, 3] < 0.01:
        start_push[2, 3] = 0.01

    # push towards the center
    center_dir = center_clutter - object_pose[:3, 3]
    center_dir = center_dir / np.linalg.norm(center_dir)
    theta_center = math.atan2(center_dir[1], center_dir[0])

    step_plan = 0
    while True:
        print("Push plan:", step_plan)
        theta_push_start = np.random.uniform(-np.pi/3, np.pi/3)
        radius_push_start = np.random.uniform(0.1, radius_max_push_start)

        # start
        start_push[:3, :3] = downward_orientation[:3,:3].dot(tra.rotation_matrix(theta_push_start + theta_center, [0, 0, 1])[:3, :3]).dot(ninety_degree_rotation[:3, :3])
        start_push[0, 3] = object_pose[0, 3] - np.cos(theta_push_start + theta_center) * radius_push_start
        start_push[1, 3] = object_pose[1, 3] - np.sin(theta_push_start + theta_center) * radius_push_start

        if robot.goto(start_push):
            # calculate end position of push
            end_push = start_push
            end_push[0, 3] = object_pose[0, 3] + np.cos(theta_push_start + theta_center) * np.random.uniform(0, max_length)
            end_push[1, 3] = object_pose[1, 3] + np.sin(theta_push_start + theta_center) * np.random.uniform(0, max_length)
            break

        step_plan += 1
        if step_plan > 30:
            return False

    print("Push start: ", start_push)
    print("Push goal: ", end_push)

    # move robot
    robot.interpolate_go_local(start_push, end_push, rospy.Duration(2.0), 5.0)

    return True

def push_object_random(robot, object_pose, radius_max_push_start=0.2, max_length=0.1):
    # get start position of push
    downward_orientation = tra.rotation_matrix(np.pi, [0, 1, 0])
    ninety_degree_rotation = tra.rotation_matrix(np.pi*0.5, [0, 0, 1])
    start_push = np.eye(4)
    start_push[:3, 3] = object_pose[:3, 3]

    # this is in base_link coordinates
    # push slightly above table surface
    start_push[2, 3] *= 0.5
    start_push[2, 3] += 0.03
    if start_push[2, 3] < 0.02:
        start_push[2, 3] = 0.02

    step_plan = 0
    while True:
        print("Push plan:", step_plan)
        theta_push_start = np.random.uniform(-np.pi, np.pi)
        radius_push_start = np.random.uniform(0.1, radius_max_push_start)

        # start
        start_push[:3, :3] = downward_orientation[:3,:3].dot(tra.rotation_matrix(theta_push_start, [0, 0, 1])[:3, :3]).dot(ninety_degree_rotation[:3, :3])
        start_push[0, 3] = object_pose[0, 3] - np.cos(theta_push_start) * radius_push_start
        start_push[1, 3] = object_pose[1, 3] - np.sin(theta_push_start) * radius_push_start

        if robot.goto(start_push):
            # calculate end position of push
            end_push = start_push
            end_push[0, 3] = object_pose[0, 3] + np.cos(theta_push_start) * np.random.uniform(0, max_length)
            end_push[1, 3] = object_pose[1, 3] + np.sin(theta_push_start) * np.random.uniform(0, max_length)
            break

        step_plan += 1
        if step_plan > 30:
            return False

    print("Push start: ", start_push)
    print("Push goal: ", end_push)

    # move robot
    robot.interpolate_go_local(start_push, end_push, rospy.Duration(2.0), 5.0)

    return True

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Data Collection')
    parser.add_argument('--target_obj', dest='target_obj',
                        help='target object to grasp',
                        required=True, type=str)
    parser.add_argument('--physical_robot', dest='physical_robot',
                        help='flag for physical robot or simulation',
                        action='store_true')
    parser.add_argument('--interact', dest='robot_interact',
                        help='flag for interaction or not',
                        action='store_true')
    parser.add_argument('--instance', dest='instance_id', help='PoseCNN instance id to use',
                        default=0, type=int)
    parser.add_argument('--world', dest='world_file',
                        help='optional world config file', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    prefix = '%02d_' % (args.instance_id)
    target_obj = args.target_obj
    tf_name = 'poserbpf/' + prefix + target_obj + '_01'

    # standoff for grasping
    STANDOFF = 0.035
    BASE_FRAME = 'measured/base_link'
    standoff_transform = np.eye(4, dtype=np.float32)
    standoff_transform[2,3] = -STANDOFF

    # adapt to new fingers
    finger_transform = np.eye(4, dtype=np.float32)
    finger_transform[2, 3] = 0.01

    # robot
    robot = Robot(is_physical_robot=args.physical_robot, object_yaml_config=args.world_file)
    robot.open_gripper()
    listener = tf.TransformListener()
    grasp_publisher = rospy.Publisher('/grasps/', MarkerArray, queue_size=10)
    RIGHT_GRIPPER_FRAME = 'measured/right_gripper'
    init_gripper = get_relative_pose_from_tf(listener, RIGHT_GRIPPER_FRAME, BASE_FRAME)

    # multiview trajectory
    center_clutter = np.array([0.6, 0, 0.025], dtype=np.float32)
    psi = np.array([0, -30, 0, 30, 15]) / 57.3
    psi += 0.1 * np.random.randn(psi.shape[0])
    theta = 50 / 57.3 * np.ones_like(psi) + 0.1 * np.random.randn(psi.shape[0])
    r = 0.5 * np.ones_like(theta) + 0.05 * np.random.randn(psi.shape[0])
    r[0] = 0.7
    r[-1] = 0.3
    theta[0] = 60 / 57.3
    theta[-1] = 75 / 57.3

    # main loop
    test_n_iter = 50
    for i_test in range(test_n_iter):
        # robot.set_speed('fast')

        local_motion = False
        for i in range(len(psi)):
            position_robot = center_clutter + np.array([-r[i] * np.cos(theta[i]) * np.cos(psi[i]),
                                                        -r[i] * np.cos(theta[i]) * np.sin(psi[i]),
                                                        r[i] * np.sin(theta[i])], dtype=np.float32)
            Z_BG = center_clutter - position_robot
            Z_BG /= np.linalg.norm(Z_BG)
            Y_BG = np.array([-np.sin(psi[i]), np.cos(psi[i]), 0], dtype=np.float32)
            X_BG = np.cross(Y_BG, Z_BG)
            R_BG = np.zeros((3, 3), dtype=np.float32)
            R_BG[:, 0] = X_BG
            R_BG[:, 1] = Y_BG
            R_BG[:, 2] = Z_BG

            pose_robot = np.eye(4, dtype=np.double)
            pose_robot[:3, 3] = position_robot.astype(np.double)
            pose_robot[:3, :3] = R_BG[:3, :3].astype(np.double)
            if local_motion == False:
                robot.goto(pose_robot, timeout=20.0)
                rospy.sleep(3.0)
                robot.poserbpf_client(0, 0)
                while 1:
                    Tbo = get_relative_pose_from_tf(listener, tf_name, BASE_FRAME)
                    if Tbo[2, 3] > 0:
                        break
                print(Tbo)
                center_clutter = Tbo[:3, 3]
                local_motion = True
            else:
                # data recording
                # robot.poserbpf_client(4, 0)
                robot.interpolate_go_local(pose_start, pose_robot, rospy.Duration(5.0), timeout=5.0,)
                rospy.sleep(0.5)
                Tbo = get_relative_pose_from_tf(listener, tf_name, BASE_FRAME)
                center_clutter = Tbo[:3, 3]
                init_step = 0
                if not i == (len(psi) - 1):
                    while (not robot.prbpf_ok) and init_step < 3:
                        pose_perturb = np.eye(4, dtype=np.float32)
                        pose_perturb[2, 3] = np.random.randn(1) * 0.025
                        robot.interpolate_go_local(pose_robot, pose_robot.dot(pose_perturb), rospy.Duration(5.0), timeout=5.0, )
                        pose_robot = pose_robot.dot(pose_perturb)
                        rospy.sleep(2.0)
                        init_step += 1
                # pose is good enough, perform grasping
                if robot.prbpf_grasp:
                    break
            pose_start = get_relative_pose_from_tf(listener, 'right_gripper', BASE_FRAME)

        init_gripper = get_relative_pose_from_tf(listener, RIGHT_GRIPPER_FRAME, BASE_FRAME)

        if not interaction:
            continue

        # robot.poserbpf_client(5, 0)
        # start grasping
        # robot.set_speed('slow')
        objects_scene_avoid = []
        for obj in objects_scene:
            if obj != target_obj:
                objects_scene_avoid.append(obj)
        if robot.is_physical_robot:
            robot._bridge.remove_obstacle()
            robot.populate_moveit_scene_real(objects_scene_avoid)
        else:
            robot.populate_moveit_scene_isaac_sim(objects_scene_avoid)

        # get the center of all the objects
        center_all = np.zeros((3,), dtype=np.float32)
        for obj in objects_scene:
            Tbo = get_relative_pose_from_tf(listener, 'poserbpf/' + obj, 'measured/base_link')
            center_all += Tbo[:3, 3]
        center_all /= len(objects_scene)

        if np.random.uniform(0, 1.0) < 0.5:
            center_all = np.array([0.58, 0, 0.0], dtype=np.float32)

        print('Object pose is ready, start grasping ... ')
        # if robot.is_physical_robot:
        #     raw_input('Press return to continue.')

        call_flag = robot.poserbpf_client(2, 0)
        print(' pause poserbpf !')
        rospy.sleep(1.0)

        files = glob.glob('100_grasps/{}.npy'.format(target_obj))
        grasps_dict = {}
        quality_dict = {}

        Tbr = get_relative_pose_from_tf(listener, 'measured/camera_link', 'measured/base_link')
        Tbo = get_relative_pose_from_tf(listener, 'poserbpf/'+target_obj, 'measured/base_link')

        for f in files:
            pure_name = f[f.rfind('/') + 5:-4]
            grasps_dict['00_' + pure_name] = np.load(f, allow_pickle=True).item()['transforms']
            order = objects
            done_objects = set()

        order = objects
        done_objects = set()

        for obj_name in order:
            if obj_name in done_objects:
                continue
            if obj_name not in grasps_dict:
                push_success = push_object(robot, Tbo, center_all)
                found_feasible_grasp = True
                if push_success:
                    robot.poserbpf_client(3, 0)
                break
                # raise ValueError('invalid obj_name {}'.format(obj_name))

            print('----------------- obj_name : {} -----------------'.format(obj_name))

            obj_pose = Tbo

            grasps = []
            standoffs = []
            pickup_poses = []
            dropoff_poses = []

            lift_height = 0.15

            # vertical grasping
            v_inner_z = []
            v_inner_x = []
            dist = []
            for i in xrange(len(grasps_dict[obj_name])):
                g = grasps_dict[obj_name][i]
                grasp_pose = obj_pose.dot(g)
                v_inner_z.append(np.dot(grasp_pose[:3, 2], np.array([0, 0, -1.0], dtype=np.float32)))
                # v_inner_x.append(np.dot(grasp_pose[:3, 1], np.array([1.0, 0, 0], dtype=np.float32)))
                p = g[:3, 3]
                z_dir = g[:3, 2]
                dist.append(np.linalg.norm(np.cross(p, z_dir)))

            dist = np.array(dist, np.float32)
            dist = 1 - 0.5 * dist / np.max(dist)

            for index in np.argsort(-1 * (np.array(v_inner_z) * dist)):
                g = grasps_dict[obj_name][index]

                # make sure the camera is not rotated
                if np.sum(g.dot(robot.grasp_frame_offset)[:, 0] * init_gripper[:, 0]) < 0:
                    g = g.dot(tra.euler_matrix(0, 0, np.pi))

                g = g.dot(finger_transform)
                grasps.append(obj_pose.dot(g))
                standoffs.append(obj_pose.dot(g.dot(standoff_transform)))
                # pickup_rt_in_base = grasps[-1].copy()

                pickup_rt_in_base = obj_pose.dot(g).dot(tra.euler_matrix(np.random.uniform(-45/57.3, 45/57.3), np.random.uniform(-45/57.3, 45/57.3), np.random.uniform(-np.pi/2, np.pi/2)))

                pickup_rt_in_base[2, 3] += lift_height
                pickup_poses.append(pickup_rt_in_base.copy())
                pickup_rt_in_base[2, 3] -= lift_height * 0.95

                # perturb_b = 0.0 * np.random.randn(3, 1)
                # perturb_o = 0.0 * np.random.randn(3, 1)
                # dropoff_pose = obj_pose.dot(tra.euler_matrix(perturb_o[0] * 0, perturb_o[1] * 0, perturb_o[2])).dot(g)
                # dropoff_pose[2, 3] = pickup_rt_in_base[2, 3]
                drop_rt_in_base = pickup_rt_in_base.copy()
                dropoff_poses.append(drop_rt_in_base)

            found = False
            robot.open_gripper()
            count = 0

            found_feasible_grasp = False
            grasp_prob = np.random.uniform(0, 1.0)

            threshold = 0.50
            push_success = True

            if grasp_prob < threshold:
                if np.random.uniform(0, 1.0) > 0.4:
                    push_success = push_object(robot, Tbo, center_all)
                else:
                    push_success = push_object_random(robot, Tbo)
                found_feasible_grasp = True
                if push_success:
                    robot.poserbpf_client(3, 0)
                else:
                    if push_object_random(robot, Tbo):
                        robot.poserbpf_client(3, 0)


            if grasp_prob >= threshold:
                for g, s, pickup, dropoff in zip(grasps, standoffs, pickup_poses, dropoff_poses):
                    print('checking grasp #{}/{}  {}'.format(count, 100, obj_name))
                    thread_stop_event = threading.Event()
                    marker_thread = threading.Thread(
                        target=marker_thread_func,
                        args=(grasp_publisher,
                              [s, g, pickup, dropoff],
                              BASE_FRAME,
                              thread_stop_event,),
                    )
                    marker_thread.start()

                    # raw_input('check grasp!')

                    if not robot.plan(s.dot(robot.grasp_frame_offset)):
                        thread_stop_event.set()
                        # marker_thread.join()
                        continue

                    if robot.grasp_object(None, s, g):
                        found = True
                        found_feasible_grasp = True
                        robot.interpolate_go_local(
                            g.dot(robot.grasp_frame_offset),
                            pickup.dot(robot.grasp_frame_offset),
                            rospy.Duration(2.0),
                            timeout=3,
                        )

                        robot.poserbpf_client(3, 0)

                        done_objects.add(obj_name)

                        # drop off in front of the robot
                        theta_drop = np.random.uniform(-np.pi, np.pi)
                        radius_drop = np.random.uniform(0.05, 0.1)
                        dropoff[0, 3] = center_all[0] + np.cos(theta_drop) * radius_drop
                        dropoff[1, 3] = center_all[1] + np.sin(theta_drop) * radius_drop
                        original_height = dropoff[2, 3]
                        dropoff[2, 3] = original_height

                        step_plan = 0
                        while not robot.plan(dropoff.dot(robot.grasp_frame_offset)):
                            theta_drop = np.random.uniform(-np.pi, np.pi)
                            radius_drop = np.random.uniform(0.05, 0.1)
                            dropoff[0, 3] = center_all[0] + np.cos(theta_drop) * radius_drop
                            dropoff[1, 3] = center_all[1] + np.sin(theta_drop) * radius_drop
                            original_height = dropoff[2, 3]
                            dropoff[2, 3] = original_height
                            step_plan += 1
                            if step_plan > 10:
                                break

                        robot.interpolate_go_local(
                            pickup.dot(robot.grasp_frame_offset),
                            dropoff.dot(robot.grasp_frame_offset),
                            rospy.Duration(3.0),
                            timeout=5,
                        )

                        robot.open_gripper()

                        # move the gripper up
                        dropoff_finish_pose = dropoff.copy()
                        dropoff_finish_pose[2, 3] += 0.25
                        robot.interpolate_go_local(
                            dropoff.dot(robot.grasp_frame_offset),
                            dropoff_finish_pose.dot(robot.grasp_frame_offset),
                            rospy.Duration(3.0),
                            timeout=5
                        )

                        found = True
                        thread_stop_event.set()
                        break
                    else:
                        raise ValueError('failed to grasp object even though there was a plan.')

            if not found_feasible_grasp:
                push_object(robot, Tbo, center_all)
                break

        # raw_input('press any key to retract....')
        robot.retract()

    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)
