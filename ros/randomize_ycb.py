#!/usr/bin/env python2.7
#
# Copyright (c) 2019, NVIDIA  All rights reserved.

from __future__ import print_function

from lula_control.world import make_basic_world
from lula_control.robot_factory import RobotFactory
from lula_control.control_visualizer import ControlVisualizer
from brain_ros.lula_tools import lula_go_local
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from isaac_bridge.srv import IsaacPose
from sensor_msgs.msg import Image, JointState, CameraInfo
from std_msgs.msg import Header
from tf import transformations
from isaac_bridge.manager import SimulationManager
from transforms3d.quaternions import mat2quat, quat2mat, qmult

import cv2
import os
import rospy
import copy
import numpy as np
import pdb
import random
import signal
import std_srvs.srv
import tf
import time

try:
    import PyKDL as kdl
    from kdl_parser_py import urdf
except ImportError as e:
    rospy.logwarn("Could not load kdl parser. Try: `sudo apt install "
                  "ros-kinetic-kdl-*`")
    raise e

pallete = [
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0.5, 0.5, 0],
    [1, 1, 1],
    [0.5, 1, 1],  # Franka
    [0, 1, 1],
    [0, 0.5, 0.5],
    [1, 0.5, 1],  # extractor hood
    [1, 1, 0.5],  # range
    [0, 0.5, 0.5],  # hitman
    [0, 0.5, 0.5],  # chewie
    [0, 0.25, 0.5],  # golf + hitman top
]


def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T


def map_seg_image(image):
    image = np.squeeze(image)
    output_image = [
        np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) for _ in range(3)]
    for i, color in enumerate(pallete):
        mask = image == (i + 1)
        for j in range(3):
            output_image[j][mask] = color[2 - j] * 255
    for i in range(3):
        output_image[i] = np.expand_dims(output_image[i], -1)

    return np.concatenate(output_image, -1)

# initialize forward kinematics
visualizer = ControlVisualizer()
base_link = 'base_link'
ee_link = 'right_gripper'

success, kdl_tree = urdf.treeFromParam('/robot_description')
if not success:
    raise RuntimeError(
        "Could not create kinematic tree from /robot_description.")

kdl_chain = kdl_tree.getChain(base_link, ee_link)
print("============ N of joints in KDL chain:",
      kdl_chain.getNrOfJoints())
kdl_fk = kdl.ChainFkSolverPos_recursive(kdl_chain)


def joint_list_to_kdl(q):
    if q is None:
        return None
    if isinstance(q, np.matrix) and q.shape[1] == 0:
        q = q.T.tolist()[0]
    q_kdl = kdl.JntArray(len(q))
    for i, q_i in enumerate(q):
        q_kdl[i] = q_i
    return q_kdl


def forward_kinematics(q):
    ee_frame = kdl.Frame()
    kinematics_status = kdl_fk.JntToCart(joint_list_to_kdl(q),
                                         ee_frame)
    if kinematics_status >= 0:
        p = ee_frame.p
        M = ee_frame.M
        return np.mat([[M[0, 0], M[0, 1], M[0, 2], p.x()],
                       [M[1, 0], M[1, 1], M[1, 2], p.y()],
                       [M[2, 0], M[2, 1], M[2, 2], p.z()],
                       [0, 0, 0, 1]])
    else:
        return None


def make_pose(trans, rot):
    """
    Helper function to get a full matrix out of this pose
    """
    pose = transformations.quaternion_matrix(rot)
    pose[:3, 3] = trans
    return pose

def go_local(T, robot, q=None, wait=False):
    # T = ros.make_pose((pos, rot))
    visualizer.send(T)
    #config_modulator.send_config(q)
    lula_go_local(robot.end_effector, T, wait_for_target=wait, high_precision=True)

def go_home(robot):
    default_home_q = [0.01200158428400755, -0.5697816014289856,
                            5.6801487517077476e-05,
                            -2.8105969429016113, -0.00025768374325707555, 3.0363450050354004,
                            0.7410701513290405]
    T = forward_kinematics(default_home_q)
    go_local(T, robot, default_home_q, True)

class generate_franka_states:
    def __init__(self):
        self.cv_bridge = CvBridge()
        # Robot joint publisher
        self.pub = rospy.Publisher("sim/desired_joint_states", JointState, queue_size=10)

        # Robot joint subscriber
        rospy.Subscriber('/robot/joint_states', JointState, self.robot_state_callback)

        # Image subscribers
        rospy.Subscriber("/sim/center_color_camera/image", Image, self.center_rgb_img_callback)
        rospy.Subscriber("/sim/left_color_camera/image", Image, self.left_rgb_img_callback)
        rospy.Subscriber("/sim/right_color_camera/image", Image, self.right_rgb_img_callback)
        rospy.Subscriber("/sim/center_depth_camera/image", Image, self.center_depth_img_callback)
        rospy.Subscriber("/sim/left_depth_camera/image", Image, self.left_depth_img_callback)
        rospy.Subscriber("/sim/right_depth_camera/image", Image, self.right_depth_img_callback)
        rospy.Subscriber("/sim/center_segmentation_camera/label_image", Image, self.segmentation_img_callback)

        # camera intrinsics
        msg = rospy.wait_for_message('/sim/center_color_camera/camera_info', CameraInfo)
        self.center_intrinsics = np.array(msg.K).reshape(3, 3).copy()
        msg = rospy.wait_for_message('/sim/left_color_camera/camera_info', CameraInfo)
        self.left_intrinsics = np.array(msg.K).reshape(3, 3).copy()
        msg = rospy.wait_for_message('/sim/right_color_camera/camera_info', CameraInfo)
        self.right_intrinsics = np.array(msg.K).reshape(3, 3).copy()

        # ROS DR service
        self.dr_sim = rospy.ServiceProxy('sim/dr', std_srvs.srv.Empty)
        # ROS pose service
        self.pose_srv = rospy.ServiceProxy("sim/pose_command", IsaacPose)
        self.sphere_names = ["cracker_box", "mustard_bottle", "potted_meat_can", "sugar_box", "tomato_soup_can"]
        # Robot
        self.robot = RobotFactory(is_sim=True).create('franka')
        # TF Listener
        self.tf_listener = tf.TransformListener()

        # Folder path
        self.rgb_img_dir = "./rendered_dataset/rgb/"
        if not os.path.exists(self.rgb_img_dir):
            os.makedirs(self.rgb_img_dir)
        self.depth_img_dir = "./rendered_dataset/depth/"
        if not os.path.exists(self.depth_img_dir):
            os.makedirs(self.depth_img_dir)
        self.segmentation_img_dir = "./rendered_dataset/segmentation/"
        if not os.path.exists(self.segmentation_img_dir):
            os.makedirs(self.segmentation_img_dir)
        self.state_dir = "./rendered_dataset/state/"
        if not os.path.exists(self.state_dir):
            os.makedirs(self.state_dir)
        self.robot_state_file = self.state_dir + "robot_state"
        self.object_state_file = self.state_dir + "object_state"

        # Data to save
        self.left_rgb_img = None
        self.left_depth_img = None
        self.right_rgb_img = None
        self.right_depth_img = None
        self.center_rgb_img = None
        self.center_depth_img = None
        self.segmentation_img = None
        self.robot_state = None
        self.obj_pose = None

        # Helper variables
        self.enable_save = False
        self.name_lst = []
        self.position_lst = []
        self.velocity_lst = []
        self.effort_lst = []
        self.pose_lst = []

    def center_rgb_img_callback(self, data):
        self.center_rgb_img = self.cv_bridge.imgmsg_to_cv2(data)

    def left_rgb_img_callback(self, data):
        self.left_rgb_img = self.cv_bridge.imgmsg_to_cv2(data)

    def right_rgb_img_callback(self, data):
        self.right_rgb_img = self.cv_bridge.imgmsg_to_cv2(data)

    def center_depth_img_callback(self, data):
        self.center_depth_img = data

    def left_depth_img_callback(self, data):
        self.left_depth_img = data

    def right_depth_img_callback(self, data):
        self.right_depth_img = data

    def segmentation_img_callback(self, data):
        self.segmentation_img = self.cv_bridge.imgmsg_to_cv2(data)

    def robot_state_callback(self, data):
        self.robot_state = data

    def get_data(self, camera):
        if camera == 'left':
            rgb_img = self.left_rgb_img
            depth_img = self.left_depth_img
            intrinsics = self.left_intrinsics
        elif camera == 'right':
            rgb_img = self.right_rgb_img
            depth_img = self.right_depth_img
            intrinsics = self.right_intrinsics
        elif camera == 'center':
            rgb_img = self.center_rgb_img
            depth_img = self.center_depth_img
            intrinsics = self.center_intrinsics

        # pose
        cls_names = []
        poses = []
        for i in range(len(self.sphere_names)):
            try:
                name = self.sphere_names[i]
                source_frame = '00_' + name
                target_frame = '00_zed_' + camera
                trans, rot = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
                RT = ros_qt_to_rt(rot, trans)
                cls_names.append(name)
                poses.append(RT)
                print(source_frame, ',', target_frame)
                print(RT)
            except:
                print('tf not found', source_frame, target_frame)
                continue
        return rgb_img, depth_img, intrinsics, cls_names, poses

    def save(self, data, type, num=0):
        num = "{:05d}".format(num)
        if type == "rgb_img":
            cv2.imwrite(self.rgb_img_dir + str(num)
                        + ".jpeg", data)

        if type == "depth_img":
            depth = np.frombuffer(data.data, dtype=np.float32).reshape(
                data.height, data.width, -1)
            max_val = np.max(depth)
            depth = depth/max_val
            depth *= 255
            depth = depth.astype(np.uint8)
            cv2.imwrite(self.depth_img_dir + str(num)
                        + ".png", depth)

        if type == "segmentation_img":
            cv2.imwrite(self.segmentation_img_dir + str(
                num) + ".png", map_seg_image(data))

        if type == "robot_state":
            self.name_lst.append(np.array(data.name))
            self.position_lst.append(np.array(data.position))
            self.velocity_lst.append(np.array(data.velocity))
            self.effort_lst.append(np.array(data.effort))
            np.save(self.robot_state_file, np.array(
                [self.name_lst, self.position_lst, self.velocity_lst, self.effort_lst]))

        if type == "object_state":
            self.pose_lst.append(np.array(data))
            np.save(self.object_state_file, np.array(self.pose_lst))


    # Helper function to find if two rectangles intersect
    def intersects(self, one_bottom_left, one_top_right, two_bottom_left, two_top_right):
        return not (one_top_right[0] < two_bottom_left[0] or one_bottom_left[0] > two_top_right[0] or one_top_right[1] < two_bottom_left[1] or one_bottom_left[1] > two_top_right[1])

    # Given (x,y) bounds, it tries to safely position all blocks
    def randomize_block_pos(self):
        r = 0.1
        x = [-0.5, 0.5]
        y = [-0.6, -0.1]
        success = True
        iteration = 0
        while success and iteration < 1000:
            center_dict = {}
            point_dict = {}
            iteration = iteration + 1
            print("Iteration: ", iteration)
            i = 0
            trial_cnt = 0
            while i < len(self.sphere_names):
                x_center = random.uniform(x[0], x[1])
                y_center = random.uniform(y[0], y[1])
                one_bottom_left = [x_center - r, y_center - r]
                one_top_right = [x_center + r, y_center + r]
                isNotOverlap = True
                for key in point_dict:
                    val = point_dict[key]
                    two_bottom_left = val[0]
                    two_top_right = val[1]
                    if self.intersects(one_bottom_left, one_top_right, two_bottom_left, two_top_right):
                        isNotOverlap = False
                if isNotOverlap:
                    point_dict[i] = [one_bottom_left, one_top_right]
                    center_dict[i] = [x_center, y_center]
                    i = i + 1
                    trial_cnt = 0
                    if len(center_dict) == len(self.sphere_names):
                        success = False
                        break
                else:
                    trial_cnt = trial_cnt + 1
                
                if trial_cnt == 1000:
                    break

        self.obj_poses = []
        if len(center_dict) == len(self.sphere_names):
            print(center_dict)
            for key in center_dict:
                val = center_dict[key]
                ball_pose = Pose()
                ball_pose.position.x = val[0]
                ball_pose.position.y = val[1]
                ball_pose.position.z = 1.0
                ball_pose.orientation.x = random.uniform(-1.0, 1.0)
                ball_pose.orientation.y = random.uniform(-1.0, 1.0)
                ball_pose.orientation.z = random.uniform(-1.0, 1.0)
                ball_pose.orientation.w = random.uniform(0.0, 1.0)
                ball_vel = Twist()
                ball_vel.linear.x = 0
                ball_vel.linear.y = 0
                ball_vel.linear.z = 0
                ball_vel.angular.x = 0
                ball_vel.angular.y = 0
                ball_vel.angular.z = 0
                self.obj_poses.append(np.array([ball_pose.position.x, ball_pose.position.y, ball_pose.position.z,
                    ball_pose.orientation.x, ball_pose.orientation.y, ball_pose.orientation.z, ball_pose.orientation.w]))

                if self.pose_srv:
                    self.pose_srv(header=Header(), names=[
                                self.sphere_names[key]], poses=[ball_pose],velocities=[ball_vel])

if __name__ == '__main__':

    rospy.init_node('randomize_ycb')
    tfs = generate_franka_states()
    should_loop = True
    timestep = 0
    r = rospy.Rate(10)
    # tfs.robot.end_effector.gripper.close(wait=False)
    enable_save = True
    while not rospy.is_shutdown() and should_loop:
        # DR
        tfs.dr_sim()

        # Lets start from home position
        tfs.robot.end_effector.retract(wait_for_target=True)
        
        # Randomize Block position
        tfs.randomize_block_pos()
        rospy.sleep(1.0)

        # Save the data
        if enable_save:

            # left camera
            rgb_img, depth_img, intrinsics, cls_names, poses = tfs.get_data(camera='left')

            # visualize

            timestep += 1

        # One trial is enough. Lets not loop anymore
        should_loop = True

    # Kill lula
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)
