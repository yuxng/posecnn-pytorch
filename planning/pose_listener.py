import rospy
import rosnode
import tf
import tf2_ros
import cv2
import numpy as np
import sys
import os
import tf.transformations as tra
from transforms3d.quaternions import mat2quat, quat2mat
from posecnn_pytorch.msg import DetectionList, Detection, BBox
from urdf_parser_py.urdf import URDF


def make_pose(tf_pose):
    """
    Helper function to get a full matrix out of this pose
    """
    trans, rot = tf_pose
    pose = tra.quaternion_matrix(rot)
    pose[:3, 3] = trans
    return pose


def make_pose_from_pose_msg(msg):
    trans = (msg.pose.position.x,
             msg.pose.position.y, msg.pose.position.z,)
    rot = (msg.pose.orientation.x,
           msg.pose.orientation.y,
           msg.pose.orientation.z,
           msg.pose.orientation.w,)
    return make_pose((trans, rot))


# query the kitchen pose
def parse_kitchen():
    kitchen = URDF.from_xml_string(open('kitchen_part_right.urdf', 'r+').read())
    link_map = kitchen.link_map
    kitchen_links = {}
    for link_name in link_map.keys():
        visual = link_map[link_name].visual
        if visual is not None:
            mesh = visual.geometry
            if hasattr(mesh, 'filename'):
                kitchen_links[link_name] = mesh.filename
    return kitchen_links


class PoseListener:

    """
    Listens on a particular message topic.
    """

    def __init__(self, topic_name, queue_size=100, max_dt=1.0):
        self.topic_name = topic_name
        self.msg = None
        self.max_dt = max_dt
        self.last_msg_time = rospy.Time(0)
        self.reset_t = rospy.Time(0).to_sec()
        self.detections = {}
        self._sub = rospy.Subscriber(self.topic_name, DetectionList, self.callback, queue_size=queue_size)

        self.listener = tf.TransformListener()
        self.kitchen_links = parse_kitchen()
        self.base_frame = 'measured/base_link'

    def ready(self):
        if self.msg is None:
            print("[SENSOR] No message received on topic", self.topic_name)
            return False
        t = rospy.Time.now()
        dt = (t - self.last_msg_time).to_sec()
        return dt < self.max_dt

    def reset(self):
        self.detections = {}
        self.reset_t = rospy.Time.now().to_sec()
        self.msg = None

    def callback(self, msg):
        """
        Records messages coming in from perception
        """

        self.last_msg_time = rospy.Time.now()
        if (self.last_msg_time.to_sec() - self.reset_t) < self.max_dt:
            return

        self.msg = msg
        # Run through all detections in the object
        for detection in self.msg.detections:
            name = detection.name
            # print(name)
            pose = make_pose_from_pose_msg(detection.pose)
            score = detection.score
            self.detections[name] = (pose, score, self.last_msg_time)

    def get_detections(self):
        return self.detections

    def get_pose(self, obj):
        # RBPF POSE
        t = rospy.Time.now()

        if obj in self.detections:
            pose, score, last_t = self.detections[obj]

            # Check failure case from PoseCNN
            if pose[2, 3] == 0:
                return False, None

            valid = (t - last_t).to_sec() <= self.max_dt
            valid = valid and (np.sum(pose[:3, 3]) != 0)
            return valid and score > 0, pose
        else:
            return False, None


    def get_tf_pose(self, target_frame, base_frame=None, is_matrix=False):
        if base_frame is None:
            base_frame = self.base_frame
        try:
            tf_pose = self.listener.lookupTransform(base_frame, target_frame, rospy.Time(0))
            if is_matrix:
                pose = make_pose(tf_pose)
            else:
                trans, rot = tf_pose
                qt = np.zeros((4,), dtype=np.float32)
                qt[0] = rot[3]
                qt[1] = rot[0]
                qt[2] = rot[1]
                qt[3] = rot[2]
                pose = np.zeros((7, ), dtype=np.float32)
                pose[:3] = trans
                pose[3:] = qt
        except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException):
            pose = None
        return pose


    def get_kitchen_pose(self, translate_z=0.0):
        # parse kitchen
        model_names = []
        object_poses = []
        object_names = []
        for link_name in self.kitchen_links.keys():
            pose = self.get_tf_pose(link_name)
            if pose is not None:
                pose[2] += translate_z
                mesh_file = self.kitchen_links[link_name].split('/')[-1]
                print(link_name, pose, mesh_file[:-4])

                model_names.append('kitchen_' + mesh_file[:-4])
                object_poses.append(pose)
                object_names.append(link_name)
        return model_names, object_poses, object_names


    def get_object_poses(self, block=False):
        while 1:
            model_names = []
            object_poses = []
            object_names = []
            for name in self.detections:
                obj_name = name.split("/")[-1]
                l = len(obj_name)
                obj_name = obj_name[3:l-3]
                if obj_name == 'cabinet_handle':
                    continue

                RT = self.detections[name][0]
                pose = np.zeros((7, ), dtype=np.float32)
                pose[:3] = RT[:3, 3]
                pose[3:] = mat2quat(RT[:3, :3])
                if pose[2] == 0:
                    continue
                print(obj_name)
                print(RT)
                object_poses.append(pose)
                model_names.append(obj_name)
                object_names.append(obj_name)
            if len(object_names) > 0:
                break
            else:
                print('cannot detect object')
                if not block:
                    break
                rospy.sleep(0.5)
        print(object_names)
        return model_names, object_poses, object_names
