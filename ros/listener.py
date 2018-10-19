import rospy
import message_filters
import cv2
import numpy as np
import torch
import torch.nn as nn

from fcn.config import cfg
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
from transforms3d.quaternions import mat2quat, quat2mat, qmult

class ImageListener:

    def __init__(self, network, dataset):

        self.net = network
        self.dataset = dataset
        self.cv_bridge = CvBridge()
        self.count = 0

        # initialize a node
        rospy.init_node("image_listener")
        self.label_pub = rospy.Publisher('posecnn_label', Image, queue_size=1)
        self.pose_pub = rospy.Publisher('posecnn_pose', Image, queue_size=1)
        rgb_sub = message_filters.Subscriber('/camera/rgb/image_color', Image, queue_size=2)
        depth_sub = message_filters.Subscriber('/camera/depth_registered/image', Image, queue_size=2)
        # depth_sub = message_filters.Subscriber('/camera/depth_registered/sw_registered/image_rect_raw', Image, queue_size=2)

        queue_size = 1
        slop_seconds = 0.025
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)

    def callback(self, rgb, depth):
        if depth.encoding == '32FC1':
            depth_32 = self.cv_bridge.imgmsg_to_cv2(depth) * 1000
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        # run network
        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        im_pose, im_label = self.test_image(im, depth_cv)

        # publish
        label_msg = self.cv_bridge.cv2_to_imgmsg(im_label)
        label_msg.header.stamp = rospy.Time.now()
        label_msg.header.frame_id = rgb.header.frame_id
        label_msg.encoding = 'rgb8'
        self.label_pub.publish(label_msg)

        pose_msg = self.cv_bridge.cv2_to_imgmsg(im_pose)
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = rgb.header.frame_id
        pose_msg.encoding = 'rgb8'
        self.pose_pub.publish(pose_msg)


    def test_image(self, im_color, im_depth):
        """segment image
        """

        num_classes = self.dataset.num_classes

        # compute image blob
        im = im_color.astype(np.float32, copy=True)
        im -= cfg.PIXEL_MEANS
        height = im.shape[0]
        width = im.shape[1]
        im = np.transpose(im / 255.0, (2, 0, 1))
        im = im[np.newaxis, :, :, :]

        # construct the meta data
        K = self.dataset._intrinsic_matrix
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros((1, 18), dtype=np.float32)
        meta_data_blob[0, 0:9] = K.flatten()
        meta_data_blob[0, 9:18] = Kinv.flatten()

        # use fake label blob
        label_blob = np.zeros((1, num_classes, height, width), dtype=np.float32)
        pose_blob = np.zeros((1, num_classes, 9), dtype=np.float32)
        gt_boxes = np.zeros((1, num_classes, 5), dtype=np.float32)

        # transfer to GPU
        inputs = torch.from_numpy(im).cuda()
        labels = torch.from_numpy(label_blob).cuda()
        meta_data = torch.from_numpy(meta_data_blob).cuda()
        extents = torch.from_numpy(self.dataset._extents).cuda()
        gt_boxes = torch.from_numpy(gt_boxes).cuda()
        poses = torch.from_numpy(pose_blob).cuda()
        points = torch.from_numpy(self.dataset._point_blob).cuda()
        symmetry = torch.from_numpy(self.dataset._symmetry).cuda()

        out_label, out_vertex, rois, out_pose, out_quaternion = self.net(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)

        # combine poses
        rois = rois.detach().cpu().numpy()
        out_pose = out_pose.detach().cpu().numpy()
        out_quaternion = out_quaternion.detach().cpu().numpy()
        num = rois.shape[0]
        poses = out_pose.copy()
        for j in xrange(num):
            cls = int(rois[j, 1])
            if cls >= 0:
                q = out_quaternion[j, 4*cls:4*cls+4]
                poses[j, :4] = q / np.linalg.norm(q)

        labels = out_label.detach().cpu().numpy()[0]
        im_pose, im_label = self.overlay_image(im_color, rois, poses, labels)

        return im_pose, im_label


    def overlay_image(self, im, rois, poses, labels):

        im = im[:, :, (2, 1, 0)]
        classes = self.dataset._classes
        class_colors = self.dataset._class_colors
        points = self.dataset._points_all
        intrinsic_matrix = self.dataset._intrinsic_matrix
        height = im.shape[0]
        width = im.shape[1]

        label_image = self.dataset.labels_to_image(labels)
        im_label = im.copy()
        I = np.where(labels != 0)
        im_label[I[0], I[1], :] = 0.5 * label_image[I[0], I[1], :] + 0.5 * im_label[I[0], I[1], :]

        for j in xrange(rois.shape[0]):
            cls = int(rois[j, 1])
            print classes[cls], rois[j, -1]
            if cls > 0 and rois[j, -1] > 0.01:

                # draw roi
                x1 = rois[j, 2]
                y1 = rois[j, 3]
                x2 = rois[j, 4]
                y2 = rois[j, 5]
                cv2.rectangle(im_label, (x1, y1), (x2, y2), class_colors[cls], 2)

                # extract 3D points
                x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                x3d[0, :] = points[cls,:,0]
                x3d[1, :] = points[cls,:,1]
                x3d[2, :] = points[cls,:,2]

                # projection
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = quat2mat(poses[j, :4])
                RT[:, 3] = poses[j, 4:7]
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x = np.round(np.divide(x2d[0, :], x2d[2, :]))
                y = np.round(np.divide(x2d[1, :], x2d[2, :]))
                index = np.where((x >= 0) & (x < width) & (y >= 0) & (y < height))[0]
                x = x[index].astype(np.int32)
                y = y[index].astype(np.int32)
                im[y, x, 0] = class_colors[cls][0]
                im[y, x, 1] = class_colors[cls][1]
                im[y, x, 2] = class_colors[cls][2]

        return im, im_label


    def vis_test(self, im, label, out_vertex, rois, poses):

        """Visualize a testing results."""
        import matplotlib.pyplot as plt

        num_classes = self.dataset.num_classes
        classes = self.dataset._classes
        class_colors = self.dataset._class_colors
        points = self.dataset._points_all
        intrinsic_matrix = self.dataset._intrinsic_matrix
        vertex_pred = out_vertex.detach().cpu().numpy()
        height = label.shape[0]
        width = label.shape[1]

        fig = plt.figure()
        # show image
        im = im[0, :, :, :].copy()
        im = im.transpose((1, 2, 0)) * 255.0
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(2, 4, 1)
        plt.imshow(im)
        ax.set_title('input image') 

        # show predicted label
        im_label = self.dataset.labels_to_image(label)
        ax = fig.add_subplot(2, 4, 2)
        plt.imshow(im_label)
        ax.set_title('predicted labels')

        if cfg.TRAIN.VERTEX_REG:

            # show predicted boxes
            ax = fig.add_subplot(2, 4, 3)
            plt.imshow(im)
            ax.set_title('predicted boxes')
            for j in range(rois.shape[0]):
                cls = rois[j, 1]
                x1 = rois[j, 2]
                y1 = rois[j, 3]
                x2 = rois[j, 4]
                y2 = rois[j, 5]
                plt.gca().add_patch(
                    plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=np.array(class_colors[int(cls)])/255.0, linewidth=3))

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                plt.plot(cx, cy, 'yo')

            # show predicted poses
            ax = fig.add_subplot(2, 4, 4)
            ax.set_title('predicted poses')
            plt.imshow(im)
            for j in xrange(rois.shape[0]):
                cls = int(rois[j, 1])
                print classes[cls], rois[j, -1]
                if cls > 0 and rois[j, -1] > 0.01:
                    # extract 3D points
                    x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                    x3d[0, :] = points[cls,:,0]
                    x3d[1, :] = points[cls,:,1]
                    x3d[2, :] = points[cls,:,2]

                    # projection
                    RT = np.zeros((3, 4), dtype=np.float32)
                    RT[:3, :3] = quat2mat(poses[j, :4])
                    RT[:, 3] = poses[j, 4:7]
                    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                    plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(class_colors[cls], 255.0), alpha=0.5)

            # show predicted vertex targets
            vertex_target = vertex_pred[0, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, num_classes):
                index = np.where(label == j)
                if len(index[0]) > 0:
                    center[0, index[0], index[1]] = vertex_target[3*j, index[0], index[1]]
                    center[1, index[0], index[1]] = vertex_target[3*j+1, index[0], index[1]]
                    center[2, index[0], index[1]] = np.exp(vertex_target[3*j+2, index[0], index[1]])

            ax = fig.add_subplot(2, 4, 5)
            plt.imshow(center[0,:,:])
            ax.set_title('predicted center x') 

            ax = fig.add_subplot(2, 4, 6)
            plt.imshow(center[1,:,:])
            ax.set_title('predicted center y')

            ax = fig.add_subplot(2, 4, 7)
            plt.imshow(center[2,:,:])
            ax.set_title('predicted z')

        plt.show()
