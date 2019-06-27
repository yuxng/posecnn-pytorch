import torch
import torch.nn as nn
import torchvision.models as models
import math
import sys
import copy
from torch.nn.init import kaiming_normal_
from layers.hard_label import HardLabel
from layers.hough_voting import HoughVoting
from layers.roi_pooling import RoIPool
from layers.point_matching_loss import PMLoss
from layers.roi_target_layer import roi_target_layer
from layers.pose_target_layer import pose_target_layer
from fcn.config import cfg

__all__ = [
    'posecnn', 'posecnn_rgbd',
]

vgg16 = models.vgg16(pretrained=False)

def conv(in_planes, out_planes, kernel_size=3, stride=1, relu=True):
    if relu:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True)


def fc(in_planes, out_planes, relu=True):
    if relu:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Linear(in_planes, out_planes)


def upsample(scale_factor):
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear')


def log_softmax_high_dimension(input):
    num_classes = input.size()[1]
    m = torch.max(input, dim=1, keepdim=True)[0]
    if input.dim() == 4:
        d = input - m.repeat(1, num_classes, 1, 1)
    else:
        d = input - m.repeat(1, num_classes)
    e = torch.exp(d)
    s = torch.sum(e, dim=1, keepdim=True)
    if input.dim() == 4:
        output = d - torch.log(s.repeat(1, num_classes, 1, 1))
    else:
        output = d - torch.log(s.repeat(1, num_classes))
    return output

def softmax_high_dimension(input):
    num_classes = input.size()[1]
    m = torch.max(input, dim=1, keepdim=True)[0]
    if input.dim() == 4:
        e = torch.exp(input - m.repeat(1, num_classes, 1, 1))
    else:
        e = torch.exp(input - m.repeat(1, num_classes))
    s = torch.sum(e, dim=1, keepdim=True)
    if input.dim() == 4:
        output = torch.div(e, s.repeat(1, num_classes, 1, 1))
    else:
        output = torch.div(e, s.repeat(1, num_classes))
    return output


class PoseCNN(nn.Module):

    def __init__(self, num_classes, num_units):
        super(PoseCNN, self).__init__()
        self.num_classes = num_classes

        # conv features
        features = list(vgg16.features)[:30]
        
        # change the first conv layer for RGBD
        if cfg.INPUT == 'RGBD':
            conv0 = conv(6, 64, kernel_size=3, relu=False)
            conv0.weight.data[:, :3, :, :] = features[0].weight.data
            conv0.weight.data[:, 3:, :, :] = features[0].weight.data
            conv0.bias.data = features[0].bias.data
            features[0] = conv0

        self.features = nn.ModuleList(features)
        self.classifier = vgg16.classifier[:-1]
        print(self.features)

        # freeze some layers
        if cfg.TRAIN.FREEZE_LAYERS:
            for i in [0, 2, 5, 7, 10, 12, 14]:
                self.features[i].weight.requires_grad = False
                self.features[i].bias.requires_grad = False

        # semantic labeling branch
        self.conv4_embed = conv(512, num_units, kernel_size=1)
        self.conv5_embed = conv(512, num_units, kernel_size=1)
        self.upsample_conv5_embed = upsample(2.0)
        self.upsample_embed = upsample(8.0)
        self.conv_score = conv(num_units, num_classes, kernel_size=1)
        self.hard_label = HardLabel(threshold=cfg.TRAIN.HARD_LABEL_THRESHOLD, sample_percentage=cfg.TRAIN.HARD_LABEL_SAMPLING)
        self.dropout = nn.Dropout()

        if cfg.TRAIN.VERTEX_REG:
            # center regression branch
            self.conv4_vertex_embed = conv(512, 2*num_units, kernel_size=1, relu=False)
            self.conv5_vertex_embed = conv(512, 2*num_units, kernel_size=1, relu=False)
            self.upsample_conv5_vertex_embed = upsample(2.0)
            self.upsample_vertex_embed = upsample(8.0)
            self.conv_vertex_score = conv(2*num_units, 3*num_classes, kernel_size=1, relu=False)
            # hough voting
            self.hough_voting = HoughVoting(is_train=0, skip_pixels=10, label_threshold=100, \
                                            inlier_threshold=0.9, voting_threshold=-1, per_threshold=0.01)

            self.roi_pool_conv4 = RoIPool(pool_height=7, pool_width=7, spatial_scale=1.0 / 8.0)
            self.roi_pool_conv5 = RoIPool(pool_height=7, pool_width=7, spatial_scale=1.0 / 16.0)
            self.fc8 = fc(4096, num_classes)
            self.fc9 = fc(4096, 4 * num_classes, relu=False)

            if cfg.TRAIN.POSE_REG:
                self.fc10 = fc(4096, 4 * num_classes, relu=False)
                self.pml = PMLoss(hard_angle=cfg.TRAIN.HARD_ANGLE)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, label_gt, meta_data, extents, gt_boxes, poses, points, symmetry):

        # conv features
        for i, model in enumerate(self.features):
            x = model(x)
            if i == 22:
                out_conv4_3 = x
            if i == 29:
                out_conv5_3 = x

        # semantic labeling branch
        out_conv4_embed = self.conv4_embed(out_conv4_3)
        out_conv5_embed = self.conv5_embed(out_conv5_3)
        out_conv5_embed_up = self.upsample_conv5_embed(out_conv5_embed)
        out_embed = self.dropout(out_conv4_embed + out_conv5_embed_up)
        out_embed_up = self.upsample_embed(out_embed)
        out_score = self.conv_score(out_embed_up)
        out_logsoftmax = log_softmax_high_dimension(out_score)
        out_prob = softmax_high_dimension(out_score)
        out_label = torch.max(out_prob, dim=1)[1].type(torch.IntTensor).cuda()
        out_weight = self.hard_label(out_prob, label_gt, torch.rand(out_prob.size()).cuda())

        if cfg.TRAIN.VERTEX_REG:
            # center regression branch
            out_conv4_vertex_embed = self.conv4_vertex_embed(out_conv4_3)
            out_conv5_vertex_embed = self.conv5_vertex_embed(out_conv5_3)
            out_conv5_vertex_embed_up = self.upsample_conv5_vertex_embed(out_conv5_vertex_embed)
            out_vertex_embed = self.dropout(out_conv4_vertex_embed + out_conv5_vertex_embed_up)
            out_vertex_embed_up = self.upsample_vertex_embed(out_vertex_embed)
            out_vertex = self.conv_vertex_score(out_vertex_embed_up)

            # hough voting
            if self.training:
                self.hough_voting.is_train = 1
                self.hough_voting.label_threshold=cfg.TRAIN.HOUGH_LABEL_THRESHOLD
                self.hough_voting.voting_threshold=cfg.TRAIN.HOUGH_VOTING_THRESHOLD
                self.hough_voting.skip_pixels=cfg.TRAIN.HOUGH_SKIP_PIXELS
            else:
                self.hough_voting.is_train = 0
                self.hough_voting.label_threshold=cfg.TEST.HOUGH_LABEL_THRESHOLD
                self.hough_voting.voting_threshold=cfg.TEST.HOUGH_VOTING_THRESHOLD
                self.hough_voting.skip_pixels=cfg.TEST.HOUGH_SKIP_PIXELS
            out_box, out_pose = self.hough_voting(out_label, out_vertex, meta_data, extents)

            # bounding box classification and regression branch
            bbox_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_target_layer(out_box, gt_boxes)
            out_roi_conv4 = self.roi_pool_conv4(out_conv4_3, out_box)
            out_roi_conv5 = self.roi_pool_conv5(out_conv5_3, out_box)
            out_roi = out_roi_conv4 + out_roi_conv5
            out_roi_flatten = out_roi.view(out_roi.size(0), -1)
            out_fc7 = self.classifier(out_roi_flatten)
            out_fc8 = self.fc8(out_fc7)
            out_logsoftmax_box = log_softmax_high_dimension(out_fc8)
            bbox_prob = softmax_high_dimension(out_fc8)
            bbox_label_weights = self.hard_label(bbox_prob, bbox_labels, torch.rand(bbox_prob.size()).cuda())
            bbox_pred = self.fc9(out_fc7)

            # rotation regression branch
            rois, poses_target, poses_weight = pose_target_layer(out_box, bbox_prob, bbox_pred, gt_boxes, poses, self.training)
            if cfg.TRAIN.POSE_REG:    
                out_qt_conv4 = self.roi_pool_conv4(out_conv4_3, rois)
                out_qt_conv5 = self.roi_pool_conv5(out_conv5_3, rois)
                out_qt = out_qt_conv4 + out_qt_conv5
                out_qt_flatten = out_qt.view(out_qt.size(0), -1)
                out_qt_fc7 = self.classifier(out_qt_flatten)
                out_quaternion = self.fc10(out_qt_fc7)
                # point matching loss
                poses_pred = nn.functional.normalize(torch.mul(out_quaternion, poses_weight))
                if self.training:
                    loss_pose = self.pml(poses_pred, poses_target, poses_weight, points, symmetry)

        if self.training:
            if cfg.TRAIN.VERTEX_REG:
                if cfg.TRAIN.POSE_REG:
                    return out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, bbox_label_weights, \
                           bbox_pred, bbox_targets, bbox_inside_weights, loss_pose, poses_weight
                else:
                    return out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, bbox_label_weights, \
                           bbox_pred, bbox_targets, bbox_inside_weights
            else:
                return out_logsoftmax, out_weight
        else:
            if cfg.TRAIN.VERTEX_REG:
                if cfg.TRAIN.POSE_REG:
                    return out_label, out_vertex, rois, out_pose, out_quaternion
                else:
                    return out_label, out_vertex, rois, out_pose
            else:
                return out_label

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class PoseCNN_RGBD(nn.Module):

    def __init__(self, num_classes, num_units):
        super(PoseCNN_RGBD, self).__init__()
        self.num_classes = num_classes

        # conv features
        features = list(vgg16.features)[:30]
        self.features_color = nn.ModuleList(features)
        self.features_depth = copy.deepcopy(self.features_color)
        self.features_depth[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.classifier = vgg16.classifier[:-1]
        print(self.features_color)
        print(self.features_depth)

        # freeze some layers
        if cfg.TRAIN.FREEZE_LAYERS:
            for i in [0, 2, 5, 7, 10, 12, 14]:
                self.features_color[i].weight.requires_grad = False
                self.features_color[i].bias.requires_grad = False
                self.features_depth[i].weight.requires_grad = False
                self.features_depth[i].bias.requires_grad = False

        # semantic labeling branch
        self.conv4_embed = conv(512, num_units, kernel_size=1)
        self.conv5_embed = conv(512, num_units, kernel_size=1)
        self.upsample_conv5_embed = upsample(2.0)
        self.upsample_embed = upsample(8.0)
        self.conv_score = conv(num_units, num_classes, kernel_size=1)
        self.hard_label = HardLabel(threshold=cfg.TRAIN.HARD_LABEL_THRESHOLD, sample_percentage=cfg.TRAIN.HARD_LABEL_SAMPLING)
        self.dropout = nn.Dropout()

        if cfg.TRAIN.VERTEX_REG:
            # center regression branch
            self.conv4_vertex_embed = conv(512, 2*num_units, kernel_size=1, relu=False)
            self.conv5_vertex_embed = conv(512, 2*num_units, kernel_size=1, relu=False)
            self.upsample_conv5_vertex_embed = upsample(2.0)
            self.upsample_vertex_embed = upsample(8.0)
            self.conv_vertex_score = conv(2*num_units, 3*num_classes, kernel_size=1, relu=False)
            # hough voting
            self.hough_voting = HoughVoting(is_train=0, skip_pixels=10, label_threshold=100, \
                                            inlier_threshold=0.9, voting_threshold=-1, per_threshold=0.01)

            self.roi_pool_conv4 = RoIPool(pool_height=7, pool_width=7, spatial_scale=1.0 / 8.0)
            self.roi_pool_conv5 = RoIPool(pool_height=7, pool_width=7, spatial_scale=1.0 / 16.0)
            self.fc8 = fc(4096, num_classes)
            self.fc9 = fc(4096, 4 * num_classes, relu=False)
            self.fc10 = fc(4096, 4 * num_classes, relu=False)
            self.pml = PMLoss(hard_angle=cfg.TRAIN.HARD_ANGLE)

        elif cfg.TRAIN.VERTEX_REG_DELTA:
            # 3D center regression
            self.conv4_vertex_embed = conv(512, 2*num_units, kernel_size=1, relu=False)
            self.conv5_vertex_embed = conv(512, 2*num_units, kernel_size=1, relu=False)
            self.upsample_conv5_vertex_embed = upsample(2.0)
            self.upsample_vertex_embed = upsample(8.0)
            self.conv_vertex_score = conv(2 * num_units, 3 * num_classes, kernel_size=1, relu=False)

            # we append features used during segmentation and pass everything through two 1x1 convolutions
            self.regression_conv1 = conv(3 * num_units, 256, kernel_size=1, relu=True)
            self.regression_conv2 = conv(256, 2 * num_units, kernel_size=1, relu=True)

            self.conv_vertex_score = conv(2 * num_units, 3 * num_classes, kernel_size=1, relu=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x_rgbd, label_gt, meta_data, extents, gt_boxes, poses, points, symmetry):

        # conv features color
        x = x_rgbd[:, :3, :, :]
        for i, model in enumerate(self.features_color):
            x = model(x)
            if i == 22:
                out_conv4_3_color = x
            if i == 29:
                out_conv5_3_color = x

        # conv features depth
        x = x_rgbd[:, 3:, :, :]
        depth_mask = x_rgbd[:, 6, :, :]
        for i, model in enumerate(self.features_depth):
            x = model(x)
            if i == 22:
                out_conv4_3_depth = x
            if i == 29:
                out_conv5_3_depth = x

        # concatenate color and depth feature
        out_conv4_3 = out_conv4_3_color + out_conv4_3_depth
        out_conv5_3 = out_conv5_3_color + out_conv5_3_depth

        # semantic labeling branch
        out_conv4_embed = self.conv4_embed(out_conv4_3)
        out_conv5_embed = self.conv5_embed(out_conv5_3)
        out_conv5_embed_up = self.upsample_conv5_embed(out_conv5_embed)
        out_embed = self.dropout(out_conv4_embed + out_conv5_embed_up)
        out_embed_up = self.upsample_embed(out_embed)
        out_score = self.conv_score(out_embed_up)
        out_logsoftmax = log_softmax_high_dimension(out_score)
        out_prob = softmax_high_dimension(out_score)
        out_label = torch.max(out_prob, dim=1)[1].type(torch.IntTensor).cuda()
        out_weight = self.hard_label(out_prob, label_gt, torch.rand(out_prob.size()).cuda())

        if cfg.TRAIN.VERTEX_REG:
            # center regression branch
            out_conv4_vertex_embed = self.conv4_vertex_embed(out_conv4_3)
            out_conv5_vertex_embed = self.conv5_vertex_embed(out_conv5_3)
            out_conv5_vertex_embed_up = self.upsample_conv5_vertex_embed(out_conv5_vertex_embed)
            out_vertex_embed = self.dropout(out_conv4_vertex_embed + out_conv5_vertex_embed_up)
            out_vertex_embed_up = self.upsample_vertex_embed(out_vertex_embed)
            out_vertex = self.conv_vertex_score(out_vertex_embed_up)

            # hough voting
            if self.training:
                self.hough_voting.is_train = 1
                self.hough_voting.label_threshold=cfg.TRAIN.HOUGH_LABEL_THRESHOLD
                self.hough_voting.voting_threshold=cfg.TRAIN.HOUGH_VOTING_THRESHOLD
                self.hough_voting.skip_pixels=cfg.TRAIN.HOUGH_SKIP_PIXELS
            else:
                self.hough_voting.is_train = 0
                self.hough_voting.label_threshold=cfg.TEST.HOUGH_LABEL_THRESHOLD
                self.hough_voting.voting_threshold=cfg.TEST.HOUGH_VOTING_THRESHOLD
                self.hough_voting.skip_pixels=cfg.TEST.HOUGH_SKIP_PIXELS
            out_box, out_pose = self.hough_voting(out_label, out_vertex, meta_data, extents)

            # bounding box classification and regression branch
            bbox_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_target_layer(out_box, gt_boxes)
            out_roi_conv4 = self.roi_pool_conv4(out_conv4_3, out_box)
            out_roi_conv5 = self.roi_pool_conv5(out_conv5_3, out_box)
            out_roi = out_roi_conv4 + out_roi_conv5
            out_roi_flatten = out_roi.view(out_roi.size(0), -1)
            out_fc7 = self.classifier(out_roi_flatten)
            out_fc8 = self.fc8(out_fc7)
            out_logsoftmax_box = log_softmax_high_dimension(out_fc8)
            bbox_prob = softmax_high_dimension(out_fc8)
            bbox_label_weights = self.hard_label(bbox_prob, bbox_labels, torch.rand(bbox_prob.size()).cuda())
            bbox_pred = self.fc9(out_fc7)

            # rotation regression branch
            rois, poses_target, poses_weight = pose_target_layer(out_box, bbox_prob, bbox_pred, gt_boxes, poses, self.training)
            out_qt_conv4 = self.roi_pool_conv4(out_conv4_3, rois)
            out_qt_conv5 = self.roi_pool_conv5(out_conv5_3, rois)
            out_qt = out_qt_conv4 + out_qt_conv5
            out_qt_flatten = out_qt.view(out_qt.size(0), -1)
            out_qt_fc7 = self.classifier(out_qt_flatten)
            out_quaternion = self.fc10(out_qt_fc7)
            # point matching loss
            poses_pred = nn.functional.normalize(torch.mul(out_quaternion, poses_weight))
            if self.training:
                loss_pose = self.pml(poses_pred, poses_target, poses_weight, points, symmetry)


        elif cfg.TRAIN.VERTEX_REG_DELTA:
            # center regression branch
            out_conv4_vertex_embed = self.conv4_vertex_embed(out_conv4_3)
            out_conv5_vertex_embed = self.conv5_vertex_embed(out_conv5_3)
            out_conv5_vertex_embed_up = self.upsample_conv5_vertex_embed(out_conv5_vertex_embed)
            out_vertex_embed = self.dropout(out_conv4_vertex_embed + out_conv5_vertex_embed_up)
            out_vertex_embed_up = self.upsample_vertex_embed(out_vertex_embed)

            out_vertex_embed_up = torch.cat((out_vertex_embed_up, out_embed_up.detach()), dim=1)
            out_vertex_embed_up = self.regression_conv1(out_vertex_embed_up)
            out_vertex_embed_up = self.regression_conv2(out_vertex_embed_up)

            out_vertex = self.conv_vertex_score(out_vertex_embed_up)

            if not self.training:
                batch_size = out_vertex.shape[0]
                stacked_size = out_vertex.shape[1]
                height = out_vertex.shape[2]
                width = out_vertex.shape[3]
                nclasses = stacked_size / 3
                out_label_indices = out_label.unsqueeze(1).long()
                label_onehot = torch.zeros(out_score.shape).cuda()
                label_onehot.scatter_(1, out_label_indices, 1)

                extents_batch = extents.repeat(batch_size, 1, 1)
                extents_largest_dim = torch.sqrt((extents_batch * extents_batch).sum(dim=2))
                extents_largest_dim = extents_largest_dim.repeat(1, 3)
                extents_largest_dim = extents_largest_dim.reshape(batch_size, 3, nclasses)
                extents_largest_dim = extents_largest_dim.transpose(1, 2).reshape(batch_size, 3 * nclasses)

                label_onehot_tiled = label_onehot.repeat(1, 1, 3, 1).view(batch_size, -1, height, width)
                xyz_images = x_rgbd[:, 3:6, :, :]
                xyz_images = xyz_images.repeat(1, stacked_size / 3, 1, 1)
                mask_repeat = depth_mask.repeat(1, stacked_size, 1, 1)
                label_onehot_tiled = label_onehot_tiled * mask_repeat

                delta_centers = out_vertex * label_onehot_tiled * extents_largest_dim.unsqueeze(2).unsqueeze(2) * 0.5
                xyz_centers = xyz_images * label_onehot_tiled

                center_predictions = xyz_centers - delta_centers

                object_centers = torch.zeros(batch_size, nclasses * 3).cuda().float()
                for b in range(batch_size):
                    for k in range(object_centers.shape[1]):
                        valid_points = torch.masked_select(center_predictions[b, k, :, :], label_onehot_tiled[b, k, :, :].byte())
                        if valid_points.shape[0]:
                            med_value = torch.median(valid_points)
                            object_centers[b, k] = med_value
                        else:
                            object_centers[b, k] = 0.0

                min_coords = object_centers - extents_largest_dim/2.0
                max_coords = object_centers + extents_largest_dim/2.0

                min_coords = min_coords.reshape(batch_size, nclasses, 3)
                max_coords = max_coords.reshape(batch_size, nclasses, 3)

                object_centers_reshape = object_centers.reshape(batch_size, nclasses, 3)

                zs = torch.clamp(object_centers_reshape[:, :, 2], min=0.001)

                x_mins = min_coords[:, :, 0]
                x_maxs = max_coords[:, :, 0]

                y_mins = min_coords[:, :, 1]
                y_maxs = max_coords[:, :, 1]

                fx = cfg.INTRINSICS[0]
                px = cfg.INTRINSICS[2]
                fy = cfg.INTRINSICS[4]
                py = cfg.INTRINSICS[5]

                col_mins = x_mins / zs * fx + px
                col_maxs = x_maxs / zs * fx + px

                row_mins = y_mins / zs * fy + py
                row_maxs = y_maxs / zs * fy + py

                col_mins = torch.clamp(col_mins, min=0.0, max=width)
                row_mins = torch.clamp(row_mins, min=0.0, max=height)

                col_maxs = torch.clamp(col_maxs, min=0.0, max=width)
                row_maxs = torch.clamp(row_maxs, min=0.0, max=height)

                col_mins = col_mins.reshape(nclasses * batch_size, 1)
                row_mins = row_mins.reshape(nclasses * batch_size, 1)
                col_maxs = col_maxs.reshape(nclasses * batch_size, 1)
                row_maxs = row_maxs.reshape(nclasses * batch_size, 1)

                class_range = torch.arange(nclasses).float()
                class_range = class_range.repeat(batch_size)
                class_range = class_range.unsqueeze_(1).reshape(nclasses * batch_size, 1)

                batch_ids = torch.arange(batch_size)
                batch_ids = batch_ids.repeat(nclasses).unsqueeze(1)
                batch_ids = batch_ids.reshape(nclasses, batch_size)
                batch_ids = batch_ids.transpose(0, 1).reshape(nclasses * batch_size, 1).float()

                bounding_boxes = torch.cat((batch_ids.cuda(), class_range.cuda(), col_mins, row_mins, col_maxs, row_maxs), dim=1)


        if self.training:
            if cfg.TRAIN.VERTEX_REG:
                return out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, bbox_label_weights, \
                       bbox_pred, bbox_targets, bbox_inside_weights, loss_pose, poses_weight
            elif cfg.TRAIN.VERTEX_REG_DELTA:
                return out_logsoftmax, out_weight, out_vertex
            else:
                return out_logsoftmax, out_weight
        else:
            if cfg.TRAIN.VERTEX_REG:
                return out_label, out_vertex, rois, out_pose, out_quaternion
            elif cfg.TRAIN.VERTEX_REG_DELTA:
                return out_label, out_vertex, object_centers, label_onehot, bounding_boxes
            else:
                return out_label

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def posecnn(num_classes, num_units, data=None):

    model = PoseCNN(num_classes, num_units)

    if data is not None:
        model_dict = model.state_dict()
        print 'model keys'
        print '================================================='
        for k, v in model_dict.items():
            print k
        print '================================================='

        print 'data keys'
        print '================================================='
        for k, v in data.items():
            print k
        print '================================================='

        pretrained_dict = {k: v for k, v in data.items() if k in model_dict and v.size() == model_dict[k].size()}
        print 'load the following keys from the pretrained model'
        print '================================================='
        for k, v in pretrained_dict.items():
            print k
        print '================================================='
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)

    return model


def posecnn_rgbd(num_classes, num_units, data=None):

    model = PoseCNN_RGBD(num_classes, num_units)

    if data is not None:
        model_dict = model.state_dict()
        print 'model keys'
        print '================================================='
        for k, v in model_dict.items():
            print k
        print '================================================='

        print 'data keys'
        print '================================================='
        for k, v in data.items():
            print k
        print '================================================='

        # construct the dictionary for update
        update_dict = dict()
        for mk, mv in model_dict.items():
            key = mk

            if key in data:
                update_dict[mk] = data[key]
            else:
                # remove color or depth in the model key
                pos = mk.find('_color')
                if pos > 0:
                    key = mk[:pos] + mk[pos+6:]

                pos = mk.find('_depth')
                if pos > 0 and not ('features_depth.0' in key):
                    key = mk[:pos] + mk[pos+6:]

                if key in data:
                    update_dict[mk] = data[key]

        print 'load the following keys from the pretrained model'
        print '================================================='
        for k, v in update_dict.items():
            print k
        print '================================================='
        model_dict.update(update_dict) 
        model.load_state_dict(model_dict)

    return model
