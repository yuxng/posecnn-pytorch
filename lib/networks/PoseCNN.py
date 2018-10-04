import torch
import torch.nn as nn
import torchvision.models as models
import math
import sys
from torch.nn.init import kaiming_normal_
from layers.hard_label import HardLabel
from layers.hough_voting import HoughVoting
from fcn.config import cfg

__all__ = [
    'posecnn',
]

vgg16 = models.vgg16(pretrained=True)

def conv(in_planes, out_planes, kernel_size=3, stride=1, relu=True):
    if relu:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True)

def upsample(scale_factor):
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear')

def log_softmax_high_dimension(input):
    num_classes = input.size()[1]
    m = torch.max(input, dim=1, keepdim=True)[0]
    d = input - m.repeat(1, num_classes, 1, 1)
    e = torch.exp(d)
    s = torch.sum(e, dim=1, keepdim=True)
    output = d - torch.log(s.repeat(1, num_classes, 1, 1))
    return output

def softmax_high_dimension(input):
    num_classes = input.size()[1]
    m = torch.max(input, dim=1, keepdim=True)[0]
    e = torch.exp(input - m.repeat(1, num_classes, 1, 1))
    s = torch.sum(e, dim=1, keepdim=True)
    output = torch.div(e, s.repeat(1, num_classes, 1, 1))
    return output


class PoseCNN(nn.Module):

    def __init__(self, num_classes, num_units):
        super(PoseCNN, self).__init__()
        self.num_classes = num_classes

        # conv features
        features = list(vgg16.features)[:30]
        self.features = nn.ModuleList(features).eval() 

        # semantic labeling branch
        self.conv4_embed = conv(512, num_units, kernel_size=1)
        self.conv5_embed = conv(512, num_units, kernel_size=1)
        self.upsample_conv5_embed = upsample(2.0)
        self.upsample_embed = upsample(8.0)
        self.conv_score = conv(num_units, num_classes, kernel_size=1)
        self.hard_label = HardLabel(threshold=1.0)

        # center regression branch
        if cfg.TRAIN.VERTEX_REG:
            self.conv4_vertex_embed = conv(512, 2*num_units, kernel_size=1, relu=False)
            self.conv5_vertex_embed = conv(512, 2*num_units, kernel_size=1, relu=False)
            self.upsample_conv5_vertex_embed = upsample(2.0)
            self.upsample_vertex_embed = upsample(8.0)
            self.conv_vertex_score = conv(2*num_units, 3*num_classes, kernel_size=1, relu=False)
            self.hough_voting = HoughVoting(is_train=0, skip_pixels=10, label_threshold=500, \
                                            inlier_threshold=0.9, voting_threshold=-1, per_threshold=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, label_gt, meta_data, extents):

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
        out_embed = out_conv4_embed + out_conv5_embed_up
        out_embed_up = self.upsample_embed(out_embed)
        out_score = self.conv_score(out_embed_up)
        out_logsoftmax = log_softmax_high_dimension(out_score)
        out_prob = softmax_high_dimension(out_score)
        out_label = torch.max(out_prob, dim=1)[1].type(torch.IntTensor).cuda()
        out_weight = self.hard_label(out_prob, label_gt)

        # center regression branch
        if cfg.TRAIN.VERTEX_REG:
            out_conv4_vertex_embed = self.conv4_vertex_embed(out_conv4_3)
            out_conv5_vertex_embed = self.conv5_vertex_embed(out_conv5_3)
            out_conv5_vertex_embed_up = self.upsample_conv5_vertex_embed(out_conv5_vertex_embed)
            out_vertex_embed = out_conv4_vertex_embed + out_conv5_vertex_embed_up
            out_vertex_embed_up = self.upsample_vertex_embed(out_vertex_embed)
            out_vertex = self.conv_vertex_score(out_vertex_embed_up)

            # hough voting
            if self.training:
                self.hough_voting.is_train = 1
            else:
                self.hough_voting.is_train = 0
            out_box, out_pose = self.hough_voting(out_label, out_vertex, meta_data, extents)

        if self.training:
            if cfg.TRAIN.VERTEX_REG:
                return out_logsoftmax, out_weight, out_vertex
            else:
                return out_logsoftmax, out_weight
        else:
            if cfg.TRAIN.VERTEX_REG:
                return out_label, out_vertex, out_box, out_pose
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
