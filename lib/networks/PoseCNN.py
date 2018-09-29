import torch
import torch.nn as nn
import torchvision.models as models
import math
import sys
from torch.nn.init import kaiming_normal_
from layers.hard_label import HardLabel


__all__ = [
    'posecnn',
]

vgg16 = models.vgg16(pretrained=True)

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
        nn.ReLU(inplace=True))

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.ReLU(inplace=True))


class PoseCNN(nn.Module):

    def __init__(self, num_classes, num_units):
        super(PoseCNN, self).__init__()
        self.num_classes = num_classes

        # conv features
        self.conv4_3 = nn.Sequential(*list(vgg16.features.children())[:23])
        self.conv5_3 = nn.Sequential(*list(vgg16.features.children())[23:30])

        # semantic labeling branch
        self.conv4_embed = conv(512, num_units, kernel_size=1)
        self.conv5_embed = conv(512, num_units, kernel_size=1)
        self.conv5_embed_up = deconv(num_units, num_units, kernel_size=4, stride=2, padding=1)
        self.deconv_embed_up = deconv(num_units, num_units, kernel_size=16, stride=8, padding=4)
        self.conv_score = conv(num_units, num_classes, kernel_size=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.hard_label = HardLabel(threshold=1.0)


    def forward(self, x, label_gt):
        # conv features
        out_conv4_3 = self.conv4_3(x)
        out_conv5_3 = self.conv5_3(out_conv4_3)

        # semantic labeling branch
        out_conv4_embed = self.conv4_embed(out_conv4_3)
        out_conv5_embed = self.conv5_embed(out_conv5_3)
        out_conv5_embed_up = self.conv5_embed_up(out_conv5_embed)
        out_embed = nn.functional.dropout(out_conv4_embed + out_conv5_embed_up)
        out_embed_up = self.deconv_embed_up(out_embed)
        out_score = self.conv_score(out_embed_up)
        out_logsoftmax = self.logsoftmax(out_score)
        out_prob = self.softmax(out_score)
        out_label = out_prob.argmax()
        out_weight = self.hard_label(out_prob, label_gt)

        if self.training:
            return out_logsoftmax, out_weight
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
        pretrained_dict = {k: v for k, v in data.items() if k in model_dict and v.size() == model_dict[k].size()}
        for k, v in pretrained_dict.items():
            print k
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)

    return model
