import torch
import torch.nn as nn
import math
import sys
from torch.nn.init import kaiming_normal_
from fcn.config import cfg

__all__ = [
    'autoencoder',
]

def conv(in_planes, out_planes, kernel_size=3, stride=1, relu=True):
    if relu:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True)


def deconv(in_planes, out_planes, kernel_size=3, stride=1, output_padding=0, relu=True):
    if relu:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, output_padding=output_padding, bias=True),
            nn.ReLU(inplace=True))
    else:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, output_padding=output_padding, bias=True)


def fc(in_planes, out_planes, relu=True):
    if relu:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Linear(in_planes, out_planes)


class AutoEncoder(nn.Module):

    def __init__(self, code_dim=128):
        super(AutoEncoder, self).__init__()

        # encoder
        self.conv1 = conv(3, 128, kernel_size=5, stride=2)
        self.conv2 = conv(128, 256, kernel_size=5, stride=2)
        self.conv3 = conv(256, 256, kernel_size=5, stride=2)
        self.conv4 = conv(256, 512, kernel_size=5, stride=2)
        self.fc1 = fc(512 * 8 * 8, code_dim)

        # decoder
        self.fc2 = fc(code_dim, 512 * 8 * 8)
        self.deconv1 = deconv(512, 256, kernel_size=5, stride=2, output_padding=1)
        self.deconv2 = deconv(256, 256, kernel_size=5, stride=2, output_padding=1)
        self.deconv3 = deconv(256, 128, kernel_size=5, stride=2, output_padding=1)
        self.deconv4 = deconv(128, 3, kernel_size=5, stride=2, output_padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        # encoder
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(-1, 512 * 8 * 8)
        embedding = self.fc1(out)

        # decoder
        out = self.fc2(embedding)
        out = out.view(-1, 512, 8, 8)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)

        return out, embedding

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def autoencoder(num_classes=1, num_units=128, data=None):
    model = AutoEncoder(num_units)
    return model
