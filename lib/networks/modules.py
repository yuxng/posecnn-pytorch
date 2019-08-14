import math
from functools import reduce
from operator import mul

import torch
from torch import nn
from torch.nn import functional as F

from networks.functional import extract_features


class PixelNorm(nn.Module):
    """
    Mentioned in '4.2 PIXELWISE FEATURE VECTOR NORMALIZATION IN GENERATOR'
    'Local response normalization'
    """

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class EqualizedBase(nn.Module):

    def __init__(self, module, equalized=True, lr_scale=1.0, bias_zero_init=True):
        r"""
        equalized (bool): if True use He's constant to normalize at runtime.
        bias_zero_init (bool): if true, bias will be initialized to zero
        """
        super(EqualizedBase, self).__init__()

        self.module = module
        self.equalized = equalized

        if bias_zero_init:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lr_scale
            self.weight = self.get_he_constant() * lr_scale

    def forward(self, x):
        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x

    def get_he_constant(self):
        r"""
        Get He's constant for the given layer
        https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
        """
        size = self.module.weight.size()
        fan_in = reduce(mul, size[1:], 1)

        return math.sqrt(2.0 / fan_in)


class EqualizedConv2d(EqualizedBase):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, **kwargs):
        module = nn.Conv2d(in_channels, out_channels, kernel_size,
                           stride=stride, padding=padding, bias=bias)
        super(EqualizedConv2d, self).__init__(module, **kwargs)


class EqualizedConv3d(EqualizedBase):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, **kwargs):
        module = nn.Conv3d(in_channels, out_channels, kernel_size,
                           stride=stride, padding=padding, bias=bias)
        super(EqualizedConv3d, self).__init__(module, **kwargs)


class EqualizedLinear(EqualizedBase):

    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        module = nn.Linear(in_channels, out_channels, bias=bias)
        super(EqualizedLinear, self).__init__(module, **kwargs)


class Interpolate(nn.Module):

    __constants__ = ['scale_factor']

    def __init__(self, scale_factor, mode='nearest'):
        super(Interpolate, self).__init__()

        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = None
        if mode == 'bilinear' or mode == 'trilinear':
            self.align_corners = False

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=self.align_corners)

    def extra_repr(self):
        return "scale_factor={self.scale_factor}"


class LayerExtractor(nn.Module):

    def __init__(self, submodule, layers):
        super(LayerExtractor, self).__init__()
        self.submodule = submodule
        self.layers = [str(l) for l in layers]

    def forward(self, x):
        return extract_features(x, self.submodule, self.layers)


class PreActivationBasicBlock(nn.Module):
    """
    Pre-activation residual block from:

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks. arXiv:1603.05027
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu_slope=0.2, scale_mode='bilinear',
                 conv_module=EqualizedConv2d):
        super(PreActivationBasicBlock, self).__init__()
        self.conv1 = conv_module(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.conv2 = conv_module(out_channels, out_channels, kernel_size, padding=1)
        self.shortcut = conv_module(in_channels, out_channels, kernel_size=1, stride=1)

        self.activation = nn.LeakyReLU(relu_slope)
        self.downscale = Interpolate(scale_factor=0.5, mode=scale_mode)

    def forward(self, x):
        shortcut = self.shortcut(self.downscale(x))
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.downscale(x)

        return x + shortcut
