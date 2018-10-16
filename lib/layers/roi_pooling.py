import math
from torch import nn
from torch.autograd import Function
import torch
import posecnn_cuda


class RoIPoolFunction(Function):
    def __init__(self, pool_height, pool_width, spatial_scale):
        self.pool_width = int(pool_width)
        self.pool_height = int(pool_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None
        self.argmax_data = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()

        outputs = posecnn_cuda.roi_pool_forward(self.pool_height, self.pool_width, self.spatial_scale, features, rois)
        top_data = outputs[0]
        self.argmax_data = outputs[1]
        return top_data

    def backward(self, top_diff):
        batch_size, num_channels, data_height, data_width = self.feature_size
        outputs = posecnn_cuda.roi_pool_backward(batch_size, data_height, data_width, self.spatial_scale, top_diff, self.rois, self.argmax_data)
        d_features = outputs[0]
        return d_features, None   

class RoIPool(nn.Module):
    def __init__(self, pool_height, pool_width, spatial_scale):
        super(RoIPool, self).__init__()

        self.pool_width = int(pool_width)
        self.pool_height = int(pool_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction(self.pool_height, self.pool_width,
                                self.spatial_scale)(features, rois)
