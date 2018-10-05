import math
from torch import nn
from torch.autograd import Function
import torch
import posecnn_cuda


class RoIAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()

        outputs = posecnn_cuda.roi_align_forward(self.aligned_height, self.aligned_width, self.spatial_scale, features, rois)
        top_data = outputs[0]
        return top_data

    def backward(self, top_diff):
        batch_size, num_channels, data_height, data_width = self.feature_size
        outputs = posecnn_cuda.roi_align_backward(batch_size, data_height, data_width, self.spatial_scale, top_diff, self.rois)
        d_features = outputs[0]
        return d_features, None   

class RoIAlign(nn.Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width,
                                self.spatial_scale)(features, rois)
