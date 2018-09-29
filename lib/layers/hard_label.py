import math
from torch import nn
from torch.autograd import Function
import torch
import posecnn_cuda


class HardLabelFunction(Function):
    @staticmethod
    def forward(ctx, threshold, prob, label):
        return posecnn_cuda.hard_label_forward(threshold, prob, label)

    @staticmethod
    def backward(ctx, top_diff):
        d_prob, d_label = posecnn_cuda.hard_label_backward(top_diff.contiguous())
        return d_prob, d_label


class HardLabel(nn.Module):
    def __init__(self):
        super(HardLabel, self).__init__()

    def forward(self, threshold, prob, label):
        return HardLabelFunction.apply(threshold, prob, label)
