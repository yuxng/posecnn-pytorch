import math
from torch import nn
from torch.autograd import Function
import torch
import posecnn_cuda


class HardLabelFunction(Function):
    @staticmethod
    def forward(ctx, prob, label, rand, threshold):
        outputs = posecnn_cuda.hard_label_forward(threshold, prob, label, rand)
        top_data = outputs[0]
        return top_data

    @staticmethod
    def backward(ctx, top_diff):
        outputs = posecnn_cuda.hard_label_backward(top_diff)
        d_prob, d_label = outputs
        return d_prob, d_label, None, None


class HardLabel(nn.Module):
    def __init__(self, threshold):
        super(HardLabel, self).__init__()
        self.threshold = threshold

    def forward(self, prob, label, rand):
        return HardLabelFunction.apply(prob, label, rand, self.threshold)
