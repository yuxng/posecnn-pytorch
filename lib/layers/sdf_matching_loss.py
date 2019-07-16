import math
from torch import nn
from torch.autograd import Function
import torch
import posecnn_cuda

class SDFLossFunction(Function):
    @staticmethod
    def forward(ctx, pose_delta, pose_init, sdf_grids, sdf_limits, points):
        outputs = posecnn_cuda.sdf_loss_forward(pose_delta, pose_init, sdf_grids, sdf_limits, points)
        loss = outputs[0]
        se3 = outputs[1]
        variables = outputs[2:]
        ctx.save_for_backward(*variables)

        return loss, se3

    @staticmethod
    def backward(ctx, grad_loss, _):
        outputs = posecnn_cuda.sdf_loss_backward(grad_loss, *ctx.saved_variables)
        d_delta = outputs[0]

        return d_delta, None, None, None, None


class SDFLoss(nn.Module):
    def __init__(self):
        super(SDFLoss, self).__init__()

    def forward(self, pose_delta, pose_init, sdf_grids, sdf_limits, points):
        return SDFLossFunction.apply(pose_delta, pose_init, sdf_grids, sdf_limits, points)
