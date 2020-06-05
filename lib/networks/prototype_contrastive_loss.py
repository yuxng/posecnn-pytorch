import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def spherical_mean(x, cluster_masks):
    """ Computes the spherical mean of a set of unit vectors. This is a PyTorch implementation
        The definition of spherical mean is minimizes cosine similarity 
            to a set of points instead of squared error.

        Solves this problem:

            argmax_{||w||^2 <= 1} (sum_i x_i)^T w

        Turns out the solution is: S_n / ||S_n||, where S_n = sum_i x_i. 
            If S_n = 0, w can be anything.


        @param x: a [batch_size x C x H x W] torch.FloatTensor of N NORMALIZED C-dimensional unit vectors
        @param cluster_masks: a [batch_size x K x H x W] torch.FloatTensor of ground truth cluster assignments in {0, ..., K-1}.
                              Note: cluster -1 (i.e. no cluster assignment) is ignored
        @param K: number of clusters

        @return: a [batch_size x C x K] torch.FloatTensor of NORMALIZED cluster means
    """

    mask = (cluster_masks[:, 1] == 1).unsqueeze(1).float() # Shape: [batch_size x 1 x H x W]
    # adding 1e-10 because if mask has nothing, it'll hit NaNs
    # * here is broadcasting
    prototypes = torch.sum(x * mask, dim=[2, 3]) / (torch.sum(mask, dim=[2, 3]) + 1e-10) 
    # normalize to compute spherical mean
    prototypes = F.normalize(prototypes, p=2, dim=1) # Note, if any vector is zeros, F.normalize will return the zero vector
    return prototypes


class PrototypeContrastiveLoss(nn.Module):

    def __init__(self, alpha, delta, lambda_intra, lambda_inter):
        super(PrototypeContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.delta = delta
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter

    def forward(self, feature_a, feature_b, label_a, label_b):
        """ Compute the clustering loss. Assumes the batch is a sequence of consecutive frames

            @param feature_a, feature_b: a [batch_size x C x H x W] torch.FloatTensor of N NORMALIZED pixel embeddings
            @param label_a, label_b: a [batch_size x K x H x W] torch.FloatTensor of ground truth cluster assignments in {0, ..., K-1}
        """

        batch_size = feature_a.shape[0]

        # compute prototypes across batch dimension
        prototypes = spherical_mean(feature_a, label_a) # shape: [batch_size x C]

        # Compute distance to prototypes
        tiled_prototypes = prototypes.unsqueeze(2).unsqueeze(3) # Shape: [batch_size x C x 1 x 1]
        distances = 0.5 * (1 - torch.sum(feature_b * tiled_prototypes, dim=1)) # Shape: [batch_size x H x W]
        mask = (label_b[:, 1] == 1).float() # Shape: [batch_size x H x W]

        ### Intra cluster loss ###
        intra_cluster_distances = mask * distances
        intra_cluster_mask = (intra_cluster_distances - self.alpha) > 0
        intra_cluster_mask = intra_cluster_mask.float()
        if torch.sum(intra_cluster_mask) > 0:
            intra_cluster_loss = torch.pow(intra_cluster_distances, 2)
            intra_cluster_weight = torch.sum(intra_cluster_mask, dim=[1, 2], keepdim=True) # Shape: [batch_size x 1 x 1]
            # Max it with 50 so it doesn't get too small
            intra_cluster_weight = torch.max(intra_cluster_weight, torch.FloatTensor([50]).to(feature_b.device)) 
            intra_cluster_loss = torch.sum(intra_cluster_loss / intra_cluster_weight) / batch_size
        else:
            intra_cluster_loss = torch.sum(torch.zeros(1, device=x.device))
        intra_cluster_loss = self.lambda_intra * intra_cluster_loss

        ### Inter cluster loss ###
        inter_cluster_distances = torch.pow(torch.clamp((self.delta - distances) * (1 - mask), min=0), 2)
        inter_cluster_weight = torch.sum((inter_cluster_distances > 0).float(), dim=[1, 2], keepdim=True)
        inter_cluster_weight = torch.max(inter_cluster_weight, torch.FloatTensor([50]).to(feature_b.device)) 
        inter_cluster_loss = torch.sum(inter_cluster_distances / inter_cluster_weight) / batch_size
        inter_cluster_loss = self.lambda_inter * inter_cluster_loss

        loss = intra_cluster_loss + inter_cluster_loss
        return loss, intra_cluster_loss, inter_cluster_loss
