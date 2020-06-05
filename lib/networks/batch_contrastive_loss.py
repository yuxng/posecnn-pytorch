import torch
from torch.autograd import Variable

class BatchContrastiveLoss(object):

    def get_loss_matched_and_non_matched(self, img_a_feat, img_b_feat,
                                         flag, M=0.5, metric='euclidean'):

        if metric == 'euclidean':
            norm_degree = 2
            distances = (img_a_feat - img_a_feat).norm(norm_degree, 1)
        elif metric == 'cosine':
            distances = 0.5 * (1 - (img_a_feat * img_b_feat).sum(dim=1))

        # positive pairs
        num_matches = torch.sum(flag)
        if num_matches > 0:
            match_loss = torch.pow(distances, 2) * flag / num_matches
        else:
            match_loss = torch.sum(torch.zeros(1, device=img_a_feat.device))

        # negative pairs
        num_non_matches = torch.sum(1 - flag)
        if num_non_matches > 0:
            non_match_loss = torch.clamp(M - distances, min=0).pow(2) * (1 - flag) / num_non_matches
        else:
            non_match_loss = torch.sum(torch.zeros(1, device=img_a_feat.device))

        return match_loss.sum(), non_match_loss.sum()
