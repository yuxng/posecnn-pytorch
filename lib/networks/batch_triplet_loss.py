import torch
from torch.autograd import Variable

class BatchTripletLoss(object):

    # feature shape: batch_size x feature_dimension

    def get_triplet_loss(self, img_a, img_p, img_n, M=0.5, metric='euclidean'):

        if metric == 'euclidean':
            norm_degree = 2
            d_positive = (img_a - img_p).norm(norm_degree, 1)
            d_negative = (img_a - img_n).norm(norm_degree, 1)
        elif metric == 'cosine':
            d_positive = 0.5 * (1 - (img_a * img_p).sum(dim=1))
            d_negative = 0.5 * (1 - (img_a * img_n).sum(dim=1))

        batch_size = img_a.shape[0]
        loss = torch.clamp(d_positive.pow(2) - d_negative.pow(2) + M, min=0)
        return loss.sum() / batch_size


    def get_contrastive_loss(self, img_a, img_p, img_n, label_positive, label_negative, M=0.5, metric='euclidean'):

        # concate features and labels
        batch_size = img_a.shape[0]
        features = torch.cat((img_a, img_p, img_n), dim=0)
        labels = torch.cat((label_positive, label_positive, label_negative), dim=0)

        if metric == 'euclidean':
            norm_degree = 2
            distances = (features.unsqueeze(1) - features.unsqueeze(0)).norm(norm_degree, 2)
        elif metric == 'cosine':
            distances = 0.5 * (1 - torch.mm(features, features.t()))
        y = torch.mm(labels, labels.t())

        # positive pairs
        num_positive = torch.sum(y) - batch_size
        loss_positive = torch.sum(y * distances.pow(2)) / num_positive

        # negative pairs
        num_hard_negative = torch.sum((1 - y) * (M - distances) > 0)
        loss_negative = torch.sum((1 - y) * torch.clamp(M - distances, min=0).pow(2)) / num_hard_negative

        return loss_positive, loss_negative


    def get_positive_loss(self, img_a, img_p, metric='euclidean'):

        if metric == 'euclidean':
            norm_degree = 2
            d_positive = (img_a - img_p).norm(norm_degree, 1)
        elif metric == 'cosine':
            d_positive = 0.5 * (1 - (img_a * img_p).sum(dim=1))

        batch_size = img_a.shape[0]
        loss = d_positive.pow(2)
        return loss.sum() / batch_size
