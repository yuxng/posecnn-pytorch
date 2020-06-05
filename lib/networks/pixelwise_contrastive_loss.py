import torch
from torch.autograd import Variable

class PixelwiseContrastiveLoss(object):

    def get_loss_matched_and_non_matched_with_l2(self, img_a_feat, img_b_feat,
                                                 matches_a, matches_b,
                                                 non_matches_a, non_matches_b,
                                                 M_descriptor=0.5, metric='euclidean'):

        match_loss, _, _ = self.match_loss(img_a_feat, img_b_feat,
                                           matches_a, matches_b, metric)

        non_match_loss, num_hard_negatives = self.non_match_loss_descriptor_only(img_a_feat, img_b_feat,
                                                                                 non_matches_a, non_matches_b,
                                                                                 M_descriptor=M_descriptor, metric=metric)

        return match_loss, non_match_loss, num_hard_negatives

    @staticmethod
    def match_loss(image_a_pred, image_b_pred, matches_a, matches_b, metric='euclidean'):
        """
        Computes the match loss given by
        1/num_matches * \sum_{matches} ||D(I_a, u_a, I_b, u_b)||_2^2
        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_b
        :return: match_loss, matches_a_descriptors, matches_b_descriptors
        :rtype: torch.Variable(),
        matches_a_descriptors is torch.FloatTensor with shape torch.Shape([num_matches, descriptor_dimension])
        """

        num_matches = matches_a.size()[0]
        matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a).squeeze()
        matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b).squeeze()

        # crazily enough, if there is only one element to index_select into
        # above, then the first dimension is collapsed down, and we end up
        # with shape [D,], where we want [1,D]
        # this unsqueeze fixes that case
        if len(matches_a) == 1:
            matches_a_descriptors = matches_a_descriptors.unsqueeze(0)
            matches_b_descriptors = matches_b_descriptors.unsqueeze(0)

        if metric == 'euclidean':
            norm_degree = 2
            distances = (matches_a_descriptors - matches_b_descriptors).norm(norm_degree, 1)
        elif metric == 'cosine':
            distances = 0.5 * (1 - (matches_a_descriptors * matches_b_descriptors).sum(dim=1))

        match_loss = torch.pow(distances, 2).sum() / num_matches

        return match_loss, matches_a_descriptors, matches_b_descriptors


    @staticmethod
    def non_match_descriptor_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_b, M=0.5, invert=False, metric='euclidean'):
        """
        Computes the max(0, M - D(I_a,I_b,u_a,u_b))^2 term
        This is effectively:       "a and b should be AT LEAST M away from each other"
        With invert=True, this is: "a and b should be AT MOST  M away from each other"
         :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a
        :param M: the margin
        :type M: float
        :return: torch.FloatTensor with shape torch.Shape([num_non_matches])
        :rtype:
        """

        non_matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a).squeeze()
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b).squeeze()

        # crazily enough, if there is only one element to index_select into
        # above, then the first dimension is collapsed down, and we end up
        # with shape [D,], where we want [1,D]
        # this unsqueeze fixes that case
        if len(non_matches_a) == 1:
            non_matches_a_descriptors = non_matches_a_descriptors.unsqueeze(0)
            non_matches_b_descriptors = non_matches_b_descriptors.unsqueeze(0)

        if metric == 'euclidean':
            norm_degree = 2
            non_match_loss = (non_matches_a_descriptors - non_matches_b_descriptors).norm(norm_degree, 1)
        elif metric == 'cosine':
            non_match_loss = 0.5 * (1 - (non_matches_a_descriptors * non_matches_b_descriptors).sum(dim=1))

        if not invert:
            non_match_loss = torch.clamp(M - non_match_loss, min=0).pow(2)
        else:
            non_match_loss = torch.clamp(non_match_loss - M, min=0).pow(2)

        hard_negative_idxs = torch.nonzero(non_match_loss)
        num_hard_negatives = len(hard_negative_idxs)

        return non_match_loss, num_hard_negatives, non_matches_a_descriptors, non_matches_b_descriptors

    def non_match_loss_descriptor_only(self, image_a_pred, image_b_pred, non_matches_a, non_matches_b,
                                       M_descriptor=0.5, invert=False, metric='euclidean'):
        """
        Computes the non-match loss, only using the desciptor norm
        :param image_a_pred:
        :type image_a_pred:
        :param image_b_pred:
        :type image_b_pred:
        :param non_matches_a:
        :type non_matches_a:
        :param non_matches_b:
        :type non_matches_b:
        :param M:
        :type M:
        :return: non_match_loss, num_hard_negatives
        :rtype: torch.Variable, int
        """
        non_match_loss_vec, num_hard_negatives, _, _ = self.non_match_descriptor_loss(image_a_pred, image_b_pred, non_matches_a,
                                                                                      non_matches_b, M=M_descriptor,
                                                                                      invert=invert, metric=metric)

        non_match_loss = non_match_loss_vec.sum()

        return non_match_loss, num_hard_negatives
