import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
from fcn.config import cfg
from layers.hard_label import HardLabel
from layers.correlation import Correlation
from networks.utils import log_softmax_high_dimension, softmax_high_dimension
from networks.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
from networks.prototype_contrastive_loss import PrototypeContrastiveLoss
from networks.batch_contrastive_loss import BatchContrastiveLoss
from . import unets
from . import resnet_dilated

__all__ = [
    'docsnet', 'seg_resnet34_8s_contrastive', 'seg_resnet34_8s_prototype',
]

encoder_archs = {
    'vgg16-based-16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 1024, 1024]
}

decoder_archs = {
    'd16': [1024, 'd512', 512, 512, 'd512', 512, 512, 'd256', 256, 256, 'd128', 128, 128, 'd64', 64, 64, 'c2']
}

class DOCSNet(nn.Module):
    '''DOCSNet a Siamese Encoder-Decoder for Object Co-segmentation.'''
    def __init__(self, input_size=512, init_weights=True, batch_norm=False, network_name='docs',
                 has_squeez=True, squeezed_out_channels=512, num_units=64, prototype=False):
        super(DOCSNet, self).__init__()
        self.network_name = network_name
        self.metric = cfg.TRAIN.EMBEDDING_METRIC
        self.contrasitve_pixelwise = cfg.TRAIN.EMBEDDING_PIXELWISE
        self.prototype = prototype

        if network_name == 'docs':

            self.hard_label = HardLabel(threshold=cfg.TRAIN.HARD_LABEL_THRESHOLD, sample_percentage=cfg.TRAIN.HARD_LABEL_SAMPLING)

            # encoder
            en_layers, en_out_channels, en_output_scale = unets.make_encoder_layers(encoder_archs['vgg16-based-16'], batch_norm=batch_norm)
            self.features = en_layers
            en_output_size = round(input_size * en_output_scale)

            disp = en_output_size-1
            self.corr = Correlation(pad_size=disp, kernel_size=1, max_displacement=disp, stride1=1, stride2=1)
            corr_out_channels = self.corr.out_channels

            self.has_squeez = has_squeez
            if has_squeez:
                self.conv_squeezed = nn.Conv2d(en_out_channels, squeezed_out_channels, 1, padding=0)
                de_in_channels = int(squeezed_out_channels + corr_out_channels)
            else:
                de_in_channels = int(en_out_channels + corr_out_channels)

            # decoder
            self.decoder = unets.make_decoder_layers(decoder_archs['d16'], de_in_channels, batch_norm)
        else:
            if prototype:
                alpha = cfg.TRAIN.EMBEDDING_ALPHA
                delta = cfg.TRAIN.EMBEDDING_DELTA
                lambda_intra = cfg.TRAIN.EMBEDDING_LAMBDA_INTRA
                lambda_inter = cfg.TRAIN.EMBEDDING_LAMBDA_INTER
                self.pcl = PrototypeContrastiveLoss(alpha, delta, lambda_intra, lambda_inter)
            elif self.contrasitve_pixelwise:
                self.pcl = PixelwiseContrastiveLoss()
            else:
                self.pcl = BatchContrastiveLoss()
            self.fcn = getattr(resnet_dilated, network_name)(num_classes=num_units)

        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img_a, img_b, label_a, label_b, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b,
                others_non_matches_a, others_non_matches_b, background_non_matches_a, background_non_matches_b, flag):
        if self.network_name == 'docs':
            self.en_a = self.features(img_a)
            self.en_b = self.features(img_b)

            self.corr_ab = self.corr(self.en_a, self.en_b)
            self.corr_ba = self.corr(self.en_b, self.en_a)

            if self.has_squeez:
                cat_a = torch.cat((self.conv_squeezed(self.en_a), self.corr_ab),dim=1)
                cat_b = torch.cat((self.conv_squeezed(self.en_b), self.corr_ba),dim=1)
            else:
                cat_a = torch.cat((self.en_a, self.corr_ab),dim=1)
                cat_b = torch.cat((self.en_b, self.corr_ba),dim=1)

            out_score_a = self.decoder(cat_a)
            out_score_b = self.decoder(cat_b)

            out_logsoftmax_a = log_softmax_high_dimension(out_score_a)
            out_logsoftmax_b = log_softmax_high_dimension(out_score_b)

            out_prob_a = softmax_high_dimension(out_score_a)
            out_prob_b = softmax_high_dimension(out_score_b)

            out_label_a = torch.max(out_prob_a, dim=1)[1].type(torch.IntTensor).cuda()
            out_label_b = torch.max(out_prob_b, dim=1)[1].type(torch.IntTensor).cuda()

            out_weight_a = self.hard_label(out_prob_a, label_a, torch.rand(out_prob_a.size()).cuda())
            out_weight_b = self.hard_label(out_prob_b, label_b, torch.rand(out_prob_b.size()).cuda())

            if self.training:
                return out_logsoftmax_a, out_logsoftmax_b, out_weight_a, out_weight_b, out_label_a, out_label_b
            else:
                return out_label_a, out_label_b
        else:
            features_a = self.fcn(img_a)
            features_a = F.normalize(features_a, p=2, dim=1)
            features_b = self.fcn(img_b)
            features_b = F.normalize(features_b, p=2, dim=1)

            if self.training and not self.contrasitve_pixelwise and not self.prototype:
                loss_match, loss_non_match = \
                    self.pcl.get_loss_matched_and_non_matched(features_a, features_b, flag, metric=self.metric)
                return loss_match, loss_non_match, features_a, features_b

            elif self.training and self.contrasitve_pixelwise:
                images_a = features_a.view(features_a.shape[0], features_a.shape[1], -1).permute(0, 2, 1)
                images_b = features_b.view(features_b.shape[0], features_b.shape[1], -1).permute(0, 2, 1)

                num_images = features_a.shape[0]
                loss_matches = []
                loss_masked_non_matches = []
                loss_others_non_matches = []
                loss_background_non_matches = []
                for i in range(num_images):
                    image_a = images_a[i].unsqueeze(0)
                    image_b = images_b[i].unsqueeze(0)

                    # matches
                    match_a = matches_a[i]
                    match_b = matches_b[i]
                    index = (match_a > 0) & (match_b > 0)
                    match_a = match_a[index]
                    match_b = match_b[index]

                    # masked non matches
                    masked_non_match_a = masked_non_matches_a[i]
                    masked_non_match_b = masked_non_matches_b[i]
                    index = (masked_non_match_a > 0) & (masked_non_match_b > 0)
                    masked_non_match_a = masked_non_match_a[index]
                    masked_non_match_b = masked_non_match_b[index]

                    # others non matches
                    others_non_match_a = others_non_matches_a[i]
                    others_non_match_b = others_non_matches_b[i]
                    index = (others_non_match_a > 0) & (others_non_match_b > 0)
                    others_non_match_a = others_non_match_a[index]
                    others_non_match_b = others_non_match_b[index]

                    # background non matches
                    background_non_match_a = background_non_matches_a[i]
                    background_non_match_b = background_non_matches_b[i]
                    index = (background_non_match_a > 0) & (background_non_match_b > 0)
                    background_non_match_a = background_non_match_a[index]
                    background_non_match_b = background_non_match_b[index]

                    match_loss, masked_non_match_loss, num_masked_hard_negatives = \
                        self.pcl.get_loss_matched_and_non_matched_with_l2(image_a, image_b, match_a, match_b, \
                            masked_non_match_a, masked_non_match_b, metric=self.metric)
                    masked_non_match_loss_scaled = masked_non_match_loss * 1.0 / max(num_masked_hard_negatives, 1)

                    others_non_match_loss, num_others_hard_negatives = \
                        self.pcl.non_match_loss_descriptor_only(image_a, image_b, others_non_match_a, others_non_match_b, metric=self.metric)
                    others_non_match_loss_scaled = others_non_match_loss * 1.0 / max(num_others_hard_negatives, 1)

                    background_non_match_loss, num_background_hard_negatives = \
                        self.pcl.non_match_loss_descriptor_only(image_a, image_b, background_non_match_a, background_non_match_b, metric=self.metric)
                    background_non_match_loss_scaled = background_non_match_loss * 1.0 / max(num_background_hard_negatives, 1)

                    loss_matches.append(match_loss)
                    loss_masked_non_matches.append(masked_non_match_loss_scaled)
                    loss_others_non_matches.append(others_non_match_loss_scaled)
                    loss_background_non_matches.append(background_non_match_loss_scaled)

                loss_match = torch.stack(loss_matches).sum()
                loss_masked_non_match = torch.stack(loss_masked_non_matches).sum()
                loss_others_non_match = torch.stack(loss_others_non_matches).sum()
                loss_background_non_match = torch.stack(loss_background_non_matches).sum()
                return loss_match, loss_masked_non_match, loss_others_non_match, loss_background_non_match, features_a, features_b
            elif self.training and self.prototype:
                loss, intra_cluster_loss, inter_cluster_loss = self.pcl(features_a, features_b, label_a, label_b)
                return loss, intra_cluster_loss, inter_cluster_loss, features_a, features_b
            else:
                return features_a, features_b

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def update_model(model, data):
    model_dict = model.state_dict()
    print('model keys')
    print('=================================================')
    for k, v in model_dict.items():
        print(k)
    print('=================================================')

    if data is not None:
        print('data keys')
        print('=================================================')
        for k, v in data.items():
            print(k)
        print('=================================================')

        pretrained_dict = {k: v for k, v in data.items() if k in model_dict and v.size() == model_dict[k].size()}
        print('load the following keys from the pretrained model')
        print('=================================================')
        for k, v in pretrained_dict.items():
            print(k)
        print('=================================================')
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)


#############################################################
def docsnet(num_classes=2, num_units=128, data=None):
    model = DOCSNet()
    update_model(model, data)
    return model

def seg_resnet34_8s_contrastive(num_classes=2, num_units=64, data=None):
    if cfg.TRAIN.EMBEDDING_PIXELWISE:
        network_name = 'Resnet34_8s'
    else:
        network_name = 'Resnet34_8s_fc'
    model = DOCSNet(network_name=network_name, num_units=num_units, prototype=False)
    update_model(model, data)
    return model

def seg_resnet34_8s_prototype(num_classes=2, num_units=64, data=None):
    model = DOCSNet(network_name='Resnet34_8s', num_units=num_units, prototype=True)
    update_model(model, data)
    return model
