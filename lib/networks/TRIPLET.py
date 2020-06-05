import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
from fcn.config import cfg
from . import resnet_dilated
from networks.batch_triplet_loss import BatchTripletLoss

__all__ = [
    'seg_resnet34_8s_triplet', 'seg_resnet34_8s_mapping',
]


class TripletNet(nn.Module):
    '''DOCSNet a Siamese Encoder-Decoder for Object Co-segmentation.'''
    def __init__(self, init_weights=True, network_name='resnet34_8s_fc', num_units=64):
        super(TripletNet, self).__init__()
        self.network_name = network_name
        self.metric = cfg.TRAIN.EMBEDDING_METRIC
        self.normalize = cfg.TRAIN.EMBEDDING_NORMALIZATION
        self.fcn = getattr(resnet_dilated, network_name)(num_classes=num_units)
        self.btl = BatchTripletLoss()

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

    def forward(self, img_a, img_p=None, img_n=None, label_positive=None, label_negative=None):
        
        features_a = self.fcn(img_a)
        if self.normalize:
            features_a = F.normalize(features_a, p=2, dim=1)

        if img_p is None or img_n is None:
            return features_a

        features_p = self.fcn(img_p)
        if self.normalize:
            features_p = F.normalize(features_p, p=2, dim=1)
        features_n = self.fcn(img_n)
        if self.normalize:
            features_n = F.normalize(features_n, p=2, dim=1)

        if self.training:
            loss_positive, loss_negative = self.btl.get_contrastive_loss(
                features_a, features_p, features_n, label_positive, label_negative, metric=self.metric)
            return loss_positive, loss_negative, features_a, features_p, features_n
        else:
            return features_a, features_p, features_n

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class SiameseNet(nn.Module):
    def __init__(self, init_weights=True, network_name='resnet34_8s_fc', num_units=64):
        super(SiameseNet, self).__init__()
        self.network_name = network_name
        self.metric = cfg.TRAIN.EMBEDDING_METRIC
        self.normalize = cfg.TRAIN.EMBEDDING_NORMALIZATION
        self.fcn = getattr(resnet_dilated, network_name)(num_classes=num_units)
        self.btl = BatchTripletLoss()

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

    def forward(self, img_a, features_p=None):
        
        features_a = self.fcn(img_a)
        if self.normalize:
            features_a = F.normalize(features_a, p=2, dim=1)

        if self.training:
            loss_positive = self.btl.get_positive_loss(features_a, features_p, metric=self.metric)
            return loss_positive, features_a
        else:
            return features_a

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


def seg_resnet34_8s_triplet(num_classes=2, num_units=64, data=None):
    network_name = 'Resnet34_8s_fc'
    model = TripletNet(network_name=network_name, num_units=num_units)
    update_model(model, data)
    return model

def seg_resnet34_8s_mapping(num_classes=2, num_units=64, data=None):
    network_name = 'Resnet34_8s_fc'
    model = SiameseNet(network_name=network_name, num_units=num_units)
    update_model(model, data)
    return model
