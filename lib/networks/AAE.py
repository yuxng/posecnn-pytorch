import torch
import torch.nn as nn
import math
import sys
from torch.nn.init import kaiming_normal_
from fcn.config import cfg

__all__ = [
    'autoencoder',
]

def conv(in_planes, out_planes, kernel_size=3, stride=1, relu=True):
    if relu:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True)


def deconv(in_planes, out_planes, kernel_size=3, stride=1, output_padding=0, relu=True):
    if relu:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, output_padding=output_padding, bias=True),
            nn.ReLU(inplace=True))
    else:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, output_padding=output_padding, bias=True)


def fc(in_planes, out_planes, relu=True):
    if relu:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Linear(in_planes, out_planes)


class Encoder(nn.Module):

    def __init__(self, code_dim=128):
        super(Encoder, self).__init__()

        # encoder
        self.conv1 = conv(3, 128, kernel_size=5, stride=2)
        self.conv2 = conv(128, 256, kernel_size=5, stride=2)
        self.conv3 = conv(256, 256, kernel_size=5, stride=2)
        self.conv4 = conv(256, 512, kernel_size=5, stride=2)
        self.fc1 = fc(512 * 8 * 8, code_dim)

    def forward(self, x):

        # encoder
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(-1, 512 * 8 * 8)
        embedding = self.fc1(out)

        return embedding

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class Decoder(nn.Module):

    def __init__(self, code_dim=128):
        super(Decoder, self).__init__()

        # decoder
        self.fc2 = fc(code_dim, 512 * 8 * 8)
        self.deconv1 = deconv(512, 256, kernel_size=5, stride=2, output_padding=1)
        self.deconv2 = deconv(256, 256, kernel_size=5, stride=2, output_padding=1)
        self.deconv3 = deconv(256, 128, kernel_size=5, stride=2, output_padding=1)
        self.deconv4 = deconv(128, 3, kernel_size=5, stride=2, output_padding=1)

    def forward(self, embedding):

        # decoder
        out = self.fc2(embedding)
        out = out.view(-1, 512, 8, 8)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)

        return out

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, 128, kernel_size=5, stride=2)
        self.conv2 = conv(128, 256, kernel_size=5, stride=2)
        self.conv3 = conv(256, 512, kernel_size=5, stride=2)
        self.conv4 = conv(512, 1, kernel_size=5, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class AutoEncoder(nn.Module):

    def __init__(self, num_classes=1, code_dim=128):
        super(AutoEncoder, self).__init__()

        # encoder and decoder
        self.num_classes = num_classes
        self.code_dim = code_dim
        self.encoder = Encoder(code_dim)
        self.discriminator = Discriminator()
        self.decoder = Decoder(code_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                print(m)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        embeddings = self.encoder(x)
        outputs = self.decoder(embeddings)
        return outputs, embeddings

    def run_discriminator(self, x):
        return self.discriminator(x)

    def weight_parameters(self):
        param = self.encoder.weight_parameters() + self.decoder.weight_parameters()
        return param

    def bias_parameters(self):
        param = self.encoder.bias_parameters() + self.decoder.bias_parameters()
        return param

    def weight_parameters_discriminator(self):
        return self.discriminator.weight_parameters()

    def bias_parameters_discriminator(self):
        return self.discriminator.bias_parameters()

    """
    :param x: batch of code from the encoder (batch size x code size)
    :param y: code book (codebook size x code size)
    :return: cosine similarity matrix (batch size x code book size)
    """
    def pairwise_cosine_distances(self, x, y, eps=1e-8):
        dot_product = torch.mm(x, torch.t(y))
        x_norm = torch.norm(x, 2, 1).unsqueeze(1)
        y_norm = torch.norm(y, 2, 1).unsqueeze(1)
        normalizer = torch.mm(x_norm, torch.t(y_norm))
        return dot_product / normalizer.clamp(min=eps)


def autoencoder(num_classes=1, num_units=128, data=None):
    model = AutoEncoder(num_classes, num_units)

    if data is not None:
        model_dict = model.state_dict()
        print 'model keys'
        print '================================================='
        for k, v in model_dict.items():
            print k
        print '================================================='

        print 'data keys'
        print '================================================='
        for k, v in data.items():
            print k
        print '================================================='

        pretrained_dict = {k: v for k, v in data.items() if k in model_dict and v.size() == model_dict[k].size()}
        print 'load the following keys from the pretrained model'
        print '================================================='
        for k, v in pretrained_dict.items():
            print k
        print '================================================='
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)

    return model
