import functools
import torch
from torch import nn
from torch.nn import functional as F

from networks.modules import EqualizedConv2d, Interpolate, PixelNorm, EqualizedLinear

__all__ = ['pggan']


class InputBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, relu_slope=0.2, padding=0):
        super(InputBlock, self).__init__()
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.activation = nn.LeakyReLU(relu_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class OutputBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(OutputBlock, self).__init__()
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,
                 scale_mode='nearest', kernel_size=5, padding=2,
                 relu_slope=0.2):
        super(Block, self).__init__()
        self.interpolate = Interpolate(scale_factor, mode=scale_mode)
        self.activation = nn.LeakyReLU(relu_slope)
        self.norm = PixelNorm()

        self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        x = self.interpolate(x)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.norm(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.norm(x)

        return x


class FC(nn.Module):

    def __init__(self, in_channels, out_channels, relu_slope=0.2):
        super(FC, self).__init__()
        self.activation = nn.LeakyReLU(relu_slope)
        self.fc = EqualizedLinear(in_channels, out_channels)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, code_dim, block_config, intermediate_inputs=False,
                 scale_mode='nearest'):
        super(Encoder, self).__init__()

        self.block_config = block_config

        self.input_blocks = nn.ModuleList()
        self.encoder_blocks = nn.ModuleList()

        for block_id, (block_in, block_out) in enumerate(zip(block_config[:-1], block_config[1:])):
            if intermediate_inputs or block_id == 0:
                self.input_blocks.append(InputBlock(in_channels, block_in))
            self.encoder_blocks.append(
                Block(block_in, block_out, scale_factor=0.5, scale_mode=scale_mode))

        self.encoder_blocks.append(FC(512 * 8 * 8, code_dim))
        self.input_level = 0

    @property
    def num_blocks(self):
        return len(self.block_config) - 1

    def forward(self, x):
        input_block = self.input_blocks[self.input_level]

        # Scale input to match input level.
        if self.input_level > 0:
            input_scale = 2 ** (-self.input_level)
            x = F.interpolate(x, scale_factor=input_scale)

        z_intermediates = []
        z = input_block(x)
        for block_id, block in enumerate(self.encoder_blocks):
            if block_id == len(self.encoder_blocks) - 1:
                z = z.view(-1, 512 * 8 * 8)
            z = block(z)
            z_intermediates.append(z)

        return z, z_intermediates

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class Decoder(nn.Module):

    def __init__(self, out_channels, code_dim, block_config, intermediate_outputs=False,
                 style_size=8, skip_connections=True, scale_mode='nearest', output_activation=None):
        super(Decoder, self).__init__()

        self.style_size = style_size
        self.skip_connections = skip_connections

        self.decoder_blocks = nn.ModuleList()
        self.output_blocks = nn.ModuleList()

        block_config = list(reversed(block_config))
        # Add size of latent style vector to first block.
        block_config[0] += self.style_size
        self.block_config = block_config

        self.decoder_blocks.append(FC(code_dim, 512 * 8 * 8))

        for block_id, (block_in, block_out) in enumerate(zip(block_config[:-1], block_config[1:])):
            if self.skip_connections and block_id >= 1:
                block_in *= 2
            self.decoder_blocks.append(
                Block(block_in, block_out, scale_factor=2, scale_mode=scale_mode))
            if intermediate_outputs or block_id == self.num_blocks - 1:
                self.output_blocks.append(OutputBlock(block_out, out_channels))

        if output_activation is None:
            self.output_activation = None
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif output_activation == 'clamp':
            self.output_activation = functools.partial(torch.clamp, min=-1, max=1)
        else:
            raise ValueError("Unknown output activation {output_activation}")

        self.output_level = 0

    @property
    def num_blocks(self):
        return len(self.block_config) - 1

    def forward(self, z_content, z_content_intermediates=None, z_style=None):
        if z_style is None and self.style_size > 0:
            raise ValueError("z_style cannot be None if style_size > 0. "
                             "(style_size={self.style_size})")

        if z_content_intermediates is None and self.skip_connections:
            raise ValueError("z_content_intermediates cannot be None if skip connections are on.")

        if z_style is not None:
            assert z_style.size(0) == z_content.size(0)
            assert z_style.size(1) == self.style_size
            # Expand z_style to the spatial size of z_content.
            z_style = z_style.view(z_style.shape, 1, 1).expand(-1, -1, z_content.shape[2:])
            # Concatenate z to x in the channel dimension.
            z = torch.cat((z_content, z_style), dim=1)
        else:
            z = z_content

        for block_id, block in enumerate(self.decoder_blocks):
            if self.skip_connections and block_id >= 1:
                z = torch.cat((z, z_content_intermediates[-block_id - 1]), dim=1)
            z = block(z)
            if block_id == 0:
                z = z.view(-1, 512, 8, 8)

        output_block = self.output_blocks[-self.output_level-1]
        y = output_block(z)

        if self.output_activation is not None:
            y = self.output_activation(y)

        return y

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def minibatch_mean_variance(x, eps=1e-8):
    mean = torch.mean(x, dim=0, keepdim=True)
    vals = torch.sqrt(torch.mean((x - mean) ** 2, dim=0) + eps)
    vals = torch.mean(vals)
    return vals


class MinibatchStatsConcat(nn.Module):

    def __init__(self, eps=1e-8):
        super(MinibatchStatsConcat, self).__init__()
        self.eps = eps

    def forward(self, x):
        mean_var = minibatch_mean_variance(x, self.eps)
        # Expand mean variance to spatial dimension of x and concatenate to end of channel
        # dimension.
        mean_var = mean_var.view(1, 1, 1, 1).expand(x.size(0), -1, x.size(2), x.size(3))
        return torch.cat((x, mean_var), dim=1)


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, norm=None,
                 minibatch_stats=False, relu_slope=0.2, padding=0):
        super(DiscriminatorBlock, self).__init__()

        self.minibatch_stats = None
        if minibatch_stats:
            self.minibatch_stats = MinibatchStatsConcat()
            in_channels += 1

        self.norm = None
        if norm:
            self.norm = norm(out_channels)

        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size, stride=stride,
                                    padding=padding)
        self.activation = nn.LeakyReLU(relu_slope)

    def forward(self, x):
        if self.minibatch_stats is not None:
            x = self.minibatch_stats(x)
        x = self.conv(x)
        # Original PatchGAN seems to have norm before activation.
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)

        return x


class Discriminator(nn.Module):

    def __init__(self, in_channels, block_config=None):
        super(Discriminator, self).__init__()

        if block_config is None:
            block_config = [64, 128, 256]

        self.in_channels = in_channels
        self.block_config = block_config

        self.blocks = nn.ModuleList()

        # Add input block.
        self.blocks.append(
            DiscriminatorBlock(in_channels, block_config[0], kernel_size=5, stride=2,
                               padding=2))

        # Add intermediate blocks.
        for block_id, (block_in, block_out) in enumerate(zip(block_config[:-1], block_config[1:])):
            is_last = (block_id == len(block_config) - 2)
            stride = 1 if is_last else 2
            block = DiscriminatorBlock(block_in, block_out, kernel_size=5, stride=stride,
                                       norm=nn.InstanceNorm2d, minibatch_stats=is_last,
                                       padding=2)
            self.blocks.append(block)

        self.output_block = EqualizedConv2d(block_config[-1], 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x, mask=None):
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
            x = mask * x

        for block in self.blocks:
            x = block(x)
        x = self.output_block(x)
        return x


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, in_channels, block_config=None, num_scales=3):
        super(MultiScaleDiscriminator, self).__init__()

        self.in_channels = in_channels
        self.block_config = block_config
        self.num_scales = num_scales

        self.discriminators = nn.ModuleList()
        for scale in range(num_scales):
            self.discriminators.append(Discriminator(in_channels, block_config))

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = cls(**checkpoint['args'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def create_checkpoint(self):
        return {
            'args': {
                'in_channels': self.in_channels,
                'block_config': self.block_config,
                'num_scales': self.num_scales,
            },
            'state_dict': self.cpu().state_dict(),
        }

    def forward(self, x, mask=None):
        if mask is not None and len(mask.shape) == 3:
            mask = mask.unsqueeze(1)

        responses = []
        for scale, discriminator in enumerate(self.discriminators):
            responses.append(discriminator(x, mask))
            if scale != len(self.discriminators) - 1:
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
                if mask is not None:
                    mask = F.interpolate(mask, scale_factor=0.5)

        return responses

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class EncoderDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, code_dim, block_config=None, intermediate_inputs=False,
                 style_size=0, skip_connections=False, scale_mode='nearest',
                 output_activation=None):
        super(EncoderDecoder, self).__init__()

        if block_config is None:
            block_config = [32, 64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.code_dim = code_dim
        self.block_config = block_config
        self.style_size = style_size

        self.skip_connections = skip_connections
        self.intermediate_inputs = intermediate_inputs
        self.scale_mode = scale_mode
        self.output_activation = output_activation

        self.encoder = Encoder(in_channels, code_dim, block_config, intermediate_inputs,
                               scale_mode=scale_mode)
        self.decoder = Decoder(out_channels, code_dim, block_config, intermediate_inputs,
                               style_size=style_size, skip_connections=skip_connections,
                               scale_mode=scale_mode, output_activation=output_activation)
        self.discriminator = MultiScaleDiscriminator(in_channels)

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                print(m)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = cls(**checkpoint['args'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def create_checkpoint(self):
        return {
            'args': {
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'block_config': self.block_config,
                'intermediate_inputs': self.intermediate_inputs,
                'style_size': self.style_size,
                'skip_connections': self.skip_connections,
                'scale_mode': self.scale_mode,
                'output_activation': self.output_activation,
            },
            'state_dict': self.cpu().state_dict(),
        }

    def forward(self, x, z_style=None):
        z_content, z_content_intermediates = self.encoder(x)
        if not self.skip_connections:
            z_content_intermediates = None
        y = self.decoder(z_content, z_content_intermediates, z_style)

        return y, z_content


    def run_discriminator(self, x):
        return self.discriminator(x)

    def weight_parameters(self):
        param = self.encoder.weight_parameters()
        param += self.decoder.weight_parameters()
        return param

    def bias_parameters(self):
        param = self.encoder.bias_parameters()
        param += self.decoder.bias_parameters()
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


def pggan(num_classes=1, num_units=128, data=None):
    model = EncoderDecoder(in_channels=3, out_channels=3, code_dim=num_units)

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
