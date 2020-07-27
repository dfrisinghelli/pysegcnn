# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:23:36 2020

@author: Daniel
"""

# externals
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSame(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # define tensorflows "SAME" padding for stride = 1
        x_pad = self.same_padding(self.dilation[1], self.kernel_size[1])
        y_pad = self.same_padding(self.dilation[0], self.kernel_size[0])

        self.padding = (y_pad, x_pad)

    def same_padding(self, d, k):
        # calculates the padding so that the convolution
        # conserves the shape of its input when stride = 1
        return int(d * (k - 1) / 2)


def conv_bn_relu(in_channels, out_channels, **kwargs):
    return nn.Sequential(
            Conv2dSame(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Conv2dSame(out_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )


class Conv2dPool(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        # initialize nn.Module class
        super().__init__()

        # create the convolutional blocks for this module
        self.conv = conv_bn_relu(in_channels, out_channels, **kwargs)

        # create the 2x2 max pooling layer
        self.pool = nn.MaxPool2d(2, return_indices=True)

    # defines the forward pass
    def forward(self, x):

        # output of the convolutional block
        x = self.conv(x)

        # output of the pooling layer
        y, i = self.pool(x)

        return (y, x, i)


class Conv2dUnpool(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        # create the convolutional blocks for this module
        self.conv = conv_bn_relu(in_channels, out_channels, **kwargs)

        # create the unpooling layer
        self.upsample = nn.MaxUnpool2d(2)

    # defines the forward pass
    def forward(self, x, feature, indices, skip):

        # upsampling with pooling indices
        x = self.upsample(x, indices, output_size=feature.shape)

        # check whether to apply the skip connection
        # skip connection: concatenate the output of a layer in the encoder to
        # the corresponding layer in the decoder (along the channel axis)
        if skip:
            x = torch.cat([x, feature], axis=1)

        # output of the convolutional layer
        x = self.conv(x)

        return x


class Conv2dUpsample(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        # create the convolutional blocks for this module
        self.conv = conv_bn_relu(in_channels, out_channels, **kwargs)

        # create the upsampling layer
        self.upsample = F.interpolate

    # defines the forward pass
    def forward(self, x, feature, indices, skip):

        # upsampling with pooling indices
        x = self.upsample(x, size=feature.shape[2:], mode='nearest')

        # check whether to apply the skip connection
        # skip connection: concatenate the output of a layer in the encoder to
        # the corresponding layer in the decoder (along the channel axis )
        if skip:
            x = torch.cat([x, feature], axis=1)

        # output of the convolutional layer
        x = self.conv(x)

        return x


class Encoder(nn.Module):

    def __init__(self, filters, block, **kwargs):
        super().__init__()

        # the number of filters for each block: the first element of filters
        # has to be the number of input channels
        self.features = filters

        # the block of operations defining a layer in the encoder
        self.block = block

        # construct the encoder layers
        self.layers = []
        for i, (l, lp1) in enumerate(zip(self.features, self.features[1:])):
            # append blocks to the encoder layers
            self.layers.append(self.block(l, lp1, **kwargs))

        # convert list of layers to ModuleList
        self.layers = nn.ModuleList(*[self.layers])

    # forward pass through the encoder
    def forward(self, x):

        # initialize a dictionary that caches the intermediate outputs, i.e.
        # features and pooling indices of each block in the encoder
        self.cache = {}

        for i, layer in enumerate(self.layers):
            # apply current encoder layer forward pass
            x, y, ind = layer.forward(x)

            # store intermediate outputs for optional skip connections
            self.cache[i] = {'feature':  y, 'indices': ind}

        return x


class Decoder(nn.Module):

    def __init__(self, filters, block, skip=True, **kwargs):
        super().__init__()

        # the block of operations defining a layer in the decoder
        self.block = block

        # the number of filters for each block is symmetric to the encoder:
        # the last two element of filters have to be equal in order to apply
        # last skip connection
        self.features = filters[::-1]
        self.features[-1] = self.features[-2]

        # whether to apply skip connections
        self.skip = skip

        # in case of skip connections, the number of input channels to
        # each block of the decoder is doubled
        n_in = 1
        if self.skip:
            n_in *= 2

        # construct decoder layers
        self.layers = []
        for l, lp1 in zip(n_in * self.features, self.features[1:]):
            self.layers.append(self.block(l, lp1, **kwargs))

        # convert list of layers to ModuleList
        self.layers = nn.ModuleList(*[self.layers])

    # forward pass through decoder
    def forward(self, x, enc_cache):

        # for each layer, upsample input and apply optional skip connection
        for i, layer in enumerate(self.layers):

            # get intermediate outputs from encoder: iterate the encoder cache
            # in reversed direction, i.e. from last to first encoder layer
            cache = enc_cache[len(self.layers) - (i+1)]
            feature, indices = cache['feature'], cache['indices']

            # apply current decoder layer forward pass
            x = layer.forward(x, feature, indices, self.skip)

        return x
