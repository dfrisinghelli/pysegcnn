# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:31:36 2020

@author: Daniel
"""
# builtins
from __future__ import absolute_import
import os

# externals
import numpy as np
import torch
import torch.nn as nn

# locals
from pytorch.layers import (Encoder, Decoder, Conv2dPool, Conv2dUnpool,
                            Conv2dUpsample, Conv2dSame)


class Network(nn.Module):

    def __init__(self):
        super().__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, state_file, optimizer, bands,
             outpath=os.path.join(os.getcwd(), '_models')):

        # check if the output path exists and if not, create it
        if not os.path.isdir(outpath):
            os.makedirs(outpath, exist_ok=True)

        # initialize dictionary to store network parameters
        model_state = {}

        # store input bands
        model_state['bands'] = bands

        # store construction parameters to instanciate the network
        model_state['params'] = {
            'skip': self.skip,
            'filters': self.nfilters,
            'nclasses': self.nclasses,
            'in_channels': self.in_channels
            }

        # store optional keyword arguments
        model_state['kwargs'] = self.kwargs

        # store model epoch
        model_state['epoch'] = self.epoch

        # store model and optimizer state
        model_state['model_state_dict'] = self.state_dict()
        model_state['optim_state_dict'] = optimizer.state_dict()

        # model state dictionary stores the values of all trainable parameters
        state = os.path.join(outpath, state_file)
        torch.save(model_state, state)
        print('Network parameters saved in {}'.format(state))

        return state

    def load(self, state_file, optimizer=None,
             inpath=os.path.join(os.getcwd(), '_models')):

        # load the model state file
        state = os.path.join(inpath, state_file)
        model_state = torch.load(state)

        # resume network parameters
        print('Loading network parameters from {} ...'.format(state))
        self.load_state_dict(model_state['model_state_dict'])
        self.epoch = model_state['epoch']

        # resume optimizer parameters
        if optimizer is not None:
            print('Loading optimizer parameters from {} ...'.format(state))
            optimizer.load_state_dict(model_state['optim_state_dict'])

        return state


class UNet(Network):

    def __init__(self, in_channels, nclasses, filters, skip, **kwargs):
        super().__init__()

        # number of input channels
        self.in_channels = in_channels

        # number of classes
        self.nclasses = nclasses

        # configuration of the convolutional layers in the network
        self.kwargs = kwargs
        self.nfilters = filters

        # convolutional layers of the encoder
        self.filters = np.hstack([np.array(in_channels), np.array(filters)])

        # whether to apply skip connections
        self.skip = skip

        # number of epochs trained
        self.epoch = 0

        # construct the encoder
        self.encoder = Encoder(filters=self.filters, block=Conv2dPool,
                               **kwargs)

        # construct the decoder
        self.decoder = Decoder(filters=self.filters, block=Conv2dUnpool,
                               skip=skip, **kwargs)

        # construct the classifier
        self.classifier = Conv2dSame(in_channels=filters[0],
                                     out_channels=self.nclasses,
                                     kernel_size=1)

    def forward(self, x):

        # forward pass: encoder
        x = self.encoder(x)

        # forward pass: decoder
        x = self.decoder(x, self.encoder.cache)

        # clear intermediate outputs
        del self.encoder.cache

        # classification
        return self.classifier(x)
