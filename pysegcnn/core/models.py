# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:31:36 2020

@author: Daniel
"""
# builtins
import os
import enum
import logging
import pathlib

# externals
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# locals
from pysegcnn.core.layers import (Encoder, Decoder, Conv2dPool, Conv2dUnpool,
                                  Conv2dUpsample, Conv2dSame)

# module level logger
LOGGER = logging.getLogger(__name__)


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # initialize state file
        self.state_file = None

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, state_file, optimizer, bands=None, **kwargs):

        # check if the output path exists and if not, create it
        state_file = pathlib.Path(state_file)
        if not state_file.parent.is_dir():
           state_file.parent.mkdir(parents=True, exist_ok=True)

        # initialize dictionary to store network parameters
        model_state = {**kwargs}

        # store the spectral bands the model is trained with
        model_state['bands'] = bands

        # store model class
        model_state['cls'] = self.__class__

        # store construction parameters to instanciate the network
        model_state['params'] = {
            'skip': self.skip,
            'filters': self.nfilters,
            'nclasses': self.nclasses,
            'in_channels': self.in_channels
            }

        # store optional keyword arguments
        model_state['params'] = {**model_state['params'], **self.kwargs}

        # store model epoch
        model_state['epoch'] = self.epoch

        # store model and optimizer state
        model_state['model_state_dict'] = self.state_dict()
        model_state['optim_state_dict'] = optimizer.state_dict()

        # model state dictionary stores the values of all trainable parameters
        torch.save(model_state, state_file)
        LOGGER.info('Network parameters saved in {}'.format(state_file))

        return state_file

    @staticmethod
    def load(state_file, optimizer=None):

        # load the pretrained model
        state_file = pathlib.Path(state_file)
        if not state_file.exists():
            raise FileNotFoundError('{} does not exist.'.format(state_file))
        LOGGER.info('Loading pretrained weights from: {}'.format(state_file))

        # load the model state
        model_state = torch.load(state_file)

        # the model class
        model_class = model_state['cls']

        # instanciate pretrained model architecture
        model = model_class(**model_state['params'])

        # store state file as instance attribute
        model.state_file = state_file

        # load pretrained model weights
        LOGGER.info('Loading model parameters ...')
        model.load_state_dict(model_state['model_state_dict'])
        model.epoch = model_state['epoch']

        # resume optimizer parameters
        if optimizer is not None:
            LOGGER.info('Loading optimizer parameters ...')
            optimizer.load_state_dict(model_state['optim_state_dict'])
        LOGGER.info('Model epoch: {:d}'.format(model.epoch))

        return model, optimizer, model_state

    @property
    def state(self):
        return self.state_file


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


class SupportedModels(enum.Enum):
    Unet = UNet


class SupportedOptimizers(enum.Enum):
    Adam = optim.Adam

class SupportedLossFunctions(enum.Enum):
    CrossEntropy = nn.CrossEntropyLoss
