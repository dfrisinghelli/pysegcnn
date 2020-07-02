# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:31:36 2020

@author: Daniel
"""
# builtins
import os
import sys

# externals
import numpy as np
import torch
import torch.nn as nn

# append path to local files to the python search path
sys.path.append('..')

# locals
from pytorch.layers import (Encoder, Decoder, Conv2dPool, Conv2dUnpool,
                            Conv2dUpsample, Conv2dSame)


class SegNet(nn.Module):

    def __init__(self, in_channels, nclasses, filters, skip, **kwargs):
        super(SegNet, self).__init__()

        # get the configuration for the convolutional layers of the encoder
        self.filters = np.hstack([np.array(in_channels), np.array(filters)])

        # number of classes to segment
        self.nclasses = nclasses

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

    def save(self, optimizer, state_file,
             outpath=os.path.join(os.getcwd(), 'models')):

        # check if the output path exists and if not, create it
        if not os.path.isdir(outpath):
            os.makedirs(outpath, exist_ok=True)

        # create a dictionary that stores the model state
        model_state = {
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optim_state_dict': optimizer.state_dict()
            }

        # model state dictionary stores the values of all trainable parameters
        state = os.path.join(outpath, state_file)
        torch.save(model_state, state)
        print('Network parameters saved in {}'.format(state))

        return state


    def load(self, optimizer, state_file,
             inpath=os.path.join(os.getcwd(), 'models')):

        # load the model state file
        state = os.path.join(inpath, state_file)
        model_state = torch.load(state)

        # resume network parameters
        print('Loading network parameters from {} ...'.format(state))
        self.load_state_dict(model_state['model_state_dict'])
        self.epoch = model_state['epoch']

        # resume optimizer parameters
        print('Loading optimizer parameters from {} ...'.format(state))
        optimizer.load_state_dict(model_state['optim_state_dict'])

        return state


if __name__ == '__main__':

    # initialize segmentation network
    net = SegNet(in_channels=3,
                 nclasses=7,
                 filters=[32, 64, 128, 256],
                 skip=True,
                 kernel_size=3)

    # print network structure
    print(net)
