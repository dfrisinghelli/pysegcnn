# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 09:45:49 2020

@author: Daniel
"""

# builtins
import os
import sys

# externals
import numpy as np
import torch
import torch.nn.functional as F

# append path to local files to the python search path
sys.path.append('..')

# local modules
from pytorch.trainer import NetworkTrainer
from pytorch.layers import Conv2dSame
from main.config import config


if __name__ == '__main__':

    # instanciate the NetworkTrainer class
    trainer = NetworkTrainer(config)
    trainer.initialize()

    # freeze the model state
    trainer.model.freeze()

    # get the number of input features to the model classifier
    in_features = trainer.model.classifier.in_channels

    # replace the classification layer
    trainer.model.classifier = Conv2dSame(in_channels=in_features,
                                          out_channels=len(trainer.dataset.labels),
                                          kernel_size=1)

    # train the model on the new dataset
    trainer.train()
