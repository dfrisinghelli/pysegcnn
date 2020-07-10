# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:33:38 2020

@author: Daniel
"""
# builtins
import os
import sys

# externals
import torch

# append path to local files to the python search path
sys.path.append('..')

# local modules
from pytorch.trainer import NetworkTrainer
from main.config import config


if __name__ == '__main__':

    # instanciate the NetworkTrainer class
    trainer = NetworkTrainer(config)

    # train the network
    print('----------------------- Network training -------------------------')
    loss, accuracy, vloss, vaccuracy = trainer.train()
