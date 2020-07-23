# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:33:38 2020

@author: Daniel
"""
# builtins
from __future__ import absolute_import
import os

# externals
import torch

# local modules
from pytorch.trainer import NetworkTrainer
from main.config import config


if __name__ == '__main__':

    # instanciate the NetworkTrainer class
    trainer = NetworkTrainer(config)
    print(trainer)

    # train the network
    training_state = trainer.train()
