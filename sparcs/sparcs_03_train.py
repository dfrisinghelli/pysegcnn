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
from pytorch.train import train
from pytorch.eval import accuracy_function
from sparcs.sparcs_00_config import loss_function, epochs, nthreads
from sparcs.sparcs_02_dataset import net, train_dl, optimizer, state_file


if __name__ == '__main__':

    # train the network
    print('----------------------- Network training -------------------------')
    losses, accuracies = train(net, train_dl, loss_function,
                               optimizer, accuracy_function, state_file,
                               epochs, nthreads)
