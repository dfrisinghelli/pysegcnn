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
from main.config import (epochs, nthreads, checkpoint, early_stop,
                         mode, delta, patience, state_path)
from main.data import state_file, trainer

if __name__ == '__main__':

    # train the network
    print('----------------------- Network training -------------------------')
    loss, accuracy, vloss, vaccuracy = trainer.train(
        state_path, state_file, epochs=epochs, resume=checkpoint,
        early_stop=early_stop, nthreads=nthreads, mode=mode,
        min_delta=delta, patience=patience)
