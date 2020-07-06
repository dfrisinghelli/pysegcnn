# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:41:20 2020

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
from pytorch.dataset import SparcsDataset
from pytorch.train import NetworkTrainer
from pytorch.models import SegNet
from sparcs.sparcs_00_config import (sparcs_path, bands, tile_size, tvratio,
                                     filters, skip_connection, kwargs,
                                     loss_function, optimizer, lr,
                                     batch_size, seed, state_file)


# instanciate the SparcsDataset class
dataset = SparcsDataset(sparcs_path, bands, tile_size)

# print the bands used for the segmentation
print('------------------------ Input bands -----------------------------')
print(*['Band {}: {}'.format(i, b) for i, b in
        enumerate(dataset.use_bands)], sep='\n')
print('------------------------------------------------------------------')

# print the classes of interest
print('-------------------------- Classes -------------------------------')
print(*['Class {}: {}'.format(k, v['label']) for k, v in
        dataset.labels.items()], sep='\n')
print('------------------------------------------------------------------')

# instanciate the segmentation network
print('------------------- Network architecture -------------------------')
net = SegNet(in_channels=len(dataset.use_bands),
             nclasses=len(dataset.labels),
             filters=filters,
             skip=skip_connection,
             **kwargs)
print(net)
print('------------------------------------------------------------------')

# instanciate the optimizer
optimizer = optimizer(net.parameters(), lr)

# add network name to state file
state_file = net.__class__.__name__ + state_file

# instanciate NetworkTrainer class
print('------------------------ Dataset split ---------------------------')
trainer = NetworkTrainer(net, dataset, loss_function, optimizer,
                         batch_size=batch_size, tvratio=tvratio, seed=seed)
print('------------------------------------------------------------------')
