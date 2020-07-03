# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:11:15 2020

@author: Daniel
"""

# builtins
import sys

# externals
from torch.utils.data import DataLoader

# append path to local files to the python search path
sys.path.append('..')

# local modules
from pytorch.dataset import SparcsDataset
from pytorch.train import train_test_split
from pytorch.models import SegNet
from sparcs.sparcs_00_config import (sparcs_path, bands, tile_size,
                                     ttratio, tvratio, seed, batch_size,
                                     filters, skip_connection, optimizer, lr,
                                     kwargs, state_file)

# instanciate the SparcsDataset class
dataset = SparcsDataset(sparcs_path, bands, tile_size)

# print the classes to segment
print('------------------- Segmentation classes -------------------------')
print(*['Class {}: {}'.format(k, v) for k, v in dataset.labels.items()],
      sep='\n')
print('------------------------------------------------------------------')

print('------------------- Dataset split --------------------------------')

# split dataset into training and test data
train_ds, test_ds = train_test_split(dataset, ttratio, seed)

# split training dataset into training and validation data
train_ds, valid_ds = train_test_split(train_ds, tvratio, seed)

# print the dataset ratios
print(*['{} set: {:.2f}%'.format(k, v) for k, v in
        {'Training': ttratio * tvratio,
         'Validation': ttratio * (1 - tvratio),
         'Test': 1 - ttratio}.items()], sep='\n')
print('------------------------------------------------------------------')

# instanciate the DataLoader class
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

print('------------------- Network architecture -------------------------')
# instanciate the segmentation network
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
