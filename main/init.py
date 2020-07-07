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
from pytorch.dataset import SparcsDataset, Cloud95Dataset
from pytorch.trainer import NetworkTrainer
from pytorch.models import SegNet
from main.config import (dataset_name, dataset_path, bands, tile_size, tvratio,
                         filters, skip_connection, kwargs, loss_function,
                         optimizer, lr, ttratio, batch_size, seed, patches)

# check which dataset the model is trained on
if dataset_name == 'Sparcs':
    # instanciate the SparcsDataset
    dataset = SparcsDataset(dataset_path, use_bands=bands, tile_size=tile_size)
elif dataset_name == 'Cloud95':
    dataset = Cloud95Dataset(dataset_path, use_bands=bands,
                             tile_size=tile_size, exclude=patches)
else:
    raise ValueError('{} is not a valid dataset. Available datasets are '
                     '"Sparcs" and "Cloud95".'.format(dataset_name))

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

# file to save model state to
# format: networkname_datasetname_t(tilesize)_b(batchsize)_bands.pt
bformat = ''.join([b[0] for b in bands]) if bands else 'all'
state_file = '{}_{}_t{}_b{}_{}.pt'.format(net.__class__.__name__,
                                          dataset.__class__.__name__,
                                          tile_size, batch_size, bformat)

# instanciate NetworkTrainer class
print('------------------------ Dataset split ---------------------------')
trainer = NetworkTrainer(net, dataset, loss_function, optimizer,
                         batch_size=batch_size, tvratio=tvratio,
                         ttratio=ttratio, seed=seed)
print('------------------------------------------------------------------')
