# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:40:35 2020

@author: Daniel
"""
# builtins
import os

# externals
import torch.nn as nn
import torch.optim as optim

# ------------------------- Dataset configuration -----------------------------
# -----------------------------------------------------------------------------

# define path to working directory
# wd = 'C:/Eurac/2020/'
wd = '/mnt/CEPH_PROJECTS/cci_snow/dfrisinghelli/'

# path to the downloaded sparcs archive
sparcs_archive = os.path.join(wd, '_Datasets/Archives/l8cloudmasks.zip')

# path to save preprocessed sparcs dataset
sparcs_path = os.path.join(wd, '_Datasets/Sparcs')

# define the bands to use to train the segmentation network:
# either a list of bands, e.g. ['red', 'green', 'nir', 'swir2', ...]
# or [], which corresponds to using all available bands
# bands = ['red', 'green', 'blue', 'nir']
bands = ['red', 'green', 'blue', 'nir']

# define the size of the network input
# if None, the size will default to the size of a scene, i.e. (1000 x 1000)
tile_size = 125

# set random seed for reproducibility of the training, validation
# and test data split
seed = 0

# (ttratio * 100) % of the dataset will be used for training and validation
ttratio = 1

# (ttratio * tvratio) * 100 % will be used as the training dataset
# (1 - ttratio * tvratio) * 100 % will be used as the validation dataset
tvratio = 0.8

# define the batch size
# determines how many samples of the dataset are processed until the weights
# of the network are updated
batch_size = 64

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# ------------------------- Network configuration -----------------------------
# -----------------------------------------------------------------------------

# define the number of filters for each convolutional layer
# the number of filters should increase with depth
filters = [32, 64, 128, 256]

# whether to apply skip connections from the encoder to the decoder
# True (recommended) or False
skip_connection = True

# configuration for each convolutional layer
kwargs = {'kernel_size': 3,  # the size of the convolving kernel
          'stride': 1,  # the step size of the kernel
          'dilation': 1  # the field of view of the kernel
          }

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# ------------------------- Training configuration ----------------------------
# -----------------------------------------------------------------------------

# whether to resume training from an existing model checkpoint
checkpoint = True

# whether to early stop training if the accuracy (loss) on the validation set
# does not increase (decrease) more than delta over patience epochs
early_stop = True
mode = 'max'
delta = 0.005
patience = 5

# define the number of epochs: the number of maximum iterations over the whole
# training dataset
epochs = 200

# define the number of threads
nthreads = os.cpu_count()

# define a loss function to calculate the network error
loss_function = nn.CrossEntropyLoss()

# define an optimizer to update the network weights
optimizer = optim.Adam

# define the learning rate
lr = 0.001

# path to save trained models
state_path = os.path.join(os.getcwd(), '_models/')

# file to save model state to: let the filename start with an underscore,
# since the network name will be appended to it
bformat = ''.join([b[0] for b in bands]) if bands else 'all'
state_file = '_Sparcs_t{}_b{}_{}.pt'.format(tile_size, batch_size, bformat)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
