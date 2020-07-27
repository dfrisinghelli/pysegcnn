"""The configuration file to train and evaluate a model.

Modify the variable values to your needs, but DO NOT modify the variable names.
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
from __future__ import absolute_import
import os
import inspect

# externals
import torch.nn as nn
import torch.optim as optim

# locals
from pysegcnn.core.models import UNet
from pysegcnn.core.transforms import Augment, FlipLr, FlipUd, Noise

# ------------------------- Dataset configuration -----------------------------
# -----------------------------------------------------------------------------

# define path to working directory
# wd = '//projectdata.eurac.edu/projects/cci_snow/dfrisinghelli/'
# wd = 'C:/Eurac/2020/'
wd = '/mnt/CEPH_PROJECTS/cci_snow/dfrisinghelli/'

# define which dataset to train on
# dataset_name = 'Sparcs'
# dataset_name = 'Cloud95'
dataset_name = 'Garmisch'

# path to the dataset
# dataset_path = os.path.join(wd, '_Datasets/Sparcs')
dataset_path = os.path.join(wd, '_Datasets/ProSnow/', dataset_name)
# dataset_path = os.path.join(wd, '_Datasets/Cloud95/Training')

# a pattern to match the ground truth file naming convention
# gt_pattern = '*mask.png'
gt_pattern = '*class.img'

# define the bands to use to train the segmentation network:
# either a list of bands, e.g. ['red', 'green', 'nir', 'swir2', ...]
# or [], which corresponds to using all available bands
bands = ['red', 'green', 'blue', 'nir']

# define the size of the network input
# if None, the size will default to the size of a scene
tile_size = 125

# whether to central pad the scenes with a constant value
# if True, padding is used if the scenes are not evenly divisible into tiles
# of size (tile_size, tile_size)
pad = True

# the constant value to pad around the ground truth mask if pad=True
cval = 3

# whether to sort the dataset in chronological order, useful for time series
# data
sort = True

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# ------------------------- Dataset augmentation ------------------------------
# -----------------------------------------------------------------------------

# whether to artificially increase the training data size using data
# augmentation methods

# Supported data augmentation methods are:
#   - FlipLr: horizontally flip an image
#   - FlipUd: vertically flip an image
#   - Noise:  add gaussian noise to an image
# More detail can be found in pytorch.transforms.py.

# transforms is a list of transformations to apply to the original data
# if transforms=[], no transformation is applied and only the original
# dataset is used
transforms = []

# list of image augmentations to apply to each sample in the original dataset
#   - each entry in should be an instance of pytorch.transforms.Augment
#   - each Augment instance in this list transforms each sample in the
#     original dataset
#   - the resulting dataset size is:
#     len(augmentations) * original_dataset_size
augmentations = [

    # an example transformation

    # the transformations are applied to the original image in the order
    # specified within the Augment class
    Augment([

        # horizontally flip image with probability p
        FlipLr(p=0.5),

        # vertically flip image with probability p
        FlipUd(p=0.5),

        # add gaussian noise to the image with probability p
        # do not add noise to pixels of values=exclude, i.e. the padded pixels
        Noise(mode='speckle', mean=0.1, var=0.05, p=0.5, exclude=cval)

        ])

    # set p=1 in each transformation in case you want to apply them without
    # randomness

    ]

# if no augmentation is required, comment the next line!
# transforms.extend(augmentations)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# ------------------------- Network configuration -----------------------------
# -----------------------------------------------------------------------------

# define the model
net = UNet

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

# path to save trained models
state_path = os.path.join(wd, 'git/deep-learning/main/_models/')

# Pretrained models: Transfer learning-----------------------------------------

# Use pretrained=True only if you wish to fine-tune a pre-trained model on a
# different dataset than the one it was trained on

# If you wish to continue training an existing model on the SAME dataset,
# set option checkpoint=True and pretrained=False and make sure the selected
# dataset is the same as the one the existing model was trained on

# whether to use a pretrained model for transfer learning
pretrained = True

# name of the pretrained model to apply to a different dataset
pretrained_model = 'UNet_SparcsDataset_t125_b128_rgbn.pt'

# Dataset split ---------------------------------------------------------------

# set random seed for reproducibility of the training, validation
# and test data split
seed = 0

# (ttratio * 100) % of the dataset will be used for training and validation
ttratio = 1

# (ttratio * tvratio) * 100 % will be used as the training dataset
# (1 - ttratio * tvratio) * 100 % will be used as the validation dataset
tvratio = 0.2

# define the batch size
# determines how many samples of the dataset are processed until the weights
# of the network are updated (via mini-batch gradient descent)
batch_size = 64

# Training configuration ------------------------------------------------------

# whether to resume training from an existing model checkpoint
checkpoint = False

# whether to early stop training if the accuracy on the validation set
# does not increase more than delta over patience epochs
early_stop = False
mode = 'max'
delta = 0
patience = 10

# define the number of epochs: the number of maximum iterations over the whole
# training dataset
epochs = 10

# define the number of threads
nthreads = os.cpu_count()

# Optimizer and loss ----------------------------------------------------------

# define a loss function to calculate the network error
loss_function = nn.CrossEntropyLoss()

# define an optimizer to update the network weights
optimizer = optim.Adam

# define the learning rate
lr = 0.001

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# ------------------------- Plotting configuration ----------------------------
# -----------------------------------------------------------------------------

# these options are only used for evaluating a trained model using main.eval.py

# whether to evaluate the model on the test set or validation set
# test=False means evaluating on the validation set
# test=True means evaluationg on the test set
test = False

# whether to compute and plot confusion matrix for the entire validation set
plot_cm = False

# whether to save plots of (input, ground truth, prediction) of the validation
# dataset to disk
# output path is: current_working_directory/_samples/
plot_samples = False

# number of samples to plot
# if nsamples = -1, all samples are plotted
nsamples = 50

# plot_bands defines the bands used to plot a false color composite of the
# input scene: red = bands[0], green = bands[1], blue = bands[2]
plot_bands = ['nir', 'red', 'green']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# DO NOT MODIFY THE FOLLOWING LINES!

# compile configuration dictionary:
config = {var: eval(var) for var in dir() if not var.startswith('_') and
          not inspect.ismodule(eval(var))}
