"""The configuration file to train and evaluate a model.

The configuration is handled by the class Config. You can either modify the
config.py file directly, or you can simply modify the config.yaml file.

Modify the variable values to your needs, but DO NOT modify the variable names.
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import os

# externals
import torch.nn as nn
import torch.optim as optim

# locals
from pysegcnn.core.models import UNet
from pysegcnn.core.transforms import Augment, FlipLr, FlipUd, Noise

# path to this file
HERE = os.path.abspath(os.path.dirname(__file__))

# path to the datasets
wd = 'C:/Eurac/2020/_Datasets/'
# wd = '//projectdata.eurac.edu/projects/cci_snow/dfrisinghelli/_Datasets/'
# wd = '/mnt/CEPH_PROJECTS/cci_snow/dfrisinghelli/_Datasets/'

# the configuration dictionary
config = {

    # ------------------------------- Dataset ---------------------------------

    # -------------------------------------------------------------------------

    # define which dataset to train on
    'dataset_name': 'Sparcs',
    # 'dataset_name': 'Cloud95'
    # 'dataset_name': 'Garmisch'

    # path to the dataset
    'dataset_path': os.path.join(wd, 'Sparcs/'),
    # 'dataset_path': os.path.join(wd, 'ProSnow/Garmisch/),
    # 'dataset_path': os.path.join(wd, 'Cloud95/Training/'),

    # a pattern to match the ground truth file naming convention
    'gt_pattern': '*mask.png',
    # 'gt_pattern': '*class.img',

    # define the bands to use to train the segmentation network:
    # either a list of bands, e.g. ['red', 'green', 'nir', 'swir2', ...]
    # or [], which corresponds to using all available bands
    'bands': ['red', 'green', 'blue', 'nir'],

    # define the size of the network input
    # if None, the size will default to the size of a scene
    'tile_size': 125,

    # whether to central pad the scenes with a constant value
    # if True, padding is used if the scenes are not evenly divisible into
    # tiles of size (tile_size, tile_size)
    'pad': True,

    # the constant value to pad around the ground truth mask if pad=True
    'cval': 99,

    # whether to sort the dataset in chronological order, useful for time
    # series data
    'sort': False,

    # the mode to split the dataset:
    #
    #    - 'random': for each scene, the tiles can be distributed among the
    #                training, validation and test set
    #
    #    - 'scene':  for each scene, all the tiles of the scene are included in
    #                either the training set, the validation set or the test
    #                set, respectively
    #
    #    - 'date':   split the scenes of a dataset based on a date, useful for
    #                time series data
    #                scenes before date build the training set, scenes after
    #                the date build the validation set, the test set is empty
    'split_mode': 'scene',

    # set random seed for reproducibility of the training, validation
    # and test data split
    # used if split_mode='random' and split_mode='scene'
    'seed': 0,

    # (ttratio * 100) % of the dataset will be used for training and
    # validation
    # used if split_mode='random' and split_mode='scene'
    'ttratio': 1,

    # (ttratio * tvratio) * 100 % will be used as for training
    # (1 - ttratio * tvratio) * 100 % will be used for validation
    # used if split_mode='random' and split_mode='scene'
    'tvratio': 0.95,

    # the date to split the scenes
    # format: 'yyyymmdd'
    # scenes before date build the training set, scenes after the date build
    # the validation set, the test set is empty
    # used if split_mode='date'
    'date': 'yyyymmdd',

    # define the batch size
    # determines how many samples of the dataset are processed until the
    # weights of the network are updated (via mini-batch gradient descent)
    'batch_size': 64,

    # whether to drop samples (during training only) with a fraction of
    # pixels equal to the constant padding value cval >= drop
    # drop=1 means, do not use a sample if all pixels = cval
    # drop=0.8 means, do not use a sample if 80% or more of the pixels are
    #                 equal to cval
    # drop=0.2 means, ...
    'drop': 1,

    # whether to artificially increase the training data size using data
    # augmentation methods

    # Supported data augmentation methods are:
    #   - FlipLr: horizontally flip an image
    #   - FlipUd: vertically flip an image
    #   - Noise:  add gaussian noise with defined mean and variance to an image
    #             two modes for adding noise are available
    #                 - speckle:  image = image + image * noise,
    #                 - gaussian: image = image + noise
    #             pixel values = exclude (default=0) are not modified by adding
    #             noise (i.e., the "no data" pixels added by the padding)
    # More detail can be found in pytorch.transforms.py

    # A probability can be assigned to each transformation so that it may or
    # may not be applied, thus
    #    - set p=1 to a transformation to always apply it
    #    - set p=0 to a transformation to never apply it
    #    - set 0 < p < 1 to apply a transformation with randomness

    # transforms is a list of transformations to apply to the original data
    # if transforms=[], no transformation is applied and only the original
    # dataset is used
    'transforms': [],

    # if you provide lists to transforms, each list represents a distinct
    # transformation of the original dataset
    # here an example if you want to perform two sets of transformations:
    #    1: FlipLr + Noise
    #    2: FlipLr + Noise + FlipUd
    # the resulting dataset will have 3 times the size of the original dataset,
    # i.e. the original dataset + the two transformed versions of it

    # 'transforms': [
    #     Augment([
    #         FlipLr(p=0.5),
    #         Noise(mode='speckle', mean=0, var=0.1, p=0.5, exclude=0)
    #         ]),
    #     Augment([
    #         FlipLr(p=0.5),
    #         Noise(mode='speckle', mean=0, var=0.1, p=0.5, exclude=0),
    #         FlipUd(p=0.5)
    #         ]),
    #     ],

    # ------------------------------ Network ----------------------------------

    # -------------------------------------------------------------------------

    # define the model
    'net': UNet,

    # define the number of filters for each convolutional layer
    # the number of filters should increase with depth
    'filters': [32, 64, 128, 256],

    # whether to apply skip connections from the encoder to the decoder
    # True (recommended) or False
    'skip_connection': True,

    # configuration for each convolutional layer
    'kwargs': {'kernel_size': 3,  # the size of the convolving kernel
               'stride': 1,  # the step size of the kernel
               'dilation': 1  # the field of view of the kernel
               },

    # ----------------------------- Training  ---------------------------------

    # -------------------------------------------------------------------------

    # path to save trained models
    'state_path': os.path.join(HERE, '_models/'),

    # Transfer learning -------------------------------------------------------

    # Use pretrained=True only if you wish to fine-tune a pre-trained model
    # on a different dataset than the one it was trained on

    # If you wish to continue training an existing model on the SAME
    # dataset, set option checkpoint=True and pretrained=False and make
    # sure the selected# dataset is the same as the one the existing model
    # was trained on

    # whether to use a pretrained model for transfer learning
    'pretrained': False,

    # name of the pretrained model to apply to a different dataset
    'pretrained_model': 'UNet_SparcsDataset_t125_b128_rgbn.pt',

    # Training ----------------------------------------------------------------

    # whether to resume training from an existing model checkpoint
    'checkpoint': False,

    # whether to early stop training if the accuracy on the validation set
    # does not increase more than delta over patience epochs
    'early_stop': False,
    'mode': 'max',
    'delta': 0,
    'patience': 10,

    # define the number of epochs: the number of maximum iterations over
    # the whole training dataset
    'epochs': 10,

    # define the number of threads
    'nthreads': os.cpu_count(),

    # define a loss function to calculate the network error
    'loss_function': nn.CrossEntropyLoss(),

    # define an optimizer to update the network weights
    'optimizer': optim.Adam,

    # define the learning rate
    'lr': 0.001,

    # ----------------------------- Evaluation --------------------------------

    # -------------------------------------------------------------------------

    # these options are only used for evaluating a trained model using
    # main.eval.py

    # whether to evaluate the model on the validation set or test set
    # test=False means evaluating on the validation set
    # test=True means evaluationg on the test set
    'test': False,

    # whether to compute and plot the confusion matrix
    'cm': True,

    # whether to predict each sample or each scene individually
    # False: each sample is predicted individually and the scenes are not
    #        reconstructed
    # True: each scene is first reconstructed and then the whole scene is
    #       predicted at once
    # NOTE: this option works only for datasets split by split_mode="scene" or
    #       split_mode="date"
    'predict_scene': True,

    # whether to save plots of (input, ground truth, prediction) of the
    # samples from the validation/test dataset to disk
    # output path is: pysegcnn/main/_samples/
    'plot_samples': False,

    # whether to save plots of (input, ground truth, prediction) for each scene
    # to disk
    # output path is: pysegcnn/main/_samples/
    'plot_scenes': True,

    # plot_bands defines the bands used to plot a false color composite of
    # the input scene: red': bands[0], green': bands[1], blue': bands[2]
    'plot_bands': ['nir', 'red', 'green'],

}
