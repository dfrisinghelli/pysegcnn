"""The configuration file to train a model on a single domain.

The configuration is handled by the configuration dictionaries.

Modify the values to your needs, but DO NOT modify the keys.

The models can be trained with :py:mod:`pysegcnn.main.train_source.py`.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import pathlib

# from pysegcnn.core.transforms import Augment, FlipLr, FlipUd, Noise

# path to this file
HERE = pathlib.Path(__file__).resolve().parent

# path to the datasets on the current machine
DRIVE_PATH = pathlib.Path('C:/Eurac/Projects/CCISNOW/_Datasets/')
# DRIVE_PATH = pathlib.Path('/mnt/CEPH_PROJECTS/cci_snow/dfrisinghelli/_Datasets/')  # nopep8

# name and paths to the datasets
DATASETS = {'Sparcs': DRIVE_PATH.joinpath('Sparcs'),
            'Alcd': DRIVE_PATH.joinpath('Alcd/60m')
            }

# name of the dataset
DS_NAME = 'Sparcs'

# spectral bands to use for training
BANDS = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']

# tile size of a single sample
TILE_SIZE = 128

# number of folds for cross validation
K_FOLDS = 1

# the source dataset configuration dictionary
ds_config = {

    # -------------------------------------------------------------------------
    # Dataset -----------------------------------------------------------------
    # -------------------------------------------------------------------------

    # name of the dataset
    'dataset_name': DS_NAME,

    # path to the dataset
    'root_dir': DATASETS[DS_NAME],

    # a regex pattern to match the ground truth file naming convention
    'gt_pattern': '(.*)mask\\.png',
    # 'gt_pattern': '(.*)class\\.img',

    # define the bands to use to train the segmentation network:
    # either a list of bands, e.g. ['red', 'green', 'nir', 'swir2', ...]
    # or [], which corresponds to using all available bands
    # IMPORTANT: the list of bands should be equal for the source and target
    #            domains, when using any sort of transfer learning
    'bands': BANDS,

    # define the size of the network input
    # if None, the size will default to the size of a scene
    'tile_size': TILE_SIZE,

    # whether to central pad the scenes with a constant value
    # if True, padding is used if the scenes are not evenly divisible into
    # tiles of size (tile_size, tile_size)
    # 'pad': False,
    'pad': True,

    # whether to sort the dataset in chronological order, useful for time
    # series data
    # 'sort': True,
    'sort': False,

    # whether to artificially increase the training data size using data
    # augmentation methods

    # Supported data augmentation methods are:
    #   - FlipLr: horizontally flip an image
    #   - FlipUd: vertically flip an image
    #   - Noise:  add gaussian noise with defined mean and variance to an image
    #             two modes for adding noise are available
    #                 - speckle:  image = image + image * noise,
    #                 - add: image = image + noise
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

    # The label mapping dictionary, where each (key, value) pair represents a
    # distinct label mapping. The keys are the labels to be mapped and the
    # values are the corresponding labels to be mapped to.
    # NOTE: Passing an empty dictionary means all labels are preserved as is
    # 'merge_labels': {}
    'merge_labels': {'Shadow_over_water': 'Shadow',
                     'Flooded': 'Land'}

    # EXAMPLE: merge label class 'Shadow over Water' to label class 'Shadow'
    # 'merge_labels': {'Shadow_over_water': 'Shadow'}
}

# the source dataset split configuration dictionary
ds_split_config = {

    # -------------------------------------------------------------------------
    # Dataset split -----------------------------------------------------------
    # -------------------------------------------------------------------------

    # the mode to split the dataset:
    #
    #    - 'tile':   for each scene, the tiles can be distributed among the
    #                training, validation and test set
    #
    #    - 'scene':  for each scene, all the tiles of the scene are included in
    #                either the training set, the validation set or the test
    #                set, respectively
    # 'split_mode': 'tile',
    'split_mode': 'scene',

    # the number of folds for cross validation
    #
    # k_folds = 1 : The model is trained with a single dataset split based on
    #               'tvratio' and 'ttratio'
    # k_folds > 1 : The model is trained via cross validation on k_folds splits
    #               of the dataset
    'k_folds': K_FOLDS,

    # the random seed for the random number generator
    # ensures reproducibility of the training, validation and test data split
    'seed': 0,

    # whether to shuffle the data before splitting
    'shuffle': True,

    # -------------------------------------------------------------------------
    # IMPORTANT: these setting only apply if 'kfolds=1'
    # -------------------------------------------------------------------------

    # (ttratio * 100) % of the dataset will be used for training and
    # validation
    # used if 'kfolds=1'
    'ttratio': 1,

    # (ttratio * tvratio) * 100 % will be used for training
    # (1 - ttratio * tvratio) * 100 % will be used for validation
    # used if 'kfolds=1'
    'tvratio': 0.8,

}


# the model configuration dictionary
model_config = {

    # -------------------------------------------------------------------------
    # Network -----------------------------------------------------------------
    # -------------------------------------------------------------------------

    # define the model
    'model_name': 'Segnet',

    # -------------------------------------------------------------------------
    # Optimizer ---------------------------------------------------------------
    # -------------------------------------------------------------------------

    # define an optimizer to update the network weights
    'optim_name': 'Adam',

    # optimizer keyword arguments
    'optim_kwargs': {
        'lr': 0.001,  # the learning rate
        'weight_decay': 0,  # the weight decay rate
        'amsgrad': False  # whether to use AMSGrad variant (for Adam)
        },

    # -------------------------------------------------------------------------
    # Training configuration --------------------------------------------------
    # -------------------------------------------------------------------------

    # whether to save the model state to disk
    # model states are saved in:    pysegcnn/main/_models
    # model log files are saved in: pysegcnn/main/_logs
    'save': True,

    # whether to resume training from an existing model checkpoint
    'checkpoint': False,

    # define the batch size
    # determines how many samples of the dataset are processed until the
    # weights of the network are updated (via mini-batch gradient descent)
    'batch_size': 128,

    # the seed for the random number generator intializing the network weights
    'torch_seed': 0,

    # whether to early stop training if the accuracy (loss) on the validation
    # set does not increase (decrease) more than delta over patience epochs
    # -------------------------------------------------------------------------
    # The early stopping metric is chosen as:
    #    - validation set accuracy if mode='max'
    #    - validation set loss if mode='min'
    # -------------------------------------------------------------------------
    'early_stop': True,
    'mode': 'max',
    'delta': 0,
    'patience': 10,

    # define the number of epochs: the number of maximum iterations over
    # the whole training dataset
    'epochs': 100,

}
