"""The configuration file to train and evaluate a model.

The configuration is handled by the configuration dictionaries.

Modify the values to your needs, but DO NOT modify the keys.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import os

# from pysegcnn.core.transforms import Augment, FlipLr, FlipUd, Noise

# path to this file
HERE = os.path.abspath(os.path.dirname(__file__))

# path to the datasets on the current machine
DRIVE_PATH = 'C:/Eurac/2020/_Datasets/'
# DRIVE_PATH = '/mnt/CEPH_PROJECTS/cci_snow/dfrisinghelli/_Datasets/'

# name and paths to the datasets
DATASETS = {'Sparcs': os.path.join(DRIVE_PATH, 'Sparcs'),
            'Cloud95': os.path.join(DRIVE_PATH, 'Cloud95/Training'),
            'Garmisch': os.path.join(DRIVE_PATH, 'ProSnow/Garmisch')}

# name of the source domain dataset
SRC_DS = 'Sparcs'

# name of the target domain dataset
TRG_DS = 'Garmisch'

# the source dataset configuration dictionary
src_ds_config = {

    # ------------------------------- Dataset ---------------------------------

    # -------------------------------------------------------------------------

    # name of the dataset
    'dataset_name': SRC_DS,

    # path to the dataset
    'root_dir': DATASETS[SRC_DS],

    # a regex pattern to match the ground truth file naming convention
    'gt_pattern': '(.*)mask\\.png',
    # 'gt_pattern': '(.*)class\\.img',

    # define the bands to use to train the segmentation network:
    # either a list of bands, e.g. ['red', 'green', 'nir', 'swir2', ...]
    # or [], which corresponds to using all available bands
    # IMPORTANT: the list of bands should be equal for the source and target
    #            domains, when using any sort of transfer learning
    'bands': ['red', 'green', 'blue', 'nir'],

    # define the size of the network input
    # if None, the size will default to the size of a scene
    'tile_size': 128,

    # whether to central pad the scenes with a constant value
    # if True, padding is used if the scenes are not evenly divisible into
    # tiles of size (tile_size, tile_size)
    # 'pad': False,
    'pad': True,

    # the random seed for the numpy random number generator
    # ensures reproducibility of the training, validation and test data split
    # used if split_mode='random' and split_mode='scene'
    'seed': 0,

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
}

# the target dataset configuration dictionary
trg_ds_config = {
    'dataset_name': TRG_DS,
    'root_dir': DATASETS[TRG_DS],
    'gt_pattern': '(.*)class\\.img',
    'bands': ['red', 'green', 'blue', 'nir'],
    'tile_size': 128,
    'pad': True,
    'seed': 0,
    'sort': True,
    'transforms': [],
}

# the source dataset split configuration dictionary
src_split_config = {

    # the mode to split the dataset:
    #
    #    - 'random': randomly split the scenes
    #                for each scene, the tiles can be distributed among the
    #                training, validation and test set
    #
    #    - 'scene':  randomly split the scenes
    #                for each scene, all the tiles of the scene are included in
    #                either the training set, the validation set or the test
    #                set, respectively
    #
    #    - 'date':   split the scenes of a dataset based on a date, useful for
    #                time series data
    #                scenes before date build the training set, scenes after
    #                the date build the validation set, the test set is empty
    # 'split_mode': 'date',
    # 'split_mode': 'random',
    'split_mode': 'scene',

    # (ttratio * 100) % of the dataset will be used for training and
    # validation
    # used if split_mode='random' and split_mode='scene'
    'ttratio': 1,

    # (ttratio * tvratio) * 100 % will be used for training
    # (1 - ttratio * tvratio) * 100 % will be used for validation
    # used if split_mode='random' and split_mode='scene'
    'tvratio': 0.8,

    # the date to split the scenes
    # format: 'yyyymmdd'
    # scenes before date build the training set, scenes after the date build
    # the validation set, the test set is empty
    # used if split_mode='date'
    'date': '20161231',
    'dateformat': '%Y%m%d',

    # whether to drop samples (during training only) with a fraction of
    # pixels equal to the constant padding value cval >= drop
    # drop=1 means, do not use a sample if all pixels = cval
    # drop=0.8 means, do not use a sample if 80% or more of the pixels are
    #                 equal to cval
    # drop=0.2 means, ...
    # drop=0 means, do not drop any samples
    'drop': 0,

    }

# the target dataset split configuration dictionary
trg_split_config = {

    'split_mode': 'date',
    # 'split_mode': 'random',
    # 'split_mode': 'scene',
    'ttratio': 1,
    'tvratio': 0.8,
    'date': '20161231',
    'dateformat': '%Y%m%d',
    'drop': 0,
    }

# the model configuration dictionary
model_config = {

    # ------------------------------ Network ----------------------------------

    # -------------------------------------------------------------------------

    # define the model
    'model_name': 'segnet',

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

    # Transfer learning -------------------------------------------------------

    # whether to apply any sort of transfer learning
    # if transfer=False, the model is only trained on the source dataset
    'transfer': True,
    # 'transfer': False

    # name of the pretrained model to apply for transfer learning
    # Required if supervised=True
    # Optional if unsupervised=True
    'pretrained_model': 'Unet_SparcsDataset_Adam_RandomSplit_s0_t10v08_t125_b128_r4g3b2n5.pt',  # nopep8
    # 'pretrained_model': 'Unet_SparcsDataset_Adam_SceneSplit_s0_t10v08_t125_b128_r4g3b2n5.pt',  # nopep8

    # Supervised vs. Unsupervised ---------------------------------------------

    # the mode of transfer learning
    # supervised=True: fine-tune a pretrained model to the target dataset
    #                  IMPORTANT: this option requires target domain labels!
    # supervised=False: adapt a model via unsupervised domain adaptation from
    #                   the source to the target domain
    # 'supervised': True,
    'supervised': False,

    # whether to freeze the pretrained model weights when fine-tuning
    # NOTE: this option only works for supervised transfer learning

    # if True, only the classification layer is fine-tuned
    # if False, all layers are fine-tuned
    'freeze': True,

    # Loss function for unsupervised domain adaptation
    # Currently supported methods:
    #   - DeepCORAL (correlation alignment)
    'uda_loss': 'coral',

    # whether to start domain adaptation from a pretrained model
    'uda_from_pretrained': False,

    # The weight of the domain adaptation, trading off adaptation with
    # classification accuracy on the source domain.
    # NOTE: the domain adaptation weight increases every epoch reaching the
    #       value you specify for 'uda_lambda' in the last epoch
    # EXAMPLES:
    #       - 'uda_lambda' = 1, means that classification and adaptation loss
    #                           have equal weight in the last epoch
    #       - 'uda_lambda' = 0.5, means that classification loss has twice the
    #                             weight than adaptation loss in the last epoch
    # IMPORTANT: The higher 'uda_lambda', the more weight is put on the target
    #            domain and the less weight on the classification accuracy on
    #            the source domain.
    'uda_lambda': 1,

    # ----------------------------- Training  ---------------------------------

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

    # whether to early stop training if the accuracy on the validation set
    # does not increase more than delta over patience epochs
    'early_stop': True,
    'mode': 'max',
    'delta': 0,
    'patience': 10,

    # define the number of epochs: the number of maximum iterations over
    # the whole training dataset
    'epochs': 100,

    # define a classification loss function to calculate the network error
    'cla_loss': 'CrossEntropy',

    # define an optimizer to update the network weights
    'optim_name': 'Adam',

    # define the learning rate
    'lr': 0.001,

}

# the evaluation configuration file
eval_config = {

    # ----------------------------- Evaluation --------------------------------

    # -------------------------------------------------------------------------

    # these options are only used for evaluating a trained model using
    # pysegcnn.main.eval.py

    # the model to evaluate
    'state_file': 'SegNet_SparcsDataset_SceneSplit_s0_t10v08_t128_b128_r4g3b2n5_uda_Unet_ProSnowGarmisch_DateSplit_20161231_t128_b128_r4g3b2n8.pt',  # nopep8

    # Evaluate on datasets defined at training time ---------------------------

    # implicit=True,  models are evaluated on the training, validation
    #                 and test datasets defined at training time
    # implicit=False, models are evaluated on an explicitly defined dataset
    #                 'ds'
    'implicit': True,

    # The options 'domain' and 'test' define on which domain (source, target)
    # and on which set (training, validation, test) to evaluate the model.
    # NOTE: If the specified set was not available at training time, an error
    #       is raised.

    # whether to evaluate the model on the labelled source domain or the
    # (un)labelled target domain
    # if domain='trg',  target domain
    # if domain='src',  source domain
    'domain': 'src',
    # 'domain': 'trg',

    # the subset to evaluate the model on
    # test=False, 0 means evaluating on the validation set
    # test=True, 1 means evaluating on the test set
    # test=None means evaluating on the training set
    'test': False,

    # Evaluate on an explicitly defined dataset -------------------------------

    # OPTIONAL: If 'ds' is specified and 'implicit'=False, the model is not
    #           evaluated on the datasets defined at training time, but on the
    #           dataset defined by 'ds'.

    # the dataset to evaluate the model on (optional)
    'ds': trg_ds_config,

    # the dataset split to use for 'ds'
    'ds_split': trg_split_config,

    # Evaluation options ------------------------------------------------------

    # whether to compute and plot the confusion matrix
    # output path is: pysegcnn/main/_graphics/
    'cm': True,

    # whether to predict each sample or each scene individually
    # False: each sample is predicted individually and the scenes are not
    #        reconstructed
    # True: each scene is first reconstructed and then the whole scene is
    #       predicted at once
    # NOTE: this option works only for datasets split by split_mode="scene" or
    #       split_mode="date"
    'predict_scene': True,

    # whether to save plots of (input, ground truth, prediction) for each
    # sample in the train/validation/test dataset to disk, applies if
    # predict_scene=False
    # output path is: pysegcnn/main/_samples/
    'plot_samples': False,

    # whether to save plots of (input, ground truth, prediction) for each scene
    # in the train/validation/test dataset to disk, applies if
    # predict_scene=True
    # output path is: pysegcnn/main/_scenes/
    'plot_scenes': True,

    # whether to create an animation of (input, ground truth, prediction) for
    # the scenes in the train/validation/test dataset. Useful when predicting a
    # time-series.
    # NOTE: this option only works if predict_scene=True and plot_scenes=True
    # output path is: pysegcnn/main/_animations/
    'animate': False,

    # plot_bands defines the bands used to plot a false color composite of
    # the input scene: red': bands[0], green': bands[1], blue': bands[2]
    'plot_bands': ['nir', 'red', 'green'],

    # size of the figures
    'figsize': (16, 9),

    # degree of constrast stretching for false color composite
    'alpha': 5

}
