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
import pathlib

# locals
from pysegcnn.core.utils import search_files

# path to this file
HERE = pathlib.Path(__file__).resolve().parent

# path to the datasets on the current machine
DRIVE_PATH = pathlib.Path('C:/Eurac/Projects/CCISNOW/Datasets/')
# DRIVE_PATH = pathlib.Path('/mnt/CEPH_PROJECTS/cci_snow/dfrisinghelli/Datasets/')  # nopep8

# name and paths to the datasets
DATASETS = {'Sparcs': DRIVE_PATH.joinpath('Sparcs'),
            'Alcd': DRIVE_PATH.joinpath('Alcd/60m')
            }

# name of the target dataset
TRG_DS = 'Alcd'

# spectral bands to use for training
BANDS = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']

# tile size of a single sample
TILE_SIZE = 128

# the target dataset configuration dictionary
trg_ds = {
    'dataset_name': 'Sparcs',
    'root_dir': DATASETS[TRG_DS],
    'gt_pattern': '(.*)Labels\\.tif',
    'bands': BANDS,
    'tile_size': TILE_SIZE,
    'pad': True,
    'sort': True,
    'transforms': [],
    'merge_labels': {'Cirrus': 'Cloud',
                     'Not_used': 'No_data'}

}

# the target dataset split configuration dictionary
trg_ds_split = {

    # 'split_mode': 'tile',
    'split_mode': 'scene',
    'k_folds': 1,  # keep k_folds=1 for evaluating models
    'seed': 0,
    'shuffle': True,
    'ttratio': 1,
    'tvratio': 0.8,

}

# the evaluation configuration
eval_config = {

    # -------------------------------------------------------------------------
    # ----------------------------- Evaluation --------------------------------
    # -------------------------------------------------------------------------

    # these options are only used for evaluating a trained model using
    # pysegcnn.main.eval.py

    # the model(s) to evaluate
    'state_files': search_files(HERE, '*.pt'),

    # Evaluate on datasets defined at training time ---------------------------

    # implicit=True,  models are evaluated on the training, validation
    #                 and test datasets defined at training time
    # implicit=False, models are evaluated on an explicitly defined dataset
    #                 'ds'
    'implicit': True,
    # 'implicit': False,

    # The options 'domain' and 'test' define on which domain (source, target)
    # and on which set (training, validation, test) to evaluate the model.
    # NOTE: If the specified set was not available at training time, an error
    #       is raised.

    # whether to evaluate the model on the labelled source domain or the
    # (un)labelled target domain
    # if domain='trg',  target domain
    # if domain='src',  source domain
    # 'domain': 'src',
    'domain': 'trg',

    # the subset to evaluate the model on
    # test=False, 0 means evaluating on the validation set
    # test=True, 1 means evaluating on the test set
    # test=None means evaluating on the training set
    # 'test': True,
    'test': None,
    # 'test': False,

    # whether to map the model labels from the model source domain to the
    # defined 'domain'
    # For models trained via unsupervised domain adaptation, the classes of the
    # source domain, i.e. the classes the model is trained with, may differ
    # from the classes of the target domain. Setting 'map_labels'=True, means
    # mapping the source classes to the target classes. Obviously, this is only
    # possible if the target classes are a subset of the source classes.
    'map_labels': False,

    # Evaluate on an explicitly defined dataset -------------------------------

    # OPTIONAL: If 'trg_ds' is specified and 'implicit'=False, the model is not
    #           evaluated on the datasets defined at training time, but on the
    #           dataset defined by 'trg_ds'.

    # the dataset to evaluate the model on (optional)
    'ds': trg_ds,

    # the dataset split to use for 'ds'
    'ds_split': trg_ds_split,

    # Evaluation options ------------------------------------------------------

    # whether to compute and plot the confusion matrix
    # output path is: pysegcnn/main/_graphics/
    # 'cm': True,
    'cm': False,

    # whether to predict each sample or each scene individually
    # False: each sample is predicted individually and the scenes are not
    #        reconstructed
    # True: each scene is first reconstructed and then the whole scene is
    #       predicted at once
    # NOTE: this option works only for datasets split by split_mode="scene"
    'predict_scene': True,

    # whether to save plots of (input, ground truth, prediction) for each scene
    # in the train/validation/test dataset to disk, applies if
    # predict_scene=True
    # output path is: pysegcnn/main/_scenes/
    'plot_scenes': True,

    # plot_bands defines the bands used to plot a false color composite of
    # the input scene: red': bands[0], green': bands[1], blue': bands[2]
    'plot_bands': ['nir', 'red', 'green'],

    # size of the figures
    'figsize': (16, 9),

    # degree of constrast stretching for false color composite
    'alpha': 5

}
