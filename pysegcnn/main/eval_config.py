"""The configuration file to evaluate a model on an explicitly defined dataset.

See pysegcnn/main/eval.py for more details.

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

# path to the datasets on the current machine
# DRIVE_PATH = pathlib.Path('C:/Eurac/Projects/CCISNOW/Datasets/')
# DRIVE_PATH = pathlib.Path('/mnt/CEPH_PROJECTS/cci_snow/dfrisinghelli/Datasets/')  # nopep8
# DRIVE_PATH = pathlib.Path('/home/dfrisinghelli/Datasets')
DRIVE_PATH = pathlib.Path('/localscratch/dfrisinghelli_eurac/Datasets/')

# name and paths to the datasets
DATASETS = {'Sparcs': DRIVE_PATH.joinpath('Sparcs'),
            'Alcd': DRIVE_PATH.joinpath('Alcd')
            }

# name of the target dataset
TRG_DS = 'Sparcs'

# spectral bands to use for evaluation
BANDS = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']

# tile size of a single sample
TILE_SIZE = 256

# the target dataset configuration dictionary
trg_ds = {
    'dataset_name': TRG_DS,
    'root_dir': DATASETS[TRG_DS],
    #'gt_pattern': '(.*)Labels\\.tif',
    'gt_pattern': '(.*)mask\\.png',
    'bands': BANDS,
    'tile_size': TILE_SIZE,
    'pad': True,
    'sort': False,
    'transforms': [],
    # 'merge_labels': {'Cirrus': 'Cloud',
    #                 'Not_used': 'No_data'}
    'merge_labels': {'Shadow_over_water': 'Shadow',
                     'Flooded': 'Land'}

}

# the target dataset split configuration dictionary
trg_ds_split = {

    # 'split_mode': 'tile',
    'split_mode': 'scene',
    'k_folds': 1,  # keep k_folds=1 for evaluating models
    'seed': 0,
    'shuffle': True,
    'ttratio': 1,
    'tvratio': 0.05,

}
