"""Configuration file for plotting.

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

# path to this file
HERE = pathlib.Path(__file__).resolve().parent

# path to save plots
PLOT_PATH = HERE.joinpath('_plots/')

# check if the output path exists
if not PLOT_PATH.exists():
    PLOT_PATH.mkdir()

# path to the datasets on the current machine
# DRIVE_PATH = pathlib.Path('C:/Eurac/Projects/CCISNOW/Datasets/')
DRIVE_PATH = pathlib.Path('F:/Studium/SS 2020/Datasets/')
# DRIVE_PATH = pathlib.Path('/mnt/CEPH_PROJECTS/cci_snow/dfrisinghelli/Datasets')  # nopep8
# DRIVE_PATH = pathlib.Path('/home/dfrisinghelli/Datasets')

# name and paths to the datasets
DATASETS = {'Sparcs': {'root_dir': DRIVE_PATH.joinpath('Sparcs'),
                       'merge_labels': {'Shadow_over_water': 'Shadow',
                                        'Flooded': 'Land'},
                       'gt_pattern': '(.*)mask\\.png'},
            'Alcd': {'root_dir': DRIVE_PATH.joinpath('Alcd'),
                     'merge_labels': {'Cirrus': 'Cloud',
                                      'Not_used': 'No_data'},
                     'gt_pattern': '(.*)Labels\\.tif'}
            }

# spectral bands to plot distribution
BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

# plot parameters
FIGSIZE = (10, 10)
ALPHA = 5

# natural with atmospheric removal
# PLOT_BANDS = ['swir2', 'nir', 'green']
PLOT_BANDS = ['red', 'green', 'blue']
