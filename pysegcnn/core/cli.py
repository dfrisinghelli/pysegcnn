"""Command line interface parsers.

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
import argparse

# epilogue to display at the end of each parser
EPILOGUE = 'Author: Daniel Frisinghelli, daniel.frisinghelli@gmail.com'


# command line argument parser: standard dataset structure
def structure_parser():
    """Command line argument parser to standardize dataset structure."""
    parser = argparse.ArgumentParser(
        description='Standardize the dataset directory structure.',
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the archive
    parser.add_argument('archive', type=pathlib.Path,
                        help='Path to the dataset archive.')

    # positional argument: path to extract and restructure the dataset
    parser.add_argument('target', type=pathlib.Path,
                        help='Path to save standardized dataset structure.')

    # optional arguments

    # default values
    default = '(default: %(default)s)'

    # optional argument: whether to overwrite existing files
    parser.add_argument('-o', '--overwrite', type=bool,
                        help='Overwrite files {}'.format(default),
                        default=False, nargs='?', const=True, metavar='')

    # optional argument: whether to copy or move extracted files
    parser.add_argument('-r', '--remove', type=bool,
                        help='Remove original dataset {}'.format(default),
                        default=True, nargs='?', const=True, metavar='')

    return parser


# command line argument parser: model evaluation
def evaluation_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description='Evaluate a model(s).',
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the directory to search for model state
    #                      files
    parser.add_argument('source', type=pathlib.Path,
                        help='The directory to search for model state files.')

    # positional argument: pattern to match
    parser.add_argument('pattern', type=str,
                        help='Pattern matching model state file names.')

    # optional arguments

    # default values
    default = '(default: %(default)s)'

    # optional argument: implicit
    # implicit=True,  models are evaluated on the training, validation
    #                 and test datasets defined at training time
    # implicit=False, models are evaluated on an explicitly defined dataset
    #                 'ds' in pysegcnn/main/eval_config.py
    parser.add_argument('-i', '--implicit', type=bool,
                        help=('Evaluate model on datasets defined at training '
                              ' time {}. If False, the model is evaluated on '
                              'the dataset defined in pysegcnn/main/eval_confi'
                              'g.py'.format(default)),
                        default=False, nargs='?', const=True, metavar='')

    # optional argument: domain
    # whether to evaluate the model on the labelled source domain or the
    # (un)labelled target domain
    # if domain='trg',  target domain
    # if domain='src',  source domain
    # The options 'domain' and 'test' define on which domain (source, target)
    # and on which set (training, validation, test) to evaluate the model.
    # NOTE: If the specified set was not available at training time, an error
    #       is raised.
    parser.add_argument('-d', '--domain', type=str,
                        help=('Evaluate model on source or target domain {}.'
                              .format(default)),
                        default='src', choices=['src', 'trg'], metavar='')

    # optional argument: subset
    # the subset to evaluate the model on
    # test=False, 0 means evaluating on the validation set
    # test=True, 1 means evaluating on the test set
    parser.add_argument('-s', '--subset', type=bool,
                        help=('Evaluate model on validation or test set. '
                              '(False=valid, True=test), {}.'.format(default)),
                        default=False, nargs='?', const=True, metavar='')

    # optional argument: aggregate
    # whether to aggregate the statistics of the different models matching the
    # defined pattern. Useful to aggregate the results of multiple model runs
    # in cross validation
    parser.add_argument('-a', '--aggregate', type=bool,
                        help=('Aggregate the statistics of the different '
                              'models matching the defined pattern {}.'
                              .format(default)),
                        default=False, nargs='?', const=True, metavar='')

    # optional argument: map labels
    # whether to map the model labels from the model source domain to the
    # defined 'domain'
    # For models trained via unsupervised domain adaptation, the classes of the
    # source domain, i.e. the classes the model is trained with, may differ
    # from the classes of the target domain. Setting 'map_labels'=True, means
    # mapping the source classes to the target classes. Obviously, this is only
    # possible if the target classes are a subset of the source classes.
    parser.add_argument('-m', '--map-labels', type=bool,
                        help=('Map labels from model source domain to target '
                              'domain {}.'.format(default)),
                        default=False, nargs='?', const=True, metavar='')

    # optional argument: confusion matrix
    # whether to compute and plot the confusion matrix
    parser.add_argument('-cm', '--confusion-matrix', type=bool,
                        help=('Compute and plot the confusion matrix {}.'
                              .format(default)),
                        default=False, nargs='?', const=True, metavar='')

    # optional argument: predict scenes
    # whether to predict each sample or each scene individually
    # False: each sample is predicted individually and the scenes are not
    #        reconstructed
    # True: each scene is first reconstructed and then the whole scene is
    #       predicted at once
    # NOTE: this option works only for datasets split by split_mode="scene"
    parser.add_argument('-ps', '--predict-scene', type=bool,
                        help=('Reconstruct and predict each scene {}.'
                              .format(default)),
                        default=False, nargs='?', const=True, metavar='')

    # optional argument: whether to overwrite existing files
    parser.add_argument('-o', '--overwrite', type=bool,
                        help=('Overwrite existing model evaluations {}.'
                              .format(default)),
                        default=False, nargs='?', const=True, metavar='')

    # optional argument: plot scenes
    parser.add_argument('-plot', '--plot-scenes', type=bool,
                        help=('Save plots for each predicted scene {}.'
                              .format(default)),
                        default=False, nargs='?', const=True, metavar='')

    # optional argument: dataset path
    parser.add_argument('-ds', '--dataset-path', type=str,
                        help=('Path to the datasets on the current machine {}.'
                              'Per default, it is assumed to be same as during'
                              ' model training .'.format(default)),
                        default='', metavar='')

    return parser
