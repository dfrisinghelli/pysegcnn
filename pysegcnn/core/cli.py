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


def structure_parser():
    """Command line argument parser to standardize dataset structure.

    Returns
    -------
    None.

    """
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
