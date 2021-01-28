"""Plot the distribution of the classes of a dataset in spectral space.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
from logging.config import dictConfig

# locals
from pysegcnn.core.trainer import DatasetConfig
from pysegcnn.core.graphics import plot_class_distribution
from pysegcnn.core.logging import log_conf
from pysegcnn.plot.plot_config import (PLOT_PATH, BANDS, FIGSIZE, ALPHA,
                                       DATASETS, DPI)


if __name__ == '__main__':

    # initialize logging
    dictConfig(log_conf())

    # iterate over the datasets
    for name, params in DATASETS.items():

        # instanciate dataset
        dc = DatasetConfig(dataset_name=name, bands=BANDS, tile_size=None,
                           **params)
        ds = dc.init_dataset()

        # plot class distribution
        fig = plot_class_distribution(ds, FIGSIZE, ALPHA)

        # save figure
        filename = PLOT_PATH.joinpath('{}_sdist.png'.format(name))
        fig.savefig(filename, dpi=DPI, bbox_inches='tight')
