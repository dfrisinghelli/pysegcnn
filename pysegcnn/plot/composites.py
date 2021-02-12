"""Plot a false color composite of each scene in a dataset.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import logging
from logging.config import dictConfig

# locals
from pysegcnn.core.trainer import DatasetConfig
from pysegcnn.core.graphics import plot_sample
from pysegcnn.core.logging import log_conf
from pysegcnn.plot.plot_config import (PLOT_PATH, BANDS, FIGSIZE, ALPHA,
                                       DATASETS, PLOT_BANDS)

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # initialize logging
    dictConfig(log_conf())

    # iterate over the datasets
    for name, params in DATASETS.items():

        # instanciate dataset
        dc = DatasetConfig(dataset_name=name, bands=BANDS, tile_size=None,
                           **params)
        ds = dc.init_dataset()

        # iterate over the scenes of the dataset
        for scene in range(len(ds)):
            # name of the current scene
            scene_id = ds.scenes[scene]['id']
            LOGGER.info(scene_id)

            # get the data of the current scene
            x, y = ds[scene]

            # plot the current scene
            fig = plot_sample(x, ds.use_bands, ds.labels, y=y,
                              hide_labels=True, bands=PLOT_BANDS,
                              alpha=ALPHA, figsize=FIGSIZE)

            # save the figure
            fig.savefig(PLOT_PATH.joinpath('.'.join([scene_id, 'pdf'])),
                        bbox_inches='tight')
