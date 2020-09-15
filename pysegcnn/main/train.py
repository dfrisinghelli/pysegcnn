"""Main script to train a model.

Steps to launch a model run:

    1. Configure the model run in :py:mod:`pysegcnn.main.config.py`
        - configure the dataset(s): ``src_ds_config`` and ``trg_ds_config``
        - configure the split(s)  : ``src_ds_config`` and ``trg_ds_config``
        - configure the model     : ``model_config``
    2. Save :py:mod:`pysegcnn.main.config.py`
    3. In a terminal, navigate to the repository's root directory
    4. Run

    .. code-block:: bash

        python pysegcnn/main/train.py


License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# locals
from pysegcnn.core.trainer import NetworkTrainer
from pysegcnn.main.config import (src_ds_config, src_split_config,
                                  trg_ds_config, trg_split_config,
                                  model_config)


if __name__ == '__main__':

    # instanciate the network trainer class
    trainer = NetworkTrainer.init_network_trainer(src_ds_config,
                                                  src_split_config,
                                                  trg_ds_config,
                                                  trg_split_config,
                                                  model_config)

    # (x) train model
    training_state = trainer.train()
