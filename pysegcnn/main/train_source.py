"""Train a model on a single domain.

Steps to launch a model run:

    1. Configure the model run in :py:mod:`pysegcnn.main.train_config.py`
        - configure the dataset   : ``ds_config``
        - configure the split     : ``ds_split_config`
        - configure the model     : ``model_config``
    2. Save :py:mod:`pysegcnn.main.train_config.py`
    3. In a terminal, navigate to the repository's root directory
    4. Run

    .. code-block:: bash

        python pysegcnn/main/train_source.py


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
from pysegcnn.core.trainer import (DatasetConfig, SplitConfig, ModelConfig,
                                   StateConfig, LogConfig,
                                   NetworkTrainer)
from pysegcnn.main.train_config import ds_config, ds_split_config, model_config
from pysegcnn.core.logging import log_conf

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # (i) instanciate the source domain configurations
    src_dc = DatasetConfig(**ds_config)   # source domain dataset
    src_sc = SplitConfig(**ds_split_config)  # source domain dataset split

    # (ii) instanciate the model configuration
    net_mc = ModelConfig(**model_config)

    # (iii) instanciate the model state configuration
    net_sc = StateConfig()

    # (iv)) instanciate the source dataset
    src_ds = src_dc.init_dataset()

    # (v) instanciate the source training, validation and test dataset folds
    src_folds = src_sc.train_val_test_split(src_ds)

    # (vi) iterate over the different folds
    for fold, src_fold in enumerate(src_folds):

        # (vii) instanciate the model state file for the current fold
        state_file = net_sc.init_state(src_dc, src_sc, net_mc, fold=fold)

        # check if the state file already exists
        if state_file.exists() and not net_mc.checkpoint:
            LOGGER.info('Fold already exists: {}'.format(state_file))
            LOGGER.info('Moving to next fold ...')
            continue

        # (viii) instanciate logging configuration
        net_lc = LogConfig(state_file)
        dictConfig(log_conf(net_lc.log_file))

        # (ix) instanciate the source dataloaders
        tra_dl, val_dl, tes_dl = src_sc.dataloaders(
            *src_fold.values(), batch_size=net_mc.batch_size, shuffle=True,
            drop_last=False)

        # (x) instanciate the model
        net, optimizer, checkpoint = net_mc.init_model(
            len(src_ds.use_bands), len(src_ds.labels), state_file)

        # (xi) instanciate the network trainer class
        trainer = NetworkTrainer(
            model=net,
            optimizer=optimizer,
            state_file=net.state_file,
            src_train_dl=tra_dl,
            src_valid_dl=val_dl,
            src_test_dl=tes_dl,
            epochs=net_mc.epochs,
            nthreads=net_mc.nthreads,
            early_stop=net_mc.early_stop,
            mode=net_mc.mode,
            delta=net_mc.delta,
            patience=net_mc.patience,
            checkpoint_state=checkpoint,
            save=net_mc.save
            )

        # (xii) train the model
        LogConfig.init_log('Fold {} / {}'.format(fold + 1, len(src_folds)))
        training_state = trainer.train()
