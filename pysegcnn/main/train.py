"""Main splcfgript to train a model.

Steps to launch a model run:

    1. Configure the model run in :py:mod:`pysegcnn.main.config.py`
        - configure the dataset: ``dataset_config``
        - configure the split  : ``split_config``
        - configure the model  : ``model_config``
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

# builtins
from logging.config import dictConfig

# locals
from pysegcnn.core.trainer import (DatasetConfig, SplitConfig, ModelConfig,
                                   StateConfig, LogConfig, NetworkTrainer)
from pysegcnn.core.logging import log_conf
from pysegcnn.main.config import (dataset_config, split_config, model_config)


if __name__ == '__main__':

    # (i) instanciate the configurations
    dstcfg = DatasetConfig(**dataset_config)      # dataset
    splcfg = SplitConfig(**split_config)          # dataset split
    mdlcfg = ModelConfig(**model_config)          # model
    sttcfg = StateConfig(dstcfg, splcfg, mdlcfg)  # state file

    # (ii) instanciate the model state file
    state_file = sttcfg.init_state()

    # (iii) initialize logging
    log = LogConfig(state_file)
    dictConfig(log_conf(log.log_file))

    # (iv) instanciate the dataset
    ds = dstcfg.init_dataset()

    # (v) instanciate the training, validation and test datasets and
    # dataloaders
    train_ds, valid_ds, test_ds = splcfg.train_val_test_split(ds)
    train_dl, valid_dl, test_dl = splcfg.dataloaders(
        train_ds, valid_ds, test_ds, batch_size=mdlcfg.batch_size,
        shuffle=True, drop_last=False)

    # (vi) instanciate the model
    model, optimizer, checkpoint_state = mdlcfg.init_model(ds, state_file)

    # (vii) instanciate the loss function
    loss_function = mdlcfg.init_loss_function()

    # (viii) initialize network trainer class for easy model training
    trainer = NetworkTrainer(model=model,
                             optimizer=optimizer,
                             loss_function=loss_function,
                             train_dl=train_dl,
                             valid_dl=valid_dl,
                             test_dl=test_dl,
                             state_file=state_file,
                             epochs=mdlcfg.epochs,
                             nthreads=mdlcfg.nthreads,
                             early_stop=mdlcfg.early_stop,
                             mode=mdlcfg.mode,
                             delta=mdlcfg.delta,
                             patience=mdlcfg.patience,
                             checkpoint_state=checkpoint_state,
                             save=mdlcfg.save
                             )

    # (ix) train model
    training_state = trainer.train()
