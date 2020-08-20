"""Main script to train a model.

Steps to launch a model run:

    (1) Configure the model run in pysegcnn/main/config.py
        (i) configure the dataset      : dictionary 'dataset_config'
        (j) configure the dataset split: dictionary 'split_config'
        (k) configure the model        : dictionary 'model_config'
    (2) Save pysegcnn/main/config.py
    (3) In a terminal, navigate to the repository's root directory
    (4) run "python pysegcnn/main/train.py"

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

    # write code that checks for list of seeds, band combinations etc. here.

    # (i) instanciate the configurations
    dc = DatasetConfig(**dataset_config)
    sc = SplitConfig(**split_config)
    mc = ModelConfig(**model_config)

    # (ii) instanciate the dataset
    ds = dc.init_dataset()

    # (iii) instanciate the model state
    state = StateConfig(ds, sc, mc)
    state_file = state.init_state()

    # (iv) initialize logging
    log = LogConfig(state_file)
    dictConfig(log_conf(log.log_file))

    # (v) instanciate the training, validation and test datasets and
    # dataloaders
    train_ds, valid_ds, test_ds = sc.train_val_test_split(ds)
    train_dl, valid_dl, test_dl = sc.dataloaders(
        train_ds, valid_ds, test_ds, batch_size=mc.batch_size, shuffle=True,
        drop_last=False)

    # (vi) instanciate the model
    model, optimizer, checkpoint_state = mc.init_model(ds, state_file)

    # (vii) instanciate the loss function
    loss_function = mc.init_loss_function()

    # (viii) initialize network trainer class for easy model training
    trainer = NetworkTrainer(model=model,
                             optimizer=optimizer,
                             loss_function=loss_function,
                             train_dl=train_dl,
                             valid_dl=valid_dl,
                             test_dl=test_dl,
                             state_file=state_file,
                             epochs=mc.epochs,
                             nthreads=mc.nthreads,
                             early_stop=mc.early_stop,
                             mode=mc.mode,
                             delta=mc.delta,
                             patience=mc.patience,
                             checkpoint_state=checkpoint_state,
                             save=mc.save
                             )

    # (ix) train model
    training_state = trainer.train()
