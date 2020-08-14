# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:33:38 2020

@author: Daniel
"""
# builtins
import logging

# locals
from pysegcnn.core.trainer import (DatasetConfig, SplitConfig, ModelConfig,
                                   StateConfig, NetworkTrainer)
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
    ds

    # (iii) instanciate the model state
    state = StateConfig(ds, sc, mc)
    state_file, loss_state = state.init_state()

    # initialize logging
    log_file = str(state_file).replace('.pt', '_train.log')
    logging.config.dictConfig(log_conf(log_file))

    # (iv) instanciate the training, validation and test datasets and
    # dataloaders
    train_ds, valid_ds, test_ds = sc.train_val_test_split(ds)
    train_dl, valid_dl, test_dl = sc.dataloaders(
        train_ds, valid_ds, test_ds, batch_size=mc.batch_size, shuffle=True,
        drop_last=False)

    # (iv) instanciate the model
    model = mc.init_model(ds)

    # (vi) instanciate the optimizer and the loss function
    optimizer = mc.init_optimizer(model)
    loss_function = mc.init_loss_function()

    # (vii) resume training from an existing model checkpoint
    (model, optimizer, checkpoint_state, max_accuracy) = mc.from_checkpoint(
        model, optimizer, state_file, loss_state)

    # (viii) initialize network trainer class for eays model training
    trainer = NetworkTrainer(model=model,
                             optimizer=optimizer,
                             loss_function=loss_function,
                             train_dl=train_dl,
                             valid_dl=valid_dl,
                             state_file=state_file,
                             loss_state=loss_state,
                             epochs=mc.epochs,
                             nthreads=mc.nthreads,
                             early_stop=mc.early_stop,
                             mode=mc.mode,
                             delta=mc.delta,
                             patience=mc.patience,
                             max_accuracy=max_accuracy,
                             checkpoint_state=checkpoint_state,
                             save=mc.save
                             )

    # (ix) train model
    training_state = trainer.train()
