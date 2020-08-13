# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:33:38 2020

@author: Daniel
"""
# locals
from pysegcnn.core.trainer import (DatasetConfig, SplitConfig, ModelConfig,
                                   TrainConfig, NetworkTrainer)
from pysegcnn.main.config import (dataset_config, split_config,
                                  model_config, train_config)


if __name__ == '__main__':

    # write code that checks for list of seeds, band combinations etc. here.

    # (i) instanciate the configurations
    dc = DatasetConfig(**dataset_config)
    sc = SplitConfig(**split_config)
    mc = ModelConfig(**model_config)
    tc = TrainConfig(**train_config)

    # (ii) instanciate the dataset
    ds = dc.init_dataset()
    ds

    # (iii) instanciate the training, validation and test datasets and
    # dataloaders
    train_ds, valid_ds, test_ds = sc.train_val_test_split(ds)
    train_dl, valid_dl, test_dl = sc.dataloaders(train_ds,
                                                 valid_ds,
                                                 test_ds,
                                                 batch_size=mc.batch_size,
                                                 shuffle=True,
                                                 drop_last=False)

    # (iv) instanciate the model state files
    state_file, loss_state = mc.init_state(ds, sc, tc)

    # (v) instanciate the model
    model = mc.init_model(ds)

    # (vi) instanciate the optimizer and the loss function
    optimizer = tc.init_optimizer(model)
    loss_function = tc.init_loss_function()

    # (vii) resume training from an existing model checkpoint
    checkpoint_state, max_accuracy = mc.load_checkpoint(state_file, loss_state,
                                                        model, optimizer)

    # (viii) initialize network trainer class for eays model training
    trainer = NetworkTrainer(model=model,
                             optimizer=optimizer,
                             loss_function=loss_function,
                             train_dl=train_dl,
                             valid_dl=valid_dl,
                             state_file=state_file,
                             loss_state=loss_state,
                             epochs=tc.epochs,
                             nthreads=tc.nthreads,
                             early_stop=tc.early_stop,
                             mode=tc.mode,
                             delta=tc.delta,
                             patience=tc.patience,
                             max_accuracy=max_accuracy,
                             checkpoint_state=checkpoint_state,
                             save=tc.save
                             )

    # (ix) train model
    training_state = trainer.train()
