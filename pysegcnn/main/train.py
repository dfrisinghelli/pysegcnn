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
                                   StateConfig, LogConfig, SingleDomainTrainer,
                                   DomainAdaptationTrainer)
from pysegcnn.core.logging import log_conf
from pysegcnn.main.config import (src_ds_config, src_split_config,
                                  trg_ds_config, trg_split_config,
                                  model_config)


if __name__ == '__main__':

    # (i) instanciate the source domain configurations
    src_dc = DatasetConfig(**src_ds_config)   # source domain dataset
    src_sc = SplitConfig(**src_split_config)  # source domain dataset split

    # (ii) instanciate the target domain configuration
    trg_dc = DatasetConfig(**trg_ds_config)   # target domain dataset
    trg_sc = SplitConfig(**trg_split_config)  # target domain dataset split

    # (iii) instanciate the model configuration
    mdlcfg = ModelConfig(**model_config)

    # (iv) instanciate the model state file
    sttcfg = StateConfig(src_dc, src_sc, trg_dc, trg_sc, mdlcfg)
    state_file = sttcfg.init_state()

    # (v) initialize logging
    log = LogConfig(state_file)
    dictConfig(log_conf(log.log_file))

    # (vi) instanciate the source and target domain datasets
    src_ds = src_dc.init_dataset()
    trg_ds = trg_dc.init_dataset()

    # (vii) instanciate the training, validation and test datasets and
    # dataloaders
    src_train_ds, src_valid_ds, src_test_ds = src_sc.train_val_test_split(
        src_ds)
    src_train_dl, src_valid_dl, src_test_dl = src_sc.dataloaders(
        src_train_ds, src_valid_ds, src_test_ds, batch_size=mdlcfg.batch_size,
        shuffle=True, drop_last=False)
    trg_train_ds, trg_valid_ds, trg_test_ds = trg_sc.train_val_test_split(
        trg_ds)
    trg_train_dl, trg_valid_dl, trg_test_dl = trg_sc.dataloaders(
        trg_train_ds, trg_valid_ds, trg_test_ds, batch_size=mdlcfg.batch_size,
        shuffle=True, drop_last=False)

    # (viii) instanciate the loss function
    cla_loss_function = mdlcfg.init_cla_loss_function()

    # (ix) instanciate the model
    if mdlcfg.transfer:

        if mdlcfg.supervised:
            model, optimizer, checkpoint_state = mdlcfg.init_model(trg_ds,
                                                                   state_file)
            trainer = SingleDomainTrainer(model=model,
                                          optimizer=optimizer,
                                          loss_function=cla_loss_function,
                                          state_file=state_file,
                                          epochs=mdlcfg.epochs,
                                          nthreads=mdlcfg.nthreads,
                                          early_stop=mdlcfg.early_stop,
                                          mode=mdlcfg.mode,
                                          delta=mdlcfg.delta,
                                          patience=mdlcfg.patience,
                                          checkpoint_state=checkpoint_state,
                                          save=mdlcfg.save,
                                          train_dl=trg_train_dl,
                                          valid_dl=trg_valid_dl,
                                          test_dl=trg_test_dl)

        else:
            model, optimizer, checkpoint_state = mdlcfg.init_model(src_ds,
                                                                   state_file)
            # instanciate the domain adaptation loss
            uda_loss_function = mdlcfg.init_uda_loss_function()
            trainer = DomainAdaptationTrainer(model=model,
                                              optimizer=optimizer,
                                              loss_function=cla_loss_function,
                                              state_file=state_file,
                                              epochs=mdlcfg.epochs,
                                              nthreads=mdlcfg.nthreads,
                                              early_stop=mdlcfg.early_stop,
                                              mode=mdlcfg.mode,
                                              delta=mdlcfg.delta,
                                              patience=mdlcfg.patience,
                                              checkpoint_state=checkpoint_state,
                                              save=mdlcfg.save,
                                              src_train_dl=src_train_dl,
                                              src_valid_dl=src_valid_dl,
                                              trg_train_dl=trg_train_dl,
                                              da_loss_function=uda_loss_function,
                                              uda_lambda=mdlcfg.uda_lambda)

    else:
        model, optimizer, checkpoint_state = mdlcfg.init_model(src_ds,
                                                               state_file)
        trainer = SingleDomainTrainer(model=model,
                                      optimizer=optimizer,
                                      loss_function=cla_loss_function,
                                      state_file=state_file,
                                      epochs=mdlcfg.epochs,
                                      nthreads=mdlcfg.nthreads,
                                      early_stop=mdlcfg.early_stop,
                                      mode=mdlcfg.mode,
                                      delta=mdlcfg.delta,
                                      patience=mdlcfg.patience,
                                      checkpoint_state=checkpoint_state,
                                      save=mdlcfg.save,
                                      train_dl=src_train_dl,
                                      valid_dl=src_valid_dl,
                                      test_dl=src_test_dl)

    # (x) train model
    training_state = trainer.train()
