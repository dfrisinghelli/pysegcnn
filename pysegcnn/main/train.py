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

# builtins
from logging.config import dictConfig

# locals
from pysegcnn.core.trainer import (DatasetConfig, SplitConfig, ModelConfig,
                                   TransferLearningConfig, StateConfig,
                                   LogConfig, DomainAdaptationTrainer)
from pysegcnn.main.config import (src_ds_config, src_split_config,
                                  trg_ds_config, trg_split_config,
                                  model_config, tlda_config)
from pysegcnn.core.logging import log_conf


if __name__ == '__main__':

    # (i) instanciate the source domain configurations
    src_dc = DatasetConfig(**src_ds_config)   # source domain dataset
    src_sc = SplitConfig(**src_split_config)  # source domain dataset split

    # (ii) instanciate the target domain configuration
    trg_dc = DatasetConfig(**trg_ds_config)   # target domain dataset
    trg_sc = SplitConfig(**trg_split_config)  # target domain dataset split

    # (iii) instanciate the model configuration
    net_mc = ModelConfig(**model_config)

    # (iv) instanciate the transfer learning configuration
    trn_sf = TransferLearningConfig(**tlda_config)

    # (v) instanciate the model state file
    net_sc = StateConfig(src_dc, src_sc, trg_dc, trg_sc, net_mc, trn_sf)
    state_file = net_sc.init_state()

    # (vi) instanciate logging configuration
    net_lc = LogConfig(state_file)
    dictConfig(log_conf(net_lc.log_file))

    # (vii) instanciate the datasets to train the model on
    src_ds = src_dc.init_dataset()
    trg_ds = trg_dc.init_dataset()

    # (viii) instanciate the training, validation and test datasets and
    # dataloaders for the source domain
    src_tra_ds, src_val_ds, src_tes_ds = src_sc.train_val_test_split(src_ds)
    src_tra_dl, src_val_dl, src_tes_dl = src_sc.dataloaders(
        src_tra_ds, src_val_ds, src_tes_ds, batch_size=net_mc.batch_size,
        shuffle=True, drop_last=False)

    # (ix) instanciate the training, validation and test datasets and
    # dataloaders dor the target domain
    trg_tra_ds, trg_val_ds, trg_tes_ds = trg_sc.train_val_test_split(trg_ds)
    trg_tra_dl, trg_val_dl, trg_tes_dl = trg_sc.dataloaders(
        trg_tra_ds, trg_val_ds, trg_tes_ds, batch_size=net_mc.batch_size,
        shuffle=True, drop_last=False)

    # (x) instanciate the model
    if trn_sf.transfer and (trn_sf.supervised or trn_sf.uda_from_pretrained):
        # check whether to load a pretrained model for (un)supervised transfer
        # learning
        net, optimizer, checkpoint = trn_sf.transfer_model(
            trn_sf.pretrained_path,
            nclasses=len(src_ds).labels,
            optim_kwargs=net_mc.optim_kwargs,
            freeze=trn_sf.freeze)
    else:
        # initialize model from scratch or from an existing model checkpoint
        net, optimizer, checkpoint = net_mc.init_model(
            len(src_ds.use_bands), len(src_ds.labels), state_file)

    # (xi) instanciate the network trainer class
    trainer = DomainAdaptationTrainer(
        model=net,
        optimizer=optimizer,
        state_file=net.state_file,
        src_train_dl=src_tra_dl,
        src_valid_dl=src_val_dl,
        src_test_dl=src_tes_dl,
        epochs=net_mc.epochs,
        nthreads=net_mc.nthreads,
        early_stop=net_mc.early_stop,
        mode=net_mc.mode,
        delta=net_mc.delta,
        patience=net_mc.patience,
        checkpoint_state=checkpoint,
        save=net_mc.save,
        supervised=trn_sf.supervised,
        trg_train_dl=trg_tra_dl,
        trg_valid_dl=trg_val_dl,
        trg_test_dl=trg_tes_dl,
        uda_loss_function=trn_sf.uda_loss_function,
        uda_lambda=trn_sf.uda_lambda,
        uda_pos=trn_sf.uda_pos)

    # (xi) train the model
    # training_state = trainer.train()
