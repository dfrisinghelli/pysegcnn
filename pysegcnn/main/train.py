# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:33:38 2020

@author: Daniel
"""
# locals
from pysegcnn.core.initconf import NetworkTrainer
from pysegcnn.main.config import (dataset_config, split_config,
                                  model_config, train_config)


if __name__ == '__main__':

    # instanciate the NetworkTrainer class
    trainer = NetworkTrainer(dconfig=dataset_config,
                             sconfig=split_config,
                             mconfig=model_config,
                             tconfig=train_config)
    trainer

    # train the network
    training_state = trainer.train()
