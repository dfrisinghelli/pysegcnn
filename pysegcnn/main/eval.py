# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:57:01 2020

@author: Daniel
"""
# builtins
import os

# locals
from pysegcnn.core.trainer import (DatasetConfig, SplitConfig, ModelConfig,
                                   TrainConfig, EvalConfig)
from pysegcnn.core.predict import predict_samples, predict_scenes
from pysegcnn.main.config import (dataset_config, split_config, model_config,
                                  train_config, eval_config, HERE)
from pysegcnn.core.graphics import plot_confusion_matrix, plot_loss


if __name__ == '__main__':

    # (i) instanciate the configurations
    dc = DatasetConfig(**dataset_config)
    sc = SplitConfig(**split_config)
    mc = ModelConfig(**model_config)
    tc = TrainConfig(**train_config)
    ec = EvalConfig(**eval_config)

    # (ii) instanciate the dataset
    ds = dc.init_dataset()
    ds

    # (iii) instanciate the training, validation and test datasets
    train_ds, valid_ds, test_ds = sc.train_val_test_split(ds)

    # (iv) instanciate the model state files
    state_file, loss_state = mc.init_state(ds, sc, tc)

    # (v) instanciate the model
    model = mc.init_model(ds)

    # (vi) instanciate the optimizer
    optimizer = tc.init_optimizer(model)

    # plot loss and accuracy
    plot_loss(loss_state, outpath=os.path.join(HERE, '_graphics/'))

    # check whether to evaluate the model on the training set, validation set
    # or the test set
    if ec.test is None:
        ds = train_ds
    else:
        ds = test_ds if ec.test else valid_ds

    # keyword arguments for plotting
    kwargs = {'bands': ec.plot_bands,
              'outpath': os.path.join(HERE, '_scenes/'),
              'stretch': True,
              'alpha': 5}

    # whether to predict each sample or each scene individually
    if ec.predict_scene:
        # reconstruct and predict the scenes in the validation/test set
        scenes, cm = predict_scenes(ds,
                                    model,
                                    optimizer,
                                    state_file,
                                    scene_id=None,
                                    cm=ec.cm,
                                    plot_scenes=ec.plot_scenes,
                                    **kwargs
                                    )

    else:
        # predict the samples in the validation/test set
        samples, cm = predict_samples(ds,
                                      model,
                                      optimizer,
                                      state_file,
                                      cm=ec.cm,
                                      plot_scenes=ec.plot_scenes,
                                      **kwargs)

    # whether to plot the confusion matrix
    if ec.cm:
        plot_confusion_matrix(cm,
                              ds.dataset.labels,
                              normalize=True,
                              state=state_file.name.replace('.pt', '.png'),
                              outpath=os.path.join(HERE, '_graphics/')
                              )
