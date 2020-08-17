# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:57:01 2020

@author: Daniel
"""
# builtins
from logging.config import dictConfig

# locals
from pysegcnn.core.models import Network
from pysegcnn.core.trainer import EvalConfig, LogConfig
from pysegcnn.core.predict import predict_samples, predict_scenes
from pysegcnn.core.logging import log_conf
from pysegcnn.core.graphics import plot_confusion_matrix, plot_loss
from pysegcnn.main.config import eval_config


if __name__ == '__main__':

    # instanciate the evaluation configuration
    ec = EvalConfig(**eval_config)

    # initialize logging
    log = LogConfig(ec.state_file)
    dictConfig(log_conf(log.log_file))

    # load the model state
    model, _, model_state = Network.load(ec.state_file)

    # plot loss and accuracy
    plot_loss(ec.state_file, outpath=ec.models_path)

    # check whether to evaluate the model on the training set, validation set
    # or the test set
    if ec.test is None:
        ds = model_state['train_ds']
    else:
        ds = model_state['test_ds'] if ec.test else model_state['valid_ds']

    # keyword arguments for plotting
    kwargs = {'bands': ec.plot_bands,
              'alpha': ec.alpha,
              'figsize': ec.figsize}

    # whether to predict each sample or each scene individually
    if ec.predict_scene:
        # reconstruct and predict the scenes in the validation/test set
        scenes, cm = predict_scenes(ds, model, scene_id=None, cm=ec.cm,
                                    plot=ec.plot_scenes,
                                    outpath=ec.scenes_path, **kwargs)

    else:
        # predict the samples in the validation/test set
        samples, cm = predict_samples(ds, model, cm=ec.cm,
                                      plot=ec.plot_samples,
                                      outpath=ec.sample_path, **kwargs)

    # whether to plot the confusion matrix
    if ec.cm:
        plot_confusion_matrix(cm, ds.dataset.labels,
                              state=ec.state_file.name.replace('.pt', '.png'),
                              outpath=ec.models_path)
