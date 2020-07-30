# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:57:01 2020

@author: Daniel
"""
# builtins
import os

# locals
from pysegcnn.core.trainer import NetworkTrainer
from pysegcnn.core.predict import predict_samples, predict_scenes
from pysegcnn.main.config import config, HERE
from pysegcnn.core.graphics import plot_confusion_matrix, plot_loss


if __name__ == '__main__':

    # instanciate the NetworkTrainer class
    trainer = NetworkTrainer(config)
    trainer

    # plot loss and accuracy
    plot_loss(trainer.loss_state, outpath=os.path.join(HERE, '_graphics/'))

    # check whether to evaluate the model on the training set, validation set
    # or the test set
    if trainer.test is None:
        ds = trainer.train_ds
    else:
        ds = trainer.test_ds if trainer.test else trainer.valid_ds

    # whether to predict each sample or each scene individually
    if trainer.predict_scene:
        # reconstruct and predict the scenes in the validation/test set
        scenes, cm = predict_scenes(ds,
                                    trainer.model,
                                    trainer.optimizer,
                                    trainer.state_path,
                                    trainer.state_file,
                                    None,
                                    trainer.cm,
                                    trainer.plot_scenes,
                                    bands=trainer.plot_bands,
                                    outpath=os.path.join(HERE, '_scenes/'),
                                    stretch=True,
                                    alpha=5)

    else:
        # predict the samples in the validation/test set
        samples, cm = predict_samples(ds,
                                      trainer.model,
                                      trainer.optimizer,
                                      trainer.state_path,
                                      trainer.state_file,
                                      trainer.cm,
                                      trainer.plot_samples,
                                      bands=trainer.plot_bands,
                                      outpath=os.path.join(HERE, '_samples/'),
                                      stretch=True,
                                      alpha=5)

    # whether to plot the confusion matrix
    if trainer.cm:
        plot_confusion_matrix(cm,
                              ds.dataset.labels,
                              normalize=True,
                              state=trainer.state_file,
                              outpath=os.path.join(HERE, '_graphics/')
                              )
