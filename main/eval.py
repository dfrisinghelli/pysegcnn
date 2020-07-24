# builtins
from __future__ import absolute_import
import os

# externals
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# local modules
from pytorch.trainer import NetworkTrainer
from pytorch.graphics import plot_confusion_matrix, plot_loss, plot_sample
from main.config import config


if __name__ == '__main__':

    # instanciate the NetworkTrainer class
    trainer = NetworkTrainer(config)
    print(trainer)

    if trainer.plot_cm:
        # predict each batch in the validation set
        cm, accuracy, loss = trainer.predict(pretrained=True, confusion=True,
                                             test=config['test'])

        # plot confusion matrix: labels of the dataset
        labels = [label['label'] for label in trainer.dataset.labels.values()]
        plot_confusion_matrix(cm, labels, state=trainer.state_file)

    # plot loss and accuracy
    plot_loss(trainer.loss_state)

    # whether to plot the samples of the validation/test dataset
    if trainer.plot_samples:

        # load pretrained model
        state = trainer.model.load(trainer.state_file,
                                   trainer.optimizer,
                                   trainer.state_path)
        trainer.model.eval()

        # base filename for each sample
        fname = trainer.state_file.split('.pt')[0]

        # set random seed for reproducibility
        np.random.seed(trainer.seed)

        # plot samples from the validation or test dataset
        dataset = trainer.test_ds if config['test'] else trainer.valid_ds
        dname = 'test' if config['test'] else 'val'

        # draw a number of samples from the validation/test set
        samples = np.arange(0, len(dataset))
        if trainer.nsamples > 0:
            samples = np.random.randint(len(dataset),
                                        size=min(trainer.nsamples,
                                                 len(dataset)))

        # iterate over the samples and plot inputs, ground truth and
        # model predictions
        for sample in samples:
            # a sample from the validation/test set
            inputs, labels = dataset[sample]

            # convert to net input shape
            net_inputs = torch.tensor(np.expand_dims(inputs, axis=0))

            # compute model predictions
            with torch.no_grad():
                y_pred = F.softmax(trainer.model(net_inputs),
                                   dim=1).argmax(dim=1).squeeze()

            # plot inputs, ground truth and model predictions
            sname = fname + '_{}_sample_{}.pt'.format(dname, sample)
            fig, ax = plot_sample(inputs,
                                  labels,
                                  trainer.dataset.use_bands,
                                  trainer.dataset.labels,
                                  y_pred=y_pred,
                                  bands=trainer.plot_bands,
                                  state=sname,
                                  stretch=True,
                                  alpha=5)
