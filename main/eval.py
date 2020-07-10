# builtins
import os
import sys

# externals
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# append path to local files to the python search path
sys.path.append('..')

# local modules
from pytorch.trainer import NetworkTrainer
from main.config import config


if __name__ == '__main__':

    # instanciate the NetworkTrainer class
    trainer = NetworkTrainer(config)

    if trainer.plot_cm:
        # predict each batch in the validation set
        cm, accuracy, loss = trainer.predict(pretrained=True, confusion=True)

        # calculate overal accuracy
        acc = (cm.diag().sum() / cm.sum()).numpy().item()
        print('After training for {:d} epochs, we achieved an overall '
              'accuracy of {:.2f}%  on the validation set!'
              .format(trainer.model.epoch, acc * 100))

        # plot confusion matrix
        trainer.dataset.plot_confusion_matrix(cm, state=trainer.state_file)

    # plot loss and accuracy
    trainer.dataset.plot_loss(trainer.loss_state)

    # whether to plot the samples of the validation dataset
    if trainer.plot_samples:

        # load pretrained model
        state = trainer.model.load(trainer.optimizer, trainer.state_file,
                                   trainer.state_path)
        trainer.model.eval()

        # base filename for each sample
        fname = trainer.state_file.split('.pt')[0]

        # set random seed for reproducibility
        np.random.seed(trainer.seed)

        # draw a number of samples from the validation set
        samples = np.arange(0, len(trainer.valid_ds))
        if trainer.nsamples > 0:
            samples = np.random.randint(len(trainer.valid_ds),
                                        size=trainer.nsamples)

        # iterate over the samples and plot inputs, ground truth and
        # model predictions
        for sample in samples:
            # a sample from the validation set
            inputs, labels = trainer.valid_ds[sample]

            # convert to net input shape
            net_inputs = torch.tensor(np.expand_dims(inputs, axis=0))

            # compute model predictions
            with torch.no_grad():
                y_pred = F.softmax(trainer.model(net_inputs),
                                   dim=1).argmax(dim=1).squeeze()

            # plot inputs, ground truth and model predictions
            sname = fname + '_sample_{}.pt'.format(sample)
            fig, ax = trainer.dataset.plot_sample(inputs, labels, y_pred,
                                                  bands=trainer.plot_bands,
                                                  state=sname,
                                                  stretch=True,
                                                  alpha=5)
