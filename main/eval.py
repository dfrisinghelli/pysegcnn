# builtins
import sys

# externals
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# append path to local files to the python search path
sys.path.append('..')

# local modules
from main.config import state_path, plot_samples, nsamples, plot_bands, seed
from main.init import state_file, trainer

if __name__ == '__main__':

    # predict each batch in the validation set
    cm, accuracy, loss = trainer.predict(state_path, state_file,
                                         confusion=True)

    # calculate overal accuracy
    acc = (cm.diag().sum() / cm.sum()).numpy().item()
    print('After training for {:d} epochs, we achieved an overall accuracy of '
          '{:.2f}%  on the validation set!'.format(trainer.model.epoch,
                                                   acc * 100))

    # plot confusion matrix
    trainer.dataset.plot_confusion_matrix(cm, state=state_file)

    # plot loss and accuracy
    trainer.dataset.plot_loss(trainer.loss_state)

    # whether to plot the samples of the validation dataset
    if plot_samples:

        # base filename for each sample
        fname = state_file.split('.pt')[0]

        # set random seed for reproducibility
        np.random.seed(seed)

        # draw a number of samples from the validation set
        samples = np.arange(0, len(trainer.valid_ds))
        if nsamples > 0:
            samples = np.random.randint(len(trainer.valid_ds), size=nsamples)

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
                                                  bands=plot_bands,
                                                  state=sname,
                                                  stretch=True)
