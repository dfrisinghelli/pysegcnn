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
from pytorch.eval import predict, accuracy_function
from sparcs.sparcs_02_dataset import (net, valid_ds, valid_dl, optimizer,
                                      state_file)


if __name__ == '__main__':

    # predict each batch in the validation set
    cm = predict(net, valid_dl, optimizer, accuracy_function, state_file)

    # calculate overal accuracy
    acc = (cm.diag().sum() / cm.sum()).numpy().item()
    print('After training for {:d} epochs, we achieved an overall accuracy of '
          '{:.2f}%  on the validation set!'.format(net.epoch, acc * 100))

    # number of samples to plot
    n = 5

    # randomly sample n integers from [0, nsamples in the validation set]
    samples = np.random.randint(len(valid_ds), size=n)

    # iterate over the samples and plot inputs, ground truth and
    # model predictions
    for sample in samples:
        # a sample from the validation set
        inputs, labels = valid_ds[sample]

        # convert to net input shape
        net_inputs = torch.tensor(np.expand_dims(inputs, axis=0))

        # compute model predictions
        with torch.no_grad():
            y_pred = F.softmax(net(net_inputs), dim=1).argmax(dim=1).squeeze()

        # plot inputs, ground truth and model predictions
        fig, ax = valid_ds.dataset.dataset.plot_sample(inputs, labels, y_pred,
                                                       bands=['nir', 'red',
                                                              'green'])

    plt.show()
