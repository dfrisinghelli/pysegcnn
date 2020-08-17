# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:04:27 2020

@author: Daniel
"""
# builtins
import os
import itertools

# externals
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm as colormap

# locals
from pysegcnn.core.trainer import accuracy_function
from pysegcnn.main.config import HERE


# this function applies percentile stretching at the alpha level
# can be used to increase constrast for visualization
def contrast_stretching(image, alpha=5):

    # compute upper and lower percentiles defining the range of the stretch
    inf, sup = np.percentile(image, (alpha, 100 - alpha))

    # normalize image intensity distribution to
    # (alpha, 100 - alpha) percentiles
    norm = ((image - inf) * (image.max() - image.min()) /
            (sup - inf)) + image.min()

    # clip: values < inf = 0, values > sup = max
    norm[norm <= image.min()] = image.min()
    norm[norm >= image.max()] = image.max()

    return norm


def running_mean(x, w):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[w:] - cumsum[:-w]) / w


# plot_sample() plots a false color composite of the scene/tile together
# with the model prediction and the corresponding ground truth
def plot_sample(x, y, use_bands, labels, y_pred=None, figsize=(10, 10),
                bands=['nir', 'red', 'green'], state=None,
                outpath=os.path.join(HERE, '_samples/'), alpha=0):

    # check whether to apply constrast stretching
    rgb = np.dstack([contrast_stretching(x[use_bands.index(band)], alpha)
                     for band in bands])

    # get labels and corresponding colors
    ulabels = [label['label'] for label in labels.values()]
    colors = [label['color'] for label in labels.values()]

    # create a ListedColormap
    cmap = ListedColormap(colors)
    boundaries = [*labels.keys(), cmap.N]
    norm = BoundaryNorm(boundaries, cmap.N)

    # create figure: check whether to plot model prediction
    if y_pred is not None:

        # compute accuracy
        acc = accuracy_function(y_pred, y)

        # plot model prediction
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[2].imshow(y_pred, cmap=cmap, interpolation='nearest', norm=norm)
        ax[2].set_title('Prediction ({:.2f}%)'.format(acc * 100), pad=15)

    else:
        fig, ax = plt.subplots(1, 2, figsize=figsize)

    # plot false color composite
    ax[0].imshow(rgb)
    ax[0].set_title('R = {}, G = {}, B = {}'.format(*bands), pad=15)

    # plot ground thruth mask
    ax[1].imshow(y, cmap=cmap, interpolation='nearest', norm=norm)
    ax[1].set_title('Ground truth', pad=15)

    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               zip(colors, ulabels)]

    # plot patches as legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2,
               frameon=False)

    # save figure
    if state is not None:
        os.makedirs(outpath, exist_ok=True)
        fig.savefig(os.path.join(outpath, state.replace('.pt', '.png')),
                    dpi=300, bbox_inches='tight')

    return fig, ax


# plot_confusion_matrix() plots the confusion matrix of the validation/test
# set returned by the pytorch.predict function
def plot_confusion_matrix(cm, labels, normalize=True,
                          figsize=(10, 10), cmap='Blues', state=None,
                          outpath=os.path.join(HERE, '_graphics/')):

    # number of classes
    labels = [label['label'] for label in labels.values()]
    nclasses = len(labels)

    # string format to plot values of confusion matrix
    fmt = '.0f'

    # minimum and maximum values of the colorbar
    vmin, vmax = 0, cm.max()

    # check whether to normalize the confusion matrix
    if normalize:
        # normalize
        norm = cm.sum(axis=1, keepdims=True)

        # check for division by zero
        norm[norm == 0] = 1
        cm = cm / norm

        # change string format to floating point
        fmt = '.2f'
        vmin, vmax = 0, 1

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # get colormap
    cmap = colormap.get_cmap(cmap, 256)

    # plot confusion matrix
    im = ax.imshow(cm, cmap=cmap, vmin=vmin, vmax=vmax)

    # threshold determining the color of the values
    thresh = (cm.max() + cm.min()) / 2

    # brightest/darkest color of current colormap
    cmap_min, cmap_max = cmap(0), cmap(256)

    # plot values of confusion matrix
    for i, j in itertools.product(range(nclasses), range(nclasses)):
        ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                color=cmap_max if cm[i, j] < thresh else cmap_min)

    # axes properties and labels
    ax.set(xticks=np.arange(nclasses),
           yticks=np.arange(nclasses),
           xticklabels=labels,
           yticklabels=labels,
           ylabel='True',
           xlabel='Predicted')

    # add colorbar axes
    cax = fig.add_axes([ax.get_position().x1 + 0.025, ax.get_position().y0,
                        0.05, ax.get_position().y1 - ax.get_position().y0])
    fig.colorbar(im, cax=cax)

    # save figure
    if state is not None:
        os.makedirs(outpath, exist_ok=True)
        fig.savefig(os.path.join(outpath, state), dpi=300, bbox_inches='tight')

    return fig, ax


def plot_loss(loss_file, figsize=(10, 10), step=5,
              colors=['lightgreen', 'green', 'skyblue', 'steelblue'],
              outpath=os.path.join(HERE, '_graphics/')):

    # load the model loss
    state = torch.load(loss_file)

    # get all non-zero elements, i.e. get number of epochs trained before
    # early stop
    loss = {k: v[np.nonzero(v)].reshape(v.shape[0], -1) for k, v in
            state.items() if k != 'epoch'}

    # compute running mean with a window equal to the number of batches in
    # an epoch
    rm = {k: running_mean(v.flatten('F'), v.shape[0]) for k, v in loss.items()}

    # sort the keys of the dictionary alphabetically
    rm = {k: rm[k] for k in sorted(rm)}

    # number of epochs trained
    epochs = np.arange(0, loss['tl'].shape[1])

    # instanciate figure
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    # create axes for each parameter to plot
    ax2 = ax1.twinx()
    ax3 = ax1.twiny()
    ax4 = ax2.twiny()

    # list of axes
    axes = [ax2, ax1, ax4, ax3]

    # plot running mean loss and accuracy of the training dataset
    [ax.plot(v, color=c) for (k, v), ax, c in zip(rm.items(), axes, colors)
     if v.any()]

    # axes properties and labels
    nbatches = loss['tl'].shape[0]
    ax3.set(xticks=[], xticklabels=[])
    ax4.set(xticks=[], xticklabels=[])
    ax1.set(xticks=np.arange(0, nbatches * epochs[-1] + 1, nbatches * step),
            xticklabels=epochs[::step],
            xlabel='Epoch',
            ylabel='Loss',
            ylim=(0, 1))
    ax2.set(ylabel='Accuracy',
            ylim=(0.5, 1))

    # compute early stopping point
    if loss['va'].any():
        esepoch = np.argmax(loss['va'].mean(axis=0)) * nbatches
        esacc = np.max(loss['va'].mean(axis=0))
        ax1.vlines(esepoch, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1],
                   ls='--', color='grey')
        ax1.text(esepoch - nbatches, ax1.get_ylim()[0] + 0.01,
                 'epoch = {}, accuracy = {:.1f}%'
                 .format(int(esepoch / nbatches), esacc * 100),
                 ha='right', color='grey')

    # create a patch (proxy artist) for every color
    ulabels = ['Training accuracy', 'Training loss',
               'Validation accuracy', 'Validation loss']
    patches = [mlines.Line2D([], [], color=c, label=l) for c, l in
               zip(colors, ulabels)]
    # plot patches as legend
    ax1.legend(handles=patches, loc='lower left', frameon=False)

    # save figure
    os.makedirs(outpath, exist_ok=True)
    fig.savefig(os.path.join(
        outpath, os.path.basename(loss_file).replace('.pt', '.png')),
                dpi=300, bbox_inches='tight')

    return fig
