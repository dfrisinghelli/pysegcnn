"""Functions to plot multispectral image data and model output.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import os
import pathlib
import itertools

# externals
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import ArtistAnimation
from matplotlib import cm as colormap

# locals
from pysegcnn.core.trainer import accuracy_function
from pysegcnn.main.config import HERE


def contrast_stretching(image, alpha=5):
    """Apply `normalization`_ to an image to increase constrast.

    Parameters
    ----------
    image : :py:class:`numpy.ndarray`
        The input image.
    alpha : `int`, optional
        The level of the percentiles. The default is `5`.

    Returns
    -------
    norm : :py:class:`numpy.ndarray`
        The normalized image.

    .. normalization:
        https://en.wikipedia.org/wiki/Normalization_(image_processing)

    """
    # compute upper and lower percentiles defining the range of the stretch
    inf, sup = np.percentile(image, (alpha, 100 - alpha))

    # normalize image intensity distribution to
    # (alpha, 100 - alpha) percentiles
    norm = ((image - inf) * ((image.max() - image.min()) / (sup - inf))
            + image.min())

    # clip: values < min = min, values > max = max
    norm[norm <= image.min()] = image.min()
    norm[norm >= image.max()] = image.max()

    return norm


def running_mean(x, w):
    """Compute a running mean of the input sequence.

    Parameters
    ----------
    x : `array_like`
        The sequence to compute a running mean on.
    w : `int`
        The window length of the running mean.

    Returns
    -------
    rm : :py:class:`numpy.ndarray`
        The running mean of the sequence ``x``.

    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[w:] - cumsum[:-w]) / w


def plot_sample(x, use_bands, labels,
                y=None,
                y_pred=None,
                figsize=(16, 9),
                bands=['nir', 'red', 'green'],
                alpha=0,
                state=None,
                fig=None,
                plot_path=os.path.join(HERE, '_samples/')
                ):
    """Plot false color composite (FCC), ground truth and model prediction.

    Parameters
    ----------
    x : :py:class:`numpy.ndarray` or :py:class:`torch.Tensor`, (b, h, w)
        Array containing the data of the tile, shape=(bands, height, width).
    use_bands : `list` of `str`
        List describing the order of the bands in ``x``.
    labels : `dict` [`int`, `dict`]
        The label dictionary. The keys are the values of the class labels
        in the ground truth ``y``. Each nested `dict` should have keys:
            ``'color'``
                A named color (`str`).
            ``'label'``
                The name of the class label (`str`).
    y : :py:class:`numpy.ndarray` or :py:class:`torch.Tensor`, optional
        Array containing the ground truth of tile ``x``, shape=(height, width).
        The default is `None`, i.e. the ground truth is not plotted.
    y_pred : :py:class:`numpy.ndarray` or :py:class:`torch.Tensor`, optional
        Array containing the prediction for tile ``x``, shape=(height, width).
        The default is `None`, i.e. the prediction is not plotted.
    figsize : `tuple`, optional
        The figure size in centimeters. The default is `(16, 9)`.
    bands : `list` [`str`], optional
        The bands to build the FCC. The default is `['nir', 'red', 'green']`.
    alpha : `int`, optional
        The level of the percentiles to increase constrast in the FCC.
        The default is `0`, i.e. no stretching is applied.
    state : `str` or `None`, optional
        Filename to save the plot to. ``state`` should be an existing model
        state file ending with `'.pt'`. The default is `None`, i.e. the plot is
        not saved to disk.
    fig : :py:class:`matplotlib.figure.Figure` or `None`, optional
        An instance of :py:class:`matplotlib.figure.Figure`. If specified, the
        images are plotted on this figure. ``fig`` needs to have one row with
        three axes, e.g.:

        .. code-block: python

            fig, _ = matplotlib.pyplot.subplots(1, 3)

        The default is `None`, which means a new figure of size ``figsize`` is
        created. Useful for creating animations.
    plot_path : `str` or :py:class:`pathlib.Path`, optional
        Output path for static plots. The default is
        `'pysegcnn/main/_samples'`.

    Raises
    ------
    TypeError
        Raised if ``fig`` is not `None` or an instance of
        :py:class:`matplotlib.figure.Figure`.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        An instance of :py:class:`matplotlib.figure.Figure`.
    ax : :py:class:`numpy.ndarray`
        A :py:class:`numpy.ndarray` of
        :py:class:`matplotlib.axes._subplots.AxesSubplot` instances.

    """
    # check whether the output path is valid
    plot_path = pathlib.Path(plot_path)
    if not plot_path.exists():
        # create output path
        plot_path.mkdir(parents=True, exist_ok=True)

    # check whether to apply constrast stretching
    rgb = np.dstack([contrast_stretching(x[use_bands.index(band), ...], alpha)
                     for band in bands])

    # get labels and corresponding colors
    ulabels = [label['label'] for label in labels.values()]
    colors = [label['color'] for label in labels.values()]

    # create a ListedColormap
    cmap = ListedColormap(colors)

    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               zip(colors, ulabels)]

    # initialize figure
    if fig is None:
        fig, _ = plt.subplots(1, 3, figsize=figsize)
    else:
        # require the passed figure to have at least three axes
        if not isinstance(fig, matplotlib.figure.Figure):
            raise TypeError('fig needs to be an instance of {}.'
                            .format(repr(plt.Figure)))

    # plot false color composite
    fig.axes[0].imshow(rgb)
    fig.axes[0].set_title('R = {}, G = {}, B = {}'.format(*bands), pad=15)

    # check whether to plot ground truth
    if y is not None:
        # plot ground thruth mask
        fig.axes[1].imshow(y, cmap=cmap, interpolation='nearest', vmin=0,
                           vmax=len(colors))
        fig.axes[1].set_title('Ground truth', pad=15)
    else:
        _del_axis(fig, 1)

    # check whether to plot model prediction
    if y_pred is not None:
        # plot model prediction
        fig.axes[2].imshow(y_pred, cmap=cmap, interpolation='nearest', vmin=0,
                           vmax=len(colors))

        # set title
        title = 'Prediction'
        if y is not None:
            acc = accuracy_function(y_pred, y)
            title += ' ({:.2f}%)'.format(acc * 100)
        fig.axes[2].set_title(title, pad=15)
    else:
        _del_axis(fig, 2)

    # if a ground truth or a model prediction is plotted, add legend
    if len(fig.axes) > 1:
        fig.axes[-1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2,
                            frameon=False)
    # save figure
    if state is not None:
        fig.savefig(plot_path.joinpath(state.replace('.pt', '.png')),
                    dpi=300, bbox_inches='tight')

    return fig


def _del_axis(fig, idx):
    """Quietly remove an axis from a figure.

    Parameters
    ----------
    fig : :py:class:`matplotlib.figure.Figure`
        An instance of :py:class:`matplotlib.figure.Figure`.
    idx : `int`
        The number of the axis to remove.

    """
    # remove axis ``idx``
    try:
        fig.delaxes(fig.axes[idx])
    # an IndexError is raised, if the axis ``idx`` does not exist
    except IndexError:
        # quietly pass
        return


def plot_confusion_matrix(cm, labels, normalize=True,
                          figsize=(10, 10), cmap='Blues', state_file=None,
                          outpath=os.path.join(HERE, '_graphics/')):
    """Plot the confusion matrix ``cm``.

    Parameters
    ----------
    cm : :py:class:`numpy.ndarray`
        The confusion matrix.
    labels : `dict` [`int`, `dict`]
        The label dictionary. The keys are the values of the class labels
        in the ground truth ``y``. Each nested `dict` should have keys:
            ``'color'``
                A named color (`str`).
            ``'label'``
                The name of the class label (`str`).
    normalize : `bool`, optional
        Whether to normalize the confusion matrix. The default is `True`.
    figsize : `tuple`, optional
        The figure size in centimeters. The default is `(10, 10)`.
    cmap : `str`, optional
        A matplotlib colormap. The default is `'Blues'`.
    state_file : `str` or `None` or :py:class:`pathlib.Path`, optional
        Filename to save the plot to. ``state`` should be an existing model
        state file ending with `'.pt'`. The default is `None`, i.e. the plot is
        not saved to disk.
    outpath : `str` or :py:class:`pathlib.Path`, optional
        Output path. The default is `'pysegcnn/main/_graphics/'`.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        An instance of :py:class:`matplotlib.figure.Figure`.
    ax : :py:class:`matplotlib.axes._subplots.AxesSubplot`
        An instance of :py:class:`matplotlib.axes._subplots.AxesSubplot`.

    """
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
    if state_file is not None:
        os.makedirs(outpath, exist_ok=True)
        fig.savefig(os.path.join(
            outpath, os.path.basename(state_file).replace('.pt', '_cm.png')),
                    dpi=300, bbox_inches='tight')

    return fig, ax


def plot_loss(state_file, figsize=(10, 10), step=5,
              colors=['lightgreen', 'green', 'skyblue', 'steelblue'],
              outpath=os.path.join(HERE, '_graphics/')):
    """Plot the observed loss and accuracy of a model run.

    Parameters
    ----------
    state_file : `str` or :py:class:`pathlib.Path`
        The model state file. Model state files are stored in
        `pysegcnn/main/_models`.
    figsize : `tuple`, optional
        The figure size in centimeters. The default is `(10, 10)`.
    step : `int`, optional
        The step to label epochs on the x-axis labels. The default is `5`, i.e.
        label each fifth epoch.
    colors : `list` [`str`], optional
        A list of four named colors supported by matplotlib.
        The default is `['lightgreen', 'green', 'skyblue', 'steelblue']`.
    outpath : `str` or :py:class:`pathlib.Path`, optional
        Output path. The default is `'pysegcnn/main/_graphics/'`.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        An instance of :py:class:`matplotlib.figure.Figure`.

    """
    # load the model state
    model_state = torch.load(state_file)

    # get all non-zero elements, i.e. get number of epochs trained before
    # early stop
    loss = {k: v[np.nonzero(v)].reshape(v.shape[0], -1) for k, v in
            model_state['state'].items()}

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
        esepoch = np.argmax(loss['va'].mean(axis=0)) * nbatches + 1
        esacc = np.max(loss['va'].mean(axis=0))
        ax1.vlines(esepoch, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1],
                   ls='--', color='grey')
        ax1.text(esepoch - nbatches, ax1.get_ylim()[0] + 0.01,
                 'epoch = {}, accuracy = {:.1f}%'
                 .format(int(esepoch / nbatches) + 1, esacc * 100),
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
        outpath, os.path.basename(state_file).replace('.pt', '_loss.png')),
                dpi=300, bbox_inches='tight')

    return fig


class Animate(object):
    """Easily create animations with :py:mod:`matplotlib`.

    .. important::

        Inspired by `celluloid`_.

    Attributes
    ----------
    axes_artists : `list`
        List of :py:mod:`matplotlib` artist names. These artists are stored for
        each frame. See the matplotlib documentation on the abstract `Artists`_
        class.
    path : :py:class:`pathlib.Path`
        Path to save animations.
    frames : `list`
        List of frames to generate animation. ``frames`` is populated by
        calling the :py:meth:`~pysegcnn.core.graphics.Animate.frame` method.
    animation : :py:class:`matplotlib.animation.ArtistAnimation`
        The animation with frames ``frames``. ``animation`` is generated by
        calling the :py:meth:`~pysegcnn.core.graphics.Animate.animate` method.

    .. _celluloid:
        https://github.com/jwkvam/celluloid

    .. _Artists:
        https://matplotlib.org/tutorials/intermediate/artists.html

    """

    # axes artists to store for each frame
    axes_artists = [
        'images', 'patches', 'texts', 'artists', 'collections', 'lines'
        ]

    def __init__(self, path):
        """Initialize.

        Parameters
        ----------
        path : `str` or :py:class:`pathlib.Path`
            Path to save animations.

        """
        self.path = pathlib.Path(path)

        # list of animation frames
        self.frames = []

    def frame(self, axes):
        """Create a snapshot of the current figure state.

        Parameters
        ----------
        axes : `list` [:py:class:`matplotlib.axes._subplots.AxesSubplot`]
            A list of :py:class:`matplotlib.axes._subplots.AxesSubplot`
            instances.

        """
        # artists in the current frame
        frame_artists = []

        # iterate over the axes
        for i, ax in enumerate(axes):
            # iterate over the artists
            for artist in self.axes_artists:
                # get current artist
                ca = getattr(ax, artist)

                # add current artist to frame if it does not already exist
                frame_artists.extend([a for a in ca if a not in
                                      np.array(self.frames).flatten()])

        # append frame to list of frames to animate
        self.frames.append(frame_artists)

    def animate(self, fig, **kwargs):
        """Create an animation.

        Parameters
        ----------
        fig : :py:class:`matplotlib.figure.Figure`
            An instance of :py:class:`matplotlib.figure.Figure`.
        **kwargs
            Additional keyword arguments passed to
            :py:class:`matplotlib.animation.ArtistAnimation`.

        Returns
        -------
        animation : :py:class:`matplotlib.animation.ArtistAnimation`
            The animation.

        """
        self.animation = ArtistAnimation(fig, self.frames, **kwargs)
        return self.animation

    def save(self, filename, **kwargs):
        """Save a `gif`_ animation to disk.

        .. important::

            This requires `ImageMagick`_ to be installed on your system.

        Parameters
        ----------
        filename : `str` or :py:class:`pathlib.Path`
            Path to save animation.
        **kwargs
            Additional keyword arguments passed to
            :py:meth:`matplotlib.animation.ArtistAnimation.save`.

        Raises
        ------
        ValueError
            Raised if ``filename`` does not point to a gif file.

        .. _ImageMagick:
            https://imagemagick.org/

        .. _gif:
            https://de.wikipedia.org/wiki/Graphics_Interchange_Format

        """
        if not str(filename).endswith('.gif'):
            raise ValueError('filename should point to a gif file, got {}.'
                             .format(pathlib.Path(filename).suffix))

        # save animation to disk
        self.animation.save(str(self.path.joinpath(filename)),
                            writer='imagemagick', **kwargs)
