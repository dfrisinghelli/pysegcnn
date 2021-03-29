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
import logging

# externals
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import ArtistAnimation
from matplotlib import cm as colormap

# locals
from pysegcnn.core.utils import (accuracy_function, check_filename_length,
                                 report2df)
from pysegcnn.main.train_config import HERE

# plot font size configuration
MEDIUM = 22
BIG = 24

# controls default font size
plt.rc('font', size=MEDIUM)

# axes labels size
plt.rc('axes', titlesize=BIG, labelsize=BIG)

# axes ticks size
plt.rc('xtick', labelsize=MEDIUM)
plt.rc('ytick', labelsize=MEDIUM)

# legend font size
plt.rc('legend', fontsize=MEDIUM)

# figure title size
plt.rc('figure', titlesize=BIG)

# training metrics
METRICS = ['train_loss', 'train_accu', 'valid_loss', 'valid_accu']

# module level logger
LOGGER = logging.getLogger(__name__)


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


def ceil_decimal(x, decimal=0):
    """Ceil to arbitrary decimal place.

    Parameters
    ----------
    x : `float`
        The floating point number to ceil.
    decimal : `int`, optional
        The decimal place to ceil. The default is 0.

    Returns
    -------
    ceiled : `float`
        The ceiled floating point number ``x``.

    """
    return np.round(x + 0.5 * 10 ** (-decimal), decimal)


def floor_decimal(x, decimal=0):
    """Floor to arbitrary decimal place.

    Parameters
    ----------
    x : `float`
        The floating point number to floor.
    decimal : `int`, optional
        The decimal place to floor. The default is 0.

    Returns
    -------
    floored : `float`
        The floored floating point number ``x``.

    """
    return np.round(x - 0.5 * 10 ** (-decimal), decimal)


def plot_sample(x, use_bands, labels,
                y=None,
                y_pred={},
                accuracy=False,
                figsize=(16, 9),
                bands=['red', 'green', 'blue'],
                alpha=0,
                state=None,
                fig=None,
                plot_path=os.path.join(HERE, '_samples/'),
                hide_labels=False
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
    y_pred : `dict` [`str`, :py:class:`numpy.ndarray`], optional
        Dictionary of predictions. The keys indicate the prediction method and
        the values are the predicted classes for tile ``x``,
        shape=(height, width). The default is `{}`, i.e. no prediction is
        plotted.
    accuracy : `bool`, optional
        Whether to calculate the accuracy of the predictions ``y_pred`` with
        respect to the ground truth ``y``. The default is `False`
    figsize : `tuple` [`int`], optional
        The figure size in centimeters. The default is `(16, 9)`.
    bands : `list` [`str`], optional
        The bands to build the FCC. The default is `['red', 'green', 'blue']`.
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
    hide_labels : `bool`
        Whether to show the axis and axis labels. The default is `False`.

    Raises
    ------
    TypeError
        Raised if ``fig`` is not `None` or an instance of
        :py:class:`matplotlib.figure.Figure`.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        An instance of :py:class:`matplotlib.figure.Figure`.

    """
    # check whether the output path is valid
    plot_path = pathlib.Path(plot_path)
    if not plot_path.exists():
        # create output path
        plot_path.mkdir(parents=True, exist_ok=True)

    # check whether to apply constrast stretching
    rgb = np.dstack([np.clip(
        contrast_stretching(x[use_bands.index(band), ...], alpha), 0, 1)
        for band in bands]).astype(np.float32)

    # sort the labels in ascending order
    sorted_labels = {k: v for k, v in sorted(labels.items())}

    # create a ListedColormap: sort the colors
    cmap = ListedColormap([v['color'] for k, v in sorted_labels.items()])

    # create a BoundaryNorm instance to map the integers to their corresponding
    # colors and add an upper boundary
    norm = BoundaryNorm(list(sorted_labels.keys()) + [255], cmap.N)

    # create a patch (proxy artist) for every color
    # patches = [mpatches.Patch(color=v['color'], label=v['label']) for _, v in
    #            sorted_labels.items()]

    # number of required axes
    naxes = len(y_pred) + 2

    # initialize figure
    if fig is None:
        fig, _ = plt.subplots(1, naxes, figsize=figsize)
    else:
        # require the passed figure to have at least three axes
        if not isinstance(fig, matplotlib.figure.Figure):
            raise TypeError('fig needs to be an instance of {}.'
                            .format(repr(plt.Figure)))

        # clear figure axes
        for ax in fig.axes:
            ax.clear()

    # plot false color composite
    fig.axes[0].imshow(rgb)

    # set title
    # fig.axes[0].text(0.5, 1.04, 'R={}, G={}, B={}'.format(*bands),
    #                  transform=fig.axes[0].transAxes, ha='center',
    #                  va='bottom')

    # check whether to plot ground truth
    if y is not None:
        removed = 2
        # plot ground thruth mask
        fig.axes[1].imshow(y, cmap=cmap, interpolation='nearest', norm=norm)

        fig.axes[1].text(0.5, 1.04, 'Ground truth',
                         transform=fig.axes[1].transAxes, ha='center',
                         va='bottom')
    else:
        removed = 1

    # check whether to plot predictions
    for i, (k, v) in enumerate(y_pred.items()):

        # axis to plot current prediction
        ax = fig.axes[int(i + removed)]

        # check whether the ground truth is specified and calculate accuracy
        if y is not None and accuracy:
            acc = accuracy_function(v, y)
            k = '{:.2f}%'.format(acc * 100)

        # plot model prediction
        ax.imshow(v, cmap=cmap, interpolation='nearest', norm=norm)
        ax.text(0.5, 1.04, k, transform=ax.transAxes, ha='center', va='bottom')

    # remove empty axes and check whether to hide axis labels
    for ax in fig.axes:

        # remove empty axes
        if not ax.images:
            fig.delaxes(ax)

        # hide both x and y-ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # hide axis and labels
        if hide_labels:
            # hide axis ticks, labels and text artists
            ax.axis('off')
            for t in ax.texts:
                t.set_visible(False)

    # adjust spacing
    fig.subplots_adjust(wspace=0.02)

    # if a ground truth or a model prediction is plotted, add legend
    # if len(fig.axes) > 1 and not hide_labels:
    #     fig.axes[-1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2,
    #                         frameon=False)

    # save figure
    if state is not None:
        # chech maximum filename length of 260 characters on Windows
        filename = check_filename_length(
            plot_path.joinpath(state.replace('.pt', '.png')))
        fig.savefig(filename, dpi=300, bbox_inches='tight')

    return fig


def plot_confusion_matrix(cm, labels, normalize=True, figsize=(10, 10),
                          cmap='Blues', state_file=None, subset=None,
                          outpath=os.path.join(HERE, '_graphics/')):
    """Plot the confusion matrix ``cm``.

    Parameters
    ----------
    cm : :py:class:`numpy.ndarray`
        The confusion matrix.
    labels : `list` [`str`]
        Names of the classes.
    normalize : `bool`, optional
        Whether to normalize the confusion matrix. The default is `True`.
    figsize : `tuple` [`int`], optional
        The figure size in centimeters. The default is `(10, 10)`.
    cmap : `str`, optional
        A matplotlib colormap. The default is `'Blues'`.
    state_file : `str` or `None` or :py:class:`pathlib.Path`, optional
        Filename to save the plot to. ``state`` should be an existing model
        state file ending with `'.pt'`. The default is `None`, i.e. the plot is
        not saved to disk.
    subset : `str` or `None`, optional
        Name of the subset ``cm`` was computed on. If ``subset`` is not `None`,
        it is added to the filename of the confusion matrix plot. The default
        is `None`.
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

    # rotate x-tick labels
    for label in ax.get_xticklabels():
        label.set_rotation(90)

    # add colorbar axes
    cax = fig.add_axes([ax.get_position().x1 + 0.025, ax.get_position().y0,
                        0.05, ax.get_position().y1 - ax.get_position().y0])
    fig.colorbar(im, cax=cax)

    # save figure
    if state_file is not None:
        os.makedirs(outpath, exist_ok=True)
        basename = os.path.basename(state_file).replace(
            '.pt', '_cm_{}.pdf'.format(subset) if subset is not None else
            '_cm.pdf')
        filename = check_filename_length(os.path.join(outpath, basename))
        fig.savefig(filename, bbox_inches='tight')

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
    figsize : `tuple` [`int`], optional
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
    state_file = pathlib.Path(state_file)
    model_state = torch.load(state_file)
    LOGGER.info('Plot model statistics: {}'.format(state_file.name))

    # get all non-zero elements, i.e. get number of epochs trained before
    # early stop
    loss = {k: v for k, v in model_state['state'].items() if k in METRICS}

    # compute running mean with a window equal to the number of batches in
    # an epoch
    rm = {k: running_mean(v.flatten('F'), v.shape[0]) for k, v in loss.items()}

    # sort the keys of the dictionary alphabetically
    rm = {k: rm[k] for k in sorted(rm)}

    # number of epochs trained
    epochs = np.arange(0, loss['train_loss'].shape[1] + 1)

    # compute number of mini-batches in training and validation set
    ntbatches = loss['train_loss'].shape[0]
    nvbatches = loss['valid_loss'].shape[0]

    # the mean loss/accuraries at each epoch
    markers = [ntbatches, ntbatches, nvbatches, nvbatches]

    # instanciate figure
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    # create axes for each parameter to plot
    ax2 = ax1.twinx()
    ax3 = ax1.twiny()
    ax4 = ax2.twiny()

    # list of axes
    axes = [ax2, ax1, ax4, ax3]

    # plot running mean loss and accuracy
    [ax.plot(v, 'o', ls='-', color=c, markevery=marker) for (k, v), ax, c,
     marker in zip(rm.items(), axes, colors, markers) if v.any()]

    # axes properties and labels: clear redundant axes labels
    ax3.set(xticks=[], xticklabels=[])
    ax4.set(xticks=[], xticklabels=[])

    # y-axis limits
    max_loss = max(rm['train_loss'].max(), rm['valid_loss'].max())
    min_loss = min(rm['train_loss'].min(), rm['valid_loss'].min())
    max_accu = max(rm['train_accu'].max(), rm['valid_accu'].max())
    min_accu = min(rm['train_accu'].min(), rm['valid_accu'].min())
    yl_max, yl_min = (ceil_decimal(max_loss, decimal=1),
                      floor_decimal(min_loss, decimal=1))
    ya_max, ya_min = (ceil_decimal(max_accu, decimal=1),
                      floor_decimal(min_accu, decimal=1))
    ax1.set(xticks=np.arange(-ntbatches, ntbatches * epochs[-1],
                             ntbatches * step),
            xticklabels=epochs[::step], xlabel='Epoch', ylabel='Loss',
            ylim=(yl_min, yl_max))

    # accuracy y-axis limits
    ax2.set(ylabel='Accuracy', ylim=(ya_min, ya_max))

    # compute early stopping point
    if loss['valid_accu'].any():
        esepoch = np.argmax(loss['valid_accu'].mean(axis=0))
        esacc = np.max(loss['valid_accu'].mean(axis=0))
        ax3.vlines(esepoch * nvbatches,
                   ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1],
                   ls='--', color='grey')
        ax3.text(esepoch * nvbatches - 1, ax1.get_ylim()[0] + 0.005,
                 'epoch = {}, accuracy = {:.1f}%'
                 .format(esepoch + 1, esacc * 100), ha='right', color='grey')

    # create a patch (proxy artist) for every color
    ulabels = ['Train accuracy', 'Train loss',
               'Valid accuracy', 'Valid loss']
    patches = [mlines.Line2D([], [], color=c, label=l) for c, l in
               zip(colors, ulabels)]
    # plot patches as legend
    ax1.legend(handles=patches, loc='upper left', frameon=False, ncol=4)

    # save figure
    os.makedirs(outpath, exist_ok=True)
    filename = check_filename_length(os.path.join(
        outpath, os.path.basename(state_file).replace('.pt', '_loss.pdf')))
    fig.savefig(filename, bbox_inches='tight')

    return fig


def plot_class_distribution(ds, figsize=(16, 9), alpha=0.5):
    """Plot the spectral distribution of the different classes in ``ds``.

    Parameters
    ----------
    ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
        An instance of the dataset to plot the class distribution over the
        different spectral bands. Make sure to initialize ``ds`` with the
        parameter ``tile_size=None``, which conserves the original size of
        each image in the dataset.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        An instance of :py:class:`matplotlib.figure.Figure`.

    """
    # compute class distribution
    cls_ds = ds.class_distribution()

    # drop classes which are not represented in the dataset
    cls_ds = {k: v for k, v in cls_ds.items() if np.any(v)}

    # number of pixels of each class
    npix_per_class = ['{:.2f}'.format(v.shape[0] * 1e-6) for k, v in
                      cls_ds.items()]

    # labels and colors for the different classes
    labels = [ds.labels[cls_id]['label'] for cls_id in cls_ds.keys()]
    colors = [ds.labels[cls_id]['color'] for cls_id in cls_ds.keys()]

    # number of spectral bands in the dataset
    nbands = len(ds.use_bands)

    # create a figure based on the number of spectral bands in the dataset
    fig, axes = plt.subplots(int(np.ceil(max(1, nbands / 2))), min(2, nbands),
                             figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    # iterate over the different bands
    for band in range(nbands):
        LOGGER.info('Plotting band: {}'.format(ds.use_bands[band]))

        # current axis
        ax = axes[band]

        # get the spectral data for each class
        data = [x[:, band] for x in cls_ds.values()]

        # plot spectral distribution of the classes in the current band
        bplot = ax.boxplot(data, labels=labels, patch_artist=True,
                           whis=[5, 95], showfliers=False, showmeans=True,
                           meanline=True, meanprops={'color': 'black',
                                                     'ls': 'dashed'},
                           medianprops={'color': 'black'})

        # set axis y-limits: physical limits are (0, 1) for reflectance data
        ax.set_ylim(-0.1, 1.25)
        ax.set_yticks(np.arange(0, 1.2, 0.2))

        # set colors of the boxes for the classes
        for k, artists in bplot.items():

            # the artists to color
            if k in ['boxes', 'medians']:

                # iterate over the artists
                for c, art in enumerate(artists):

                    # patch artists
                    if isinstance(art, matplotlib.patches.Patch):
                        # set the colors of the patches
                        art.set_facecolor(colors[c])
                        art.set_alpha(alpha)

        # add name of the spectral band to the plot
        ax.text(x=0.6, y=ax.get_ylim()[1] - 0.05, s='({}: {})'.format(
            getattr(ds.sensor, ds.use_bands[band]).number,
            getattr(ds.sensor, ds.use_bands[band]).name),
            ha='left', va='top', weight='bold')

        # add mean value to plot
        upper_whisker = [item.get_ydata()[1] for item in
                         bplot['whiskers'][1::2]]
        for i, (d, w) in enumerate(zip(data, upper_whisker)):
            # calculate mean
            mu = d.mean()

            # add mean value to plot at the top of each whisker
            ax.text(x=i + 1, y=w.item() + 0.025, s='{:.2f}'.format(mu),
                    ha='center', va='bottom')

    # add text with number of pixels per class to plot
    for i, lab in enumerate(npix_per_class):
        axes[-2].text(i + 1, axes[-2].get_ylim()[0] - 0.5, lab, ha='center')

    # adjust space between subplots
    fig.subplots_adjust(hspace=0.075, wspace=0.025)

    # remove empty axes
    for ax in axes:
        if not ax.lines:
            # hide axis ticks, labels and text artists
            ax.axis('off')
            for t in ax.texts:
                t.set_visible(False)

    return fig


def plot_classification_report(report, labels, figsize=(10, 10),
                               cmap='Blues', **kwargs):
    """Plot the :py:func:`sklearn.metrics.classification_report` as heatmap.

    Parameters
    ----------
    report : `dict` or :py:class:`pandas.DataFrame`
        The dictionary returned by setting ``output_dict=True`` in
        :py:func:`sklearn.metrics.classification_report` or the
        :py:class:`pandas.DataFrame` returned by
        :py:func:`pysegcnn.core.utils.report2df`.
    labels : `list` [`str`]
        Names of the classes.
    figsize : `tuple` [`int`], optional
        The figure size in centimeters. The default is `(10, 10)`.
    cmap : `str`, optional
        A matplotlib colormap. The default is `'Blues'`.
    **kwargs :
        Additional keyword arguments passed to :py:func:`seaborn.heatmap`.

    Returns
    -------
     fig : :py:class:`matplotlib.figure.Figure`
        An instance of :py:class:`matplotlib.figure.Figure`.

    """
    # check input type
    df = report
    if isinstance(report, dict):
        # convert to DataFrame
        df = report2df(report, labels)

    # get overall accuracy
    try:
        overall_accuracy = df.loc['accuracy'].loc['f1-score']
    except KeyError:
        overall_accuracy = df.loc['micro avg'].loc['f1-score']

    # drop micro avg/accuracy from dataframe
    metrics = df.loc[labels + ['macro avg', 'weighted avg']]

    # create a figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # plot class wise statistics as heatmap
    sns.heatmap(metrics, vmin=0, vmax=1, annot=True, fmt='.2f',
                ax=ax, xticklabels=[c.capitalize() for c in metrics.columns],
                yticklabels=[r.capitalize() for r in metrics.index], cmap=cmap,
                **kwargs)

    # add a white line separating class-wise and average statistics
    ax.plot(np.arange(0, len(metrics.columns) + 1),
            np.tile(len(labels), len(metrics.columns) + 1),
            color='white', lw=3)

    # set figure title
    ax.set_title('Accuracy: {:.2f}'.format(overall_accuracy), pad=20)

    # rotate x-tick labels
    for label in ax.get_xticklabels():
        label.set_rotation(90)

    # rotate y-tick labels
    for label in ax.get_yticklabels():
        label.set_rotation(0)

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
