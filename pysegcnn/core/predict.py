"""Functions for model inference.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import enum
import pathlib
import logging

# externals
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torch.nn.functional as F

# locals
from pysegcnn.core.utils import reconstruct_scene, accuracy_function
from pysegcnn.core.graphics import plot_sample, Animate
from pysegcnn.core.split import RandomSubset, SceneSubset

# module level logger
LOGGER = logging.getLogger(__name__)


def predict_samples(ds, model, label_map=None, cm=False, plot=False, **kwargs):
    """Classify each sample in ``ds`` with model ``model``.

    Parameters
    ----------
    ds : :py:class:`pysegcnn.core.split.RandomSubset` or
    :py:class:`pysegcnn.core.split.SceneSubset`
        An instance of :py:class:`pysegcnn.core.split.RandomSubset` or
        :py:class:`pysegcnn.core.split.SceneSubset`.
    model : :py:class:`pysegcnn.core.models.Network`
        An instance of :py:class:`pysegcnn.core.models.Network`.
    label_map : `dict` [`int`, `int`], :py:class:`enum.EnumMeta` or `None`
        Dictionary with labels of ``ds`` as keys and corresponding labels of
        the ``model`` as values. If specified, ``label_map`` is used to map the
        model label predictions to the actual labels of the dataset ``ds``. The
        default is `None`, i.e. ``model`` and ``ds`` share the same labels.
    cm : `bool`, optional
        Whether to compute the confusion matrix. The default is `False`.
    plot : `bool`, optional
        Whether to plot a false color composite, ground truth and model
        prediction for each sample. The default is `False`.
    **kwargs
        Additional keyword arguments passed to
        :py:func:`pysegcnn.core.graphics.plot_sample`.

    Raises
    ------
    TypeError
        Raised if ``ds`` is not an instance of
        :py:class:`pysegcnn.core.split.RandomSubset` or
        :py:class:`pysegcnn.core.split.SceneSubset`.

    Returns
    -------
    output : `dict`
        Output dictionary with keys:
            ``'input'``
                Model input data
            ``'labels'``
                The ground truth
            ``'prediction'``
                Model prediction
    conf_mat : :py:class:`numpy.ndarray`
        The confusion matrix. Note that the confusion matrix ``conf_mat`` is
        only computed if ``cm=True``.

    """

    # set the model to evaluation mode
    LOGGER.info('Setting model to evaluation mode ...')
    model.eval()
    model.to(device)

    # base filename for each sample
    fname = model.state_file.stem

    # create the dataloader
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    # iterate over the samples and plot inputs, ground truth and
    # model predictions
    output = {}
    LOGGER.info('Predicting samples of the {} dataset ...'.format(ds.name))
    for batch, (inputs, labels) in enumerate(dataloader):

        # send inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # compute model predictions
        with torch.no_grad():
            prd = F.softmax(model(inputs), dim=1).argmax(dim=1).squeeze()

        # map model labels to dataset labels
        if label_map is not None:
            for k, v in label_map.items():
                prd[torch.where(prd == k)] = v

        # store output for current batch
        output[batch] = {'input': inputs, 'labels': labels, 'prediction': prd}

        LOGGER.info('Sample: {:d}/{:d}, Accuracy: {:.2f}'.format(
            batch + 1, len(dataloader), accuracy_function(prd, labels)))

        # update confusion matrix
        if cm:
            for ytrue, ypred in zip(labels.view(-1), prd.view(-1)):
                conf_mat[ytrue.long(), ypred.long()] += 1

        # save plot of current batch to disk
        if plot:

            # plot inputs, ground truth and model predictions
            sname = fname + '_{}_{}.pt'.format(ds.name, batch)

            if isinstance(label_map, enum.Enum):
                _ = plot_sample(inputs.numpy().squeeze().clip(0, 1),
                                ds.dataset.use_bands,
                                labels,
                                y=prd.numpy(),
                                state=sname,
                                **kwargs)
            else:
                _ = plot_sample(inputs.numpy().squeeze().clip(0, 1),
                                ds.dataset.use_bands,
                                labels,
                                y=labels.numpy().squeeze(),
                                y_pred={'SegNet': prd.numpy()},
                                state=sname,
                                **kwargs)

    return output, conf_mat
