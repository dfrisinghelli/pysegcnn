"""A collection of functions for model inference."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import logging

# externals
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torch.nn.functional as F

# locals
from pysegcnn.core.utils import reconstruct_scene, accuracy_function
from pysegcnn.core.graphics import plot_sample
from pysegcnn.core.split import RandomSubset, SceneSubset

# module level logger
LOGGER = logging.getLogger(__name__)


def _get_scene_tiles(ds, scene_id):
    """Return the tiles of the scene with id = ``scene_id``.

    Parameters
    ----------
    ds : `pysegcnn.core.dataset.ImageDataset`
        An instance of `~pysegcnn.core.dataset.ImageDataset`.
    scene_id : `str`
        A valid scene identifier.

    Returns
    -------
    indices : `list` [`int`]
        List of indices of the tiles from scene with id ``scene_id`` in ``ds``.

    """
    # iterate over the scenes of the dataset
    indices = []
    for i, scene in enumerate(ds.scenes):
        # if the scene id matches a given id, save the index of the scene
        if scene['id'] == scene_id:
            indices.append(i)

    return indices


def predict_samples(ds, model, cm=False, plot=False, **kwargs):
    """Classify each sample in ``ds`` with model ``model``.

    Parameters
    ----------
    ds : `pysegcnn.core.split.RandomSubset` or
    `pysegcnn.core.split.SceneSubset`
        An instance of `~pysegcnn.core.split.RandomSubset` or
        `~pysegcnn.core.split.SceneSubset`.
    model : `pysegcnn.core.models.Network`
        An instance of `~pysegcnn.core.models.Network`.
    cm : `bool`, optional
        Whether to compute the confusion matrix. The default is False.
    plot : `bool`, optional
        Whether to plot a false color composite, ground truth and model
        prediction for each sample. The default is False.
    **kwargs
        Additional keyword arguments passed to
        `pysegcnn.core.graphics.plot_sample`.

    Raises
    ------
    TypeError
        Raised if ``ds`` is not an instance of
        `~pysegcnn.core.split.RandomSubset` or
        `~pysegcnn.core.split.SceneSubset`.

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
    conf_mat : `numpy.ndarray`
        The confusion matrix. Note that the confusion matrix ``conf_mat`` is
        only computed if ``cm`` = True.
    """
    # check whether the dataset is a valid subset, i.e.
    # an instance of pysegcnn.core.split.SceneSubset or
    # an instance of pysegcnn.core.split.RandomSubset
    if not isinstance(ds, RandomSubset) or not isinstance(ds, SceneSubset):
        raise TypeError('ds should be an instance of {} or of {}.'
                        .format(repr(RandomSubset), repr(SceneSubset)))

    # the device to compute on, use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set the model to evaluation mode
    LOGGER.info('Setting model to evaluation mode ...')
    model.eval()
    model.to(device)

    # base filename for each sample
    fname = model.state_file.name.split('.pt')[0]

    # initialize confusion matrix
    conf_mat = np.zeros(shape=(model.nclasses, model.nclasses))

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
            fig, ax = plot_sample(inputs.numpy().clip(0, 1),
                                  labels,
                                  ds.dataset.use_bands,
                                  ds.dataset.labels,
                                  y_pred=prd,
                                  state=sname,
                                  **kwargs)

    return output, conf_mat


def predict_scenes(ds, model, scene_id=None, cm=False, plot=False, **kwargs):
    """Classify each scene in ``ds`` with model ``model``.

    Parameters
    ----------
    ds : `pysegcnn.core.split.SceneSubset`
        An instance of `~pysegcnn.core.split.SceneSubset`.
    model : `pysegcnn.core.models.Network`
        An instance of `~pysegcnn.core.models.Network`.
    scene_id : `str` or `None`
        A valid scene identifier.
    cm : `bool`, optional
        Whether to compute the confusion matrix. The default is False.
    plot : `bool`, optional
        Whether to plot a false color composite, ground truth and model
        prediction for each scene. The default is False.
    **kwargs
        Additional keyword arguments passed to
        `pysegcnn.core.graphics.plot_sample`.

    Raises
    ------
    TypeError
        Raised if ``ds`` is not an instance of
        `~pysegcnn.core.split.SceneSubset`.

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
    conf_mat : `numpy.ndarray`
        The confusion matrix. Note that the confusion matrix ``conf_mat`` is
        only computed if ``cm`` = True.
    """
    # check whether the dataset is a valid subset, i.e. an instance of
    # pysegcnn.core.split.SceneSubset
    if not isinstance(ds, SceneSubset):
        raise TypeError('ds should be an instance of {}.'
                        .format(repr(SceneSubset)))

    # the device to compute on, use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set the model to evaluation mode
    LOGGER.info('Setting model to evaluation mode ...')
    model.eval()
    model.to(device)

    # base filename for each scene
    fname = model.state_file.name.split('.pt')[0]

    # initialize confusion matrix
    conf_mat = np.zeros(shape=(model.nclasses, model.nclasses))

    # check whether a scene id is provided
    if scene_id is None:
        scene_ids = ds.ids
    else:
        # the name of the selected scene
        scene_ids = [scene_id]

    # spatial size of scene
    scene_size = (ds.dataset.height, ds.dataset.width)

    # iterate over the scenes
    LOGGER.info('Predicting scenes of the {} dataset ...'.format(ds.name))
    output = {}
    for i, sid in enumerate(scene_ids):

        # filename for the current scene
        sname = fname + '_{}_{}.pt'.format(ds.name, sid)

        # get the indices of the tiles of the scene
        indices = _get_scene_tiles(ds, sid)
        indices.sort()

        # create a subset of the dataset
        scene_ds = Subset(ds, indices)

        # create the dataloader
        scene_dl = DataLoader(scene_ds, batch_size=len(scene_ds),
                              shuffle=False, drop_last=False)

        # predict the current scene
        for b, (inp, lab) in enumerate(scene_dl):

            # send inputs and labels to device
            inp = inp.to(device)
            lab = lab.to(device)

            # apply forward pass: model prediction
            with torch.no_grad():
                prd = F.softmax(model(inp), dim=1).argmax(dim=1).squeeze()

            # update confusion matrix
            if cm:
                for ytrue, ypred in zip(lab.view(-1), prd.view(-1)):
                    conf_mat[ytrue.long(), ypred.long()] += 1

        # reconstruct the entire scene
        inputs = reconstruct_scene(inp, scene_size, nbands=inp.shape[1])
        labels = reconstruct_scene(lab, scene_size, nbands=1)
        prdtcn = reconstruct_scene(prd, scene_size, nbands=1)

        # print progress
        LOGGER.info('Scene {:d}/{:d}, Id: {}, Accuracy: {:.2f}'.format(
            i + 1, len(scene_ids), sid, accuracy_function(prdtcn, labels)))

        # save outputs to dictionary
        output[sid] = {'input': inputs, 'labels': labels, 'prediction': prdtcn}

        # plot current scene
        if plot:
            fig, ax = plot_sample(inputs.clip(0, 1),
                                  labels,
                                  ds.dataset.use_bands,
                                  ds.dataset.labels,
                                  y_pred=prdtcn,
                                  state=sname,
                                  **kwargs)

    return output, conf_mat
