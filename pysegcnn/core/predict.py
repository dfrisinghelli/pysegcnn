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


def _get_scene_tiles(ds, scene_id):
    """Return the tiles of the scene with id ``scene_id``.

    Parameters
    ----------
    ds : :py:class:`pysegcnn.core.split.CustomSubset`
        A instance of a subclass of
        :py:class:`pysegcnn.core.split.CustomSubset`.
    scene_id : `str`
        A valid scene identifier.

    Raises
    ------
    ValueError
        Raised if ``scene_id`` is not a valid scene identifier for the dataset
        ``ds``.

    Returns
    -------
    indices : `list` [`int`]
        List of indices of the tiles of the scene with id ``scene_id`` in
        ``ds``.
    date : :py:class:`datetime.datetime`
        The date of the scene with id ``scene_id``.

    """
    # check if the scene id is valid
    scene_meta = ds.dataset.parse_scene_id(scene_id)
    if scene_meta is None:
        raise ValueError('{} is not a valid scene identifier'.format(scene_id))

    # iterate over the scenes of the dataset
    indices = []
    for i, scene in enumerate(ds.scenes):
        # if the scene id matches a given id, save the index of the scene
        if scene['id'] == scene_id:
            indices.append(i)

    return indices, scene_meta['date']


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
    label_map : `dict` [`int`, `int`] or `None`
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
    # check whether the dataset is a valid subset, i.e.
    # an instance of pysegcnn.core.split.SceneSubset or
    # an instance of pysegcnn.core.split.RandomSubset
    if not (isinstance(ds, RandomSubset) or isinstance(ds, SceneSubset)):
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
    if label_map is None:
        conf_mat = np.zeros(shape=2 * (model.nclasses,))
    else:
        conf_mat = np.zeros(shape=2 * (len(ds.dataset.labels),))

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
            _ = plot_sample(inputs.numpy().squeeze().clip(0, 1),
                            ds.dataset.use_bands,
                            ds.dataset.labels,
                            y=labels.numpy().squeeze(),
                            y_pred=prd.numpy(),
                            state=sname,
                            **kwargs)

    return output, conf_mat


def predict_scenes(ds, model, label_map=None, scene_id=None, cm=False,
                   plot=False, animate=False, anim_path=None, **kwargs):
    """Classify each scene of the dataset ``ds`` with model ``model``.

    Parameters
    ----------
    ds : :py:class:`pysegcnn.core.split.SceneSubset`
        An instance of :py:class:`pysegcnn.core.split.SceneSubset`.
    model : :py:class:`pysegcnn.core.models.Network`
        An instance of :py:class:`pysegcnn.core.models.Network`.
    label_map : `dict` [`int`, `int`] or `None`
        Dictionary with labels of ``ds`` as keys and corresponding labels of
        the ``model`` as values. If specified, ``label_map`` is used to map the
        model label predictions to the actual labels of the dataset ``ds``. The
        default is `None`, i.e. ``model`` and ``ds`` share the same labels.
    scene_id : `str` or `None`
        A valid scene identifier.
    cm : `bool`, optional
        Whether to compute the confusion matrix. The default is `False`.
    plot : `bool`, optional
        Whether to plot a false color composite, ground truth and model
        prediction for each scene. The default is `False`.
    animate : `bool`, optional
        Whether to create an animation of (input, ground truth, prediction) for
        the scenes of the dataset ``ds``. The default is `False`.
    anim_path : `str` or :py:class:`pathlib.Path` or `None`, optional
        Path to save animations. The default is `None`.
    **kwargs
        Additional keyword arguments passed to
        :py:func:`pysegcnn.core.graphics.plot_sample`.

    Raises
    ------
    TypeError
        Raised if ``ds`` is not an instance of
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
    # check whether the dataset is a valid subset, i.e. an instance of
    # pysegcnn.core.split.SceneSubset
    # if not isinstance(ds, SceneSubset):
    #     raise TypeError('ds should be an instance of {}.'
    #                     .format(repr(SceneSubset)))

    # the device to compute on, use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set the model to evaluation mode
    LOGGER.info('Setting model to evaluation mode ...')
    model.eval()
    model.to(device)

    # base filename for each scene
    fname = model.state_file.stem

    # initialize confusion matrix
    if label_map is None:
        conf_mat = np.zeros(shape=2 * (model.nclasses,))
    else:
        conf_mat = np.zeros(shape=2 * (len(ds.dataset.labels),))

    # check whether a scene id is provided
    if scene_id is None:
        scene_ids = ds.ids
    else:
        # the name of the selected scene
        scene_ids = [scene_id]

    # instanciate figure
    fig, _ = plt.subplots(1, 3, figsize=kwargs['figsize'])

    # check whether to animate figures
    if animate:
        # check whether the output path is valid
        anim_path = pathlib.Path(anim_path)
        if not anim_path.exists():
            # create output path
            anim_path.mkdir(parents=True, exist_ok=True)
        anim = Animate(anim_path)

    # iterate over the scenes
    LOGGER.info('Predicting scenes of the {} dataset ...'.format(ds.name))
    output = {}
    for i, sid in enumerate(scene_ids):

        # filename for the current scene
        sname = fname + '_{}_{}.pt'.format(ds.name, sid)

        # get the indices of the tiles of the scene
        indices, date = _get_scene_tiles(ds, sid)
        indices.sort()

        # check whether the dataset is a time series
        date = date if ds.split_mode == 'date' else None

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

            # map model labels to dataset labels
            if label_map is not None:
                for k, v in label_map.items():
                    prd[torch.where(prd == k)] = v

            # update confusion matrix
            if cm:
                for ytrue, ypred in zip(lab.view(-1), prd.view(-1)):
                    conf_mat[ytrue.long(), ypred.long()] += 1

        # reconstruct the entire scene
        inputs = reconstruct_scene(inp)
        labels = reconstruct_scene(lab)
        prdctn = reconstruct_scene(prd)

        # print progress
        LOGGER.info('Scene {:d}/{:d}, Id: {}, Accuracy: {:.2f}'.format(
            i + 1, len(scene_ids), sid, accuracy_function(prdctn, labels)))

        # save outputs to dictionary
        output[sid] = {'input': inputs, 'labels': labels, 'prediction': prdctn}

        # plot current scene
        if plot:
            # plot inputs, ground truth and model predictions
            _ = plot_sample(inputs.clip(0, 1),
                            ds.dataset.use_bands,
                            ds.dataset.labels,
                            y=labels,
                            y_pred=prdctn,
                            date=date,
                            state=sname,
                            fig=fig,
                            **kwargs)
            # save current figure state as frame for animation
            if animate:
                anim.frame(fig.axes)

    # save animation
    if animate:
        anim.animate(fig, interval=1000, repeat=True, blit=True)
        anim.save(fname + '_{}.gif'.format(ds.name), dpi=200)

    return output, conf_mat
