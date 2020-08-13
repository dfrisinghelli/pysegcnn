# builtins
import os
import pathlib

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


def get_scene_tiles(ds, scene_id):

    # iterate over the scenes of the dataset
    indices = []
    for i, scene in enumerate(ds.scenes):
        # if the scene id matches a given id, save the index of the scene
        if scene['id'] == scene_id:
            indices.append(i)

    return indices


def predict_samples(ds, model, optimizer, state_file, cm=False,
                    plot=False, **kwargs):

    # check whether the dataset is a valid subset, i.e.
    # an instance of pysegcnn.core.split.SceneSubset or
    # an instance of pysegcnn.core.split.RandomSubset
    _name = type(ds).__name__
    if not isinstance(ds, RandomSubset) or not isinstance(ds, SceneSubset):
        raise TypeError('ds should be an instance of {} or of {}.'
                        .format(repr(RandomSubset), repr(SceneSubset)))

    # convert state file to pathlib.Path object
    state_file = pathlib.Path(state_file)

    # the device to compute on, use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the pretrained model state
    if not state_file.exists():
        raise FileNotFoundError('{} does not exist.'.format(state_file))
    _ = model.load(state_file.name, optimizer, state_file.parent)

    # set the model to evaluation mode
    print('Setting model to evaluation mode ...')
    model.eval()
    model.to(device)

    # base filename for each sample
    fname = state_file.name.split('.pt')[0]

    # initialize confusion matrix
    cmm = np.zeros(shape=(model.nclasses, model.nclasses))

    # create the dataloader
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    # iterate over the samples and plot inputs, ground truth and
    # model predictions
    output = {}
    print('Predicting samples of the {} dataset ...'.format(ds.name))
    for batch, (inputs, labels) in enumerate(dataloader):

        # send inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # compute model predictions
        with torch.no_grad():
            prd = F.softmax(model(inputs), dim=1).argmax(dim=1).squeeze()

        # store output for current batch
        output[batch] = {'input': inputs, 'labels': labels, 'prediction': prd}

        print('Sample: {:d}/{:d}, Accuracy: {:.2f}'
              .format(batch + 1, len(dataloader),
                      accuracy_function(prd, labels)))

        # update confusion matrix
        if cm:
            for ytrue, ypred in zip(labels.view(-1), prd.view(-1)):
                cmm[ytrue.long(), ypred.long()] += 1

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

    return output, cmm


def predict_scenes(ds, model, optimizer, state_file,
                   scene_id=None, cm=False, plot_scenes=False, **kwargs):

    # check whether the dataset is a valid subset, i.e. an instance of
    # pysegcnn.core.split.SceneSubset
    if not isinstance(ds, SceneSubset):
        raise TypeError('ds should be an instance of {}.'
                        .format(repr(SceneSubset)))

    # convert state file to pathlib.Path object
    state_file = pathlib.Path(state_file)

    # the device to compute on, use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the pretrained model state
    if not state_file.exists():
        raise FileNotFoundError('{} does not exist.'.format(state_file))
    _ = model.load(state_file.name, optimizer, state_file.parent)

    # set the model to evaluation mode
    print('Setting model to evaluation mode ...')
    model.eval()
    model.to(device)

    # base filename for each scene
    fname = state_file.name.split('.pt')[0]

    # initialize confusion matrix
    cmm = np.zeros(shape=(model.nclasses, model.nclasses))

    # check whether a scene id is provided
    if scene_id is None:
        scene_ids = ds.ids
    else:
        # the name of the selected scene
        scene_ids = [scene_id]

    # spatial size of scene
    scene_size = (ds.dataset.height, ds.dataset.width)

    # iterate over the scenes
    print('Predicting scenes of the {} dataset ...'.format(ds.name))
    scenes = {}
    for i, sid in enumerate(scene_ids):

        # filename for the current scene
        sname = fname + '_{}_{}.pt'.format(ds.name, sid)

        # get the indices of the tiles of the scene
        indices = get_scene_tiles(ds, sid)
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
                    cmm[ytrue.long(), ypred.long()] += 1

        # reconstruct the entire scene
        inputs = reconstruct_scene(inp, scene_size, nbands=inp.shape[1])
        labels = reconstruct_scene(lab, scene_size, nbands=1)
        prdtcn = reconstruct_scene(prd, scene_size, nbands=1)

        # print progress
        print('Scene {:d}/{:d}, Id: {}, Accuracy: {:.2f}'.format(
            i + 1, len(scene_ids), sid, accuracy_function(prdtcn, labels)))

        # save outputs to dictionary
        scenes[sid] = {'input': inputs, 'labels': labels, 'prediction': prdtcn}

        # plot current scene
        if plot_scenes:
            fig, ax = plot_sample(inputs.clip(0, 1),
                                  labels,
                                  ds.dataset.use_bands,
                                  ds.dataset.labels,
                                  y_pred=prdtcn,
                                  state=sname,
                                  **kwargs)

    return scenes, cmm
