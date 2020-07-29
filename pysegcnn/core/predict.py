# builtins
import os

# externals
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torch.nn.functional as F

# locals
from pysegcnn.core.utils import reconstruct_scene
from pysegcnn.core.graphics import plot_sample


def get_scene_tiles(ds, scene_id):

    # iterate over the scenes of the dataset
    indices = []
    for i, scene in enumerate(ds.scenes):
        # if the scene id matches a given id, save the index of the scene
        if scene['id'] == scene_id:
            indices.append(i)

    return indices


def predict_samples(ds, model, optimizer, state_path, state_file, nsamples,
                    seed, batch_size=None, cm=False, plot_samples=False,
                    **kwargs):

    # the device to compute on, use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the pretrained model state
    state = os.path.join(state_path, state_file)
    if not os.path.exists(state):
        raise FileNotFoundError('{} does not exist.'.format(state))
    state = model.load(state_file, optimizer, state_path)

    # set the model to evaluation mode
    print('Setting model to evaluation mode ...')
    model.eval()
    model.to(device)

    # base filename for each sample
    fname = state_file.split('.pt')[0]

    # initialize confusion matrix
    cmm = np.zeros(shape=(model.nclasses, model.nclasses))

    # set random seed for reproducibility
    np.random.seed(seed)

    # draw a number of samples from the dataset
    samples = np.arange(0, len(ds))
    if nsamples > 0:
        batch_size = nsamples
        samples = np.random.choice(samples, size=min(nsamples, len(ds)))

    # create a subset of the dataset
    smpl_subset = Subset(ds, samples.tolist())
    if batch_size is None:
        raise ValueError('If you specify "nsamples"=-1, you have to provide '
                         'a batch size, e.g. trainer.batch_size.')
    smpl_loader = DataLoader(smpl_subset, batch_size=batch_size, shuffle=False)

    # iterate over the samples and plot inputs, ground truth and
    # model predictions
    output = {}
    for batch, (inputs, labels) in enumerate(smpl_loader):

        # send inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # compute model predictions
        with torch.no_grad():
            prd = F.softmax(model(inputs), dim=1).argmax(dim=1).squeeze()

        # store output for current batch
        output[batch] = {'input': inputs, 'labels': labels, 'prediction': prd}

        # update confusion matrix
        if cm:
            for ytrue, ypred in zip(labels.view(-1), prd.view(-1)):
                cmm[ytrue.long(), ypred.long()] += 1

        # save plot of current batch to disk
        if plot_samples:

            # check whether the dataset is a subset
            if isinstance(ds, Subset):
                use_bands = ds.dataset.use_bands
                ds_labels = ds.dataset.labels
            else:
                use_bands = ds.use_bands
                ds_labels = ds.labels

            # plot inputs, ground truth and model predictions
            sname = fname + '_sample_{}.pt'.format(batch)
            fig, ax = plot_sample(inputs.numpy().clip(0, 1),
                                  labels,
                                  use_bands,
                                  ds_labels,
                                  y_pred=prd,
                                  state=sname,
                                  **kwargs)

    return output, cmm


def predict_scenes(ds, model, optimizer, state_path, state_file,
                   scene_id=None, cm=False, plot_scenes=False, **kwargs):

    # the device to compute on, use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the pretrained model state
    state = os.path.join(state_path, state_file)
    if not os.path.exists(state):
        raise FileNotFoundError('{} does not exist.'.format(state))
    state = model.load(state_file, optimizer, state_path)

    # set the model to evaluation mode
    print('Setting model to evaluation mode ...')
    model.eval()
    model.to(device)

    # base filename for each scene
    fname = state_file.split('.pt')[0]

    # initialize confusion matrix
    cmm = np.zeros(shape=(model.nclasses, model.nclasses))

    # check whether a scene id is provided
    if scene_id is None:

        # check if the dataset is an instance of torch.data.dataset.Subset
        if not isinstance(ds, Subset):
            raise TypeError('ds should be of type {}'.format(Subset))

        print('Predicting scenes of the subset ...')

        # get the names of the scenes
        try:
            scene_ids = ds.ids
        except AttributeError:
            raise TypeError('predict_scenes does only work for datasets split '
                            'by "scene" or by "date".')

    else:

        # the name of the selected scene
        scene_ids = [scene_id]

    # spatial size of scene
    scene_size = (ds.dataset.height, ds.dataset.width)

    # iterate over the scenes
    scene = {}
    for sid in scene_ids:

        # filename for the current scene
        sname = fname + '_' + sid + '.pt'

        # get the indices of the tiles of the scene
        indices = get_scene_tiles(ds, sid)
        indices.sort()

        # create a subset of the dataset
        scene_ds = Subset(ds, indices)

        # create the dataloader
        scene_dl = DataLoader(scene_ds, batch_size=len(scene_ds),
                              shuffle=False)

        # predict the current scene
        for i, (inp, lab) in enumerate(scene_dl):
            print('Predicting scene ({}/{}), id: {}'.format(i + 1,
                                                            len(scene_ids),
                                                            sid))

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

        # save outputs to dictionary
        scene[sid] = {'input': inputs, 'labels': labels, 'prediction': prdtcn}

        # plot current scene
        if plot_scenes:
            fig, ax = plot_sample(inputs.clip(0, 1),
                                  labels,
                                  ds.dataset.use_bands,
                                  ds.dataset.labels,
                                  y_pred=prdtcn,
                                  state=sname,
                                  **kwargs)

    return scene, cmm
