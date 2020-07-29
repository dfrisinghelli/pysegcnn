# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:02:32 2020

@author: Daniel
"""
# builtins
import datetime

# externals
import numpy as np
from torch.utils.data.dataset import Subset


# function calculating number of samples in a dataset given a ratio
def _ds_len(ds, ratio):
    return int(np.round(len(ds) * ratio))


# randomly split the tiles of a dataset across the training, validation and
# test dataset
# for each scene, the tiles can be distributed among the training, validation
# and test set
def random_tile_split(ds, tvratio, ttratio=1, seed=0):

    # set the random seed for reproducibility
    np.random.seed(seed)

    # randomly permute indices to access dataset
    indices = np.random.permutation(len(ds))

    # length of the training and validation dataset
    # number of samples: (ttratio * len(ds))
    trav_len = _ds_len(indices, ttratio)
    trav_indices = indices[:trav_len]

    # length of the training dataset
    # number of samples: (ttratio * tvratio * len(ds))
    train_len = _ds_len(trav_indices, tvratio)
    train_indices = trav_indices[:train_len]

    # length of the validation dataset
    # number of samples: (ttratio * (1- tvratio) * len(ds))
    valid_indices = trav_indices[train_len:]

    # length of the test dataset
    # number of samples: ((1 - ttratio) * len(ds))
    test_indices = indices[trav_len:]

    # get the tiles of the scenes of each dataset
    subsets = []
    for dataset in [train_indices, valid_indices, test_indices]:

        # build the subset: store the scenes
        sbst = Subset(dataset=ds, indices=list(dataset))
        sbst.scenes = [ds.scenes[i] for i in dataset]

        # add to list of subsets
        subsets.append(sbst)

    # check if the splits are disjoint
    assert pairwise_disjoint([s.indices for s in subsets])

    return subsets


# randomly split the tiles of a dataset across the training, validation and
# test dataset
# for each scene, all the tiles of the scene are included in either the
# training set, the validation set or the test set, respectively
def random_scene_split(ds, tvratio, ttratio=1, seed=0):

    # set the random seed for reproducibility
    np.random.seed(seed)

    # get the names of the scenes and generate random permutation
    scene_ids = np.random.permutation(np.unique([s['id'] for s in ds.scenes]))

    # the training and validation scenes
    # number of samples: (ttratio * nscenes)
    trav_len = _ds_len(scene_ids, ttratio)
    trav_scenes = scene_ids[:trav_len]

    # the training scenes
    # number of samples: (ttratio * tvratio * nscenes)
    train_len = _ds_len(trav_scenes, tvratio)
    train_scenes = trav_scenes[:train_len]

    # the validation scenes
    # number of samples: (ttratio * (1- tvratio) * nscenes)
    valid_scenes = trav_scenes[train_len:]

    # the test scenes
    # number of samples:((1 - ttratio) * nscenes)
    test_scenes = scene_ids[trav_len:]

    # get the tiles of the scenes of each dataset
    subsets = []
    for dataset in [train_scenes, valid_scenes, test_scenes]:
        # the indices of the scenes in the dataset
        indices = []
        tiles = []

        # iterate over the scenes of the whole dataset
        for i, scene in enumerate(ds.scenes):
            if scene['id'] in dataset:
                indices.append(i)
                tiles.append(scene)

        # build the subset: store scene ids
        sbst = Subset(dataset=ds, indices=indices)
        sbst.scenes = tiles
        sbst.ids = dataset

        # add to list of subsets
        subsets.append(sbst)

    # check if the splits are disjoint
    assert pairwise_disjoint([s.indices for s in subsets])

    return subsets


# split the scenes of a dataset based on a date, useful for time series data
# scenes before date build the training set, scenes after the date build the
# validation set, the test set is empty
def date_scene_split(ds, date, dateformat='%Y%m%d'):

    # convert date to datetime object
    date = datetime.datetime.strptime(date, dateformat)

    # the training, validation and test scenes
    train_scenes = {i: s for i, s in enumerate(ds.scenes) if s['date'] <= date}
    valid_scenes = {i: s for i, s in enumerate(ds.scenes) if s['date'] > date}
    test_scenes = {}

    # build the training and test datasets
    subsets = []
    for scenes in [train_scenes, valid_scenes, test_scenes]:
        # build the subset: store the scenes
        sbst = Subset(dataset=ds, indices=list(scenes.keys()))
        sbst.scenes = list(scenes.values())
        sbst.ids = np.unique([s['id'] for s in scenes.values()])

        # add to list of subsets
        subsets.append(sbst)

    # check if the splits are disjoint
    assert pairwise_disjoint([s.indices for s in subsets])

    return subsets


def pairwise_disjoint(sets):
    union = set().union(*sets)
    n = sum(len(u) for u in sets)
    return n == len(union)
