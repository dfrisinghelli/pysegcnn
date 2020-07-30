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

# the names of the subsets
SUBSET_NAMES = ['train', 'valid', 'test']


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
    train_ind = trav_indices[:train_len]

    # length of the validation dataset
    # number of samples: (ttratio * (1- tvratio) * len(ds))
    valid_ind = trav_indices[train_len:]

    # length of the test dataset
    # number of samples: ((1 - ttratio) * len(ds))
    test_ind = indices[trav_len:]

    # get the tiles of the scenes of each dataset
    subsets = {}
    for name, dataset in enumerate([train_ind, valid_ind, test_ind]):

        # store the indices and corresponding tiles of the current subset to
        # dictionary
        subsets[SUBSET_NAMES[name]] = {k: ds.scenes[k] for k in dataset}

    # check if the splits are disjoint
    assert pairwise_disjoint([s.keys() for s in subsets.values()])

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
    subsets = {}
    for name, dataset in enumerate([train_scenes, valid_scenes, test_scenes]):

        # store the indices and corresponding tiles of the current subset to
        # dictionary
        subsets[SUBSET_NAMES[name]] = {k: v for k, v in enumerate(ds.scenes)
                                       if v['id'] in dataset}

    # check if the splits are disjoint
    assert pairwise_disjoint([s.keys() for s in subsets.values()])

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
    subsets = {}
    for name, scenes in enumerate([train_scenes, valid_scenes, test_scenes]):

        # store the indices and corresponding tiles of the current subset to
        # dictionary
        subsets[SUBSET_NAMES[name]] = scenes
        # sbst.ids = np.unique([s['id'] for s in scenes.values()])

    # check if the splits are disjoint
    assert pairwise_disjoint([s.keys() for s in subsets.values()])

    return subsets


def pairwise_disjoint(sets):
    union = set().union(*sets)
    n = sum(len(u) for u in sets)
    return n == len(union)


class Split(object):

    # the valid modes
    valid_modes = ['random', 'scene', 'date']

    def __init__(self, ds, mode, **kwargs):

        # check which mode is provided
        if mode not in self.valid_modes:
            raise ValueError('{} is not supported. Valid modes are {}, see '
                             'pysegcnn.main.config.py for a description of '
                             'each mode.'.format(mode, self.valid_modes))
        self.mode = mode

        # the dataset to split
        self.ds = ds

        # the keyword arguments
        self.kwargs = kwargs

        # initialize split
        self._init_split()

    def _init_split(self):

        if self.mode == 'random':
            self.subset = RandomSubset
            self.split_function = random_tile_split
            self.allowed_kwargs = ['tvratio', 'ttratio', 'seed']

        if self.mode == 'scene':
            self.subset = SceneSubset
            self.split_function = random_scene_split
            self.allowed_kwargs = ['tvratio', 'ttratio', 'seed']

        if self.mode == 'date':
            self.subset = SceneSubset
            self.split_function = date_scene_split
            self.allowed_kwargs = ['date', 'dateformat']

        self._check_kwargs()

    def _check_kwargs(self):

        # check if the correct keyword arguments are provided
        if not set(self.allowed_kwargs).issubset(self.kwargs.keys()):
            raise TypeError('__init__() expecting keyword arguments: {}.'
                            .format(', '.join(kwa for kwa in
                                              self.allowed_kwargs)))
        # select the correct keyword arguments
        self.kwargs = {k: self.kwargs[k] for k in self.allowed_kwargs}

    # function apply the split
    def split(self):

        # create the subsets
        subsets = self.split_function(self.ds, **self.kwargs)

        # build the subsets
        ds_split = []
        for name, sub in subsets.items():

            # the scene identifiers of the current subset
            ids = np.unique([s['id'] for s in sub.values()])

            # build the subset
            subset = self.subset(self.ds, list(sub.keys()), name,
                                 list(sub.values()), ids)
            ds_split.append(subset)

        return ds_split


class SceneSubset(Subset):

    def __init__(self, ds, indices, name, scenes, scene_ids):
        super().__init__(dataset=ds, indices=indices)

        # the name of the subset
        self.name = name

        # the scene in the subset
        self.scenes = scenes

        # the names of the scenes
        self.ids = scene_ids


class RandomSubset(Subset):

    def __init__(self, ds, indices, name, scenes):
        super().__init__(dataset=ds, indices=indices)

        # the name of the subset
        self.name = name

        # the scene in the subset
        self.scenes = scenes
