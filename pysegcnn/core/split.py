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

# valid split modes
VALID_SPLIT_MODES = ['random', 'scene', 'date']


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

    # check if the splits are disjoint
    assert pairwise_disjoint([s.keys() for s in subsets.values()])

    return subsets


def pairwise_disjoint(sets):
    union = set().union(*sets)
    n = sum(len(u) for u in sets)
    return n == len(union)


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

    def __init__(self, ds, indices, name, scenes, scene_ids):
        super().__init__(dataset=ds, indices=indices)

        # the name of the subset
        self.name = name

        # the scene in the subset
        self.scenes = scenes


class Split(object):

    def __init__(self, ds):

        # the dataset to split
        self.ds = ds

    def split(self):

        # build the subsets
        ds_split = []
        for name, sub in self.subsets().items():

            # the scene identifiers of the current subset
            ids = np.unique([s['id'] for s in sub.values()])

            # build the subset
            sbst = self.subset_type()(self.ds, list(sub.keys()), name,
                                      list(sub.values()), ids)
            ds_split.append(sbst)

        return ds_split

    @property
    def subsets(self):
        raise NotImplementedError

    def subset_type(self):
        raise NotImplementedError

    def __repr__(self):

        # representation string to print
        fs = self.__class__.__name__ + '(\n    '

        # dataset split
        fs += '\n    '.join(
            '- {}: {:d} batches ({:.2f}%)'
            .format(k, len(v), len(v) * 100 / len(self.ds))
                    for k, v in self.subsets().items())
        fs += '\n)'
        return fs

class DateSplit(Split):

    def __init__(self, ds, date, dateformat):
        super().__init__(ds)

        # the date to split the dataset
        # before: training set
        # after : validation set
        self.date = date

        # the format of the date
        self.dateformat = dateformat

    def subsets(self):
        return date_scene_split(self.ds, self.date, self.dateformat)

    def subset_type(self):
        return SceneSubset


class RandomSplit(Split):

    def __init__(self, ds, ttratio, tvratio, seed):
        super().__init__(ds)

        # the training, validation and test set ratios
        self.ttratio = ttratio
        self.tvratio = tvratio

        # the random seed: useful for reproducibility
        self.seed = seed


class RandomTileSplit(RandomSplit):

    def __init__(self, ds, ttratio, tvratio, seed):
        super().__init__(ds, ttratio, tvratio, seed)

    def subsets(self):
        return random_tile_split(self.ds, self.tvratio, self.ttratio,
                                 self.seed)

    def subset_type(self):
        return RandomSubset


class RandomSceneSplit(RandomSplit):

    def __init__(self, ds, ttratio, tvratio, seed):
        super().__init__(ds, ttratio, tvratio, seed)

    def subsets(self):
        return random_scene_split(self.ds, self.tvratio, self.ttratio,
                                  self.seed)

    def subset_type(self):
        return SceneSubset
