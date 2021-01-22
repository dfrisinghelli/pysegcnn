"""Split the dataset into training, validation and test set.

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

# externals
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset

# the names of the subsets
SUBSET_NAMES = ['train', 'valid', 'test']


def _ds_len(ds, ratio):
    """Calcute number of samples in a dataset given a ratio.

    Parameters
    ----------
    ds : :py:class:`collections.Sized`
        An object with a :py:meth:__len__ method.
    ratio : `float`
        A ratio to multiply with the length of ``ds``.

    Returns
    -------
    n_samples: `int`
        Length of ``ds * ratio``.

    """
    return int(np.round(len(ds) * ratio))


def pairwise_disjoint(sets):
    """Check if ``sets`` are pairwise disjoint.

    Sets are pairwise disjoint if the length of their union equals the sum of
    their lengths.

    Parameters
    ----------
    sets : `list` [:py:class:`collections.Sized`]
        A list of sized objects.

    Returns
    -------
    disjoint : `bool`
        Whether the sets are pairwise disjoint.

    """
    union = set().union(*sets)
    n = sum(len(u) for u in sets)
    return n == len(union)


def index_dict(indices):
    """Generate the training, validation and test set index dictionary.

    Parameters
    ----------
    indices : `list` [:py:class:`numpy.ndarray`]
        An ordered list composed of three :py:class:`numpy.ndarray` containing
        the indices to the training, validation and test set.

    Returns
    -------
    index_dict : `dict`
        The index dictionary, where the keys are equal to ``SUBSET_NAMES`` and
        the values are py:class:`numpy.ndarray` containing the indices to the
        training, validation and test set.

    """
    return {k: v for k, v in zip(SUBSET_NAMES, indices)}


def random_split(ds, tvratio=0.8, ttratio=1, seed=0, shuffle=True):
    """Randomly split an iterable into training, validation and test set.

    The parameters ``ttratio`` and ``tvratio`` control the size of the
    training, validation and test datasets.

    Test dataset size      : ``(1 - ttratio) * len(ds)``
    Train dataset size     : ``ttratio * tvratio * len(ds)``
    Validation dataset size: ``ttratio * (1 - tvratio) * len(ds)``

    Parameters
    ----------
    ds : :py:class:`collections.Sized`
        An object with a :py:meth:`__len__` method.
    tvratio : `float`, optional
        The ratio of training data to validation data, e.g. ``tvratio=0.8``
        means 80% training, 20% validation. The default is `0.8`.
    ttratio : `float`, optional
        The ratio of training and validation data to test data, e.g.
        ``ttratio=0.6`` means 60% for training and validation, 40% for
        testing. The default is `1`.
    seed : `int`, optional
        The random seed for reproducibility. The default is `0`.
    shuffle : `bool`, optional
        Whether to shuffle the data before splitting into batches. The default
        is `True`.

    Raises
    ------
    AssertionError
        Raised if the splits are not pairwise disjoint.

    Returns
    -------
    indices : `list` [`dict`]
        List of index dictionaries as composed by
        :py:func:`pysegcnn.core.split.index_dict`.

    """
    # set the random seed for reproducibility
    np.random.seed(seed)

    # whether to shuffle the data before splitting
    indices = np.arange(len(ds))
    if shuffle:
        # randomly permute indices to access the iterable
        indices = np.random.permutation(indices)

    # the training and validation scenes
    # number of samples: (ttratio * len(ds))
    trav_len = _ds_len(ds, ttratio)
    trav_ids = indices[:trav_len]

    # the training dataset indices
    # number of samples: (ttratio * tvratio * len(ds))
    train_len = _ds_len(trav_ids, tvratio)
    train_ids = trav_ids[:train_len]

    # the validation dataset indices
    # number of samples: (ttratio * (1- tvratio) * len(ds))
    valid_ids = trav_ids[train_len:]

    # the test dataset indices
    # number of samples:((1 - ttratio) * len(ds))
    test_ids = trav_ids[trav_len:]

    # check whether the different datasets or pairwise disjoint
    indices = index_dict([train_ids, valid_ids, test_ids])
    assert pairwise_disjoint(indices.values())

    return [indices]


def kfold_split(ds, k_folds=5, seed=0, shuffle=True):
    """Randomly split an iterable into ``k_folds`` folds.

    This function uses the cross validation index generator
    :py:class:`sklearn.model_selection.KFold`.

    Parameters
    ----------
    ds : :py:class:`collections.Sized`
        An object with a :py:meth:`__len__` method.
    k_folds: `int`, optional
        The number of folds. Must be a least 2. The default is `5`.
    seed : `int`, optional
        The random seed for reproducibility. The default is `0`.
    shuffle : `bool`, optional
        Whether to shuffle the data before splitting into batches. The default
        is `True`.

    Raises
    ------
    AssertionError
        Raised if the (training, validation) folds are not pairwise disjoint.

    """
    # set the random seed for reproducibility
    np.random.seed(seed)

    # cross validation index generator from scikit-learn
    kf = KFold(k_folds, random_state=seed, shuffle=shuffle)

    # generate the indices of the different folds
    folds = []
    for i, (train, valid) in enumerate(kf.split(ds)):
        folds.append(index_dict([train, valid, np.array([])]))
        assert pairwise_disjoint(folds[i].values())
    return folds


class RandomSplit(object):
    """Base class for random splits of a `torch.utils.data.Dataset`."""

    def __init__(self, ds, k_folds, seed=0, shuffle=True, tvratio=0.8,
                 ttratio=1):
        """Randomly split a dataset into training, validation and test set.

        Parameters
        ----------
        ds : :py:class:`collections.Sized`
            An object with a :py:meth:`__len__` method.
        k_folds: `int`
            The number of folds.
        seed : `int`, optional
            The random seed for reproducibility. The default is `0`.
        shuffle : `bool`, optional
            Whether to shuffle the data before splitting into batches. The
            default is `True`.
        tvratio : `float`, optional
            The ratio of training data to validation data, e.g. ``tvratio=0.8``
            means 80% training, 20% validation. The default is `0.8`. Used if
            ``k_folds=1``.
        ttratio : `float`, optional
            The ratio of training and validation data to test data, e.g.
            ``ttratio=0.6`` means 60% for training and validation, 40% for
            testing. The default is `1`. Used if ``k_folds=1``.

        """

        # instance attributes
        self.ds = ds
        self.k_folds = k_folds
        self.seed = seed
        self.shuffle = shuffle

        # instance attributes: training/validation/test split ratios
        # used if kfolds=1
        self.tvratio = tvratio
        self.ttratio = ttratio

    def generate_splits(self):

        # check whether to generate a single or multiple folds
        if self.k_folds > 1:
            # k-fold split
            indices = kfold_split(
                self.indices_to_split, self.k_folds, self.seed, self.shuffle)
        else:
            # single-fold split
            indices = random_split(
                self.indices_to_split, self.tvratio, self.ttratio, self.seed,
                self.shuffle)

        return indices

    @property
    def indices_to_split(self):
        raise NotImplementedError

    @property
    def indices(self):
        raise NotImplementedError

    def split(self):

        # initialize training, validation and test subsets
        subsets = []

        # the training, validation and test indices
        for folds in self.indices:
            subsets.append(
                index_dict([Subset(self.ds, ids) for ids in folds.values()]))

        return subsets


class RandomTileSplit(RandomSplit):
    """Split a :py:class:`pysegcnn.core.dataset.ImageDataset` into tiles."""

    def __init__(self, ds, k_folds, seed=0, shuffle=True, tvratio=0.8,
                 ttratio=1):
        # initialize super class
        super().__init__(ds, k_folds, seed, shuffle, tvratio, ttratio)

    @property
    def indices_to_split(self):
        return np.arange(len(self.ds))

    @property
    def indices(self):
        return self.generate_splits()


class RandomSceneSplit(RandomSplit):
    """Split a :py:class:`pysegcnn.core.dataset.ImageDataset` into scenes."""

    def __init__(self, ds, k_folds, seed=0, shuffle=True, tvratio=0.8,
                 ttratio=1):
        # initialize super class
        super().__init__(ds, k_folds, seed, shuffle, tvratio, ttratio)

        # the number of the scenes in the dataset
        self.scenes = np.array([v['scene'] for v in self.ds.scenes])

    @property
    def indices_to_split(self):
        return np.unique(self.scenes)

    @property
    def indices(self):
        # indices of the different scene identifiers
        indices = self.generate_splits()

        # iterate over the different folds
        scene_indices = []
        for folds in indices:
            # iterate over the training, validation and test set
            subset = {}
            for name, ids in folds.items():
                subset[name] = np.where(np.isin(self.scenes, ids))[0]
            scene_indices.append(subset)

        return scene_indices


class SupportedSplits(enum.Enum):
    """Names and corresponding classes of the implemented split modes."""

    tile = RandomTileSplit
    scene = RandomSceneSplit
