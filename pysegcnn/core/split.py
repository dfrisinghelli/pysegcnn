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
import datetime
import enum

# externals
import numpy as np
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


def random_tile_split(ds, tvratio, ttratio=1, seed=0):
    """Randomly split the tiles of a dataset.

    For each scene, the tiles of the scene can be distributed among the
    training, validation and test set.

    The parameters ``ttratio`` and ``tvratio`` control the size of the
    training, validation and test datasets.

    Test dataset size      : ``(1 - ttratio) * len(ds)``
    Train dataset size     : ``ttratio * tvratio * len(ds)``
    Validation dataset size: ``ttratio * (1 - tvratio) * len(ds)``

    Parameters
    ----------
    ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
        An instance of :py:class:`pysegcnn.core.dataset.ImageDataset`.
    tvratio : `float`
        The ratio of training data to validation data, e.g. ``tvratio=0.8``
        means 80% training, 20% validation.
    ttratio : `float`, optional
        The ratio of training and validation data to test data, e.g.
        ``ttratio=0.6`` means 60% for training and validation, 40% for
        testing. The default is `1`.
    seed : `int`, optional
        The random seed for reproducibility. The default is `0`.

    Raises
    ------
    AssertionError
        Raised if the splits are not pairwise disjoint.

    Returns
    -------
    subsets : `dict`
        Subset dictionary with keys:
            ``'train'``
                The training scenes (`dict`).
            ``'valid'``
                The validation scenes (`dict`).
            ``'test'``
                The test scenes (`dict`).

    """
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


def random_scene_split(ds, tvratio, ttratio=1, seed=0):
    """Semi-randomly split the tiles of a dataset.

    For each scene, all the tiles of the scene are included in either the
    training, validation or test set, respectively.

    The parameters ``ttratio`` and ``tvratio`` control the size of the
    training, validation and test datasets.

    Test dataset size      : ``(1 - ttratio) * len(ds)``
    Train dataset size     : ``ttratio * tvratio * len(ds)``
    Validation dataset size: ``ttratio * (1 - tvratio) * len(ds)``

    Parameters
    ----------
    ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
        An instance of :py:class:`pysegcnn.core.dataset.ImageDataset`.
    tvratio : `float`
        The ratio of training data to validation data, e.g. ``tvratio=0.8``
        means 80% training, 20% validation.
    ttratio : `float`, optional
        The ratio of training and validation data to test data, e.g.
        ``ttratio=0.6`` means 60% for training and validation, 40% for
        testing. The default is `1`.
    seed : `int`, optional
        The random seed for reproducibility. The default is `0`.

    Raises
    ------
    AssertionError
        Raised if the splits are not pairwise disjoint.

    Returns
    -------
    subsets : `dict`
        Subset dictionary with keys:
            ``'train'``
                The training scenes (`dict`).
            ``'valid'``
                The validation scenes (`dict`).
            ``'test'``
                The test scenes (`dict`).

    """
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


def date_scene_split(ds, date, dateformat='%Y%m%d'):
    """Split the dataset based on a date.

    Scenes before ``date`` build the training dataset, scenes after ``date``
    the validation dataset. The test set is empty.

    Useful for time series data.

    Parameters
    ----------
    ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
        An instance of :py:class:`pysegcnn.core.dataset.ImageDataset`.
    date : `str`
        A date in the format ``dateformat``.
    dateformat : `str`, optional
        The format of ``date``. ``dateformat`` is used by
        :py:func:`datetime.datetime.strptime' to parse ``date`` to a
        :py:class:`datetime.datetime` object. The default is `'%Y%m%d'`.

    Raises
    ------
    AssertionError
        Raised if the splits are not pairwise disjoint.

    Returns
    -------
    subsets : `dict`
        Subset dictionary with keys:
            ``'train'``
                The training scenes (`dict`).
            ``'valid'``
                The validation scenes (`dict`).
            ``'test'``
                The test scenes (`dict`).

    """
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


class CustomSubset(Subset):
    """Generic custom subset inheriting :py:class:`torch.utils.data.Subset`.

    .. important::

        The training, validation and test datasets should be subclasses of
        :py:class:`pysegcnn.core.split.CustomSubset`.

    See :py:class:`pysegcnn.core.split.RandomTileSplit` for an example
    implementing the :py:class:`pysegcnn.core.split.RandomSubset` subset
    class.

    Attributes
    ----------
    dataset : :py:class:`pysegcnn.core.dataset.ImageDataset`
        The dataset to split into subsets.
    split_mode : `str`
        The mode to split the dataset.
    indices : `list` [`int`]
        List of indices to access the dataset.
    name : `str`
        Name of the subset.
    scenes : `list` [`dict`]
        List of the subset tiles.
    ids : `list` or :py:class:`numpy.ndarray`
        Container of the scene identifiers.

    """

    def __init__(self, ds, split_mode, indices, name, scenes, scene_ids):
        """Initialize.

        Parameters
        ----------
        ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
            An instance of :py:class:`pysegcnn.core.dataset.ImageDataset`.
        split_mode : `str`
            The mode to split the dataset.
        indices : `list` [`int`]
            List of indices to access ``ds``. ``indices`` must be pairwise
            disjoint for each subset derived from the same dataset ``ds``.
        name : `str`
            Name of the subset.
        scenes : `list` [`dict`]
            List of the subset tiles.
        scene_ids : `list` or :py:class:`numpy.ndarray`
            Container of the scene identifiers.

        """
        super().__init__(dataset=ds, indices=indices)

        # the mode to split the dataset
        self.split_mode = split_mode

        # the name of the subset
        self.name = name

        # the scene in the subset
        self.scenes = scenes

        # the names of the scenes
        self.ids = scene_ids

    def __repr__(self):
        """Representation string.

        Returns
        -------
        fs : `str`
            The representation string.

        """
        fs = '- {}: {:d} tiles ({:.2f}%), mode = {}'.format(
            self.name, len(self.scenes), 100 * len(self.scenes) /
            len(self.dataset), self.split_mode)

        return fs


class SceneSubset(CustomSubset):
    """A custom subset for dataset splits where the scenes are preserved."""

    def __init__(self, ds, split_mode, indices, name, scenes, scene_ids):
        super().__init__(ds, split_mode, indices, name, scenes, scene_ids)


class RandomSubset(CustomSubset):
    """A custom subset for random dataset splits."""

    def __init__(self, ds, split_mode, indices, name, scenes, scene_ids):
        super().__init__(ds, split_mode, indices, name, scenes, scene_ids)


class Split(object):
    """Generic class handling how ``ds`` is split.

    Each dataset should be split by a subclass of
    :py:class:`pysegcnn.core.split.Split`, by calling the
    :py:meth:`pysegcnn.core.split.Split.split` method.

    .. important::

        The :py:meth:`~pysegcnn.core.split.Split.subsets` and
        :py:meth:`~pysegcnn.core.split.Split.subset_type` methods have to be
        implemented when inheriting :py:class:`pysegcnn.core.split.Split`.
        Furthermore, a class attribute ``split_mode`` (`str`) has to be
        defined and added to :py:class:`pysegcnn.core.split.SupportedSplits`.

    See :py:class:`pysegcnn.core.split.RandomTileSplit` for an example.

    Attributes
    ----------
    ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
        The dataset to split into training, validation and test set.

    """

    def __init__(self, ds):
        """Initialize.

        Parameters
        ----------
        ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
            An instance of :py:class:`pysegcnn.core.dataset.ImageDataset`.

        """
        # the dataset to split
        self.ds = ds

    def split(self):
        """Split dataset into training, validation and test set.

        :py:meth:`~pysegcnn.core.split.Split.split` works only if
        :py:meth:`~pysegcnn.core.split.Split.subsets` and
        :py:meth:`~pysegcnn.core.split.Split.subset_type` are implemented.

        """
        # build the subsets
        ds_split = []
        for name, sub in self.subsets().items():

            # the scene identifiers of the current subset
            ids = np.unique([s['id'] for s in sub.values()])

            # build the subset
            sbst = self.subset_type()(self.ds, self.split_mode,
                                      list(sub.keys()), name,
                                      list(sub.values()), ids)
            ds_split.append(sbst)

        return ds_split

    def subsets(self):
        """Define training, validation and test sets.

        Wrapper method for
        :py:func:`pysegcnn.core.split.Split.random_tile_split`,
        :py:func:`pysegcnn.core.split.Split.random_scene_split` or
        :py:func:`pysegcnn.core.split.Split.date_scene_split`.

        Raises
        ------
        NotImplementedError
            Raised if :py:class:`pysegcnn.core.split.Split` is not inherited.

        Returns
        -------
        None.

        """
        raise NotImplementedError

    def subset_type(self):
        """Define the type of each subset.

        Wrapper method for :py:class:`pysegcnn.core.split.RandomSubset` or
        :py:class:`pysegcnn.core.split.SceneSubset`.

        Raises
        ------
        NotImplementedError
            Raised if :py:class:`pysegcnn.core.split.Split` is not inherited.

        Returns
        -------
        None.

        """
        raise NotImplementedError


class DateSplit(Split):
    """Split a dataset based on a date.

    .. important::

        Scenes before ``date`` build the training dataset, scenes after
        ``date`` the validation dataset. The test set is empty.

    Useful for time series data.

    Class wrapper for :py:func:`pysegcnn.core.split.date_scene_split`.

    Attributes
    ----------
    split_mode : `str`
        The mode to split the dataset, i.e. `'date'`.
    ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
        The dataset to split into training, validation and test set.
    date : `str`
        The date used to split the dataset.
    dateformat : `str`
        The format of ``date``.

    """

    # the split mode
    split_mode = 'date'

    def __init__(self, ds, date, dateformat):
        """Initialize.

        Parameters
        ----------
        ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
            An instance of :py:class:`pysegcnn.core.dataset.ImageDataset`.
        date : `str`
            A date in the format ``dateformat``.
        dateformat : `str`
            The format of ``date``. ``dateformat`` is used by
            :py:func:`datetime.datetime.strptime' to parse ``date`` to a
            :py:class:`datetime.datetime` object.

        """
        super().__init__(ds)

        # the date to split the dataset
        # before: training set
        # after : validation set
        self.date = date

        # the format of the date
        self.dateformat = dateformat

    def subsets(self):
        """Wrap :py:func:`pysegcnn.core.split.Split.date_scene_split`.

        Returns
        -------
        subsets : `dict`
            Subset dictionary with keys:
                ``'train'``
                    The training scenes (`dict`).
                ``'valid'``
                    The validation scenes (`dict`).
                ``'test'``
                    The test scenes, empty (`dict`).

        """
        return date_scene_split(self.ds, self.date, self.dateformat)

    def subset_type(self):
        """Wrap :py:class:`pysegcnn.core.split.SceneSubset`.

        Returns
        -------
        SceneSubset : :py:class:`pysegcnn.core.split.SceneSubset`
            The subset type.

        """
        return SceneSubset


class RandomSplit(Split):
    """Generic class for random dataset splits."""

    def __init__(self, ds, ttratio, tvratio, seed):
        """Initialize.

        Parameters
        ----------
        ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
            An instance of :py:class:`pysegcnn.core.dataset.ImageDataset`.
        tvratio : `float`
            The ratio of training data to validation data, e.g.
            ``tvratio=0.8`` means 80% training, 20% validation.
        ttratio : `float`
            The ratio of training and validation data to test data, e.g.
            ``ttratio=0.6`` means 60% for training and validation, 40% for
            testing.
        seed : `int`
            The random seed used to generate the split. Useful for
            reproducibility.

        """
        super().__init__(ds)

        # the training, validation and test set ratios
        self.ttratio = ttratio
        self.tvratio = tvratio

        # the random seed: useful for reproducibility
        self.seed = seed


class RandomTileSplit(RandomSplit):
    """Randomly split the dataset.

    .. important::

        For each scene, the tiles of the scene can be distributed among the
        training, validation and test set.

    Class wrapper for :py:func:`pysegcnn.core.split.random_tile_split`.

    Attributes
    ----------
    split_mode : `str`
        The mode to split the dataset, i.e. `'random'`.
    ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
        The dataset to split into training, validation and test set.
    tvratio : `float`
        The ratio of training data to validation data.
    ttratio : `float`
        The ratio of training and validation data to test data.
    seed : `int`
        The random seed used to generate the split.

    """

    # the split mode
    split_mode = 'random'

    def __init__(self, ds, ttratio, tvratio, seed):
        super().__init__(ds, ttratio, tvratio, seed)

    def subsets(self):
        """Wrap :py:func:`pysegcnn.core.split.Split.random_tile_split`.

        Returns
        -------
        subsets : `dict`
            Subset dictionary with keys:
                ``'train'``
                    The training scenes (`dict`).
                ``'valid'``
                    The validation scenes (`dict`).
                ``'test'``
                    The test scenes (`dict`).

        """
        return random_tile_split(self.ds, self.tvratio, self.ttratio,
                                 self.seed)

    def subset_type(self):
        """Wrap :py:class:`pysegcnn.core.split.RandomSubset`.

        Returns
        -------
        SceneSubset : :py:class:`pysegcnn.core.split.RandomSubset`
            The subset type.

        """
        return RandomSubset


class RandomSceneSplit(RandomSplit):
    """Semi-randomly split the dataset.

    .. important::

        For each scene, all the tiles of the scene are included in either the
        training, validation or test set, respectively.

    Class wrapper for :py:func:`pysegcnn.core.split.random_scene_split`.

    Attributes
    ----------
    split_mode : `str`
        The mode to split the dataset, i.e. `'scene'`.
    ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
        The dataset to split into training, validation and test set.
    tvratio : `float`
        The ratio of training data to validation data.
    ttratio : `float`
        The ratio of training and validation data to test data.
    seed : `int`
        The random seed used to generate the split.

    """

    # the split mode
    split_mode = 'scene'

    def __init__(self, ds, ttratio, tvratio, seed):
        super().__init__(ds, ttratio, tvratio, seed)

    def subsets(self):
        """Wrap :py:func:`pysegcnn.core.split.Split.random_scene_split`.

        Returns
        -------
        subsets : `dict`
            Subset dictionary with keys:
                ``'train'``
                    The training scenes (`dict`).
                ``'valid'``
                    The validation scenes (`dict`).
                ``'test'``
                    The test scenes (`dict`).

        """
        return random_scene_split(self.ds, self.tvratio, self.ttratio,
                                  self.seed)

    def subset_type(self):
        """Wrap :py:class:`pysegcnn.core.split.SceneSubset`.

        Returns
        -------
        SceneSubset : :py:class:`pysegcnn.core.split.SceneSubset`
            The subset type.

        """
        return SceneSubset


class SupportedSplits(enum.Enum):
    """Names and corresponding classes of the implemented split modes."""

    random = RandomTileSplit
    scene = RandomSceneSplit
    date = DateSplit
