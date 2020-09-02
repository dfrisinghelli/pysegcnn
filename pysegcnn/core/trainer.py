"""Model configuration and training.

This module provides an end-to-end framework of dataclasses designed to train
segmentation models on image datasets.

See pysegcnn/main/train.py for a complete walkthrough.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import os
import dataclasses
import pathlib
import logging
import datetime

# externals
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer

# locals
from pysegcnn.core.dataset import SupportedDatasets, ImageDataset
from pysegcnn.core.transforms import Augment
from pysegcnn.core.utils import img2np, item_in_enum, accuracy_function
from pysegcnn.core.split import SupportedSplits, CustomSubset
from pysegcnn.core.models import (SupportedModels, SupportedOptimizers,
                                  SupportedLossFunctions, Network)
from pysegcnn.core.layers import Conv2dSame
from pysegcnn.main.config import HERE

# module level logger
LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class BaseConfig:
    """Base :py:class:`dataclasses.dataclass` for each configuration."""

    def __post_init__(self):
        """Check the type of each argument.

        Raises
        ------
        TypeError
            Raised if the conversion to the specified type of the argument
            fails.

        """
        # check input types
        for field in dataclasses.fields(self):
            # the value of the current field
            value = getattr(self, field.name)

            # check whether the value is of the correct type
            if not isinstance(value, field.type):
                # try to convert the value to the correct type
                try:
                    setattr(self, field.name, field.type(value))
                except TypeError:
                    # raise an exception if the conversion fails
                    raise TypeError('Expected {} to be {}, got {}.'
                                    .format(field.name, field.type,
                                            type(value)))


@dataclasses.dataclass
class DatasetConfig(BaseConfig):
    """Dataset configuration class.

    Instanciate a dataset.

    Attributes
    ----------
    dataset_name : `str`
        The name of the dataset.
    root_dir : `str`
        The root directory, path to the dataset.
    bands : `list` [`str`]
        A list of the spectral bands to use.
    tile_size : `int` or `None`
        The size of the tiles. If not `None`, each scene is divided into
        square tiles of shape ``(tile_size, tile_size)``.
    gt_pattern : `str`
        A regural expression to match the ground truth naming convention.
        All directories and subdirectories in ``root_dir`` are searched for
        files matching ``gt_pattern``.
    seed : `int`
        The random seed. Used to split the dataset into training,
        validation and test set. Useful for reproducibility.
    sort : `bool`
        Whether to chronologically sort the samples. Useful for time series
        data. The default is `False`.
    transforms : `list`
        List of :py:class:`pysegcnn.core.transforms.Augment` instances.
        Each item in ``transforms`` generates a distinct transformed
        version of the dataset. The total dataset is composed of the
        original untransformed dataset together with each transformed
        version of it. If ``transforms=[]``, only the original dataset is
        used. The default is `[]`.
    pad : `bool`
        Whether to center pad the input image. Set ``pad=True``, if the
        images are not evenly divisible by the ``tile_size``. The image
        data is padded with a constant padding value of zero. For each
        image, the corresponding ground truth image is padded with a
        "no data" label. The default is `False`.
    dataset_class : :py:class:`pysegcnn.core.dataset.ImageDataset`
        A subclass of :py:class:`pysegcnn.core.dataset.ImageDataset`.

    """

    dataset_name: str
    root_dir: pathlib.Path
    bands: list
    tile_size: int
    gt_pattern: str
    seed: int
    sort: bool = False
    transforms: list = dataclasses.field(default_factory=list)
    pad: bool = False

    def __post_init__(self):
        """Check the type of each argument.

        Raises
        ------
        ValueError
            Raised if ``dataset_name`` is not supported.
        FileNotFoundError
            Raised if ``root_dir`` does not exist.
        TypeError
            Raised if not each item in ``transforms`` is an instance of
            :py:class:`pysegcnn.core.split.Augment` in case ``transforms`` is
            not empty.

        """
        # check input types
        super().__post_init__()

        # check whether the dataset is currently supported
        self.dataset_class = item_in_enum(self.dataset_name, SupportedDatasets)

        # check whether the root directory exists
        if not self.root_dir.exists():
            raise FileNotFoundError('{} does not exist.'.format(self.root_dir))

        # check whether the transformations inherit from the correct class
        if not all([isinstance(t, Augment) for t in self.transforms if
                    self.transforms]):
            raise TypeError('Each transformation is expected to be an instance'
                            ' of {}.'.format('.'.join([Augment.__module__,
                                                       Augment.__name__])))

    def init_dataset(self):
        """Instanciate the dataset.

        Returns
        -------
        dataset : :py:class:`pysegcnn.core.dataset.ImageDataset`
            An instance of :py:class:`pysegcnn.core.dataset.ImageDataset`.

        """
        # instanciate the dataset
        dataset = self.dataset_class(
                    root_dir=str(self.root_dir),
                    use_bands=self.bands,
                    tile_size=self.tile_size,
                    seed=self.seed,
                    sort=self.sort,
                    transforms=self.transforms,
                    pad=self.pad,
                    gt_pattern=self.gt_pattern
                    )

        return dataset


@dataclasses.dataclass
class SplitConfig(BaseConfig):
    """Dataset split configuration class.

    Split a dataset into training, validation and test set.

    Attributes
    ----------
    split_mode : `str`
        The mode to split the dataset.
    ttratio : `float`
        The ratio of training and validation data to test data, e.g.
        ``ttratio= 0.6`` means 60% for training and validation, 40% for
        testing.
    tvratio : `float`
        The ratio of training data to validation data, e.g. ``tvratio=0.8``
        means 80% training, 20% validation.
    date : `str`
        A date. Used if ``split_mode='date'``. The default is  `yyyymmdd`.
    dateformat : `str`
        The format of ``date``. ``dateformat`` is used by
        :py:func:`datetime.datetime.strptime' to parse ``date`` to a
        :py:class:`datetime.datetime` object. The default is `'%Y%m%d'`.
    drop : `float`
        Whether to drop samples (during training only) with a fraction of
        pixels equal to the constant padding value >= ``drop``. ``drop=0``
        means, do not drop any samples. The default is `0`.
    split_class : :py:class:`pysegcnn.core.split.Split`
        A subclass of :py:class:`pysegcnn.core.split.Split`.
    dropped : `list` [`dict`]
        List of the dropped samples.

    """

    split_mode: str
    ttratio: float
    tvratio: float
    date: str = 'yyyymmdd'
    dateformat: str = '%Y%m%d'
    drop: float = 0

    def __post_init__(self):
        """Check the type of each argument.

        Raises
        ------
        ValueError
            Raised if ``split_mode`` is not supported.

        """
        # check input types
        super().__post_init__()

        # check if the split mode is valid
        self.split_class = item_in_enum(self.split_mode, SupportedSplits)

        # list of dropped samples
        self.dropped = []

    @staticmethod
    def drop_samples(ds, drop_threshold=1):
        """Drop samples with a fraction of pixels equal to the padding value.

        Parameters
        ----------
        ds : :py:class:`pysegcnn.core.split.CustomSubset`
            An instance of :py:class:`pysegcnn.core.split.CustomSubset`.
        drop_threshold : `float`, optional
            The threshold above which samples are dropped. ``drop_threshold=1``
            means a sample is dropped, if all pixels are equal to the padding
            value. ``drop_threshold=0.8`` means, drop a sample if 80% of the
            pixels are equal to the padding value, etc. The default is `1`.

        Returns
        -------
        dropped : `list` [`dict`]
            List of the dropped samples.

        """
        # iterate over the scenes returned by self.compose_scenes()
        dropped = []
        for pos, i in enumerate(ds.indices):

            # the current scene
            s = ds.dataset.scenes[i]

            # the current tile in the ground truth
            tile_gt = img2np(s['gt'], ds.dataset.tile_size, s['tile'],
                             ds.dataset.pad, ds.dataset.cval)

            # percent of pixels equal to the constant padding value
            npixels = (tile_gt[tile_gt == ds.dataset.cval].size / tile_gt.size)

            # drop samples where npixels >= self.drop
            if npixels >= drop_threshold:
                LOGGER.info('Skipping scene {}, tile {}: {:.2f}% padded pixels'
                            ' ...'.format(s['id'], s['tile'], npixels * 100))
                dropped.append(s)
                _ = ds.indices.pop(pos)

        return dropped

    def train_val_test_split(self, ds):
        """Split ``ds`` into training, validation and test set.

        Parameters
        ----------
        ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
            An instance of :py:class:`pysegcnn.core.dataset.ImageDataset`.

        Raises
        ------
        TypeError
            Raised if ``ds`` is not an instance of
            :py:class:`pysegcnn.core.dataset.ImageDataset`.

        Returns
        -------
        train_ds : :py:class:`pysegcnn.core.split.CustomSubset`.
            The training set.
        valid_ds : :py:class:`pysegcnn.core.split.CustomSubset`.
            The validation set.
        test_ds : :py:class:`pysegcnn.core.split.CustomSubset`.
            The test set.

        """
        if not isinstance(ds, ImageDataset):
            raise TypeError('Expected "ds" to be {}.'
                            .format('.'.join([ImageDataset.__module__,
                                              ImageDataset.__name__])))

        if self.split_mode == 'random' or self.split_mode == 'scene':
            subset = self.split_class(ds,
                                      self.ttratio,
                                      self.tvratio,
                                      ds.seed)

        else:
            subset = self.split_class(ds, self.date, self.dateformat)

        # the training, validation and test dataset
        train_ds, valid_ds, test_ds = subset.split()

        # whether to drop training samples with a fraction of pixels equal to
        # the constant padding value cval >= drop
        if ds.pad and self.drop > 0:
            self.dropped = self.drop_samples(train_ds, self.drop)

        return train_ds, valid_ds, test_ds

    @staticmethod
    def dataloaders(*args, **kwargs):
        """Build :py:class:`torch.utils.data.DataLoader` instances.

        Parameters
        ----------
        *args : `list` [:py:class:`torch.utils.data.Dataset`]
            List of instances of :py:class:`torch.utils.data.Dataset`.
        **kwargs
            Additional keyword arguments passed to
            :py:class:`torch.utils.data.DataLoader`.

        Raises
        ------
        TypeError
            Raised if not each item in ``args`` is an instance of
            :py:class:`torch.utils.data.Dataset`.

        Returns
        -------
        loaders : `list` [:py:class:`torch.utils.data.DataLoader`]
            List of instances of :py:class:`torch.utils.data.DataLoader`. If an
            instance of :py:class:`torch.utils.data.Dataset` in ``args`` is
            empty, `None` is appended to ``loaders`` instead of an instance of
            :py:class:`torch.utils.data.DataLoader`.

        """
        # check whether each dataset in args has the correct type
        loaders = []
        for ds in args:
            if not isinstance(ds, Dataset):
                raise TypeError('Expected {}, got {}.'
                                .format(repr(Dataset), type(ds)))

            # check if the dataset is not empty
            if len(ds) > 0:
                # build the dataloader
                loader = DataLoader(ds, **kwargs)
            else:
                loader = None
            loaders.append(loader)

        return loaders


@dataclasses.dataclass
class ModelConfig(BaseConfig):
    """Model configuration class.

    Instanciate a (pretrained) model.

    Attributes
    ----------
    model_name : `str`
        The name of the model.
    filters : `list` [`int`]
        List of input channels to the convolutional layers.
    torch_seed : `int`
        The random seed to initialize the model weights.
        Useful for reproducibility.
    optim_name : `str`
        The name of the optimizer to update the model weights.
    loss_name : `str`
        The name of the loss function measuring the model error.
    skip_connection : `bool`
        Whether to apply skip connections. The default is `True`.
    kwargs: `dict`
        The configuration for each convolution in the model. The default is
        `{'kernel_size': 3, 'stride': 1, 'dilation': 1}`.
    batch_size : `int`
        The model batch size. Determines the number of samples to process
        before updating the model weights. The default is `64`.
    checkpoint : `bool`
        Whether to resume training from an existing model checkpoint. The
        default is `False`.
    transfer : `bool`
        Whether to use a model for transfer learning on a new dataset. If True,
        the model architecture of ``pretrained_model`` is adjusted to a new
        dataset. The default is `False`.
    pretrained_model : `str`
        The name of the pretrained model to use for transfer learning.
        The default is `''`.
    lr : `float`
        The learning rate used by the gradient descent algorithm.
        The default is `0.001`.
    early_stop : `bool`
        Whether to apply `Early Stopping`_. The default is `False`.
    mode : `str`
        The early stopping mode. Depends on the metric measuring
        performance. When using model loss as metric, use ``mode='min'``,
        however, when using accuracy as metric, use ``mode='max'``. For now,
        only ``mode='max'`` is supported. Only used if ``early_stop=True``.
        The default is `'max'`.
    delta : `float`
        Minimum change in early stopping metric to be considered as an
        improvement. Only used if ``early_stop=True``. The default is `0`.
    patience : `int`
        The number of epochs to wait for an improvement in the early stopping
        metric. If the model does not improve over more than ``patience``
        epochs, quit training. Only used if ``early_stop=True``. The default is
        `10`.
    epochs : `int`
        The maximum number of epochs to train. The default is `50`.
    nthreads : `int`
        The number of cpu threads to use during training. The default is
        :py:func:`torch.get_num_threads()`.
    save : `bool`
        Whether to save the model state to disk. Model states are saved in
        pysegcnn/main/_models. The default is `True`.
    model_class : :py:class:`pysegcnn.core.models.Network`
        A subclass of :py:class:`pysegcnn.core.models.Network`.
    optim_class : :py:class:`torch.optim.Optimizer`
        A subclass of :py:class:`torch.optim.Optimizer`.
    loss_class : :py:class:`torch.nn.Module`
        A subclass of :py:class:`torch.nn.Module`
    state_path : :py:class:`pathlib.Path`
        Path to save model states.
    pretrained_path : :py:class:`pathlib.Path`
        Path to the ``pretrained_model`` used if ``transfer=True``.

    .. _Early Stopping:
        https://en.wikipedia.org/wiki/Early_stopping

    """

    model_name: str
    filters: list
    torch_seed: int
    optim_name: str
    loss_name: str
    skip_connection: bool = True
    kwargs: dict = dataclasses.field(
        default_factory=lambda: {'kernel_size': 3, 'stride': 1, 'dilation': 1})
    batch_size: int = 64
    checkpoint: bool = False
    transfer: bool = False
    pretrained_model: str = ''
    lr: float = 0.001
    early_stop: bool = False
    mode: str = 'max'
    delta: float = 0
    patience: int = 10
    epochs: int = 50
    nthreads: int = torch.get_num_threads()
    save: bool = True

    def __post_init__(self):
        """Check the type of each argument.

        Configure path to save model state.

        Raises
        ------
        ValueError
            Raised if the model ``model_name``, the optimizer ``optim_name`` or
            the loss function ``loss_name`` is not supported.

        """
        # check input types
        super().__post_init__()

        # check whether the model is currently supported
        self.model_class = item_in_enum(self.model_name, SupportedModels)

        # check whether the optimizer is currently supported
        self.optim_class = item_in_enum(self.optim_name, SupportedOptimizers)

        # check whether the loss function is currently supported
        self.loss_class = item_in_enum(self.loss_name, SupportedLossFunctions)

        # path to model states
        self.state_path = pathlib.Path(HERE).joinpath('_models/')

        # path to pretrained model
        self.pretrained_path = self.state_path.joinpath(self.pretrained_model)

    def init_optimizer(self, model):
        """Instanciate the optimizer.

        Parameters
        ----------
        model : :py:class:`torch.nn.Module`
            An instance of :py:class:`torch.nn.Module`.

        Returns
        -------
        optimizer : :py:class:`torch.optim.Optimizer`
            An instance of :py:class:`torch.optim.Optimizer`.

        """
        LOGGER.info('Optimizer: {}.'.format(repr(self.optim_class)))

        # initialize the optimizer for the specified model
        optimizer = self.optim_class(model.parameters(), self.lr)

        return optimizer

    def init_loss_function(self):
        """Instanciate the loss function.

        Returns
        -------
        loss_function : :py:class:`torch.nn.Module`
            An instance of :py:class:`torch.nn.Module`.

        """
        LOGGER.info('Loss function: {}.'.format(repr(self.loss_class)))

        # instanciate the loss function
        loss_function = self.loss_class()

        return loss_function

    def init_model(self, ds, state_file):
        """Instanciate the model and the optimizer.

        If the model checkpoint ``state_file`` exists, the pretrained model and
        optimizer states are loaded, otherwise the model and the optimizer are
        initialized from scratch.

        Parameters
        ----------
        ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
            An instance of :py:class:`pysegcnn.core.dataset.ImageDataset`.
        state_file : :py:class:`pathlib.Path`
            Path to a model checkpoint.

        Returns
        -------
        model : :py:class:`pysegcnn.core.models.Network`
            An instance of :py:class:`pysegcnn.core.models.Network`.
        optimizer : :py:class:`torch.optim.Optimizer`
            An instance of :py:class:`torch.optim.Optimizer`.
        checkpoint_state : `dict` [`str`, :py:class:`numpy.ndarray`]
            If the model checkpoint ``state_file`` exists, ``checkpoint_state``
            has keys:
                ``'ta'``
                    The accuracy on the training set
                    (:py:class:`numpy.ndarray`).
                ``'tl'``
                    The loss on the training set (:py:class:`numpy.ndarray`).
                ``'va'``
                    The accuracy on the validation set
                    (:py:class:`numpy.ndarray`).
                ``'vl'``
                    The loss on the validation set (:py:class:`numpy.ndarray`).

        """
        # write an initialization string to the log file
        LogConfig.init_log('{}: Initializing model run. ')

        # case (1): build a new model
        if not self.transfer:

            # set the random seed for reproducibility
            torch.manual_seed(self.torch_seed)
            LOGGER.info('Initializing model: {}'.format(state_file.name))

            # instanciate the model
            model = self.model_class(
                in_channels=len(ds.use_bands),
                nclasses=len(ds.labels),
                filters=self.filters,
                skip=self.skip_connection,
                **self.kwargs)

        # case (2): load a pretrained model for transfer learning
        else:
            # load pretrained model
            LOGGER.info('Loading pretrained model for transfer learning from: '
                        '{}'.format(self.pretrained_path))
            model = self.transfer_model(self.pretrained_path, ds)

        # initialize the optimizer
        optimizer = self.init_optimizer(model)

        # whether to resume training from an existing model checkpoint
        checkpoint_state = {}
        if self.checkpoint:
            model, optimizer, checkpoint_state = self.load_checkpoint(
                model, optimizer, state_file)

        return model, optimizer, checkpoint_state

    @staticmethod
    def load_checkpoint(model, optimizer, state_file):
        """Load an existing model checkpoint.

        If the model checkpoint ``state_file`` exists, the pretrained model and
        optimizer states are loaded.

        Parameters
        ----------
        model : :py:class:`pysegcnn.core.models.Network`
            An instance of :py:class:`pysegcnn.core.models.Network`.
        optimizer : :py:class:`torch.optim.Optimizer`
            An instance of :py:class:`torch.optim.Optimizer`.
        state_file : :py:class:`pathlib.Path`
            Path to the model checkpoint.

        Returns
        -------
        model : :py:class:`pysegcnn.core.models.Network`
            An instance of :py:class:`pysegcnn.core.models.Network`.
        optimizer : :py:class:`torch.optim.Optimizer`
            An instance of :py:class:`torch.optim.Optimizer`.
        checkpoint_state : `dict` [`str`, :py:class:`numpy.ndarray`]
            If the model checkpoint ``state_file`` exists, ``checkpoint_state``
            has keys:
                ``'ta'``
                    The accuracy on the training set
                    (:py:class:`numpy.ndarray`).
                ``'tl'``
                    The loss on the training set (:py:class:`numpy.ndarray`).
                ``'va'``
                    The accuracy on the validation set
                    (:py:class:`numpy.ndarray`).
                ``'vl'``
                    The loss on the validation set (:py:class:`numpy.ndarray`).

        """
        # whether to resume training from an existing model checkpoint
        checkpoint_state = {}

        # if no checkpoint exists, file a warning and continue with a model
        # initialized from scratch
        if not state_file.exists():
            LOGGER.warning('Checkpoint for model {} does not exist. '
                           'Initializing new model.'
                           .format(state_file.name))
        else:
            # load model checkpoint
            model, optimizer, model_state = Network.load(state_file)

            # load model loss and accuracy

            # get all non-zero elements, i.e. get number of epochs trained
            # before the early stop
            checkpoint_state = {k: v[np.nonzero(v)].reshape(v.shape[0], -1)
                                for k, v in model_state['state'].items()}

        return model, optimizer, checkpoint_state

    @staticmethod
    def transfer_model(state_file, ds):
        """Adjust a pretrained model to a new dataset.

        The classification layer of the pretrained model in ``state_file`` is
        initilialized from scratch with the classes of the new dataset ``ds``.

        The remaining model weights are preserved.

        Parameters
        ----------
        state_file : :py:class:`pathlib.Path`
            Path to a pretrained model.
        ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
            An instance of :py:class:`pysegcnn.core.dataset.ImageDataset`.

        Raises
        ------
        TypeError
            Raised if ``ds`` is not an instance of
            :py:class:`pysegcnn.core.dataset.ImageDataset`.
        ValueError
            Raised if the bands of ``ds`` do not match the bands of the dataset
            the pretrained model was trained with.

        Returns
        -------
        model : :py:class:`pysegcnn.core.models.Network`
            An instance of :py:class:`pysegcnn.core.models.Network`. The
            pretrained model adjusted to the new dataset.

        """
        # check input type
        if not isinstance(ds, ImageDataset):
            raise TypeError('Expected "ds" to be {}.'
                            .format('.'.join([ImageDataset.__module__,
                                              ImageDataset.__name__])))

        # load the pretrained model
        model, _, model_state = Network.load(state_file)
        LOGGER.info('Configuring model for new dataset: {}.'.format(
            ds.__class__.__name__))

        # check whether the current dataset uses the correct spectral bands
        if ds.use_bands != model_state['bands']:
            raise ValueError('The pretrained network was trained with '
                             'bands {}, not with bands {}.'
                             .format(model_state['bands'], ds.use_bands))

        # get the number of convolutional filters
        filters = model_state['params']['filters']

        # reset model epoch to 0, since the model is trained on a different
        # dataset
        model.epoch = 0

        # adjust the number of classes in the model
        model.nclasses = len(ds.labels)
        LOGGER.info('Replacing classification layer to classes: {}.'
                    .format(', '.join('({}, {})'.format(k, v['label'])
                                      for k, v in ds.labels.items())))

        # adjust the classification layer to the classes of the new dataset
        model.classifier = Conv2dSame(in_channels=filters[0],
                                      out_channels=model.nclasses,
                                      kernel_size=1)

        return model


@dataclasses.dataclass
class StateConfig(BaseConfig):
    """Model state configuration class.

    Generate the model state filename according to the following naming
    convention:

    `model_dataset_optimizer_splitmode_splitparams_tilesize_batchsize_bands.pt`

    Attributes
    ----------
    dc : :py:class:`pysegcnn.core.trainer.DatasetConfig`
        An instance of :py:class:`pysegcnn.core.trainer.DatasetConfig`.
    sc : :py:class:`pysegcnn.core.trainer.SplitConfig`
        An instance of :py:class:`pysegcnn.core.trainer.SplitConfig`.
    mc : :py:class:`pysegcnn.core.trainer.ModelConfig`
        An instance of :py:class:`pysegcnn.core.trainer.SplitConfig`.

    """

    dc: DatasetConfig
    sc: SplitConfig
    mc: ModelConfig

    def __post_init__(self):
        """Check the type of each argument."""
        super().__post_init__()

    def init_state(self):
        """Generate the model state filename.

        Returns
        -------
        state : :py:class:`pathlib.Path`
            The path to the model state file.

        """
        # file to save model state to:
        # network_dataset_optim_split_splitparams_tilesize_batchsize_bands.pt

        # model state filename
        state_file = '{}_{}_{}_{}Split_{}_t{}_b{}_{}.pt'

        # get the band numbers
        if self.dc.bands:
            bformat = ''.join(band[0] +
                              str(self.dc.get_sensor().__members__[band].value)
                              for band in self.dc.bands)
        else:
            bformat = 'all'

        # check which split mode was used
        if self.sc.split_mode == 'date':
            # store the date that was used to split the dataset
            state_file = state_file.format(self.mc.model_name,
                                           self.dc.dataset_class.__name__,
                                           self.mc.optim_name,
                                           self.sc.split_mode.capitalize(),
                                           self.sc.date,
                                           self.dc.tile_size,
                                           self.mc.batch_size,
                                           bformat)
        else:
            # store the random split parameters
            split_params = 's{}_t{}v{}'.format(
                self.dc.seed, str(self.sc.ttratio).replace('.', ''),
                str(self.sc.tvratio).replace('.', ''))

            # model state filename
            state_file = state_file.format(self.mc.model_name,
                                           self.dc.dataset_class.__name__,
                                           self.mc.optim_name,
                                           self.sc.split_mode.capitalize(),
                                           split_params,
                                           self.dc.tile_size,
                                           self.mc.batch_size,
                                           bformat)

        # check whether a pretrained model was used and change state filename
        # accordingly
        if self.mc.transfer:
            # add the configuration of the pretrained model to the state name
            state_file = (state_file.replace('.pt', '_') +
                          'pretrained_' + self.mc.pretrained_model)

        # path to model state
        state = self.mc.state_path.joinpath(state_file)

        return state


@dataclasses.dataclass
class EvalConfig(BaseConfig):
    """Model inference configuration.

    Evaluate a model.

    Attributes
    ----------
    state_file : :py:class:`pathlib.Path`
        Path to the model to evaluate.
    test : `bool` or `None`
        Whether to evaluate the model on the training(``test=None``), the
        validation (``test=False``) or the test set (``test=True``).
    predict_scene : `bool`
        The model prediction order. If False, the samples (tiles) of a dataset
        are predicted in any order and the scenes are not reconstructed.
        If True, the samples (tiles) are ordered according to the scene they
        belong to and a model prediction for each entire reconstructed scene is
        returned. The default is `False`.
    plot_samples : `bool`
        Whether to save a plot of false color composite, ground truth and model
        prediction for each sample (tile). Only used if ``predict_scene=False``
        . The default is `False`.
    plot_scenes : `bool`
        Whether to save a plot of false color composite, ground truth and model
        prediction for each entire scene. Only used if ``predict_scene=True``.
        The default is `False`.
    plot_bands : `list` [`str`]
        The bands to build the false color composite. The default is
        `['nir', 'red', 'green']`.
    cm : `bool`
        Whether to compute and plot the confusion matrix. The default is `True`
        .
    figsize : `tuple`
        The figure size in centimeters. The default is `(10, 10)`.
    alpha : `int`
        The level of the percentiles for contrast stretching of the false color
        compsite. The default is `0`, i.e. no stretching.
    animate : `bool`
        Whether to create an animation of (input, ground truth, prediction) for
        the scenes in the train/validation/test dataset.
    base_path : :py:class:`pathlib.Path`
        Root path to store model output.
    sample_path : :py:class:`pathlib.Path`
        Path to store plots of model predictions for single samples.
    scenes_path : :py:class:`pathlib.Path`
        Path to store plots of model predictions for entire scenes.
    perfmc_path : :py:class:`pathlib.Path`
        Path to store plots of model performance, e.g. confusion matrix.
    models_path : :py:class:`pathlib.Path`
        Path to search for model state files, i.e. pretrained models.
    animtn_path : :py:class:`pathlib.Path`
        Path to store animations.

    """

    state_file: pathlib.Path
    test: object
    predict_scene: bool = False
    plot_samples: bool = False
    plot_scenes: bool = False
    plot_bands: list = dataclasses.field(
        default_factory=lambda: ['nir', 'red', 'green'])
    cm: bool = True
    figsize: tuple = (10, 10)
    alpha: int = 5
    animate: bool = False

    def __post_init__(self):
        """Check the type of each argument.

        Configure figure output paths.

        Raises
        ------
        TypeError
            Raised if ``test`` is not of type `bool` or `None`.

        """
        super().__post_init__()

        # check whether the test input parameter is correctly specified
        if self.test not in [None, False, True]:
            raise TypeError('Expected "test" to be None, True or False, got '
                            '{}.'.format(self.test))

        # the output paths for the different graphics
        self.base_path = pathlib.Path(HERE)
        self.sample_path = self.base_path.joinpath('_samples')
        self.scenes_path = self.base_path.joinpath('_scenes')
        self.perfmc_path = self.base_path.joinpath('_graphics')
        self.animtn_path = self.base_path.joinpath('_animations')

        # input path for model state files
        self.models_path = self.base_path.joinpath('_models')
        self.state_file = self.models_path.joinpath(self.state_file)

    @staticmethod
    def replace_dataset_path(ds, dataset_path):
        """Replace the path to the scenes of a dataset.

        Useful to evaluate models on machines, that are different from the
        machine the model was trained on.

        Parameters
        ----------
        ds : :py:class:`pysegcnn.core.split.CustomSubset`
            A subset of an instance of
            :py:class:`pysegcnn.core.dataset.ImageDataset`.
        dataset_path : `str`
            Path to the dataset on the current machine.

        Raises
        ------
        TypeError
            Raised if ``ds`` is not an instance of
            :py:class:`pysegcnn.core.split.CustomSubset` and if ``ds`` is not
            a subset of an instance of
            :py:class:`pysegcnn.core.dataset.ImageDataset`.

        """
        # check input type
        if isinstance(ds, CustomSubset):
            # check the type of the dataset
            if not isinstance(ds.dataset, ImageDataset):
                raise TypeError('ds should be a subset created from a {}.'
                                .format(repr(ImageDataset)))
        else:
            raise TypeError('ds should be an instance of {}.'
                            .format(repr(CustomSubset)))

        # iterate over the scenes of the dataset
        for scene in ds.dataset.scenes:
            for k, v in scene.items():
                # do only look for paths
                if isinstance(v, str) and k != 'id':

                    # iterate over the path for as long as the basename does
                    # not match any scene identifier
                    dpath = os.path.dirname(v)
                    while (ds.dataset.parse_scene_id(os.path.basename(dpath))
                           is not None):
                        dpath = os.path.dirname(dpath)

                    # replace dataset path
                    if dpath != dataset_path:
                        scene[k] = v.replace(str(dpath), dataset_path)


@dataclasses.dataclass
class LogConfig(BaseConfig):
    """Logging configuration class.

    Generate the model log file.

    Attributes
    ----------
    state_file : :py:class:`pathlib.Path`
        Path to a model state file.
    log_path : :py:class:`pathlib.Path`
        Path to store model logs.
    log_file : :py:class:`pathlib.Path`
        Path to the log file of the model ``state_file``.
    """

    state_file: pathlib.Path

    def __post_init__(self):
        """Check the type of each argument.

        Generate model log file.

        """
        super().__post_init__()

        # the path to store model logs
        self.log_path = pathlib.Path(HERE).joinpath('_logs')

        # the log file of the current model
        self.log_file = self.log_path.joinpath(
            self.state_file.name.replace('.pt', '.log'))

    @staticmethod
    def now():
        """Return the current date and time.

        Returns
        -------
        date : :py:class:`datetime.datetime`
            The current date and time.

        """
        return datetime.datetime.strftime(datetime.datetime.now(),
                                          '%Y-%m-%dT%H:%M:%S')

    @staticmethod
    def init_log(init_str):
        """Generate a string to identify a new model run.

        Parameters
        ----------
        init_str : `str`
            The string to write to the model log file.

        """
        LOGGER.info(80 * '-')
        LOGGER.info(init_str.format(LogConfig.now()))
        LOGGER.info(80 * '-')


@dataclasses.dataclass
class NetworkTrainer(BaseConfig):
    """Model training class.

    Generic class to train an instance of
    :py:class:`pysegcnn.core.models.Network` on a dataset of type
    :py:class:`pysegcnn.core.dataset.ImageDataset`.

    Attributes
    ----------
    model : :py:class:`pysegcnn.core.models.Network`
        The model to train. An instance of
        :py:class:`pysegcnn.core.models.Network`.
    optimizer : :py:class:`torch.optim.Optimizer`
        The optimizer to update the model weights. An instance of
        :py:class:`torch.optim.Optimizer`.
    loss_function : :py:class:`torch.nn.Module`
        The loss function to compute the model error. An instance of
        :py:class:`torch.nn.Module`.
    train_dl : :py:class:`torch.utils.data.DataLoader`
        The training :py:class:`torch.utils.data.DataLoader` instance build
        from an instance of :py:class:`pysegcnn.core.split.CustomSubset`.
    valid_dl : :py:class:`torch.utils.data.DataLoader`
        The validation :py:class:`torch.utils.data.DataLoader` instance build
        from an instance of :py:class:`pysegcnn.core.split.CustomSubset`.
    test_dl : :py:class:`torch.utils.data.DataLoader`
        The test :py:class:`torch.utils.data.DataLoader` instance build from an
        instance of :py:class:`pysegcnn.core.split.CustomSubset`.
    state_file : :py:class:`pathlib.Path`
        Path to save the model state.
    epochs : `int`
        The maximum number of epochs to train. The default is `1`.
    nthreads : `int`
        The number of cpu threads to use during training. The default is
        :py:func:`torch.get_num_threads()`.
    early_stop : `bool`
        Whether to apply `Early Stopping`_. The default is `False`.
    mode : `str`
        The early stopping mode. Depends on the metric measuring
        performance. When using model loss as metric, use ``mode='min'``,
        however, when using accuracy as metric, use ``mode='max'``. For now,
        only ``mode='max'`` is supported. Only used if ``early_stop=True``.
        The default is `'max'`.
    delta : `float`
        Minimum change in early stopping metric to be considered as an
        improvement. Only used if ``early_stop=True``. The default is `0`.
    patience : `int`
        The number of epochs to wait for an improvement in the early stopping
        metric. If the model does not improve over more than ``patience``
        epochs, quit training. Only used if ``early_stop=True``. The default is
        `10`.
    checkpoint_state : `dict` [`str`, :py:class:`numpy.ndarray`]
        A model checkpoint for ``model``. If specified, ``checkpoint_state``
        should be a dictionary with keys:
            ``'ta'``
                The accuracy on the training set (:py:class:`numpy.ndarray`).
            ``'tl'``
                The loss on the training set (:py:class:`numpy.ndarray`).
            ``'va'``
                The accuracy on the validation set (:py:class:`numpy.ndarray`).
            ``'vl'``
                The loss on the validation set (:py:class:`numpy.ndarray`).
        The default is `{}`.
    save : `bool`
        Whether to save the model state to ``state_file``. The default is
        `True`.
    device : `str`
        The device to train the model on, i.e. `cpu` or `cuda`.
    max_accuracy : `float`
        Maximum accuracy on the validation dataset.
    es : `None` or :py:class:`pysegcnn.core.trainer.EarlyStopping`
        The early stopping instance if ``early_stop=True``, else `None`.
    training_state : `dict` [`str`, `numpy.ndarray`]
            The training state dictionary with keys:
            ``'ta'``
                The accuracy on the training set (:py:class:`numpy.ndarray`).
            ``'tl'``
                The loss on the training set (:py:class:`numpy.ndarray`).
            ``'va'``
                The accuracy on the validation set (:py:class:`numpy.ndarray`).
            ``'vl'``
                The loss on the validation set (:py:class:`numpy.ndarray`).

    .. _Early Stopping:
        https://en.wikipedia.org/wiki/Early_stopping

    """

    model: Network
    optimizer: Optimizer
    loss_function: nn.Module
    train_dl: DataLoader
    valid_dl: DataLoader
    test_dl: DataLoader
    state_file: pathlib.Path
    epochs: int = 1
    nthreads: int = torch.get_num_threads()
    early_stop: bool = False
    mode: str = 'max'
    delta: float = 0
    patience: int = 10
    checkpoint_state: dict = dataclasses.field(default_factory=dict)
    save: bool = True

    def __post_init__(self):
        """Check the type of each argument.

        Configure the device to train the model on, i.e. train on the gpu if
        available.

        Configure early stopping if required.

        """
        super().__post_init__()

        # whether to use the gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

        # send the model to the gpu if available
        self.model = self.model.to(self.device)

        # maximum accuracy on the validation dataset
        self.max_accuracy = 0
        if self.checkpoint_state:
            self.max_accuracy = self.checkpoint_state['va'].mean(
                axis=0).max().item()

        # whether to use early stopping
        self.es = None
        if self.early_stop:
            self.es = EarlyStopping(self.mode, self.max_accuracy, self.delta,
                                    self.patience)

        # log representation
        LOGGER.info(repr(self))

    def train(self):
        """Train the model.

        Returns
        -------
        training_state : `dict` [`str`, `numpy.ndarray`]
            The training state dictionary with keys:
            ``'ta'``
                The accuracy on the training set (:py:class:`numpy.ndarray`).
            ``'tl'``
                The loss on the training set (:py:class:`numpy.ndarray`).
            ``'va'``
                The accuracy on the validation set (:py:class:`numpy.ndarray`).
            ``'vl'``
                The loss on the validation set (:py:class:`numpy.ndarray`).

        """
        LOGGER.info(35 * '-' + ' Training ' + 35 * '-')

        # set the number of threads
        LOGGER.info('Device: {}'.format(self.device))
        LOGGER.info('Number of cpu threads: {}'.format(self.nthreads))
        torch.set_num_threads(self.nthreads)

        # create dictionary of the observed losses and accuracies on the
        # training and validation dataset
        tshape = (len(self.train_dl), self.epochs)
        vshape = (len(self.valid_dl), self.epochs)
        self.training_state = {'tl': np.zeros(shape=tshape),
                               'ta': np.zeros(shape=tshape),
                               'vl': np.zeros(shape=vshape),
                               'va': np.zeros(shape=vshape)
                               }

        # initialize the training: iterate over the entire training dataset
        for epoch in range(self.epochs):

            # set the model to training mode
            LOGGER.info('Setting model to training mode ...')
            self.model.train()

            # iterate over the dataloader object
            for batch, (inputs, labels) in enumerate(self.train_dl):

                # send the data to the gpu if available
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # reset the gradients
                self.optimizer.zero_grad()

                # perform forward pass
                outputs = self.model(inputs)

                # compute loss
                loss = self.loss_function(outputs, labels.long())
                observed_loss = loss.detach().numpy().item()
                self.training_state['tl'][batch, epoch] = observed_loss

                # compute the gradients of the loss function w.r.t.
                # the network weights
                loss.backward()

                # update the weights
                self.optimizer.step()

                # calculate predicted class labels
                ypred = F.softmax(outputs, dim=1).argmax(dim=1)

                # calculate accuracy on current batch
                observed_accuracy = accuracy_function(ypred, labels)
                self.training_state['ta'][batch, epoch] = observed_accuracy

                # print progress
                LOGGER.info('Epoch: {:d}/{:d}, Mini-batch: {:d}/{:d}, '
                            'Loss: {:.2f}, Accuracy: {:.2f}'.format(
                                epoch + 1,
                                self.epochs,
                                batch + 1,
                                len(self.train_dl),
                                observed_loss,
                                observed_accuracy))

            # update the number of epochs trained
            self.model.epoch += 1

            # whether to evaluate model performance on the validation set and
            # early stop the training process
            if self.early_stop:

                # model predictions on the validation set
                vacc, vloss = self.predict()

                # append observed accuracy and loss to arrays
                self.training_state['va'][:, epoch] = vacc.squeeze()
                self.training_state['vl'][:, epoch] = vloss.squeeze()

                # metric to assess model performance on the validation set
                epoch_acc = vacc.squeeze().mean()

                # whether the model improved with respect to the previous epoch
                if self.es.increased(epoch_acc, self.max_accuracy, self.delta):
                    self.max_accuracy = epoch_acc

                    # save model state if the model improved with
                    # respect to the previous epoch
                    self.save_state()

                # whether the early stopping criterion is met
                if self.es.stop(epoch_acc):
                    break

            else:
                # if no early stopping is required, the model state is
                # saved after each epoch
                self.save_state()

        return self.training_state

    def predict(self):
        """Model inference at training time.

        Returns
        -------
        accuracies : :py:class:`numpy.ndarray`
            The mean model prediction accuracy on each mini-batch in the
            validation set.
        losses : :py:class:`numpy.ndarray`
            The model loss for each mini-batch in the validation set.

        """
        # set the model to evaluation mode
        LOGGER.info('Setting model to evaluation mode ...')
        self.model.eval()

        # create arrays of the observed losses and accuracies
        accuracies = np.zeros(shape=(len(self.valid_dl), 1))
        losses = np.zeros(shape=(len(self.valid_dl), 1))

        # iterate over the validation/test set
        LOGGER.info('Calculating accuracy on the validation set ...')
        for batch, (inputs, labels) in enumerate(self.valid_dl):

            # send the data to the gpu if available
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # calculate network outputs
            with torch.no_grad():
                outputs = self.model(inputs)

            # compute loss
            loss = self.loss_function(outputs, labels.long())
            losses[batch, 0] = loss.detach().numpy().item()

            # calculate predicted class labels
            pred = F.softmax(outputs, dim=1).argmax(dim=1)

            # calculate accuracy on current batch
            acc = accuracy_function(pred, labels)
            accuracies[batch, 0] = acc

            # print progress
            LOGGER.info('Mini-batch: {:d}/{:d}, Accuracy: {:.2f}'
                        .format(batch + 1, len(self.valid_dl), acc))

        # calculate overall accuracy on the validation/test set
        LOGGER.info('Epoch: {:d}, Mean accuracy: {:.2f}%.'
                    .format(self.model.epoch, accuracies.mean() * 100))

        return accuracies, losses

    def save_state(self):
        """Save the model state."""
        # whether to save the model state
        if self.save:

            # append the model performance before the checkpoint to the model
            # state, if a checkpoint is passed
            if self.checkpoint_state:

                # append values from checkpoint to current training state
                state = {k1: np.hstack([v1, v2]) for (k1, v1), (k2, v2) in
                         zip(self.checkpoint_state.items(),
                             self.training_state.items()) if k1 == k2}
            else:
                state = self.training_state

            # save model state
            _ = self.model.save(
                self.state_file,
                self.optimizer,
                bands=self.train_dl.dataset.dataset.use_bands,
                train_ds=self.train_dl.dataset,
                valid_ds=self.valid_dl.dataset,
                test_ds=self.test_dl.dataset,
                state=state,
                )

    def __repr__(self):
        """Representation.

        Returns
        -------
        fs : `str`
            Representation string.

        """
        # representation string to print
        fs = self.__class__.__name__ + '(\n'

        # dataset
        fs += '    (dataset):\n        '
        fs += ''.join(
            repr(self.train_dl.dataset.dataset)).replace('\n', '\n        ')

        # batch size
        fs += '\n    (batch):\n        '
        fs += '- batch size: {}\n        '.format(self.train_dl.batch_size)
        fs += '- mini-batch shape (b, c, h, w): {}'.format(
            (self.train_dl.batch_size,
             len(self.train_dl.dataset.dataset.use_bands),
             self.train_dl.dataset.dataset.tile_size,
             self.train_dl.dataset.dataset.tile_size)
            )

        # dataset split
        fs += '\n    (split):'
        fs += '\n        ' + repr(self.train_dl.dataset)
        fs += '\n        ' + repr(self.valid_dl.dataset)
        fs += '\n        ' + repr(self.test_dl.dataset)

        # model
        fs += '\n    (model):\n        '
        fs += ''.join(repr(self.model)).replace('\n', '\n        ')

        # optimizer
        fs += '\n    (optimizer):\n        '
        fs += ''.join(repr(self.optimizer)).replace('\n', '\n        ')

        # early stopping
        fs += '\n    (early stop):\n        '
        fs += ''.join(repr(self.es)).replace('\n', '\n        ')

        fs += '\n)'
        return fs


class EarlyStopping(object):
    """`Early Stopping`_ algorithm.

    This implementation of the early stopping algorithm advances a counter each
    time a metric did not improve over a training epoch. If the metric does not
    improve over more than ``patience`` epochs, the early stopping criterion is
    met.

    See the :py:meth:`pysegcnn.core.trainer.NetworkTrainer.train` method for an
    example implementation.

    .. _Early Stopping:
        https://en.wikipedia.org/wiki/Early_stopping

    Attributes
    ----------
    mode : `str`
        The early stopping mode.
    best : `float`
        Best metric score.
    min_delta : `float`
        Minimum change in early stopping metric to be considered as an
        improvement.
    patience : `int`
        The number of epochs to wait for an improvement.
    is_better : `function`
        Function indicating whether the metric improved.
    early_stop : `bool`
        Whether the early stopping criterion is met.
    counter : `int`
        The counter advancing each time a metric does not improve.

    """

    def __init__(self, mode='max', best=0, min_delta=0, patience=10):
        """Initialize.

        Parameters
        ----------
        mode : `str`, optional
            The early stopping mode. Depends on the metric measuring
            performance. When using model loss as metric, use ``mode='min'``,
            however, when using accuracy as metric, use ``mode='max'``. For
            now, only ``mode='max'`` is supported. Only used if
            ``early_stop=True``. The default is `'max'`.
        best : `float`, optional
            Threshold indicating the best metric score. At instanciation, set
            ``best`` to the worst possible score of the metric. ``best`` will
            be overwritten during training. The default is `0`.
        min_delta : `float`, optional
            Minimum change in early stopping metric to be considered as an
            improvement. Only used if ``early_stop=True``. The default is `0`.
        patience : `int`, optional
            The number of epochs to wait for an improvement in the early
            stopping metric. If the model does not improve over more than
            ``patience`` epochs, quit training. Only used if
            ``early_stop=True``. The default is `10`.

        Raises
        ------
        ValueError
            Raised if ``mode`` is not either 'min' or 'max'.

        """
        # check if mode is correctly specified
        if mode not in ['min', 'max']:
            raise ValueError('Mode "{}" not supported. '
                             'Mode is either "min" (check whether the metric '
                             'decreased, e.g. loss) or "max" (check whether '
                             'the metric increased, e.g. accuracy).'
                             .format(mode))

        # mode to determine if metric improved
        self.mode = mode

        # whether to check for an increase or a decrease in a given metric
        self.is_better = self.decreased if mode == 'min' else self.increased

        # minimum change in metric to be considered as an improvement
        self.min_delta = min_delta

        # number of epochs to wait for improvement
        self.patience = patience

        # initialize best metric
        self.best = best

        # initialize early stopping flag
        self.early_stop = False

        # initialize the early stop counter
        self.counter = 0

    def stop(self, metric):
        """Advance early stopping counter.

        Parameters
        ----------
        metric : `float`
            The current metric score.

        Returns
        -------
        early_stop : `bool`
            Whether the early stopping criterion is met.

        """
        # if the metric improved, reset the epochs counter, else, advance
        if self.is_better(metric, self.best, self.min_delta):
            self.counter = 0
            self.best = metric
        else:
            self.counter += 1
            LOGGER.info('Early stopping counter: {}/{}'.format(
                self.counter, self.patience))

        # if the metric did not improve over the last patience epochs,
        # the early stopping criterion is met
        if self.counter >= self.patience:
            LOGGER.info('Early stopping criterion met, stopping training.')
            self.early_stop = True

        return self.early_stop

    def decreased(self, metric, best, min_delta):
        """Whether a metric decreased with respect to a best score.

        Measure improvement for metrics that are considered as 'better' when
        they decrease, e.g. model loss, mean squared error, etc.

        Parameters
        ----------
        metric : `float`
            The current score.
        best : `float`
            The current best score.
        min_delta : `float`
            Minimum change to be considered as an improvement.

        Returns
        -------
        `bool`
            Whether the metric improved.

        """
        return metric < best - min_delta

    def increased(self, metric, best, min_delta):
        """Whether a metric increased with respect to a best score.

        Measure improvement for metrics that are considered as 'better' when
        they increase, e.g. accuracy, precision, recall, etc.

        Parameters
        ----------
        metric : `float`
            The current score.
        best : `float`
            The current best score.
        min_delta : `float`
            Minimum change to be considered as an improvement.

        Returns
        -------
        `bool`
            Whether the metric improved.

        """
        return metric > best + min_delta

    def __repr__(self):
        """Representation.

        Returns
        -------
        fs : `str`
            Representation string.

        """
        fs = self.__class__.__name__
        fs += '(mode={}, best={:.2f}, delta={}, patience={})'.format(
            self.mode, self.best, self.min_delta, self.patience)

        return fs
