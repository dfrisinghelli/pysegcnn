"""Model configuration and training.

This module provides an end-to-end framework of dataclasses designed to train
segmentation models on image datasets.

See :py:meth:`pysegcnn.core.trainer.NetworkTrainer.init_network_trainer` for a
complete walkthrough.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import re
import dataclasses
import pathlib
import logging
import datetime
from logging.config import dictConfig

# externals
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import matplotlib.pyplot as plt

# locals
from pysegcnn.core.dataset import SupportedDatasets, ImageDataset
from pysegcnn.core.transforms import Augment
from pysegcnn.core.utils import (img2np, item_in_enum, accuracy_function,
                                 reconstruct_scene)
from pysegcnn.core.split import SupportedSplits, CustomSubset, SceneSubset
from pysegcnn.core.models import (SupportedModels, SupportedOptimizers,
                                  SupportedLossFunctions, Network)
from pysegcnn.core.uda import SupportedUdaMethods, CoralLoss, UDA_POSITIONS
from pysegcnn.core.layers import Conv2dSame
from pysegcnn.core.logging import log_conf
from pysegcnn.core.graphics import (plot_loss, plot_confusion_matrix,
                                    plot_sample, Animate)
from pysegcnn.core.constants import map_labels
from pysegcnn.main.config import HERE, DRIVE_PATH

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
    merge_labels : `dict` [`str`, `str`], optional
        The label mapping dictionary, where each (key, value) pair represents a
        distinct label mapping. The keys are the labels to be mapped and the
        values are the corresponding labels to be mapped to. The default is
        `{}`, which means each label is preserved as is.
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
    transforms: list = dataclasses.field(default_factory=[])
    pad: bool = False
    merge_labels: dict = dataclasses.field(default_factory={})

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
                    gt_pattern=self.gt_pattern,
                    merge_labels=self.merge_labels
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
    cla_loss : `str`
        The name of the loss function measuring the model error.
    uda_loss : `str`
        The name of the unsupervised domain adaptation loss.
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
    supervised : `bool`
        Whether to fine-tune a pretrained model (True) or whether to train a
        model via unsupervised domain adapation methods (False). The default is
        `False`. Only used if ``transfer=True``.
    pretrained_model : `str`
        The name of the pretrained model to use for transfer learning.
        The default is `''`, i.e. do not use a pretrained model.
    uda_from_pretrained : `bool`
        Whether to start domain adaptation from ``pretrained_model``. The
        default is `False`, i.e. train from scratch.
    uda_lambda : `float`
        The weight of the domain adaptation, trading off adaptation with
        classification accuracy on the source domain.
    uda_pos : `str`
        The layer where to compute the domain adaptation loss.
    freeze : `bool`
        Whether to freeze the pretrained weights.
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
    cla_loss_class : :py:class:`torch.nn.Module`
        A subclass of :py:class:`torch.nn.Module`
    uda_loss_class : :py:class:`torch.nn.Module`
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
    cla_loss: str
    uda_loss: str = ''
    skip_connection: bool = True
    kwargs: dict = dataclasses.field(
        default_factory=lambda: {'kernel_size': 3, 'stride': 1, 'dilation': 1})
    batch_size: int = 64
    checkpoint: bool = False
    transfer: bool = False
    supervised: bool = False
    pretrained_model: str = ''
    uda_from_pretrained: bool = False
    uda_lambda: float = 0.5
    uda_pos: str = 'enc'
    freeze: bool = True
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
            Raised if the model ``model_name``, the optimizer ``optim_name``,
            the loss function ``cla_loss`` or the domain adaptation loss
            ``uda_loss`` is not supported.

        """
        # check input types
        super().__post_init__()

        # check whether the model is currently supported
        self.model_class = item_in_enum(self.model_name, SupportedModels)

        # check whether the optimizer is currently supported
        self.optim_class = item_in_enum(self.optim_name, SupportedOptimizers)

        # check whether the loss function is currently supported
        self.cla_loss_class = item_in_enum(self.cla_loss,
                                           SupportedLossFunctions)

        # check whether the domain adaptation loss is currently supported
        if self.transfer and not self.supervised:
            self.uda_loss_class = item_in_enum(self.uda_loss,
                                               SupportedUdaMethods)

            # check whether the position to compute the domain adaptation loss
            # is supported
            if self.uda_pos not in UDA_POSITIONS:
                raise ValueError('Position {} to compute domain adaptation '
                                 'loss is not supported. Valid positions are '
                                 '{}.'.format(self.uda_pos,
                                              ', '.join(UDA_POSITIONS)))

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

    def init_cla_loss_function(self):
        """Instanciate the classification loss function.

        Returns
        -------
        cla_loss_function : :py:class:`torch.nn.Module`
            An instance of :py:class:`torch.nn.Module`.

        """
        LOGGER.info('Classification loss function: {}.'
                    .format(repr(self.cla_loss_class)))

        # instanciate the classification loss function
        cla_loss_function = self.cla_loss_class()

        return cla_loss_function

    def init_uda_loss_function(self, uda_lambda):
        """Instanciate the domain adaptation loss function.

        Returns
        -------
        uda_loss_function : :py:class:`torch.nn.Module`
            An instance of :py:class:`torch.nn.Module`.

        """
        LOGGER.info('Domain adaptation loss function: {}.'
                    .format(repr(self.uda_loss_class)))

        # instanciate the loss function
        uda_loss_function = self.uda_loss_class(uda_lambda)

        return uda_loss_function

    def init_model(self, ds, state_file):
        """Instanciate the model and the optimizer.

        If the model checkpoint ``state_file`` exists, the pretrained model and
        optimizer states are loaded, otherwise the model and the optimizer are
        initialized from scratch.

        Parameters
        ----------
        ds : :py:class:`pysegcnn.core.dataset.ImageDataset`
            The target domain dataset.
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
            is the model loss and accuracy time series.

        """
        # write an initialization string to the log file
        LogConfig.init_log('{}: Initializing model run. ')

        # copies of configuration variables to avoid boilerplate code
        checkpoint = self.checkpoint
        transfer = self.transfer

        # initialize checkpoint state, i.e. no model checkpoint
        checkpoint_state = {}

        # check whether to load a model checkpoint
        if checkpoint:
            try:
                # load model checkpoint
                model, optimizer, model_state = Network.load(state_file)
                checkpoint_state = self.load_checkpoint(model_state)

            except FileNotFoundError:
                LOGGER.warning('Checkpoint for model {} does not exist. '
                               'Initializing new model.'
                               .format(state_file.name))
                checkpoint = False

        # case (1): initialize a model for transfer learning
        if transfer and not checkpoint:

            # check whether to fine-tune (supervised) or adapt (unsupervised)
            if self.supervised:
                # load pretrained model to fine-tune on new dataset
                LOGGER.info('Loading pretrained model for supervised transfer '
                            'learning from: {}'.format(self.pretrained_path))
                model, optimizer, model_state = self.transfer_model(
                    self.pretrained_path, ds, self.freeze)
            else:
                # adapt a pretrained model to the target domain
                if self.uda_from_pretrained:
                    LOGGER.info('Loading pretrained model for unsupervised '
                                'domain adaptation from: {}'
                                .format(self.pretrained_path))
                    model, optimizer, model_state = Network.load(
                        self.pretrained_path)
                else:
                    transfer = False

        # case (2): initialize a model to train on the source/target domain
        #           only
        if not transfer and not checkpoint:

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

        # initialize the optimizer
        optimizer = self.init_optimizer(model)

        return model, optimizer, checkpoint_state

    @staticmethod
    def load_checkpoint(model_state):
        """Load an existing model checkpoint.

        Load the pretrained model loss and accuracy time series.

        Parameters
        ----------
        model_state : `dict`
            A dictionary containing the model and optimizer state.

        Returns
        -------
        checkpoint_state : `dict` [`str`, :py:class:`numpy.ndarray`]
            The model checkpoint loss and accuracy time series.

        """
        # load model loss and accuracy

        # get all non-zero elements, i.e. get number of epochs trained
        # before the early stop
        checkpoint_state = {k: v[np.nonzero(v)].reshape(v.shape[0], -1)
                            for k, v in model_state['state'].items()}

        return checkpoint_state

    @staticmethod
    def transfer_model(state_file, ds, freeze=True):
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
        freeze : `bool`, optional
            Whether to freeze the pretrained weights. If `True`, only the last
            layer (classification layer) is trained. The default is `True`.

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
        optimizer : :py:class:`torch.optim.Optimizer`
           The optimizer used to train the model.
        model_state : `dict`
            A dictionary containing the model and optimizer state.

        """
        # check input type
        if not isinstance(ds, ImageDataset):
            raise TypeError('Expected "ds" to be {}.'
                            .format('.'.join([ImageDataset.__module__,
                                              ImageDataset.__name__])))

        # load the pretrained model
        model, optimizer, model_state = Network.load(state_file)
        LOGGER.info('Configuring model for new dataset: {}.'.format(
            ds.__class__.__name__))

        # check whether the current dataset uses the correct spectral bands
        if ds.use_bands != model_state['bands']:
            raise ValueError('The model was trained with bands {}, not with '
                             'bands {}.'
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

        # whether to freeze the pretrained model weigths
        if freeze:
            LOGGER.info('Freezing pretrained model weights ...')
            model.freeze()

        # adjust the classification layer to the classes of the new dataset
        model.classifier = Conv2dSame(in_channels=filters[0],
                                      out_channels=model.nclasses,
                                      kernel_size=1)

        return model, optimizer, model_state


@dataclasses.dataclass
class StateConfig(BaseConfig):
    """Model state configuration class.

    Generate the model state filename according to the following naming
    convention:

    `model_dataset_optimizer_splitmode_splitparams_tilesize_batchsize_bands.pt`

    Attributes
    ----------
    src_dc : :py:class:`pysegcnn.core.trainer.DatasetConfig`
        The source domain dataset configuration.
        An instance of :py:class:`pysegcnn.core.trainer.DatasetConfig`.
    src_sc : :py:class:`pysegcnn.core.trainer.SplitConfig`
        The source domain dataset split configuration.
        An instance of :py:class:`pysegcnn.core.trainer.SplitConfig`.
    trg_dc : :py:class:`pysegcnn.core.trainer.DatasetConfig`
        The target domain dataset configuration.
        An instance of :py:class:`pysegcnn.core.trainer.DatasetConfig`.
    trg_sc : :py:class:`pysegcnn.core.trainer.SplitConfig`
        The target domain dataset split configuration.
        An instance of :py:class:`pysegcnn.core.trainer.SplitConfig`.
    mc : :py:class:`pysegcnn.core.trainer.ModelConfig`
        The model configuration.
        An instance of :py:class:`pysegcnn.core.trainer.SplitConfig`.

    """

    src_dc: DatasetConfig
    src_sc: SplitConfig
    trg_dc: DatasetConfig
    trg_sc: SplitConfig
    mc: ModelConfig

    def __post_init__(self):
        """Check the type of each argument.

        Raises
        ------
        ValueError
            Raised if the spectral bands of the source ``src_dc`` and target
            ``trg_dc`` datasets are not equal.

        """
        super().__post_init__()

        # base model state filename
        # Model_Dataset_SplitMode_SplitParams_TileSize_BatchSize_Bands
        self.state_file = '{}_{}_{}Split_{}_t{}_b{}_{}.pt'

        # check that the spectral bands are the same for both source and target
        # domains
        if self.src_dc.bands != self.trg_dc.bands:
            raise ValueError('Spectral bands of the source and target domain '
                             'have to be equal: \n source: {} \n target: {}'
                             .format(', '.join(self.src_dc.bands),
                                     ', '.join(self.trg_dc.bands))
                             )

    def init_state(self):
        """Generate the model state filename.

        Returns
        -------
        state : :py:class:`pathlib.Path`
            The path to the model state file.

        """
        # state file name for model trained on the source domain only
        state_src = self._format_state_file(
                self.state_file, self.src_dc, self.src_sc, self.mc)

        # check whether the model is trained on the source domain only
        if not self.mc.transfer:
            # state file for models trained only on the source domain
            state = state_src
        else:
            # state file for model trained on target domain
            state_trg = self._format_state_file(
                self.state_file, self.trg_dc, self.trg_sc, self.mc)

            # check whether a pretrained model is used to fine-tune to the
            # target domain
            if self.mc.supervised:
                # state file for models fine-tuned to target domain
                state = state_trg.replace('.pt', '_pretrained_{}'.format(
                    self.mc.pretrained_model))
            else:
                # state file for models trained via unsupervised domain
                # adaptation
                state = state_src.replace('.pt', '_uda_{}{}'.format(
                    self.mc.uda_pos, state_trg.replace(self.mc.model_name, ''))
                    )

                # check whether unsupervised domain adaptation is initialized
                # from a pretrained model state
                if self.mc.uda_from_pretrained:
                    state.replace('.pt', '_pretrained_{}'.format(
                        self.mc.pretrained_model))

        # path to model state
        state = self.mc.state_path.joinpath(state)

        return state

    def _format_state_file(self, state_file, dc, sc, mc):
        """Format base model state filename.

        Parameters
        ----------
        state_file : `str`
            The base model state filename.
        dc : :py:class:`pysegcnn.core.trainer.DatasetConfig`
            The domain dataset configuration.
        sc : :py:class:`pysegcnn.core.trainer.SplitConfig`
            The domain dataset split configuration.
        mc : :py:class:`pysegcnn.core.trainer.ModelConfig`
            The model configuration.

        Returns
        -------
        file : `str`
            The formatted model state filename.

        """
        # get the band numbers
        if dc.bands:
            bformat = ''.join(band[0] +
                              str(dc.dataset_class.get_sensor().
                                  __members__[band].value)
                              for band in dc.bands)
        else:
            bformat = 'all'

        # check which split mode was used
        if sc.split_mode == 'date':
            # store the date of the split
            split_params = sc.date
        else:
            # store the random split parameters
            split_params = 's{}_t{}v{}'.format(
                dc.seed, str(sc.ttratio).replace('.', ''),
                str(sc.tvratio).replace('.', ''))

        # model state filename
        file = state_file.format(mc.model_name,
                                 dc.dataset_class.__name__ +
                                 '_m{}'.format(len(dc.merge_labels)),
                                 sc.split_mode.capitalize(),
                                 split_params,
                                 dc.tile_size,
                                 mc.batch_size,
                                 bformat)

        return file


@dataclasses.dataclass
class NetworkInference(BaseConfig):
    """Model inference configuration.

    Evaluate a model.

    Attributes
    ----------
    state_file : :py:class:`pathlib.Path`
        Path to the model to evaluate.
    implicit : `bool`
        Whether to evaluate the model on the datasets defined at training time.
    domain : `str`
        Whether to evaluate on the source domain (``domain='src'``), i.e. the
        domain the model in ``state_file`` was trained on, or a target domain
        (``domain='trg'``).
    test : `bool` or `None`
        Whether to evaluate the model on the training(``test=None``), the
        validation (``test=False``) or the test set (``test=True``).
    map_labels : `bool`
        Whether to map the model labels from the model source domain to the
        defined ``domain`` in case the domain class labels differ.
    ds : `dict`
        The dataset configuration dictionary passed to
        :py:class:`pysegcnn.core.trainer.DatasetConfig` when evaluating on
        an explicitly defined dataset, i.e. ``implicit=False``.
    ds_split : `dict`
        The dataset split configuration dictionary passed to
        :py:class:`pysegcnn.core.trainer.SplitConfig` when evaluating on
        an explicitly defined dataset, i.e. ``implicit=False``.
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
        the scenes in the train/validation/test dataset. Only works if
        ``predict_scene=True`` and ``plot_scene=True``.
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
    kwargs : `dict`
        Keyword arguments for :py:func:`pysegcnn.core.graphics.plot_sample`

    """

    state_file: pathlib.Path
    implicit: bool
    domain: str
    test: object
    map_labels: bool
    ds: dict = dataclasses.field(default_factory={})
    ds_split: dict = dataclasses.field(default_factory={})
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
        ValueError
            Raised if ``domain`` is not 'src' or 'trg'.

        """
        super().__post_init__()

        # check whether the test input parameter is correctly specified
        if self.test not in [None, False, True]:
            raise TypeError('Expected "test" to be None, True or False, got '
                            '{}.'.format(self.test))

        # check whether the domain is correctly specified
        if self.domain not in ['src', 'trg']:
            raise ValueError('Expected "domain" to be "src" or "trg", got {}.'
                             .format(self.domain))

        # the device to compute on, use gpu if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else
                                   "cpu")

        # the output paths for the different graphics
        self.base_path = pathlib.Path(HERE)
        self.sample_path = self.base_path.joinpath('_samples')
        self.scenes_path = self.base_path.joinpath('_scenes')
        self.perfmc_path = self.base_path.joinpath('_graphics')
        self.animtn_path = self.base_path.joinpath('_animations')

        # input path for model state files
        self.models_path = self.base_path.joinpath('_models')
        self.state_file = self.models_path.joinpath(self.state_file)

        # initialize logging
        log = LogConfig(self.state_file)
        dictConfig(log_conf(log.log_file))
        log.init_log('{}: ' + 'Evaluating model: {}.'
                     .format(self.state_file.name))

        # plotting keyword arguments
        self.kwargs = {'bands': self.plot_bands,
                       'alpha': self.alpha,
                       'figsize': self.figsize}

        # base filename for each plot
        self.basename = self.state_file.stem

        # load the model state
        self.model, _, self.model_state = Network.load(self.state_file)

        # load the target dataset: dataset to evaluate the model on
        self.trg_ds = self.load_dataset()

        # load the source dataset: dataset the model was trained on
        self.src_ds = self.model_state['src_train_dl'].dataset.dataset

        # create a figure to use for plotting
        self.fig, _ = plt.subplots(1, 3, figsize=self.kwargs['figsize'])

        # check if the animate parameter is correctly specified
        if self.animate:
            if not self.plot:
                LOGGER.warning('animate requires plot_scenes=True or '
                               'plot_samples=True. Can not create animation.')
            else:
                # check whether the output path is valid
                if not self.anim_path.exists():
                    # create output path
                    self.anim_path.mkdir(parents=True, exist_ok=True)
                    self.anim = Animate(self.anim_path)

    @staticmethod
    def get_scene_tiles(ds, scene_id):
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
            Raised if ``scene_id`` is not a valid scene identifier for the
            dataset ``ds``.

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
            raise ValueError('{} is not a valid scene identifier'
                             .format(scene_id))

        # iterate over the scenes of the dataset
        indices = []
        for i, scene in enumerate(ds.scenes):
            # if the scene id matches a given id, save the index of the scene
            if scene['id'] == scene_id:
                indices.append(i)

        return indices, scene_meta['date']

    @staticmethod
    def replace_dataset_path(ds, drive_path):
        """Replace the path to the datasets.

        Useful to evaluate models on machines, that are different from the
        machine the model was trained on.

        .. important::

            This function assumes that the datasets are stored in a directory
            named "Datasets" on each machine.

        See ``DRIVE_PATH`` in :py:mod:`pysegcnn.main.config`.

        Parameters
        ----------
        ds : :py:class:`pysegcnn.core.split.CustomSubset`
            A subset of an instance of
            :py:class:`pysegcnn.core.dataset.ImageDataset`.
        drive_path : `str`
            Base path to the datasets on the current machine. ``drive_path``
            should end with `'Datasets'`.

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

                    # drive path: match path before "Datasets"
                    # dpath = re.search('^(.*)(?=(/.*Datasets))', v)

                    # drive path: match path up to "Datasets"
                    dpath = re.search('^(.*?Datasets)', v)[0]

                    # replace drive path
                    if dpath != drive_path:
                        scene[k] = v.replace(str(dpath), drive_path)

    def load_dataset(self):
        """Load the defined dataset.

        Raises
        ------
        ValueError
            Raised if the requested dataset was not available at training time,
            if ``implicit=True``.

            Raised if the dataset ``ds`` does not have the same spectral bands
            as the model to evaluate, if ``implicit=False``.

        Returns
        -------
        ds : :py:class:`pysegcnn.core.split.CustomSubset`
            The dataset to evaluate the model on.

        """
        # check whether to evaluate on the datasets defined at training time
        if self.implicit:

            # check whether to evaluate the model on the training, validation
            # or test set
            if self.test is None:
                ds_set = 'train'
            else:
                ds_set = 'test' if self.test else 'valid'

            # the dataset to evaluate the model on
            ds = self.model_state[
                self.domain + '_{}_dl'.format(ds_set)].dataset
            if ds is None:
                raise ValueError('Requested dataset "{}" is not available.'
                                 .format(self.domain + '_{}_dl'.format(ds_set))
                                 )

            # log dataset representation
            LOGGER.info('Evaluating on {} set of the {} domain defined at '
                        'training time.'.format(ds_set, self.domain))

        else:
            # explicitly defined dataset
            ds = DatasetConfig(**self.ds).init_dataset()

            # check if the spectral bands match
            if ds.use_bands != self.model_state['bands']:
                raise ValueError('The model was trained with bands {}, not '
                                 'with bands {}.'.format(
                                     self.model_state['bands'], ds.use_bands))

            # split configuration
            sc = SplitConfig(**self.ds_split)
            train_ds, valid_ds, test_ds = sc.train_val_test_split(ds)

            # check whether to evaluate the model on the training, validation
            # or test set
            if self.test is None:
                ds = train_ds
            else:
                ds = test_ds if self.test else valid_ds

            # log dataset representation
            LOGGER.info('Evaluating on {} set of explicitly defined dataset: '
                        '\n {}'.format(ds.name, repr(ds.dataset)))

        # check the dataset path: replace by path on current machine
        self.replace_dataset_path(ds, DRIVE_PATH)

        return ds

    @property
    def source_labels(self):
        """Class labels of the source domain the model was trained on.

        Returns
        -------
        source_labels : `dict` [`int`, `dict`]
            The class labels of the source domain.

        """
        return self.src_ds.labels

    @property
    def target_labels(self):
        """Class labels of the dataset to evaluate.

        Returns
        -------
        target_labels :  `dict` [`int`, `dict`]
            The class labels of the dataset to evaluate.

        """
        return self.trg_ds.dataset.labels

    @property
    def label_map(self):
        """Label mapping from the source to the target domain.

        See :py:func:`pysegcnn.core.constants.map_labels`.

        Returns
        -------
        label_map : `dict` [`int`, `int`]
            Dictionary with source labels as keys and corresponding target
            labels as values.

        """
        # check whether the source domain labels are the same as the target
        # domain labels
        return map_labels(self.src_ds.get_labels(),
                          self.trg_ds.dataset.get_labels())

    @property
    def source_is_target(self):
        """Whether the source and target domain labels are the same.

        Returns
        -------
        source_is_target : `bool`
            `True` if the source and target domain labels are the same, `False`
            if not.

        """
        return self.label_map is None

    @property
    def apply_label_map(self):
        """Whether to map source labels to target labels.

        Returns
        -------
        apply_label_map : `bool`
            `True` if source and target labels differ and label mapping is
            requested, `False` otherwise.

        """
        return not self.source_is_target and self.map_labels

    @property
    def use_labels(self):
        """Labels to be predicted.

        Returns
        -------
        use_labels : `dict` [`int`, `dict`]
            The labels of the classes to be predicted.

        """
        return (self.target_labels if self.apply_label_map else
                self.source_labels)

    @property
    def bands(self):
        """Spectral bands the model was trained with.

        Returns
        -------
        bands : `list` [`str`]
            A list of the named spectral bands used to train the model.

        """
        return self.src_ds.use_bands

    @property
    def compute_cm(self):
        """Whether to compute the confusion matrix.

        Returns
        -------
        compute_cm : `bool`
            Whether the confusion matrix can be computed. For datasets with
            labels different from the source domain labels, the confusion
            matrix can not be computed.

        """
        return (False if not self.source_is_target and not self.map_labels
                else self.cm)

    @property
    def plot(self):
        """Whether to save plots of (input, ground truth, prediction).

        Returns
        -------
        plot : `bool`
            Save plots for each sample or for each scene of the target dataset,
            depending on ``self.predict_scene``.

        """
        return self.plot_scenes if self.predict_scene else self.plot_samples

    @property
    def is_scene_subset(self):
        """Check the type of the target dataset.

        Whether ``self.trg_ds`` is an instance of
        :py:class:`pysegcnn.core.split.SceneSubset`, as required when
        ``self.predict_scene=True``.

        Returns
        -------
        is_scene_subset : `bool`
            Whether ``self.trg_ds`` is an instance of
            :py:class:`pysegcnn.core.split.SceneSubset`.

        """
        return isinstance(self.trg_ds, SceneSubset)

    @property
    def dataloader(self):
        """Dataloader instance for model inference.

        Returns
        -------
        dataloader : :py:class:`torch.utils.data.DataLoader`
            The dataset for model inference.

        """
        # build the dataloader for model inference
        return DataLoader(self.trg_ds, batch_size=self._batch_size,
                          shuffle=False, drop_last=False)

    @property
    def _original_source_labels(self):
        """Original source domain labels.

        Since PyTorch requires class labels to be an ascending sequence
        starting from 0, the actual class labels in the ground truth may differ
        from the class labels fed to the model.

        Returns
        -------
        original_source_labels : `dict` [`int`, `dict`]
            The original class labels of the source domain.

        """
        return self.src_ds._labels

    @property
    def _original_target_labels(self):
        """Original target domain labels.

        Returns
        -------
        original_target_labels : `dict` [`int`, `dict`]
            The original class labels of the target domain.

        """
        return self.trg_ds.dataset._labels

    @property
    def _label_log(self):
        """Log if a label mapping is applied.

        Returns
        -------
        log : `str`
            Represenation of the label mapping.

        """
        log = 'Retaining model labels ({})'.format(
            ', '.join([v['label'] for _, v in self.source_labels.items()]))
        if self.apply_label_map:
            log = (log.replace('Retaining', 'Mapping') + ' to target labels '
                   '({}).'.format(', '.join([v['label'] for _, v in
                                  self.target_labels.items()])))
        return log

    @property
    def _batch_size(self):
        """Batch size of the inference dataloader.

        Returns
        -------
        batch_size : `int`
            The batch size of the dataloader used for model inference. Depends
            on whether to predict each sample of the target dataset
            individually or whether to reconstruct each scene in the target
            dataset.

        """
        return (self.trg_ds.dataset.tiles if self.predict_scene and
                self.is_scene_subset else 1)

    def map_to_target(self, prd):
        """Map source domain labels to target domain labels.

        Parameters
        ----------
        prd : :py:class:`torch.Tensor`
            The source domain class labels as predicted by ```self.model``.

        Returns
        -------
        prd : :py:class:`torch.Tensor`
            The predicted target domain labels.

        """
        # map actual source labels to original source labels
        for aid, oid in zip(self.source_labels.keys(),
                            self._original_source_labels.keys()):
            prd[torch.where(prd == aid)] = oid

        # apply the label mapping
        for src_label, trg_label in self.label_map.items():
            prd[torch.where(prd == src_label)] = trg_label

        # map original target labels to actual target labels
        for oid, aid in zip(self._original_target_labels.keys(),
                            self.target_labels.keys()):
            prd[torch.where(prd == oid)] = aid

        return prd

    def predict(self):
        """Classify the samples of the target dataset.

        Returns
        -------
        output : `dict` [`str`, `dict`]
            The inference output dictionary. The keys are either the number of
            the samples (``self.predict_scene=False``) or the name of the
            scenes of the target dataset (``self.predict_scene=True``). The
            values are dictionaries with keys:
                ``'input'``
                    Model input data of the sample (:py:class:`numpy.ndarray`).
                ``'labels'
                    Ground truth class labels (:py:class:`numpy.ndarray`).
                ``'prediction'``
                    Model prediction class labels (:py:class:`numpy.ndarray`).

        """
        # iterate over the samples of the target dataset
        output = {}
        for batch, (inputs, labels) in enumerate(self.dataloader):

            # send inputs and labels to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # compute model predictions
            with torch.no_grad():
                prdctn = F.softmax(self.model(inputs),
                                   dim=1).argmax(dim=1).squeeze()

            # map source labels to target dataset labels
            if self.apply_label_map:
                prdctn = self.map_to_target(prdctn)

            # update confusion matrix
            if self.compute_cm:
                for ytrue, ypred in zip(labels.view(-1), prdctn.view(-1)):
                    self.conf_mat[ytrue.long(), ypred.long()] += 1

            # convert torch tensors to numpy arrays
            inputs = inputs.numpy()
            labels = labels.numpy()
            prdctn = prdctn.numpy()

            # progress string to log
            progress = 'Sample: {:d}/{:d}'.format(batch + 1,
                                                  len(self.dataloader))

            # check whether to reconstruct the scene
            date = None
            if self.dataloader.batch_size > 1:

                # id and date of the current scene
                batch = self.trg_ds.ids[batch]
                if self.trg_ds.split_mode == 'date':
                    date = self.trg_ds.dataset.parse_scene_id(batch)['date']

                # modify the progress string
                progress = progress.replace('Sample', 'Scene')
                progress += ' Id: {}'.format(batch)

                # reconstruct the entire scene
                inputs = reconstruct_scene(inputs)
                labels = reconstruct_scene(labels)
                prdctn = reconstruct_scene(prdctn)

            # save current batch to output dictionary
            output[batch] = {'input': inputs, 'labels': labels,
                             'prediction': prdctn}

            # filename for the plot of the current batch
            batch_name = self.basename + '_{}_{}.pt'.format(self.trg_ds.name,
                                                            batch)

            # in case the source and target class labels are the same or a
            # label mapping is applied, the accuracy of the prediction can be
            # calculated
            if self.source_is_target or self.apply_label_map:
                progress += ', Accuracy: {:.2f}'.format(
                    accuracy_function(prdctn, labels))
            LOGGER.info(progress)

            # plot current scene
            if self.plot:

                # in case the source and target class labels are the same or a
                # label mapping is applied, plot the ground truth, otherwise
                # just plot the model prediction
                gt = None
                if self.source_is_target or self.apply_label_map:
                    gt = labels

                # plot inputs, ground truth and model predictions
                plot_sample(inputs.clip(0, 1),
                            self.bands,
                            self.use_labels,
                            y=gt,
                            y_pred={self.model.__class__.__name__: prdctn},
                            state=batch_name,
                            date=date,
                            fig=self.fig,
                            plot_path=self.scenes_path,
                            **self.kwargs)

                # save current figure state as frame for animation
                if self.animate:
                    self.anim.frame(self.fig.axes)

        # save animation
        if self.animate:
            self.anim.animate(self.fig, interval=1000, repeat=True, blit=True)
            self.anim.save(self.basename + '_{}.gif'.format(self.trg_ds.name),
                           dpi=200)

        return output

    def evaluate(self):
        """Evaluate a pretrained model on a defined dataset.

        Returns
        -------
        output : `dict` [`str`, `dict`]
            The inference output dictionary. The keys are either the number of
            the samples (``self.predict_scene=False``) or the name of the
            scenes of the target dataset (``self.predict_scene=True``). The
            values are dictionaries with keys:
                ``'input'``
                    Model input data of the sample (:py:class:`numpy.ndarray`).
                ``'labels'
                    Ground truth class labels (:py:class:`numpy.ndarray`).
                ``'prediction'``
                    Model prediction class labels (:py:class:`numpy.ndarray`).

        """
        # plot loss and accuracy
        plot_loss(self.state_file, outpath=self.perfmc_path)

        # set the model to evaluation mode
        LOGGER.info('Setting model to evaluation mode ...')
        self.model.eval()
        self.model.to(self.device)

        # initialize confusion matrix
        self.conf_mat = np.zeros(shape=2 * (len(self.use_labels), ))

        # log which labels the model predicts
        LOGGER.info(self._label_log)

        # evaluate the model on the target dataset
        output = self.predict()

        # whether to plot the confusion matrix
        if self.compute_cm:
            plot_confusion_matrix(self.conf_mat, self.use_labels,
                                  state_file=self.state_file,
                                  outpath=self.perfmc_path)

        return output


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
class MetricTracker(BaseConfig):
    """Log training metrics.

    Attributes
    ----------
    train_metrics : `list` [`str`]
        List of metric names on the training dataset.
    valid_metrics : `list` [`str`]
        List of metric names on the validation dataset.
    metrics : `list` [`str`]
        Union of ``train_metrics`` and ``valid_metrics``.
    state : `dict` [`str`, `list`]
        Dictionary of the logged metrics. The keys are ``metrics`` and the
        values are lists of the corresponding metric observed during training.

    """

    train_metrics: list = dataclasses.field(default_factory=['train_loss',
                                                             'train_accu'])
    valid_metrics: list = dataclasses.field(default_factory=['valid_loss',
                                                             'valid_accu'])

    def __post_init__(self):
        """Check the type of each argument."""
        super().__post_init__()

    def initialize(self):
        """Store the metrics as instance attributes."""
        # initialize the metrics
        self.metrics = self.train_metrics + self.valid_metrics
        for metric in self.metrics:
            setattr(self, str(metric), [])

    def update(self, metric, value):
        """Update a metric.

        Parameters
        ----------
        metric : `str`
            Name of a metric in ``metrics``.
        value : `float` or `list` [`float`]
            The observed value(s) of ``metric``.

        """
        if isinstance(value, list):
            getattr(self, str(metric)).extend(value)
        else:
            getattr(self, str(metric)).append(value)

    def batch_update(self, metrics, values):
        """Update a list of metrics.

        Parameters
        ----------
        metrics : `list` [`str`]
            List of metric names.
        values : `list` [`float`] or `list` [`list` [`float`]]
            The corresponfing observed values of ``metrics``.

        """
        for metric, value in zip(metrics, values):
            self.update(metric, value)

    @property
    def state(self):
        """Return a dictionary of the logged metrics.

        Returns
        -------
        state : `dict` [`str`, `list`]
            Dictionary of the logged metrics. The keys are ``metrics`` and the
            values are lists of the corresponding metric observed during
            training.

        """
        return {k: getattr(self, k) for k in self.metrics}

    def np_state(self, tmbatch, vmbatch):
        """Return a dictionary of the logged metrics.

        Parameters
        ----------
        tmbatch : `int`
            Number of mini-batches in the training dataset.
        vmbatch : `int`
            Number of mini-batches in the validation dataset.

        Returns
        -------
        state : `dict` [`str`, :py:class:`numpy.ndarray`]
            Dictionary of the logged metrics. The keys are ``metrics`` and the
            values are :py:class:`numpy.ndarray`'s of the corresponding metric
            observed during training with shape=(mini_batch, epoch).

        """
        return {**{k: np.asarray(getattr(self, k)).reshape(tmbatch, -1,
                                                           order='F')
                   for k in self.train_metrics},
                **{k: np.asarray(getattr(self, k)).reshape(vmbatch, -1,
                                                           order='F')
                   for k in self.valid_metrics}}


@dataclasses.dataclass
class NetworkTrainer(BaseConfig):
    """Model training class.

    Train an instance of :py:class:`pysegcnn.core.models.Network` on a dataset
    of type :py:class:`pysegcnn.core.dataset.ImageDataset`.

    Supports training a model on a single source domain only and on a source
    and target domain using deep domain adaptation.

    Attributes
    ----------
    model : :py:class:`pysegcnn.core.models.Network`
        The model to train. An instance of
        :py:class:`pysegcnn.core.models.Network`.
    optimizer : :py:class:`torch.optim.Optimizer`
        The optimizer to update the model weights. An instance of
        :py:class:`torch.optim.Optimizer`.
    state_file : :py:class:`pathlib.Path`
        Path to save the model state.
    src_train_dl : :py:class:`torch.utils.data.DataLoader`
        The source domain training :py:class:`torch.utils.data.DataLoader`
        instance build from an instance of
        :py:class:`pysegcnn.core.split.CustomSubset`.
    src_valid_dl : :py:class:`torch.utils.data.DataLoader`
        The source domain validation :py:class:`torch.utils.data.DataLoader`
        instance build from an instance of
        :py:class:`pysegcnn.core.split.CustomSubset`.
    src_test_dl : :py:class:`torch.utils.data.DataLoader`
        The source domain test :py:class:`torch.utils.data.DataLoader`
        instance build from an instance of
        :py:class:`pysegcnn.core.split.CustomSubset`.
    cla_loss_function : :py:class:`torch.nn.Module`
        The classification loss function to compute the model error. An
        instance of :py:class:`torch.nn.Module`.
    trg_train_dl : :py:class:`torch.utils.data.DataLoader`
        The target domain training :py:class:`torch.utils.data.DataLoader`
        instance build from an instance of
        :py:class:`pysegcnn.core.split.CustomSubset`. The default is an empty
        :py:class:`torch.utils.data.DataLoader`.
    trg_valid_dl : :py:class:`torch.utils.data.DataLoader`
        The target domain validation :py:class:`torch.utils.data.DataLoader`
        instance build from an instance of
        :py:class:`pysegcnn.core.split.CustomSubset`. The default is an empty
        :py:class:`torch.utils.data.DataLoader`.
    trg_test_dl : :py:class:`torch.utils.data.DataLoader`
        The target domain test :py:class:`torch.utils.data.DataLoader`
        instance build from an instance of
        :py:class:`pysegcnn.core.split.CustomSubset`. The default is an empty
        :py:class:`torch.utils.data.DataLoader`.
    uda_loss_function : :py:class:`torch.nn.Module`
        The domain adaptation loss function. An instance of
        :py:class:`torch.nn.Module`.
        The default is :py:class:`pysegcnn.core.uda.CoralLoss`.
    uda_lambda : `float`
        The weight of the domain adaptation, trading off adaptation with
        classification accuracy on the source domain. The default is `0`.
    uda_pos : `str`
        The layer where to compute the domain adaptation loss. The default
        is `'enc'`, i.e. compute the domain adaptation loss using the output of
        the model encoder.
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
        should be a dictionary with keys describing the training metric.
        The default is `{}`.
    save : `bool`
        Whether to save the model state to ``state_file``. The default is
        `True`.
    device : `str`
        The device to train the model on, i.e. `cpu` or `cuda`.
    tracker : :py:class:`pysegcnn.core.trainer.MetricTracker`
        A :py:class:`pysegcnn.core.trainer.MetricTracker` instance tracking
        training metrics, i.e. loss and accuracy.
    uda : `bool`
        Whether to apply deep domain adaptation.
    max_accuracy : `float`
        Maximum accuracy of ``model`` on the validation dataset.
    es : `None` or :py:class:`pysegcnn.core.trainer.EarlyStopping`
        The early stopping instance if ``early_stop=True``, else `None`.
    tmbatch : `int`
        Number of mini-batches in the training dataset.
    vmbatch : `int`
        Number of mini-batches in the validation dataset.
    bands : `list` [`str`]
        The spectral bands used to train ``model``.
    training_state : `dict` [`str`, :py:class:`numpy.ndarray`]
        The training state dictionary. The keys describe the type of the
        training metric.

    .. _Early Stopping:
        https://en.wikipedia.org/wiki/Early_stopping

    """

    model: Network
    optimizer: Optimizer
    state_file: pathlib.Path
    src_train_dl: DataLoader
    src_valid_dl: DataLoader
    src_test_dl: DataLoader
    cla_loss_function: nn.Module
    trg_train_dl: DataLoader = DataLoader(None)
    trg_valid_dl: DataLoader = DataLoader(None)
    trg_test_dl: DataLoader = DataLoader(None)
    uda_loss_function: nn.Module = CoralLoss
    uda_lambda: float = 0
    uda_pos: str = 'enc'
    epochs: int = 1
    nthreads: int = torch.get_num_threads()
    early_stop: bool = False
    mode: str = 'max'
    delta: float = 0
    patience: int = 10
    checkpoint_state: dict = dataclasses.field(default_factory={})
    save: bool = True

    def __post_init__(self):
        """Check the type of each argument.

        Configure the device to train the model on, i.e. train on the gpu if
        available.

        Configure early stopping if required.

        Initialize training metric tracking.

        """
        super().__post_init__()

        # the device to train the model on
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else
                                   'cpu')
        # set the number of threads
        torch.set_num_threads(self.nthreads)

        # send the model to the gpu if available
        self.model = self.model.to(self.device)

        # instanciate metric tracker
        self.tracker = MetricTracker(
            train_metrics=['train_loss', 'train_accu'],
            valid_metrics=['valid_loss', 'valid_accu'])

        # whether to train using supervised transfer learning or
        # deep domain adaptation

        # dummy variables for easy model evaluation
        self.uda = False
        if self.trg_train_dl.dataset is not None and self.uda_lambda > 0:

            # set the device for computing domain adaptation loss
            self.uda_loss_function.device = self.device

            # adjust metrics and initialize metric tracker
            self.tracker.train_metrics.extend(['cla_loss', 'uda_loss'])

            # train using deep domain adaptation
            self.uda = True

            # forward function for deep domain adaptation
            self.uda_forward = self._uda_frwd()

        # initialize metric tracker
        self.tracker.initialize()

        # maximum accuracy on the validation set
        self.max_accuracy = 0
        if self.checkpoint_state:
            self.max_accuracy = self.checkpoint_state['valid_accu'].mean(
                axis=0).max().item()

        # whether to use early stopping
        self.es = None
        if self.early_stop:
            self.es = EarlyStopping(self.mode, self.max_accuracy, self.delta,
                                    self.patience)

        # number of mini-batches in the training and validation sets
        self.tmbatch = len(self.src_train_dl)
        self.vmbatch = len(self.src_valid_dl)

        # the spectral bands used for training
        self.bands = self.src_train_dl.dataset.dataset.use_bands

        # log representation
        LOGGER.info(repr(self))

        # initialize training log
        LOGGER.info(35 * '-' + ' Training ' + 35 * '-')

        # log the device and number of threads
        LOGGER.info('Device: {}'.format(self.device))
        LOGGER.info('Number of cpu threads: {}'.format(self.nthreads))

    def _train_source_domain(self, epoch):
        """Train a model for an epoch on the source domain.

        Parameters
        ----------
        epoch : `int`
            The current epoch.

        """
        # iterate over the dataloader object
        for batch, (inputs, labels) in enumerate(self.src_train_dl):

            # send the data to the gpu if available
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # reset the gradients
            self.optimizer.zero_grad()

            # perform forward pass
            outputs = self.model(inputs)

            # compute loss
            loss = self.cla_loss_function(outputs, labels.long())

            # compute the gradients of the loss function w.r.t.
            # the network weights
            loss.backward()

            # update the weights
            self.optimizer.step()

            # calculate predicted class labels
            ypred = F.softmax(outputs, dim=1).argmax(dim=1)

            # calculate accuracy on current batch
            acc = accuracy_function(ypred, labels)

            # print progress
            LOGGER.info('Epoch: {:d}/{:d}, Mini-batch: {:d}/{:d}, '
                        'Loss: {:.2f}, Accuracy: {:.2f}'
                        .format(epoch + 1, self.epochs, batch + 1,
                                self.tmbatch, loss.item(), acc))

            # update training metrics
            self.tracker.batch_update(self.tracker.train_metrics,
                                      [loss.item(), acc])

    def _enc_uda(self, src_input, trg_input):

        # perform forward pass: encoded source domain features
        src_feature = self.model.encoder(src_input)
        src_dec_feature = self.model.decoder(src_feature,
                                             self.model.encoder.cache)
        # model logits on source domain
        src_prdctn = self.model.classifier(src_dec_feature)
        del self.model.encoder.cache  # clear intermediate encoder outputs

        # perform forward pass: encoded target domain features
        trg_feature = self.model.encoder(trg_input)

        return src_feature, trg_feature, src_prdctn

    def _dec_uda(self, src_input, trg_input):

        # perform forward pass: decoded source domain features
        src_feature = self.model.encoder(src_input)
        src_feature = self.model.decoder(src_feature,
                                         self.model.encoder.cache)
        # model logits on source domain
        src_prdctn = self.model.classifier(src_feature)
        del self.model.encoder.cache  # clear intermediate encoder outputs

        # perform forward pass: decoded target domain features
        trg_feature = self.model.encoder(trg_input)
        trg_feature = self.model.decoder(trg_feature,
                                         self.model.encoder.cache)
        del self.model.encoder.cache

        return src_feature, trg_feature, src_prdctn

    def _cla_uda(self, src_input, trg_input):

        # perform forward pass: classified source domain features
        src_feature = self.model(src_input)

        # perform forward pass: target domain features
        trg_feature = self.model(trg_input)

        return src_feature, trg_feature, src_feature

    def _uda_frwd(self):
        if self.uda_pos == 'enc':
            forward = self._enc_uda

        if self.uda_pos == 'dec':
            forward = self._enc_dec

        if self.uda_pos == 'cla':
            forward = self._cla_uda

        return forward

    def _train_domain_adaptation(self, epoch):
        """Train a model for an epoch on the source and target domain.

        This function implements deep domain adaptation by extending the
        standard classification loss by a "domain adaptation loss" calculated
        from unlabelled target domain samples.

        Parameters
        ----------
        epoch : `int`
            The current epoch.

        """
        # create target domain iterator
        target = iter(self.trg_train_dl)

        # increase domain adaptation weight with increasing epochs
        uda_lambda = self.uda_lambda * ((epoch + 1) / self.epochs)

        # iterate over the number of samples
        for batch, (src_input, src_label) in enumerate(self.src_train_dl):

            # get the target domain input data
            try:
                trg_input, _ = target.next()
            # in case the iterator is finished, re-instanciate it
            except StopIteration:
                target = iter(self.trg_train_dl)
                trg_input, _ = target.next()

            # send the data to the gpu if available
            src_input, src_label = (src_input.to(self.device),
                                    src_label.to(self.device))
            trg_input = trg_input.to(self.device)

            # reset the gradients
            self.optimizer.zero_grad()

            # forward pass
            src_feature, trg_feature, src_prdctn = self.uda_forward(src_input,
                                                                    trg_input)

            # compute classification loss
            cla_loss = self.cla_loss_function(src_prdctn, src_label.long())

            # compute domain adaptation loss:
            # the difference between source and target domain is computed
            # from the compressed representation of the model encoder
            uda_loss = self.uda_loss_function(src_feature, trg_feature)

            # total loss
            tot_loss = cla_loss + uda_lambda * uda_loss

            # compute the gradients of the loss function w.r.t.
            # the network weights
            tot_loss.backward()

            # update the weights
            self.optimizer.step()

            # calculate predicted class labels
            ypred = F.softmax(src_prdctn, dim=1).argmax(dim=1)

            # calculate accuracy on current batch
            acc = accuracy_function(ypred, src_label)

            # print progress
            LOGGER.info('Epoch: {:d}/{:d}, Mini-batch: {:d}/{:d}, '
                        'Cla_loss: {:.2f}, Uda_loss: {:.2f}, '
                        'Tot_loss: {:.2f}, Acc: {:.2f}'
                        .format(epoch + 1, self.epochs, batch + 1,
                                self.tmbatch, cla_loss.item(),
                                uda_loss.item(), tot_loss.item(), acc))

            # update training metrics
            self.tracker.batch_update(self.tracker.train_metrics,
                                      [tot_loss.item(), acc,
                                       cla_loss.item(), uda_loss.item()])

    def train_epoch(self, epoch):
        """Wrap the function to train a model for a single epoch.

        Depends on whether to apply deep domain adaptation.

        Parameters
        ----------
        epoch : `int`
            The current epoch.

        Returns
        -------
        `function`
            The function to train a model for a single epoch.

        """
        if self.uda:
            self._train_domain_adaptation(epoch)
        else:
            self._train_source_domain(epoch)

    def train(self):
        """Train the model.

        Returns
        -------
        training_state : `dict` [`str`, :py:class:`numpy.ndarray`]
            The training state dictionary. The keys describe the type of the
            training metric. See
            :py:meth:`~pysegcnn.core.trainer.NetworkTrainer.training_state`.

        """
        # initialize the training: iterate over the entire training dataset
        for epoch in range(self.epochs):

            # set the model to training mode
            LOGGER.info('Setting model to training mode ...')
            self.model.train()

            # train model for a single epoch
            self.train_epoch(epoch)

            # update the number of epochs trained
            self.model.epoch += 1

            # whether to evaluate model performance on the validation set and
            # early stop the training process
            if self.early_stop:

                # model predictions on the validation set
                valid_accu, valid_loss = self.predict(self.src_valid_dl)

                # update validation metrics
                self.tracker.batch_update(self.tracker.valid_metrics,
                                          [valid_loss, valid_accu])

                # metric to assess model performance on the validation set
                epoch_acc = np.mean(valid_accu)

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

    def predict(self, dataloader):
        """Model inference at training time.

        Parameters
        ----------
        dataloader : :py:class:`torch.utils.data.DataLoader`
            The validation dataloader to evaluate the model predictions.

        Returns
        -------
        accuracy : :py:class:`numpy.ndarray`
            The mean model prediction accuracy on each mini-batch in the
            validation set.
        loss : :py:class:`numpy.ndarray`
            The model loss for each mini-batch in the validation set.

        """
        # set the model to evaluation mode
        LOGGER.info('Setting model to evaluation mode ...')
        self.model.eval()

        # create arrays of the observed loss and accuracy
        accuracy = []
        loss = []

        # iterate over the validation/test set
        LOGGER.info('Calculating accuracy on the validation set ...')
        for batch, (inputs, labels) in enumerate(dataloader):

            # send the data to the gpu if available
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # calculate network outputs
            with torch.no_grad():
                outputs = self.model(inputs)

            # compute loss
            cla_loss = self.cla_loss_function(outputs, labels.long())
            loss.append(cla_loss.item())

            # calculate predicted class labels
            pred = F.softmax(outputs, dim=1).argmax(dim=1)

            # calculate accuracy on current batch
            acc = accuracy_function(pred, labels)
            accuracy.append(acc)

            # print progress
            LOGGER.info('Mini-batch: {:d}/{:d}, Accuracy: {:.2f}'
                        .format(batch + 1, len(dataloader), acc))

        # calculate overall accuracy on the validation/test set
        LOGGER.info('Epoch: {:d}, Mean accuracy: {:.2f}%.'
                    .format(self.model.epoch, np.mean(accuracy) * 100))

        return accuracy, loss

    @property
    def training_state(self):
        """Model training metrics.

        Returns
        -------
        state : `dict` [`str`, :py:class:`numpy.ndarray`]
            The training state dictionary. The keys describe the type of the
            training metric and the values are :py:class:`numpy.ndarray`'s of
            the corresponding metric observed during training with
            shape=(mini_batch, epoch).

        """
        # current training state
        state = self.tracker.np_state(self.tmbatch, self.vmbatch)

        # optional: training state of the model checkpoint
        if self.checkpoint_state:
            # prepend values from checkpoint to current training state
            state = {k1: np.hstack([v1, v2]) for (k1, v1), (k2, v2) in
                     zip(self.checkpoint_state.items(), state.items())
                     if k1 == k2}

        return state

    def save_state(self):
        """Save the model state."""
        if self.save:
            _ = self.model.save(self.state_file,
                                self.optimizer,
                                self.bands,
                                src_train_dl=self.src_train_dl,
                                src_valid_dl=self.src_valid_dl,
                                src_test_dl=self.src_test_dl,
                                trg_train_dl=self.trg_train_dl,
                                trg_valid_dl=self.trg_valid_dl,
                                trg_test_dl=self.trg_test_dl,
                                state=self.training_state,
                                uda_lambda=self.uda_lambda
                                )

    @staticmethod
    def init_network_trainer(src_ds_config, src_split_config, trg_ds_config,
                             trg_split_config, model_config):
        """Prepare network training.

        Parameters
        ----------
        src_ds_config : :py:class:`pysegcnn.core.trainer.DatasetConfig`
            The source domain dataset configuration.
        src_split_config : :py:class:`pysegcnn.core.trainer.SplitConfig`
            The source domain dataset split configuration.
        trg_ds_config : :py:class:`pysegcnn.core.trainer.DatasetConfig`
            The target domain dataset configuration..
        trg_split_config : :py:class:`pysegcnn.core.trainer.SplitConfig`
            The target domain dataset split configuration.
        model_config : :py:class:`pysegcnn.core.trainer.ModelConfig`
            The model configuration.

        Returns
        -------
        trainer : :py:class:`pysegcnn.core.trainer.NetworkTrainer`
            A network trainer instance.

        See :py:mod:`pysegcnn.main.train.py` for an example on how to
        instanciate a :py:class:`pysegcnn.core.trainer.NetworkTrainer`
        instance.

        """
        # (i) instanciate the source domain configurations
        src_dc = DatasetConfig(**src_ds_config)   # source domain dataset
        src_sc = SplitConfig(**src_split_config)  # source domain dataset split

        # (ii) instanciate the target domain configuration
        trg_dc = DatasetConfig(**trg_ds_config)   # target domain dataset
        trg_sc = SplitConfig(**trg_split_config)  # target domain dataset split

        # (iii) instanciate the model configuration
        mdlcfg = ModelConfig(**model_config)

        # (iv) instanciate the model state file
        sttcfg = StateConfig(src_dc, src_sc, trg_dc, trg_sc, mdlcfg)
        state_file = sttcfg.init_state()

        # (v) instanciate logging configuration
        logcfg = LogConfig(state_file)
        dictConfig(log_conf(logcfg.log_file))

        # (vi) instanciate the source and target domain datasets
        src_ds = src_dc.init_dataset()
        trg_ds = trg_dc.init_dataset()

        # (vii) instanciate the training, validation and test datasets and
        # dataloaders for the source domain
        (src_train_ds,
         src_valid_ds,
         src_test_ds) = src_sc.train_val_test_split(src_ds)
        (src_train_dl,
         src_valid_dl,
         src_test_dl) = src_sc.dataloaders(src_train_ds,
                                           src_valid_ds,
                                           src_test_ds,
                                           batch_size=mdlcfg.batch_size,
                                           shuffle=True, drop_last=False)

        # (viii) instanciate the loss function
        cla_loss_function = mdlcfg.init_cla_loss_function()

        # (ix) instanciate the domain adaptation loss
        uda_loss_function = mdlcfg.init_uda_loss_function(mdlcfg.uda_lambda)

        # (x) check whether to apply transfer learning
        if mdlcfg.transfer:

            # (a) instanciate the training, validation and test datasets and
            # dataloaders for the target domain
            (trg_train_ds,
             trg_valid_ds,
             trg_test_ds) = trg_sc.train_val_test_split(trg_ds)
            (trg_train_dl,
             trg_valid_dl,
             trg_test_dl) = trg_sc.dataloaders(trg_train_ds,
                                               trg_valid_ds,
                                               trg_test_ds,
                                               batch_size=mdlcfg.batch_size,
                                               shuffle=True, drop_last=False)

            # (b) instanciate the model: supervised transfer learning
            if mdlcfg.supervised:
                model, optimizer, checkpoint_state = mdlcfg.init_model(
                    trg_ds, state_file)

                # (xi) instanciate the network trainer
                trainer = NetworkTrainer(model,
                                         optimizer,
                                         state_file,
                                         trg_train_dl,
                                         trg_valid_dl,
                                         trg_test_dl,
                                         cla_loss_function,
                                         epochs=mdlcfg.epochs,
                                         nthreads=mdlcfg.nthreads,
                                         early_stop=mdlcfg.early_stop,
                                         mode=mdlcfg.mode,
                                         delta=mdlcfg.delta,
                                         patience=mdlcfg.patience,
                                         checkpoint_state=checkpoint_state,
                                         save=mdlcfg.save)

            # (c) instanciate the model: unsupervised transfer learning
            else:
                model, optimizer, checkpoint_state = mdlcfg.init_model(
                    src_ds, state_file)

                # (xi) instanciate the network trainer
                trainer = NetworkTrainer(model,
                                         optimizer,
                                         state_file,
                                         src_train_dl,
                                         src_valid_dl,
                                         src_test_dl,
                                         cla_loss_function,
                                         trg_train_dl,
                                         trg_valid_dl,
                                         trg_test_dl,
                                         uda_loss_function,
                                         mdlcfg.uda_lambda,
                                         mdlcfg.uda_pos,
                                         mdlcfg.epochs,
                                         mdlcfg.nthreads,
                                         mdlcfg.early_stop,
                                         mdlcfg.mode,
                                         mdlcfg.delta,
                                         mdlcfg.patience,
                                         checkpoint_state,
                                         mdlcfg.save)

        else:
            # (x) instanciate the model
            model, optimizer, checkpoint_state = mdlcfg.init_model(
                src_ds, state_file)

            # (xi) instanciate the network trainer
            trainer = NetworkTrainer(model,
                                     optimizer,
                                     state_file,
                                     src_train_dl,
                                     src_valid_dl,
                                     src_test_dl,
                                     cla_loss_function,
                                     epochs=mdlcfg.epochs,
                                     nthreads=mdlcfg.nthreads,
                                     early_stop=mdlcfg.early_stop,
                                     mode=mdlcfg.mode,
                                     delta=mdlcfg.delta,
                                     patience=mdlcfg.patience,
                                     checkpoint_state=checkpoint_state,
                                     save=mdlcfg.save)

        return trainer

    def _build_ds_repr(self, train_dl, valid_dl, test_dl):
        """Build the dataset representation.

        Returns
        -------
        fs : `str`
            Representation string.

        """
        # dataset configuration
        fs = '    (dataset):\n        '
        fs += ''.join(repr(train_dl.dataset.dataset)).replace('\n',
                                                              '\n' + 8 * ' ')
        fs += '\n    (batch):\n        '
        fs += '- batch size: {}\n        '.format(train_dl.batch_size)
        fs += '- mini-batch shape (b, c, h, w): {}'.format(
            ((train_dl.batch_size, len(train_dl.dataset.dataset.use_bands),) +
             2 * (train_dl.dataset.dataset.tile_size,)))

        # dataset split
        fs += '\n    (split):'
        for dl in [train_dl, valid_dl, test_dl]:
            if dl.dataset is not None:
                fs += '\n' + 8 * ' ' + repr(dl.dataset)

        return fs

    def _build_model_repr_(self):
        """Build the model representation.

        Returns
        -------
        fs : `str`
            Representation string.

        """
        # model
        fs = '\n    (model):' + '\n' + 8 * ' '
        fs += ''.join(repr(self.model)).replace('\n', '\n' + 8 * ' ')

        # optimizer
        fs += '\n    (optimizer):' + '\n' + 8 * ' '
        fs += ''.join(repr(self.optimizer)).replace('\n', '\n' + 8 * ' ')

        # early stopping
        fs += '\n    (early stop):' + '\n' + 8 * ' '
        fs += ''.join(repr(self.es)).replace('\n', '\n' + 8 * ' ')

        # domain adaptation
        if self.uda:
            fs += '\n    (adaptation)' + '\n' + 8 * ' '
            fs += repr(self.uda_loss_function).replace('\n', '\n' + 8 * ' ')

        return fs

    def __repr__(self):
        """Representation.

        Returns
        -------
        fs : `str`
            Representation string.

        """
        # representation string to print
        fs = self.__class__.__name__ + '(\n'

        # source domain
        fs += '    (source domain)\n    '
        fs += self._build_ds_repr(
            self.src_train_dl, self.src_valid_dl, self.src_test_dl).replace(
                '\n', '\n' + 4 * ' ')

        # target domain
        if self.uda:
            fs += '\n    (target domain)\n    '
            fs += self._build_ds_repr(
                self.trg_train_dl,
                self.trg_valid_dl,
                self.trg_test_dl).replace('\n', '\n' + 4 * ' ')

        # model configuration
        fs += self._build_model_repr_()

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
