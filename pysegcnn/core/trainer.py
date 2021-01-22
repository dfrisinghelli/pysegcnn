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
from torch.utils.data import DataLoader
from torch.optim import Optimizer

# locals
from pysegcnn.core.dataset import SupportedDatasets
from pysegcnn.core.transforms import Augment
from pysegcnn.core.utils import (item_in_enum, accuracy_function,
                                 reconstruct_scene, check_filename_length,
                                 array_replace)
from pysegcnn.core.split import SupportedSplits
from pysegcnn.core.models import (SupportedModels, SupportedOptimizers,
                                  Network)
from pysegcnn.core.uda import SupportedUdaMethods, CoralLoss, UDA_POSITIONS
from pysegcnn.core.layers import Conv2dSame
from pysegcnn.core.logging import log_conf
from pysegcnn.core.graphics import (plot_loss, plot_confusion_matrix,
                                    plot_sample)
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
    tile_size: object
    gt_pattern: str
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
    k_folds: `int`, optional
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
    split_class : :py:class:`pysegcnn.core.split.RandomSplit`
        A subclass of :py:class:`pysegcnn.core.split.RandomSplit`.

    """

    split_mode: str
    k_folds: int = 1
    seed: int = 0
    shuffle: bool = True
    tvratio: float = 0.8
    ttratio: float = 1

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
        # instanciate the split class
        split = self.split_class(ds, self.k_folds, self.seed, self.shuffle,
                                 self.tvratio, self.ttratio)

        # the training, validation and test dataset
        subsets = split.split()

        return subsets

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

            # check if the dataset is not empty
            if len(ds) > 0:
                # build the dataloader
                loader = DataLoader(ds, **kwargs)
            else:
                loader = DataLoader(None)
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
    optim_name : `str`
        The name of the optimizer to update the model weights.
    torch_seed : `int`
        The random seed to initialize the model weights. Useful for
        reproducibility. The default is `0`.
    batch_size : `int`
        The model batch size. Determines the number of samples to process
        before updating the model weights. The default is `64`.
    checkpoint : `bool`
        Whether to resume training from an existing model checkpoint. The
        default is `False`.
    optim_kwargs : `dict`
        Keyword arguments passed to the optimizer. The default is `{}`,
        which is equivalent to using the defaults of a
        :py:class:`torch.optim.Optimizer`.
    early_stop : `bool`
        Whether to apply `Early Stopping`_. The default is `False`.
    mode : `str`
        The early stopping mode. Depends on the metric measuring
        performance. When using model loss as metric, use ``mode='min'``,
        however, when using accuracy as metric, use ``mode='max'``.
        Only used if ``early_stop=True``. The default is `'max'`, which means
        using the validation set accuracy as metric.
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
    state_path : :py:class:`pathlib.Path`
        Path to save model states.

    .. _Early Stopping:
        https://en.wikipedia.org/wiki/Early_stopping

    """

    model_name: str
    optim_name: str
    torch_seed: int = 0
    batch_size: int = 64
    checkpoint: bool = False
    optim_kwargs: dict = dataclasses.field(default_factory={})
    early_stop: bool = False
    mode: str = 'max'
    delta: float = 0
    patience: int = 10
    epochs: int = 50
    nthreads: int = torch.get_num_threads()
    save: bool = True

    def __post_init__(self):
        """Check the type of each argument.

        Raises
        ------
        ValueError
            Raised if the model ``model_name`` or the optimizer ``optim_name``
            are not supported.

        """
        # check input types
        super().__post_init__()

        # check whether the model is currently supported
        self.model_class = item_in_enum(self.model_name, SupportedModels)

        # check whether the optimizer is currently supported
        self.optim_class = item_in_enum(self.optim_name, SupportedOptimizers)

        # path to save model states
        self.state_path = pathlib.Path(HERE).joinpath('_models/')

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
        optimizer = self.optim_class(model.parameters(), **self.optim_kwargs)

        return optimizer

    def init_model(self, in_channels, nclasses, state_file):
        """Instanciate the model and the optimizer.

        If ``self.checkpoint`` is set to True, a model checkpoint called
        ``state_file`` is loaded, if it exists. Otherwise, the model is
        initiated from scratch on the dataset ``ds``.

        If ``self.transfer`` is True, the pretrained model in
        ``self.pretrained_path`` is adjusted to the dataset ``ds``.

        Parameters
        ----------
        in_channels : `int`
            Number of input features.
        nclasses : `int`
            Number of classes.
        state_file : :py:class:`pathlib.Path`
            Path to save the trained model.

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

        # set the random seed for reproducibility
        torch.manual_seed(self.torch_seed)
        LOGGER.info('Initializing model: {}'.format(state_file.name))

        # initialize checkpoint state, i.e. no model checkpoint
        checkpoint_state = {}

        # instanciate a model from scratch
        model = self.model_class(
            state_file=state_file, in_channels=in_channels, nclasses=nclasses)

        # initialize the optimizer
        optimizer = self.init_optimizer(model)

        # check whether to load a model checkpoint
        if self.checkpoint:
            try:
                # load model checkpoint
                model_state = Network.load(state_file)
                checkpoint_state = self.load_checkpoint(model_state)

                # load the pretrained model
                model, optimizer = Network.load_pretrained_model(state_file)

            except FileNotFoundError:
                # model checkpoint does not exist
                LOGGER.info('Checkpoint {} does not exist. Intializing model '
                            'from scratch ...')

        return model, optimizer, checkpoint_state

    @staticmethod
    def load_checkpoint(model_state):
        """Load an existing model checkpoint.

        Load the pretrained model loss and accuracy time series.

        Parameters
        ----------
        model_state : `dict`
            A dictionary containing the model and optimizer state, as
            constructed by :py:meth:`~pysegcnn.core.Network.save`.

        Returns
        -------
        checkpoint_state : `dict` [`str`, :py:class:`numpy.ndarray`]
            The model checkpoint loss and accuracy time series.

        """
        # get all non-zero elements, i.e. get number of epochs trained
        # before the early stop
        checkpoint_state = {k: v[np.nonzero(v)].reshape(v.shape[0], -1)
                            for k, v in model_state['state'].items()}

        return checkpoint_state


@dataclasses.dataclass
class TransferLearningConfig(BaseConfig):
    """Transfer learning configuration.

    Configure a pretrained model for supervised domain adaptation (fine-tuning
    of a pretrained model to a target domain) or configure a (pretrained) model
    for unsupervised domain adaptation.

    Attributes
    ----------
    transfer : `bool`
        Whether to use a model for transfer learning on a new dataset. If True,
        the model architecture of ``pretrained_model`` is adjusted to a new
        dataset. The default is `False`.
    supervised : `bool`
        Whether to train a model supervised (``supervised=True``) or
        semi-supervised (``supervised=False``). The default is `True`.
    pretrained_model : `str`
        The name of the pretrained model to use for transfer learning.
        The default is `''`, i.e. do not use a pretrained model.
    uda_loss : `str`
        The name of the unsupervised domain adaptation loss. The default is
        `'coral'`, which means using the `DeepCoral`_ algorithm, see
        :py:class:`pysegcnn.core.uda.CoralLoss`. Note that ``uda_loss`` is only
        used when ``supervised=False``.
    uda_from_pretrained : `bool`
        Whether to start domain adaptation from ``pretrained_model``. The
        default is `False`, i.e. train from scratch.
    uda_lambda : `float`
        The weight of the domain adaptation, trading off adaptation with
        classification accuracy on the source domain. The default is `0.5`.
    uda_pos : `str`
        The layer where to compute the domain adaptation loss. The default is
        `enc`, which means calculating the adaptation loss after the encoder
        layers.
    freeze : `bool`
        Whether to freeze the pretrained weights. The default is `False`.
    pretrained_path : :py:class:`pathlib.Path`
        Path to the ``pretrained_model`` used if ``transfer=True``.
    state_path : :py:class:`pathlib.Path`
        Path to save model states.
    uda_loss_class : :py:class:`torch.nn.Module`
        The domain adaptation loss function class.

    .. DeepCoral:

    """

    transfer: bool = False
    supervised: bool = True
    pretrained_model: str = ''
    uda_loss: str = 'coral'
    uda_from_pretrained: bool = False
    uda_lambda: float = 0.5
    uda_pos: str = 'enc'
    freeze: bool = True

    def __post_init__(self):
        """Check the type of each argument.

        Configure path to pretrained model and initialize unsupervised domain
        adaptation loss function.

        Raises
        ------
        ValueError
            Raised if the domain adaptation loss function ``uda_loss`` is not
            supported.
        ValueError
            Raised if the position ``uda_pos``, where the domain adaptation
            loss is computed, is not supported.

        """
        # check input types
        super().__post_init__()

        # check whether to apply transfer learning
        if not self.transfer:
            # if transfer learning is not required, training is automatically
            # set to supervised mode
            self.supervised = True

        # path to model states
        self.state_path = pathlib.Path(HERE).joinpath('_models/')

        # path to pretrained model
        self.pretrained_path = self.state_path.joinpath(self.pretrained_model)

        # domain adaptation loss function
        self.uda_loss_class = item_in_enum(self.uda_loss, SupportedUdaMethods)

        # check whether the position to compute the domain adaptation
        # loss is supported
        if self.uda_pos not in UDA_POSITIONS:
            raise ValueError('Position {} to compute domain adaptation'
                             ' loss is not supported. Valid positions '
                             'are {}.'.format(self.uda_pos,
                                              ', '.join(UDA_POSITIONS))
                             )

        # instanciate domain adaptation loss
        self.uda_loss_function = self.uda_loss_class(self.uda_lambda)

    @staticmethod
    def transfer_model(state_file, nclasses, optim_kwargs={}, freeze=False):
        """Adjust a pretrained model and optimizer to a new number of classes.

        This function is designed to work with models of class
        :py:class:`pysegcnn.core.models.ConvolutionalAutoEncoder`.

        If the number of classes in the pretrained model ``model`` does not
        match the number of classes ``nclasses``, the classification layer is
        initilialized from scratch with ``nclasses`` classes. The remaining
        model weights are preserved.

        Parameters
        ----------
        state_file : :py:class:`pathlib.Path`
            Path to the pretrained model.
        nclasses : `int`
            Number of classes to adjust the model to.
        optim_kwargs : `dict`
            Keyword arguments passed to the optimizer. The default is `{}`,
            which is equivalent to using the defaults of a
            :py:class:`torch.optim.Optimizer`.
        freeze : `bool`, optional
            Whether to freeze the pretrained weights. If `True`, all weights of
            the pretrained model ``model`` are frozen. Only the classification
            layer is retrained when it is initialized from scratch, otherwise
            its weights are also frozen. The default is `False`.

        Returns
        -------
        model : :py:class:`pysegcnn.core.models.ConvolutionalAutoEncoder`
            An instance of the (adjusted) pretrained model.
        optimizer : :py:class:`torch.optim.Optimizer`
            An instance of :py:class:`torch.optim.Optimizer`.
        checkpoint_state : `dict` [`str`, :py:class:`numpy.ndarray`]
            A dictionary describing the training state. Not required when
            training from a pretrained model. See
            :py:meth:`pysegcnn.core.trainer.ModelConfig.init_model`.

        """
        # load the pretrained model and optimizer
        model, optimizer = Network.load_pretrained_model(state_file)

        # reset model epoch to 0
        model.epoch = 0

        # whether to freeze the pretrained model weigths
        if freeze:
            LOGGER.info('Freezing pretrained model weights ...')
            model.freeze()

        # check whether the number of classes the model was trained with
        # matches the number of classes of the new dataset
        if model.nclasses != nclasses:

            # adjust the number of classes in the model
            LOGGER.info('Replacing number of classes from {} to {}.'
                        .format(model.nclasses, nclasses))
            model.nclasses = nclasses

            # adjust the classification layer to the new number of classes
            model.classifier = Conv2dSame(in_channels=model.filters[0],
                                          out_channels=model.nclasses,
                                          kernel_size=1)

        # reset the optimizer parameters
        optimizer = optimizer.__class__(model.parameters(), **optim_kwargs)

        return model, optimizer, {}


@dataclasses.dataclass
class StateConfig(BaseConfig):
    """Model state configuration class.

    Generate the model state filename according to the following naming
    conventions:

        - For source domain without domain adaptation:
            Model_Optim_SourceDataset_ModelParams.pt

        - For supervised domain adaptation to a target domain:
            NameOfPretrainedModel_sda_TargetDataset.pt

        - For unsupervised domain adaptation to a target domain:
            Model_Optim_SourceDataset_uda_TargetDataset_ModelParams.pt

        - For unsupervised domain adaptation to a target domain using a
        pretrained model:
            Model_Optim_SourceDataset_TargetDataset_ModelParams_prt_
            NameOfPretrainedModel.pt

    """

    def __post_init__(self):
        """Check the type of each argument.

        Raises
        ------
        ValueError
            Raised if the spectral bands of the source ``src_dc`` and target
            ``trg_dc`` datasets are not equal.

        """
        super().__post_init__()

        # base dataset state filename: Dataset_SplitMode_SplitParams
        self.ds_state_file = '{}_{}_{}'

        # base model state filename: Model_Optim_BatchSize
        self.ml_state_file = '{}_{}_b{}'

        # base model state filename extentsion: TileSize_Bands
        self.ds_state_ext = 't{}_{}.pt'

    def init_state(self, src_dc, src_sc, mc, trg_dc=None, trg_sc=None, tc=None,
                   fold=0):
        """Generate the model state filename.

        Parameters
        ----------
        src_dc : :py:class:`pysegcnn.core.trainer.DatasetConfig`
            The source domain dataset configuration.
        src_sc : :py:class:`pysegcnn.core.trainer.SplitConfig`
            The source domain dataset split configuration.
        mc : :py:class:`pysegcnn.core.trainer.ModelConfig`
            The model configuration.
        trg_dc : :py:class:`pysegcnn.core.trainer.DatasetConfig`
            The target domain dataset configuration.
        trg_sc : :py:class:`pysegcnn.core.trainer.SplitConfig`
            The target domain dataset split configuration.
        tc : :py:class:`pysegcnn.core.trainer.TransferLearningConfig`
            The transfer learning configuration.

        Returns
        -------
        state : :py:class:`pathlib.Path`
            The path to the model state file.

        """
        # source domain dataset state filename and extension
        src_ds_state, src_ds_ext = self.format_ds_state(
            src_dc, src_sc, fold)

        # model state file name: common to both source and target domain
        ml_state = self.format_model_state(mc)

        # state file for models trained only on the source domain
        state = '_'.join([ml_state, src_ds_state, src_ds_ext])

        # check whether the model is trained on the source domain only
        if tc is not None:

            # check whether the target domain configurations are correctly
            # specified
            if trg_dc is None or trg_sc is None:
                raise ValueError('Target domain configurations required.')

            # target domain dataset state filename
            trg_ds_state, _ = self.format_ds_state(
                trg_dc, trg_sc, fold)

            # check whether a pretrained model is used to fine-tune to the
            # target domain
            if tc.supervised:
                # state file for models fine-tuned to target domain
                # DatasetConfig_PretrainedModel.pt

                # TODO: Is this correct? Trainer is initialized with source
                # dataloaders
                state = '_'.join([tc.pretrained_model,
                                  'sda_{}'.format(trg_ds_state)])
            else:
                # state file for models trained via unsupervised domain
                # adaptation
                state = '_'.join([state.replace(
                    src_ds_ext, 'uda_{}'.format(tc.uda_pos)),
                    trg_ds_state, src_ds_ext])

                # check whether unsupervised domain adaptation is initialized
                # from a pretrained model state
                if tc.uda_from_pretrained:
                    state = '_'.join(state.replace('.pt', ''),
                                     'prt_{}'.format(
                                         tc.pretrained_model))

        # path to model state
        state = mc.state_path.joinpath(state)

        return state

    def format_model_state(self, mc):
        """Format base model state filename.

        Parameters
        ----------
        mc : :py:class:`pysegcnn.core.trainer.ModelConfig`
            The model configuration.

        """
        return self.ml_state_file.format(mc.model_name, mc.optim_name,
                                         mc.batch_size)

    def format_ds_state(self, dc, sc, fold=None):
        """Format base dataset state filename.

        Parameters
        ----------
        dc : :py:class:`pysegcnn.core.trainer.DatasetConfig`
            The dataset configuration.
        sc : :py:class:`pysegcnn.core.trainer.SplitConfig`
            The dataset split configuration.
        fold : `int` or `None`, optional
            The number of the current fold. The default is `None`, which means
            the fold is not reported in the model state filename.

        Returns
        -------
        file : `str`
            The formatted dataset state filename.

        """
        # store the random seed for reproducibility
        split_params = 's{}'.format(sc.seed)

        # check whether the model is trained via cross validation
        if sc.k_folds > 1 and fold is not None:
            split_params += 'f{}'.format(fold)
        else:
            # construct dataset split parameters
            split_params += 't{}v{}'.format(str(sc.ttratio).replace('.', ''),
                                            str(sc.tvratio).replace('.', ''))

        # get the band numbers
        if dc.bands:
            # the spectral bands used to train the model
            bands = dc.dataset_class.get_sensor().band_dict()
            bformat = ''.join([(v[0] + str(k)) for k, v in bands.items() if
                               v in dc.bands])
        else:
            # all available spectral bands are used to train the model
            bformat = 'all'

        # dataset state filename
        file = self.ds_state_file.format(dc.dataset_class.__name__ +
                                         '_m{}'.format(len(dc.merge_labels)),
                                         sc.split_mode.capitalize(),
                                         split_params,
                                         )

        # dataset state filename extension: common to both source and target
        # domain
        ext = self.ds_state_ext.format(dc.tile_size, bformat)

        return file, ext


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
        self.log_file = check_filename_length(self.log_path.joinpath(
            self.state_file.name.replace('.pt', '.log')))

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
class ClassificationNetworkTrainer(BaseConfig):
    """Base model training class for classification problems.

    Train an instance of :py:class:`pysegcnn.core.models.Network` on a
    classification problem. The `categorical cross-entropy loss`_
    is used as the loss function in combination with the `softmax`_ output
    layer activation function.

    In case of a binary classification problem, the categorical cross-entropy
    loss reduces to the binary cross-entropy loss and the softmax function to
    the standard `logistic function`_.

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
        however, when using accuracy as metric, use ``mode='max'``.
        Only used if ``early_stop=True``. The default is `'max'`, which means
        using the validation set accuracy as metric.
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
    cla_loss_function : :py:class:`torch.nn.Module`
        The classification loss function to compute the model error. An
        instance of :py:class:`torch.nn.CrossEntropyLoss`.
    tracker : :py:class:`pysegcnn.core.trainer.MetricTracker`
        A :py:class:`pysegcnn.core.trainer.MetricTracker` instance tracking
        training metrics, i.e. loss and accuracy.
    max_accuracy : `float`
        Maximum accuracy of ``model`` on the validation dataset.
    es : `None` or :py:class:`pysegcnn.core.trainer.EarlyStopping`
        The early stopping instance if ``early_stop=True``, else `None`.
    tmbatch : `int`
        Number of mini-batches in the training dataset.
    vmbatch : `int`
        Number of mini-batches in the validation dataset.
    training_state : `dict` [`str`, :py:class:`numpy.ndarray`]
        The training state dictionary. The keys describe the type of the
        training metric.
    params_to_save : `dict`
        The parameters to save in the model ``state_file``, in addition to the
        model and optimizer weights.

    .. _Early Stopping:
        https://en.wikipedia.org/wiki/Early_stopping

    .. _categorical cross-entropy loss:
        https://gombru.github.io/2018/05/23/cross_entropy_loss/

    .. _softmax:
        https://peterroelants.github.io/posts/cross-entropy-softmax/

    .. _logistic function:
        https://en.wikipedia.org/wiki/Logistic_function

    """

    model: Network
    optimizer: Optimizer
    state_file: pathlib.Path
    src_train_dl: DataLoader
    src_valid_dl: DataLoader
    src_test_dl: DataLoader
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

        # instanciate multiclass classification loss function: categorical
        # cross-entropy loss function
        self.cla_loss_function = nn.CrossEntropyLoss()
        LOGGER.info('Classification loss function: {}.'
                    .format(repr(nn.CrossEntropyLoss)))

        # instanciate metric tracker
        self.tracker = MetricTracker(
            train_metrics=['train_loss', 'train_accu'],
            valid_metrics=['valid_loss', 'valid_accu'])

        # initialize metric tracker
        self.tracker.initialize()

        # check which metric to use for early stopping
        self.best, self.metric, self.mfn = (
            (0, 'valid_accu', np.max) if self.mode == 'max' else
            (np.inf, 'valid_loss', np.min))

        # best metric score on the validation set
        if self.checkpoint_state:
            self.best = self.mfn(
                self.checkpoint_state[self.metric].mean(axis=0))

        # whether to use early stopping
        self.es = None
        if self.early_stop:
            self.es = EarlyStopping(
                self.mode, self.best, self.delta, self.patience)

        # number of mini-batches in the training and validation sets
        self.tmbatch = len(self.src_train_dl)
        self.vmbatch = len(self.src_valid_dl)

        # log representation
        LOGGER.info(repr(self))

        # initialize training log
        LOGGER.info(35 * '-' + ' Training ' + 35 * '-')

        # log the device and number of threads
        LOGGER.info('Device: {}'.format(self.device))
        LOGGER.info('Number of cpu threads: {}'.format(self.nthreads))

    def train_source_domain(self, epoch):
        """Train a model for a single epoch on the source domain.

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

    def train_epoch(self, epoch):
        """Train a model for a single epoch on the source domain.

        Parameters
        ----------
        epoch : `int`
            The current epoch.

        """
        self.train_source_domain(epoch)

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
                epoch_best = self.training_state[self.metric][:, -1].mean()

                # whether the model improved with respect to the previous epoch
                if self.es.is_better(epoch_best, self.best, self.delta):
                    self.best = epoch_best

                    # save model state if the model improved with
                    # respect to the previous epoch
                    if self.save:
                        self.save_state()

                # whether the early stopping criterion is met
                if self.es.stop(epoch_best):
                    break

            else:
                # if no early stopping is required, the model state is
                # saved after each epoch
                if self.save:
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

    def save_state(self):
        """Save the model state."""
        _ = self.model.save(self.state_file,
                            self.optimizer,
                            state=self.training_state,
                            **self.params_to_save)

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

    @property
    def params_to_save(self):
        """The parameters and variables to save in the model state file."""
        return {'src_train_dl': self.src_train_dl,
                'src_valid_dl': self.src_valid_dl,
                'src_test_dl': self.src_test_dl}

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

        # loss function
        fs += '\n    (loss function):' + '\n' + 8 * ' '
        fs += ''.join(repr(self.cla_loss_function)).replace('\n',
                                                            '\n' + 8 * ' ')

        # early stopping
        fs += '\n    (early stop):' + '\n' + 8 * ' '
        fs += ''.join(repr(self.es)).replace('\n', '\n' + 8 * ' ')

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

        # model configuration
        fs += self._build_model_repr_()

        fs += '\n)'
        return fs


@dataclasses.dataclass
class DomainAdaptationTrainer(ClassificationNetworkTrainer):
    """Model training class for domain adaptation.

    Train an instance of :py:class:`pysegcnn.core.models.EncoderDecoderNetwork`
    on an instance of :py:class:`pysegcnn.core.dataset.ImageDataset`.

    Attributes
    ----------
    supervised : `bool`
        Whether the model is trained supervised or semi-supervised, where
        ``supervised=True`` corresponds to a supervised training on the
        source domain only and ``supervised=False`` corresponds to a
        supervised training on the source domain combined with an unsupervised
        domain adaptation to the target domain. The default is `True`.
    trg_train_dl : `None` or :py:class:`torch.utils.data.DataLoader`
        The target domain training :py:class:`torch.utils.data.DataLoader`
        instance build from an instance of
        :py:class:`pysegcnn.core.split.CustomSubset`. The default is an empty
        :py:class:`torch.utils.data.DataLoader`.
    trg_valid_dl : `None` or  :py:class:`torch.utils.data.DataLoader`
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

    """

    supervised: bool = True
    trg_train_dl: DataLoader = DataLoader(None)
    trg_valid_dl: DataLoader = DataLoader(None)
    trg_test_dl: DataLoader = DataLoader(None)
    uda_loss_function: nn.Module = CoralLoss(uda_lambda=0)
    uda_lambda: float = 0
    uda_pos: str = 'enc'

    def __post_init__(self):
        """Check the type of each argument.

        Configure the device to train the model on, i.e. train on the gpu if
        available.

        Configure early stopping if required.

        Initialize training metric tracking.

        """
        # initialize super class
        super().__post_init__()

        # set the device for computing domain adaptation loss
        if not self.supervised:
            LOGGER.info('Domain adaptation loss function: {}.'
                        .format(repr(self.uda_loss_function)))
            self.uda_loss_function.device = self.device

        # adjust metrics to accomodate domain adaptation metrics
        self.tracker.train_metrics.extend(['cla_loss', 'uda_loss'])

    def _inp_uda(self, src_input, trg_input):
        """Domain adaptation at input feature level."""

        # perform forward pass: classified source domain features
        src_prdctn = self.model(src_input)

        return src_input, trg_input, src_prdctn

    def _enc_uda(self, src_input, trg_input):
        """Domain adaptation at encoder feature level."""

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
        """Domain adaptation at decoder feature level."""

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
        """Domain adaptation at classifier feature level."""

        # perform forward pass: classified source domain features
        src_feature = self.model(src_input)

        # perform forward pass: target domain features
        trg_feature = self.model(trg_input)

        return src_feature, trg_feature, src_feature

    def uda_frwd(self, src_input, trg_input):
        """Forward function for deep domain adaptation.

        Parameters
        ----------
        src_input : :py:class:`torch.Tensor`
            Source domain input features.
        trg_input : :py:class:`torch.Tensor`
            Target domain input features.

        """
        if self.uda_pos == 'inp':
            self._inp_uda(src_input, trg_input)

        if self.uda_pos == 'enc':
            self._enc_uda(src_input, trg_input)

        if self.uda_pos == 'dec':
            self._dec_uda(src_input, trg_input)

        if self.uda_pos == 'cla':
            self._cla_uda(src_input, trg_input)

    def train_domain_adaptation(self, epoch):
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
        if not self.supervised:
            self.train_domain_adaptation(epoch)
        else:
            self.train_source_domain(epoch)

    @property
    def params_to_save(self):
        """The parameters and variables to save in the model state file."""
        return {'src_train_dl': self.src_train_dl,
                'src_valid_dl': self.src_valid_dl,
                'src_test_dl': self.src_test_dl,
                'trg_train_dl': self.trg_train_dl,
                'trg_valid_dl': self.trg_valid_dl,
                'trg_test_dl': self.trg_test_dl,
                'uda': self.uda,
                'uda_pos': self.uda_pos,
                'uda_lambda': self.uda_lambda}

    def __repr__(self):
        """Representation.

        Returns
        -------
        fs : `str`
            Representation string.

        """
        # representation string to print
        fs = self.__class__.__name__ + '(\n'

        # model configuration
        fs += self._build_model_repr_()

        # domain adaptation
        if not self.supervised:
            fs += '\n    (adaptation)' + '\n' + 8 * ' '
            fs += repr(self.uda_loss_function).replace('\n', '\n' + 8 * ' ')

        fs += '\n)'
        return fs


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


@dataclasses.dataclass
class NetworkInference(BaseConfig):
    """Model inference configuration.

    Evaluate model(s) trained by an instance of
    :py:class:`pysegcnn.core.trainer.DomainAdaptationTrainer` on an instance of
    a :py:class:`pysegcnn.core.dataset.ImageDataset` dataset.

    Attributes
    ----------
    state_files : `list´ [:py:class:`pathlib.Path`]
        Path to the model(s) to evaluate.
    implicit : `bool`
        Whether to evaluate the model on the datasets defined at training time.
        The default is `True`.
    domain : `str`
        Whether to evaluate on the source domain (``domain='src'``), i.e. the
        domain a model was trained on, or a target domain (``domain='trg'``).
        The default is `'src'`.
    test : `bool` or `None`
        Whether to evaluate the model on the training (``test=None``), the
        validation (``test=False``) or the test set (``test=True``). The
        default is `False`.
    ds : `dict`
        The dataset configuration dictionary passed to
        :py:class:`pysegcnn.core.trainer.DatasetConfig` when evaluating on
        an explicitly defined dataset, i.e. ``implicit=False``. The default is
        `{}`.
    ds_split : `dict`
        The dataset split configuration dictionary passed to
        :py:class:`pysegcnn.core.trainer.SplitConfig` when evaluating on
        an explicitly defined dataset, i.e. ``implicit=False``. The default is
        `{}`.
    map_labels : `bool`
        Whether to map the model labels from the model source domain to the
        defined ``domain`` in case the domain class labels differ. The default
        is `False`.
    predict_scene : `bool`
        The model prediction order. If ``predict_scene=False``, the samples of
        a dataset are predicted in any order.If ``predict_scene=True``, the
        samples are ordered according to their scene and a model prediction for
        each entire reconstructed scene is returned. The default is `True`.
    plot_scenes : `bool`
        Whether to save a plot of false color composite, ground truth and model
        prediction for each entire scene. Only used if ``predict_scene=True``.
        The default is `False`.
    plot_bands : `list` [`str`]
        The bands to build the false color composite. The default is
        `['nir', 'red', 'green']`.
    cm : `bool`
        Whether to compute the confusion matrix. The default is `True`.
    figsize : `tuple`
        The figure size in centimeters. The default is `(10, 10)`.
    alpha : `int`
        The level of the percentiles for contrast stretching of the false color
        compsite. The default is `0`, i.e. no stretching.
    animate : `bool`
        Whether to create an animation of (input, ground truth, prediction) for
        the scenes in the train/validation/test dataset. Only works if
        ``predict_scene=True`` and ``plot_scene=True``.
    device : `str`
        The device to evaluate the model on, i.e. `cpu` or `cuda`.
    base_path : :py:class:`pathlib.Path`
        Root path to store model output.
    sample_path : :py:class:`pathlib.Path`
        Path to store plots of model predictions for single samples.
    scenes_path : :py:class:`pathlib.Path`
        Path to store plots of model predictions for entire scenes.
    perfmc_path : :py:class:`pathlib.Path`
        Path to store plots of model performance, e.g. confusion matrix.
    models_path : :py:class:`pathlib.Path`
        Path to search for model state files ``state_files``.
    plot_kwargs : `dict`
        Keyword arguments for :py:func:`pysegcnn.core.graphics.plot_sample`
    trg_ds : :py:class:`pysegcnn.core.split.CustomSubset`
        The dataset to evaluate ``model`` on.
    src_ds : :py:class:`pysegcnn.core.split.CustomSubset`
        The model source domain training dataset.

    """

    state_files: list
    implicit: bool = True
    domain: str = 'src'
    test: object = False
    ds: dict = dataclasses.field(default_factory={})
    ds_split: dict = dataclasses.field(default_factory={})
    map_labels: bool = False
    predict_scene: bool = True
    plot_scenes: bool = False
    plot_bands: list = dataclasses.field(
        default_factory=lambda: ['nir', 'red', 'green'])
    cm: bool = True
    figsize: tuple = (10, 10)
    alpha: int = 5

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
        self.device = torch.device(
            'cuda:' if torch.cuda.is_available() else 'cpu')

        # the output paths for the different graphics
        self.base_path = pathlib.Path(HERE)
        self.sample_path = self.base_path.joinpath('_samples')
        self.scenes_path = self.base_path.joinpath('_scenes')
        self.perfmc_path = self.base_path.joinpath('_graphics')

        # input path for model state files
        self.models_path = self.base_path.joinpath('_models')
        self.state_files = [self.models_path.joinpath(s) for s in
                            self.state_files]

        # plotting keyword arguments
        self.plot_kwargs = {'bands': self.plot_bands,
                            'alpha': self.alpha,
                            'figsize': self.figsize}

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

    def load_dataset(self, state, implicit=True, test=False, domain='src'):
        """Load the defined dataset.

        Raises
        ------
        ValueError
            Raised if the requested dataset was not available at training time,
            if ``implicit=True``.

        Returns
        -------
        ds : :py:class:`pysegcnn.core.split.CustomSubset`
            The dataset to evaluate the model on.

        """
        # load model state
        model_state = Network.load(state)

        # check whether to evaluate on the datasets defined at training time
        if implicit:

            # check whether to evaluate the model on the training, validation
            # or test set
            if test is None:
                ds_set = 'train'
            else:
                ds_set = 'test' if test else 'valid'

            # the dataset to evaluate the model on
            ds = model_state[domain + '_{}_dl'.format(ds_set)].dataset
            if ds is None:
                raise ValueError('Requested dataset "{}" is not available.'
                                 .format(domain + '_{}_dl'.format(ds_set))
                                 )

            # log dataset representation
            LOGGER.info('Evaluating on {} set of the {} domain defined at '
                        'training time.'.format(ds_set, domain))

        else:
            # explicitly defined dataset
            ds = DatasetConfig(**self.ds).init_dataset()

            # split configuration
            sc = SplitConfig(**self.ds_split)
            train_ds, valid_ds, test_ds = sc.train_val_test_split(ds)

            # check whether to evaluate the model on the training, validation
            # or test set
            if test is None:
                ds = train_ds
            else:
                ds = test_ds if test else valid_ds

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
        """Class labels of the target domain the model is evaluated on.

        Returns
        -------
        target_labels : `dict` [`int`, `dict`]
            The class labels of the target domain.

        """
        return self.trg_ds.labels

    @property
    def source_label_map(self):
        """Mapping of the original source labels to the model source labels.

        See
        :py:meth:`pysegcnn.core.trainer.NetworkInference._original_source_labels`.

        Returns
        -------
        source_labels : :py:class:`numpy.ndarray`
            The mapping from the original source class identifiers to the
            identifiers used during training.

        """
        return np.array([list(self.source_labels.keys()),
                         list(self._original_source_labels.keys())]).T

    @property
    def target_label_map(self):
        """Mapping of the original target labels to the model target labels.

         See
        :py:meth:`pysegcnn.core.trainer.NetworkInference._original_source_labels`.

        Returns
        -------
        target_labels : :py:class:`numpy.ndarray`
            The mapping from the original target class identifiers to the
            identifiers used for evaluation.

        """
        return np.array([list(self._original_target_labels_labels.keys()),
                         list(self.target_labels.keys())]).T

    @property
    def label_map(self):
        """Label mapping dictionary from the source to the target domain.

        See :py:class:`pysegcnn.core.constants.LabelMapping`.

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
        return self.map_labels and not self.source_is_target

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
    def plot(self):
        """Whether to save plots of (input, ground truth, prediction).

        Returns
        -------
        plot : `bool`
            Save plots for each sample or for each scene of the target dataset,
            depending on ``self.predict_scene``.

        """
        return self.plot_scenes if self.predict_scene else False

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
        return self.trg_ds.dataset.tiles if self.predict_scene else 1

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

    def map_to_target(self, prd):
        """Map source domain labels to target domain labels.

        Parameters
        ----------
        prd : :py:class:`torch.Tensor`
            The source domain class labels as predicted by the model.

        Returns
        -------
        prd : :py:class:`torch.Tensor`
            The predicted target domain labels.

        """
        # map actual source labels to original source labels
        prd = array_replace(prd, self.source_label_map)

        # apply the label mapping
        prd = array_replace(prd, self.label_map.to_numpy())

        # map original target labels to actual target labels
        prd = array_replace(prd, self.target_label_map)

        return prd

    def predict(self, model):
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
        # set the model to evaluation mode
        LOGGER.info('Setting model to evaluation mode ...')
        model.eval()
        model.to(self.device)

        # iterate over the samples of the target dataset
        output = {}
        for batch, (inputs, labels) in enumerate(self.dataloader):

            # send inputs and labels to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # compute model predictions
            with torch.no_grad():
                prdctn = F.softmax(
                    model(inputs), dim=1).argmax(dim=1).squeeze()

            # progress string to log
            progress = 'Sample: {:d}/{:d}'.format(batch + 1,
                                                  len(self.dataloader))

            # check whether to reconstruct the scene
            if self.dataloader.batch_size > 1:

                # id of the current scene
                batch = self.trg_ds.ids[batch]

                # modify the progress string
                progress = progress.replace('Sample', 'Scene')
                progress += ' Id: {}'.format(batch)

                # reconstruct the entire scene
                inputs = reconstruct_scene(inputs)
                labels = reconstruct_scene(labels)
                prdctn = reconstruct_scene(prdctn)

            # check whether the source and target domain labels differ
            if self.apply_label_map:
                prdctn = self.map_to_target(prdctn)

            # save current batch to output dictionary
            output[batch] = {'x': inputs, 'y': labels, 'y_pred': prdctn}

            # filename for the plot of the current batch
            batch_name = '_'.join(model.state_file.stem,
                                  '{}_{}.pt'.format(self.trg_ds.name, batch))

            # check if the current batch name exceeds the Windows limit of
            # 255 characters
            batch_name = check_filename_length(batch_name)

            # calculate the accuracy of the prediction
            progress += ', Accuracy: {:.2f}'.format(
                accuracy_function(prdctn, labels))
            LOGGER.info(progress)

            # plot current scene
            if self.plot:

                # plot inputs, ground truth and model predictions
                _ = plot_sample(inputs.clip(0, 1),
                                self.bands,
                                self.source_labels,
                                y=labels,
                                y_pred={self.model.__class__.__name__: prdctn},
                                accuracy=True,
                                state=batch_name,
                                plot_path=self.scenes_path,
                                **self.kwargs)
        return output

    def eval_file(self, state_file):
        return pathlib.Path(str(state_file).replace('.pt', '_eval.pt'))

    def evaluate(self):
        """Evaluate the models on a defined dataset.

        Returns
        -------
        inference : `dict` [`str`, `dict`]
            The inference output dictionary. The keys are the names of the
            models in ``self.state_file`` and the values are dictionaries
            where the keys are either the number of the batches
            (``self.predict_scene=False``) or the name of the
            scenes of the target dataset (``self.predict_scene=True``). The
            values of the nested dictionaries are again dictionaries with keys:
                ``'x'``
                    Model input data of the sample (:py:class:`numpy.ndarray`).
                ``'y'
                    Ground truth class labels (:py:class:`numpy.ndarray`).
                ``'y_pred'``
                    Model prediction class labels (:py:class:`numpy.ndarray`).
                ``'cm'``
                    The confusion matrix of the model, which is only present if
                    ``self.cm=True`` (:py:class:`numpy.ndarray`).

        """
        # iterate over the models to evaluate
        inference = {}
        for state in self.state_files:

            # initialize logging
            log = LogConfig(state)
            dictConfig(log_conf(log.log_file))
            log.init_log('{}: ' + 'Evaluating model: {}.'.format(state.name))

            # check whether model was already evaluated
            if self.eval_file(state).exists():

                # load existing model evaluation
                LOGGER.info('Found existing model evaluation: {}.'
                            .format(self.eval_file(state)))
                inference[state.stem] = torch.load(self.eval_file(state))
                continue

            # plot loss and accuracy
            plot_loss(check_filename_length(state), outpath=self.perfmc_path)

            # load the target dataset to evaluate the model on
            self.trg_ds = self.load_dataset(
                state, implicit=self.implicit, test=self.test,
                domain=self.domain)

            # load the source dataset the model was trained on
            self.src_ds = self.load_dataset(state, test=None)

            # load the pretrained model
            model, _ = Network.load_pretrained_model(state)

            # evaluate the model on the target dataset
            output = self.predict()

            # check whether to calculate confusion matrix
            if self.cm:

                # initialize confusion matrix
                conf_mat = np.zeros(shape=2 * (len(self.src_ds.labels), ))

                # calculate confusion matrix
                for ytrue, ypred in zip(output['y'].flatten(),
                                        output['y_pred'].flatten()):
                    # update confusion matrix entries
                    conf_mat[ytrue.long(), ypred.long()] += 1

                # add confusion matrix to model output
                output['cm'] = conf_mat

                # plot confusion matrix
                plot_confusion_matrix(
                    conf_mat, self.source_labels, state_file=state,
                    subset=self.domain + '_' + self.trg_ds.name,
                    outpath=self.perfmc_path)

            # save model predictions to file
            torch.save(output, self.eval_file(state))

            # save model predictions to list
            inference[state.stem] = output

        return inference
