"""Neural networks for semantic image segmentation.

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
import logging
import pathlib

# externals
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# locals
from pysegcnn.core.layers import (Encoder, Decoder, Conv2dPool, Conv2dUnpool,
                                  Conv2dSame)

# module level logger
LOGGER = logging.getLogger(__name__)


class Network(nn.Module):
    """Generic Network class.

    The base class for each model. If you want to implement a new model,
    inherit the ``~pysegcnn.core.models.Network`` class.

    Returns
    -------
    None.

    """

    def __init__(self):
        super().__init__()

        # initialize state file
        self.state_file = None

    def freeze(self):
        """Freeze the weights of a model.

        Disables gradient computation: useful when using a pretrained model for
        inference.

        Returns
        -------
        None.

        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze the weights of a model.

        Enables gradient computation: useful when adjusting a pretrained model
        to a new dataset.

        Returns
        -------
        None.

        """
        for param in self.parameters():
            param.requires_grad = True

    def save(self, state_file, optimizer, bands=None, **kwargs):
        """Save the model state.

        Saves the model and optimizer states together with the model
        construction parameters, to easily re-instanciate the model.

        Optional ``kwargs`` are also saved.

        Parameters
        ----------
        state_file : `str` or `pathlib.Path`
            Path to save the model state.
        optimizer : `torch.optim.Optimizer`
            The optimizer used to train the model.
        bands : `list` [`str`] or `None`, optional
            List of bands the model is trained with. The default is None.
        **kwargs
            Arbitrary keyword arguments. Each keyword argument will be saved
            as (key, value) pair in ``state_file``.

        Returns
        -------
        model_state : `dict`
            A dictionary containing the model and optimizer state

        """
        # check if the output path exists and if not, create it
        state_file = pathlib.Path(state_file)
        if not state_file.parent.is_dir():
            state_file.parent.mkdir(parents=True, exist_ok=True)

        # initialize dictionary to store network parameters
        model_state = {**kwargs}

        # store the spectral bands the model is trained with
        model_state['bands'] = bands

        # store model class
        model_state['cls'] = self.__class__

        # store construction parameters to instanciate the network
        model_state['params'] = {
            'skip': self.skip,
            'filters': self.nfilters,
            'nclasses': self.nclasses,
            'in_channels': self.in_channels
            }

        # store optional keyword arguments
        model_state['params'] = {**model_state['params'], **self.kwargs}

        # store model epoch
        model_state['epoch'] = self.epoch

        # store model and optimizer state
        model_state['model_state_dict'] = self.state_dict()
        model_state['optim_state_dict'] = optimizer.state_dict()

        # model state dictionary stores the values of all trainable parameters
        torch.save(model_state, state_file)
        LOGGER.info('Network parameters saved in {}'.format(state_file))

        return model_state

    @staticmethod
    def load(state_file, optimizer=None):
        """Load a model state.

        Returns the model in ``state_file`` with the pretrained model weights.
        If ``optimizer`` is specified, the optimizer parameters are also loaded
        from ``state_file``. This is useful when resuming training an existing
        model.

        Parameters
        ----------
        state_file : `str` or `pathlib.Path`
           The model state file. Model state files are stored in
           pysegcnn/main/_models.
        optimizer : `torch.optim.Optimizer` or `None`, optional
           The optimizer used to train the model.

        Raises
        ------
        FileNotFoundError
            Raised if ``state_file`` does not exist.

        Returns
        -------
        model : `pysegcnn.core.models.Network`
            The pretrained model.
        optimizer : `torch.optim.Optimizer` or `None`
           The optimizer used to train the model.
        model_state : '`dict`
            A dictionary containing the model and optimizer state, as
            constructed by `~pysegcnn.core.Network.save`.

        """
        # load the pretrained model
        state_file = pathlib.Path(state_file)
        if not state_file.exists():
            raise FileNotFoundError('{} does not exist.'.format(state_file))
        LOGGER.info('Loading pretrained weights from: {}'.format(state_file))

        # load the model state
        model_state = torch.load(state_file)

        # the model class
        model_class = model_state['cls']

        # instanciate pretrained model architecture
        model = model_class(**model_state['params'])

        # store state file as instance attribute
        model.state_file = state_file

        # load pretrained model weights
        LOGGER.info('Loading model parameters ...')
        model.load_state_dict(model_state['model_state_dict'])
        model.epoch = model_state['epoch']

        # resume optimizer parameters
        if optimizer is not None:
            LOGGER.info('Loading optimizer parameters ...')
            optimizer.load_state_dict(model_state['optim_state_dict'])
        LOGGER.info('Model epoch: {:d}'.format(model.epoch))

        return model, optimizer, model_state

    @property
    def state(self):
        """Return the model state file.

        Returns
        -------
        state_file : `pathlib.Path` or `None`
            The model state file.

        """
        return self.state_file


class UNet(Network):
    """A PyTorch implementation of `U-Net`_.

    .. _U-Net:
        https://arxiv.org/abs/1505.04597

    Parameters
    ----------
    in_channels : `int`
        Number of channels of the input images.
    nclasses : `int`
        Number of classes.
    filters : `list` [`int`]
        List of input channels to each convolutional block.
    skip : `bool`
        Whether to apply skip connections from the encoder to the decoder.
    **kwargs: 'dict' [`str`]
        Additional keyword arguments passed to
        `pysegcnn.core.layers.Conv2dSame`.

    Returns
    -------
    None.

    """

    def __init__(self, in_channels, nclasses, filters, skip, **kwargs):
        super().__init__()

        # number of input channels
        self.in_channels = in_channels

        # number of classes
        self.nclasses = nclasses

        # configuration of the convolutional layers in the network
        self.kwargs = kwargs
        self.nfilters = filters

        # convolutional layers of the encoder
        self.filters = np.hstack([np.array(in_channels), np.array(filters)])

        # whether to apply skip connections
        self.skip = skip

        # number of epochs trained
        self.epoch = 0

        # construct the encoder
        self.encoder = Encoder(filters=self.filters, block=Conv2dPool,
                               **kwargs)

        # construct the decoder
        self.decoder = Decoder(filters=self.filters, block=Conv2dUnpool,
                               skip=skip, **kwargs)

        # construct the classifier
        self.classifier = Conv2dSame(in_channels=filters[0],
                                     out_channels=self.nclasses,
                                     kernel_size=1)

    def forward(self, x):
        """Forward propagation of U-Net.

        Parameters
        ----------
        x : `torch.Tensor`
            The input image, shape=(batch_size, channels, height, width).

        Returns
        -------
        y : 'torch.tensor'
            The classified image, shape=(batch_size, height, width).

        """
        # forward pass: encoder
        x = self.encoder(x)

        # forward pass: decoder
        x = self.decoder(x, self.encoder.cache)

        # clear intermediate outputs
        del self.encoder.cache

        # classification
        return self.classifier(x)


class SupportedModels(enum.Enum):
    """Names and corresponding classes of the implemented models."""

    Unet = UNet


class SupportedOptimizers(enum.Enum):
    """Names and corresponding classes of the tested optimizers."""

    Adam = optim.Adam


class SupportedLossFunctions(enum.Enum):
    """Names and corresponding classes of the tested loss functions."""

    CrossEntropy = nn.CrossEntropyLoss
