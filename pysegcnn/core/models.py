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
from pysegcnn.core.layers import (Encoder, Decoder, ConvBnReluMaxPool,
                                  ConvBnReluMaxUnpool, Conv2dSame)
from pysegcnn.core.utils import check_filename_length, item_in_enum

# module level logger
LOGGER = logging.getLogger(__name__)


class Network(nn.Module):
    """Generic neural network class for image classification tasks.

    The base class for each model. If you want to implement a new model,
    inherit the :py:class:`pysegcnn.core.models.Network` class.

    Attributes
    ----------
    state_file : `str` or `None` or :py:class:`pathlib.Path`
        The model state file, where the model parameters are saved.
    in_channels : `int`
        Number of input features.
    nclasses : `int`
        Number of classes.
    epoch : `int`
        Number of epochs the network was trained.

    """

    def __init__(self, state_file, in_channels, nclasses):
        """Initialize.

        Parameters
        ----------
        state_file : `str` or `None` or :py:class:`pathlib.Path`
            The model state file, where the model parameters are saved.
        in_channels : `int`
            Number of input features.
        nclasses : `int`
            Number of classes.

        """
        super().__init__()

        # initialize state file
        self.state_file = state_file

        # number of spectral bands of the input images
        self.in_channels = in_channels

        # number of output classes
        self.nclasses = nclasses

        # number of epochs trained
        self.epoch = 0

    def freeze(self, name=None):
        """Freeze the weights of a model.

        Disables gradient computation: useful when using a pretrained model for
        inference.

        Parameters
        ----------
        name : `str` or `None`
            The name of a part of the model. If specified, only the weights of
            that specific part of the model are frozen. If `None`, all model
            weights are frozen. The default is `None`.

        """
        if name is None:
            # freeze all the model weights
            for param in self.parameters():
                param.requires_grad = False
        else:
            # freeze the weights of a part of the model, e.g. encoder weights
            for param in getattr(self, str(name)).parameters():
                param.requires_grad = False

    def unfreeze(self, name=None):
        """Unfreeze the weights of a model.

        Enables gradient computation: useful when adjusting a pretrained model
        to a new dataset.

        Parameters
        ----------
        name : `str` or `None`
            The name of a part of the model. If specified, only the weights of
            that specific part of the model are unfrozen. If `None`, all model
            weights are unfrozen. The default is `None`.

        """
        if name is None:
            # freeze all the model weights
            for param in self.parameters():
                param.requires_grad = True
        else:
            # freeze the weights of a part of the model, e.g. encoder weights
            for param in getattr(self, str(name)).parameters():
                param.requires_grad = True

    def save(self, state_file, optimizer, **kwargs):
        """Save the model and optimizer state.

        Optional ``kwargs`` are also saved.

        Parameters
        ----------
        state_file : `str` or :py:class:`pathlib.Path`
            Path to save the model state.
        optimizer : :py:class:`torch.optim.Optimizer`
            The optimizer used to train the model.
        **kwargs
            Arbitrary keyword arguments. Each keyword argument will be saved
            as (key, value) pair in ``state_file``.

        Returns
        -------
        model_state : `dict`
            A dictionary containing the model and optimizer state.

        """
        # check if the output path exists and if not, create it
        state_file = pathlib.Path(check_filename_length(state_file))
        if not state_file.parent.is_dir():
            state_file.parent.mkdir(parents=True, exist_ok=True)

        # initialize dictionary to store network parameters
        model_state = {**kwargs}

        # store model epoch
        model_state['epoch'] = self.epoch

        # store model construction parameters
        model_state['in_channels'] = self.in_channels
        model_state['nclasses'] = self.nclasses

        # store model and optimizer state
        model_state['model_state_dict'] = self.state_dict()
        model_state['optim_state_dict'] = optimizer.state_dict()

        # model state dictionary stores the values of all trainable parameters
        torch.save(model_state, state_file)
        LOGGER.info('Network parameters saved in {}'.format(state_file))

        return model_state

    @staticmethod
    def load(state_file):
        """Load a model state file.

        Parameters
        ----------
        state_file : `str` or :py:class:`pathlib.Path`
           The model state file containing the pretrained parameters.

        Raises
        ------
        FileNotFoundError
            Raised if ``state_file`` does not exist.

        Returns
        -------
        model_state : `dict`
            A dictionary containing the model and optimizer state, as
            constructed by :py:meth:`~pysegcnn.core.Network.save`.

        """
        # load the pretrained model
        state_file = pathlib.Path(check_filename_length(state_file))
        if not state_file.exists():
            raise FileNotFoundError('{} does not exist.'.format(state_file))

        # load the model state
        model_state = torch.load(state_file)

        return model_state

    @staticmethod
    def load_pretrained_model_weights(model, model_state):
        """Load the pretrained model weights from a state file.

        Parameters
        ----------
        model : :py:class:`pysegcnn.core.models.Network`
           An instance of the model for which the pretrained weights are
           stored in ``model_state``.
        model_state : `dict`
            A dictionary containing the model and optimizer state, as
            constructed by :py:meth:`~pysegcnn.core.Network.save`.

        Returns
        -------
        model : :py:class:`pysegcnn.core.models.Network`
           An instance of the pretrained model in ``model_state``.

        """
        # load pretrained model weights
        LOGGER.info('Loading model parameters ...')
        model.load_state_dict(model_state['model_state_dict'])

        # set model epoch
        model.epoch = model_state['epoch']
        LOGGER.info('Model epoch: {:d}'.format(model.epoch))

        return model

    @staticmethod
    def load_pretrained_optimizer_weights(optimizer, model_state):
        """Load the pretrained optimizer weights from a state file.

        Parameters
        ----------
        optimizer : :py:class:`torch.optim.Optimizer`
           An instance of the optimizer used to train ``model`` for which the
           pretrained weights are stored in ``model_state``.
        model_state : `dict`
            A dictionary containing the model and optimizer state, as
            constructed by :py:meth:`~pysegcnn.core.Network.save`.

        Returns
        -------
        optimizer : :py:class:`torch.optim.Optimizer`
           An instance of the pretrained optimizer in ``model_state``.

        """
        # resume optimizer parameters
        LOGGER.info('Loading optimizer parameters ...')
        optimizer.load_state_dict(model_state['optim_state_dict'])

        return optimizer

    @staticmethod
    def load_pretrained_model(state_file):
        """Load an instance of the pretrained model in ``state_file``.

        Parameters
        ----------
        state_file : `str` or :py:class:`pathlib.Path`
           The model state file containing the pretrained parameters.

        Returns
        -------
        model : :py:class:`pysegcnn.core.models.Network`
            An instance of the pretrained model in ``state_file``.
        optimizer : :py:class:`torch.optim.Optimizer`
           An instance of the pretrained optimizer in ``state_file``.

        """
        # get the model class of the pretrained model
        state_file = pathlib.Path(state_file)
        model_class = item_in_enum(str(state_file.stem).split('_')[0],
                                   SupportedModels)

        # get the optimizer class of the pretrained model
        optim_class = item_in_enum(str(state_file.stem).split('_')[1],
                                   SupportedOptimizers)

        # load the pretrained model configuration
        LOGGER.info('Loading pretrained weights from: {}'
                    .format(state_file.name))
        model_state = Network.load(state_file)

        # instanciate the pretrained model architecture
        model = model_class(state_file=state_file,
                            in_channels=model_state['in_channels'],
                            nclasses=model_state['nclasses'])

        # instanciate the optimizer
        optimizer = optim_class(model.parameters())

        # load pretrained model weights
        model = Network.load_pretrained_model_weights(model, model_state)

        # load pretrained optimizer weights
        optimizer = Network.load_pretrained_optimizer_weights(optimizer,
                                                              model_state)

        return model, optimizer


class ConvolutionalAutoEncoder(Network):
    """Generic convolutional autoencoder.

    Attributes
    ----------
    state_file : `str` or `None` or :py:class:`pathlib.Path`
        The model state file, where the model parameters are saved.
    in_channels : `int`
        Number of channels of the input images.
    nclasses : `int`
        Number of classes.
    filters : `list` [`int`]
        List of the number of convolutional filters in each block.
    skip : `bool`
        Whether to apply skip connections from the encoder to the decoder.
    kwargs : `dict` [`str`]
        Additional keyword arguments passed to
        :py:class:`pysegcnn.core.layers.Conv2dSame`.
    epoch : `int`
        Number of epochs the model was trained.
    encoder : :py:class:`pysegcnn.core.layers.Encoder`
        The convolutional encoder.
    decoder : :py:class:`pysegcnn.core.layers.Decoder`
        The convolutional decoder.
    classifier : :py:class:`pysegcnn.core.layers.Conv2dSame`
        The classification layer, a 1x1 convolution.

    """

    def __init__(self, state_file, in_channels, nclasses, encoder_block,
                 decoder_block, filters, skip, **kwargs):
        """Initialize.

        Parameters
        ----------
        state_file : `str` or `None` or :py:class:`pathlib.Path`
            The model state file, where the model parameters are saved.
        in_channels : `int`
            Number of channels of the input images.
        nclasses : `int`
            Number of classes.
        encoder_block : :py:class:`pysegcnn.core.layers.EncoderBlock`
            The convolutional block defining a layer in the encoder.
            A subclass of :py:class:`pysegcnn.core.layers.EncoderBlock`, e.g.
            :py:class:`pysegcnn.core.layers.ConvBnReluMaxPool`.
        decoder_block : :py:class:`pysegcnn.core.layers.DecoderBlock`
            The convolutional block defining a layer in the decoder.
            A subclass of :py:class:`pysegcnn.core.layers.DecoderBlock`, e.g.
            :py:class:`pysegcnn.core.layers.ConvBnReluMaxUnpool`.
        filters : `list` [`int`]
            List of input channels to each convolutional block.
        skip : `bool`
            Whether to apply skip connections from the encoder to the decoder.
        **kwargs: `dict` [`str`]
            Additional keyword arguments passed to
            :py:class:`pysegcnn.core.layers.Conv2dSame`.

        """
        super().__init__(state_file, in_channels, nclasses)

        # number of convolutional filters for each block
        self.filters = np.hstack([np.array(in_channels), np.array(filters)])

        # whether to apply skip connections
        self.skip = skip

        # configuration of the convolutional layers in the network
        self.kwargs = kwargs

        # construct the encoder
        self.encoder = Encoder(filters=self.filters, block=encoder_block,
                               **kwargs)

        # construct the decoder
        self.decoder = Decoder(filters=self.filters, block=decoder_block,
                               skip=self.skip, **kwargs)

        # construct the classifier
        self.classifier = Conv2dSame(in_channels=filters[0],
                                     out_channels=self.nclasses,
                                     kernel_size=1)

    def forward(self, x):
        """Forward propagation of a convolutional encoder-decoder network.

        Parameters
        ----------
        x : `torch.Tensor`
            The input image, shape=(batch_size, channels, height, width).

        Returns
        -------
        y : 'torch.Tensor'
            Logits, shape=(batch_size, channels, height, width).

        """
        # forward pass: encoder
        x = self.encoder(x)

        # forward pass: decoder
        x = self.decoder(x, self.encoder.cache)

        # clear intermediate outputs
        del self.encoder.cache

        # classification
        return self.classifier(x)


class SegNet(ConvolutionalAutoEncoder):
    """An implementation of `SegNet`_ in PyTorch.

    .. _SegNet:
        https://arxiv.org/abs/1511.00561

    Attributes
    ----------
    state_file : `str` or `None` or :py:class:`pathlib.Path`
        The model state file, where the model parameters are saved.
    in_channels : `int`
        Number of channels of the input images.
    nclasses : `int`
        Number of classes.
    filters : `list` [`int`]
        List of the number of convolutional filters in each block.
    skip : `bool`
        Whether to apply skip connections from the encoder to the decoder.
    kwargs : `dict` [`str`]
        Additional keyword arguments passed to
        :py:class:`pysegcnn.core.layers.Conv2dSame`.
    epoch : `int`
        Number of epochs the model was trained.
    encoder : :py:class:`pysegcnn.core.layers.Encoder`
        The convolutional encoder.
    decoder : :py:class:`pysegcnn.core.layers.Decoder`
        The convolutional decoder.
    classifier : :py:class:`pysegcnn.core.layers.Conv2dSame`
        The classification layer, a 1x1 convolution.

    """

    def __init__(self, state_file, in_channels, nclasses,
                 filters=[32, 64, 128, 256], skip=True,
                 kwargs={'kernel_size': 3, 'stride': 1, 'dilation': 1}):
        """Initialize.

        Parameters
        ----------
        state_file : `str` or `None` or :py:class:`pathlib.Path`
            The model state file, where the model parameters are saved.
        in_channels : `int`
            Number of channels of the input images.
        nclasses : `int`
            Number of classes.
        filters : `list` [`int`], optional
            List of input channels to each convolutional block. The default is
            `[32, 64, 128, 256]`.
        skip : `bool`, optional
            Whether to apply skip connections from the encoder to the decoder.
            The default is `True`.
        kwargs: `dict` [`str`: `int`]
            Additional keyword arguments passed to
            :py:class:`pysegcnn.core.layers.Conv2dSame`. The default is
            `{'kernel_size': 3, 'stride': 1, 'dilation': 1}`.

        """
        super().__init__(state_file=state_file,
                         in_channels=in_channels,
                         nclasses=nclasses,
                         encoder_block=ConvBnReluMaxPool,
                         decoder_block=ConvBnReluMaxUnpool,
                         filters=filters,
                         skip=skip,
                         **kwargs)


class SupportedModels(enum.Enum):
    """Names and corresponding classes of the implemented models."""

    Segnet = SegNet


class SupportedOptimizers(enum.Enum):
    """Names and corresponding classes of the tested optimizers."""

    Adam = optim.Adam
    AdamW = optim.AdamW
