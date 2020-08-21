"""Layers of a convolutional encoder-decoder network.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# externals
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSame(nn.Conv2d):
    """A convolution preserving the shape of its input.

    Given the kernel size, the dilation and a stride of 1, the padding is
    calculated such that the output of the convolution has the same spatial
    dimensions as the input.

    Parameters
    ----------
    *args: `list` [`str`]
        positional arguments passed to `torch.nn.Conv2d`:
            ``'in_channels'``: `int`
                Number of input channels.
            ``'out_channels'``: `int`
                Number of output channels.
            ``'kernel_size'``: `int` or `tuple` [`int`]
                Size of the convolving kernel.
    **kwargs: 'dict' [`str`]
        Additional keyword arguments passed to `torch.nn.Conv2d`_.

    .. _torch.nn.Conv2d:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

    Returns
    -------
    None.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # define tensorflows "SAME" padding for stride = 1
        x_pad = self.same_padding(self.dilation[1], self.kernel_size[1])
        y_pad = self.same_padding(self.dilation[0], self.kernel_size[0])

        self.padding = (y_pad, x_pad)

    @staticmethod
    def same_padding(d, k):
        """Calculate the amount of padding.

        Parameters
        ----------
        d : `int`
            The dilation of the convolution.
        k : `int`
            The kernel size.

        Returns
        -------
        p : `int`
            the amount of padding.

        """
        # calculates the padding so that the convolution
        # conserves the shape of its input when stride = 1
        return int(d * (k - 1) / 2)


def conv_bn_relu(in_channels, out_channels, **kwargs):
    """Block of convolution, batch normalization and rectified linear unit.

    Parameters
    ----------
    in_channels : `int`
        Number of input channels.
    out_channels : `int`
        Number of output channels.
    **kwargs: 'dict' [`str`]
        Additional arguments passed to `pysegcnn.core.layers.Conv2dSame`.

    Returns
    -------
    block : `torch.nn.Sequential` [`torch.nn.Module`]
        An instance of `torch.nn.Sequential` containing a sequence of
        convolution, batch normalization and rectified linear unit layers.

    """
    return nn.Sequential(
            Conv2dSame(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Conv2dSame(out_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )


class Block(nn.Module):
    """Basic convolutional block.

    Parameters
    ----------
    in_channels : `int`
        Number of input channels.
    out_channels : `int`
        Number of output channels.
    **kwargs: 'dict' [`str`]
         Additional arguments passed to `pysegcnn.core.layers.Conv2dSame`.

    Raises
    ------
    TypeError
        Raised if `~pysegcnn.core.layers.Block.layers` method does not return
        an instance of `torch.nn.Sequential`.

    Returns
    -------
    None.

    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        # number of input and output channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # keyword arguments configuring convolutions
        self.kwargs = kwargs

        # the layers of the block
        self.conv = self.layers()
        if not isinstance(self.conv, nn.Sequential):
            raise TypeError('{}.layers() should return an instance of {}.'
                            .format(self.__class__.__name__,
                                    repr(nn.Sequential)))

    def layers(self):
        """Define the layers of the block.

        Raises
        ------
        NotImplementedError
            Raised if `pysegcnn.core.layers.Block` is not inherited.

        Returns
        -------
        layers : `torch.nn.Sequential` [`torch.nn.Module`]
            Return an instance of `torch.nn.Sequential` containing a sequence
            of layer (`torch.nn.Module` ) instances.

        """
        raise NotImplementedError('Return an instance of {}.'
                                  .format(repr(nn.Sequential)))

    def forward(self):
        """Forward pass of the block.

        Raises
        ------
        NotImplementedError
            Raised if `pysegcnn.core.layers.Block` is not inherited.

        Returns
        -------
        None.

        """
        raise NotImplementedError('Implement the forward pass.')


class EncoderBlock(Block):
    """Block of a convolutional encoder.

    Parameters
    ----------
    in_channels : `int`
        Number of input channels.
    out_channels : `int`
        Number of output channels.
    **kwargs: 'dict' [`str`]
         Additional arguments passed to `pysegcnn.core.layers.Conv2dSame`.

    Returns
    -------
    None.

    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

    def forward(self, x):
        """Forward pass of an encoder block.

        Parameters
        ----------
        x : `torch.Tensor`
            Input tensor, e.g. output of the previous block/layer.

        Returns
        -------
        y : `torch.Tensor`
            Output of the encoder block.
        x : `torch.Tensor`
            Intermediate output before applying downsampling. Useful to
            implement skip connections.
        indices : `torch.Tensor` or `None`
            Optional indices of the downsampling method, e.g. indices of the
            maxima when using `torch.nn.functional.max_pool2d`. Useful for
            upsampling later. If no indices are required to upsample, simply
            return ``indices`` = `None`.

        """
        # the forward pass of the layers of the block
        x = self.conv(x)

        # the downsampling layer
        y, indices = self.downsample(x)

        return (y, x, indices)

    def downsample(self, x):
        """Define the downsampling method.

        The `~pysegcnn.core.layers.EncoderBlock.downsample` `method should
        implement the spatial pooling operation.

        Use one of the following functions to downsample:
            - `torch.nn.functional.max_pool2d`
            - `torch.nn.functional.interpolate`

        See `pysegcnn.core.layers.ConvBnReluMaxPool` for an example
        implementation.

        Parameters
        ----------
        x : `torch.Tensor`
            Input tensor, e.g. output of a convolutional block.

        Raises
        ------
        NotImplementedError
            Raised if `pysegcnn.core.layers.EncoderBlock` is not inherited.

        Returns
        -------
        x : `torch.Tensor`
            The spatially downsampled tensor.
        indices : `torch.Tensor` or `None`
            Optional indices of the downsampling method, e.g. indices of the
            maxima when using `torch.nn.functional.max_pool2d`. Useful for
            upsampling later. If no indices are required to upsample, simply
            return ``indices`` = `None`.

        """
        raise NotImplementedError('Implement the downsampling function.')


class DecoderBlock(Block):
    """Block of a convolutional decoder.

    Parameters
    ----------
    in_channels : `int`
        Number of input channels.
    out_channels : `int`
        Number of output channels.
    **kwargs: 'dict' [`str`]
         Additional arguments passed to `pysegcnn.core.layers.Conv2dSame`.

    Returns
    -------
    None.

    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

    def forward(self, x, feature, indices, skip):
        """Forward pass of a decoder block.

        Parameters
        ----------
        x : `torch.Tensor`
            Input tensor.
        feature : `torch.Tensor`, shape=(batch, channel, height, width)
            Intermediate output of a layer in the encoder.
            If ``skip`` = True, ``feature`` is concatenated (along the channel
            axis) to the output of the respective upsampling layer in the
            decoder (skip connection).
        indices : `torch.Tensor` or `None`
            Indices of the encoder downsampling method.
        skip : `bool`
            Whether to apply the skip connection.

        Returns
        -------
        x : `torch.Tensor`
            Output of the decoder block.

        """
        # upsample
        x = self.upsample(x, feature, indices)

        # check whether to apply the skip connection
        # skip connection: concatenate the output of a layer in the encoder to
        # the corresponding layer in the decoder (along the channel axis)
        if skip:
            x = torch.cat([x, feature], axis=1)

        # output of the convolutional layer
        x = self.conv(x)

        return x

    def upsample(self, x, feature, indices):
        """Define the upsampling method.

        The `~pysegcnn.core.layers.DecoderBlock.upsample` `method should
        implement the spatial upsampling operation.

        Use one of the following functions to upsample:
            - `torch.nn.functional.max_unpool2d`
            - `torch.nn.functional.interpolate`

        See `pysegcnn.core.layers.ConvBnReluMaxUnpool` or
        `pysegcnn.core.layers.ConvBnReluUpsample` for an example
        implementation.

        Parameters
        ----------
        x : `torch.Tensor`
            Input tensor, e.g. output of a convolutional block.
        feature : `torch.Tensor`, shape=(batch, channel, height, width)
            Intermediate output of a layer in the encoder. Used to implement
            skip connections.
        indices : `torch.Tensor` or `None`
            Indices of the encoder downsampling method.

        Raises
        ------
        NotImplementedError
            Raised if `pysegcnn.core.layers.DecoderBlock` is not inherited.

        Returns
        -------
        x : `torch.Tensor`
            The spatially upsampled tensor.

        """
        raise NotImplementedError('Implement the upsampling function')


class Encoder(nn.Module):
    """Generic convolutional encoder.

    When instanciating an encoder-decoder architechure, ``filters`` should be
    the same for `pysegcnn.core.layers.Encoder` and
    `pysegcnn.core.layers.Decoder`.

    See `pysegcnn.core.models.UNet` for an example implementation.

    Parameters
    ----------
    filters : `list` [`int`]
        List of input channels to each convolutional block. The length of
        ``filters`` determines the depth of the encoder. The first element of
        ``filters`` has to be the number of channels of the input images.
    block : `pysegcnn.core.layers.EncoderBlock`
        The convolutional block defining a layer in the encoder.
        A subclass of `pysegcnn.core.layers.EncoderBlock`, e.g.
        `pysegcnn.core.layers.ConvBnReluMaxPool`.
    **kwargs: 'dict' [`str`]
        Additional arguments passed to `pysegcnn.core.layers.Conv2dSame`.

    Returns
    -------
    None.

    """

    def __init__(self, filters, block, **kwargs):
        super().__init__()

        # the number of filters for each block: the first element of filters
        # has to be the number of input channels
        self.features = np.asarray(filters)

        # the block of operations defining a layer in the encoder
        if not issubclass(block, EncoderBlock):
            raise TypeError('"block" expected to be a subclass of {}.'
                            .format(repr(EncoderBlock)))
        self.block = block

        # construct the encoder layers
        self.layers = []
        for lyr, lyrp1 in zip(self.features, self.features[1:]):
            # append blocks to the encoder layers
            self.layers.append(self.block(lyr, lyrp1, **kwargs))

        # convert list of layers to ModuleList
        self.layers = nn.ModuleList(*[self.layers])

    def forward(self, x):
        """Forward pass of the encoder.

        Stores intermediate outputs in a dictionary. The keys of the dictionary
        are the number of the network layers and the values are dictionaries
        with the following (key, value) pairs:
            ``"feature"``
                The intermediate encoder outputs (`torch.Tensor`).
            ``"indices"``
                The indices of the max pooling layer, if required
                (`torch.Tensor`).

        Parameters
        ----------
        x : `torch.Tensor`
            Input image.

        Returns
        -------
        x : `torch.Tensor`
            Output of the encoder.

        """
        # initialize a dictionary that caches the intermediate outputs, i.e.
        # features and pooling indices of each block in the encoder
        self.cache = {}
        for i, layer in enumerate(self.layers):
            # apply current encoder layer forward pass
            x, y, ind = layer.forward(x)

            # store intermediate outputs for optional skip connections
            self.cache[i] = {'feature':  y, 'indices': ind}

        return x


class Decoder(nn.Module):
    """Generic convolutional decoder.

    When instanciating an encoder-decoder architechure, ``filters`` should be
    the same for `pysegcnn.core.layers.Encoder` and
    `pysegcnn.core.layers.Decoder`.

    See `pysegcnn.core.models.UNet` for an example implementation.

    Parameters
    ----------
    filters : `list` [`int`]
        List of input channels to each convolutional block. The length of
        ``filters`` determines the depth of the decoder. The first element of
        ``filters`` has to be the number of channels of the input images.
    block : `pysegcnn.core.layers.DecoderBlock`
        The convolutional block defining a layer in the decoder.
        A subclass of `pysegcnn.core.layers.DecoderBlock`, e.g.
        `pysegcnn.core.layers.ConvBnReluMaxUnpool`.
    skip : `bool`
        Whether to apply skip connections from the encoder to the decoder.
    **kwargs: 'dict' [`str`]
        Additional arguments passed to `pysegcnn.core.layers.Conv2dSame`.

    Returns
    -------
    None.

    """

    def __init__(self, filters, block, skip=True, **kwargs):
        super().__init__()

        # the block of operations defining a layer in the decoder
        if not issubclass(block, DecoderBlock):
            raise TypeError('"block" expected to be a subclass of {}.'
                            .format(repr(DecoderBlock)))
        self.block = block

        # the number of filters for each block is symmetric to the encoder:
        # the last two element of filters have to be equal in order to apply
        # last skip connection
        self.features = np.asarray(filters)[::-1]
        self.features[-1] = self.features[-2]

        # whether to apply skip connections
        self.skip = skip

        # in case of skip connections, the number of input channels to
        # each block of the decoder is doubled
        n_in = 2 if self.skip else 1

        # construct decoder layers
        self.layers = []
        for lyr, lyrp1 in zip(n_in * self.features, self.features[1:]):
            self.layers.append(self.block(lyr, lyrp1, **kwargs))

        # convert list of layers to ModuleList
        self.layers = nn.ModuleList(*[self.layers])

    def forward(self, x, enc_cache):
        """Forward pass of the decoder.

        Parameters
        ----------
        x : `torch.Tensor`
            Output of the encoder.
        enc_cache : `dict` [`dict`]
            Cache dictionary. The keys of the dictionary are the number of the
            network layers and the values are dictionaries with the following
            (key, value) pairs:
                ``"feature"``
                    The intermediate encoder outputs (`torch.Tensor`).
                ``"indices"``
                    The indices of the max pooling layer (`torch.Tensor`).

        Returns
        -------
        x : `torch.Tensor`
            Output of the decoder.

        """
        # for each layer, upsample input and apply optional skip connection
        for i, layer in enumerate(self.layers):

            # get intermediate outputs from encoder: iterate the encoder cache
            # in reversed direction, i.e. from last to first encoder layer
            cache = enc_cache[len(self.layers) - (i+1)]
            feature, indices = cache['feature'], cache['indices']

            # apply current decoder layer forward pass
            x = layer.forward(x, feature, indices, self.skip)

        return x


class ConvBnReluMaxPool(EncoderBlock):
    """Block of convolution, batchnorm, relu and 2x2 max pool.

    Parameters
    ----------
    in_channels : `int`
        Number of input channels.
    out_channels : `int`
        Number of output channels.
    **kwargs: 'dict' [`str`]
        Additional keyword arguments passed to
        `pysegcnn.core.layers.Conv2dSame`.

    Returns
    -------
    None.

    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

    def layers(self):
        """Sequence of convolution, batchnorm and relu layers.

        Returns
        -------
        layers : `torch.nn.Sequential` [`torch.nn.Module`]
            An instance of `torch.nn.Sequential` containing the sequence
            of convolution, batchnorm and relu layer (`torch.nn.Module`)
            instances.

        """
        return conv_bn_relu(self.in_channels, self.out_channels, **self.kwargs)

    def downsample(self, x):
        """2x2 max pooling layer, `torch.nn.functional.max_pool2d`.

        Parameters
        ----------
        x : `torch.Tensor`
            Input tensor.

        Returns
        -------
        x : `torch.Tensor`
            The 2x2 max pooled tensor.
        indices : `torch.Tensor` or `None`
            The indices of the maxima. Useful for upsampling with
            `torch.nn.functional.max_unpool2d`.
        """
        x, indices = F.max_pool2d(x, kernel_size=2, return_indices=True)
        return x, indices


class ConvBnReluMaxUnpool(DecoderBlock):
    """Block of convolution, batchnorm, relu and 2x2 max unpool.

    Parameters
    ----------
    in_channels : `int`
        Number of input channels.
    out_channels : `int`
        Number of output channels
    **kwargs: 'dict' [`str`]
        Additional keyword arguments passed to
        `pysegcnn.core.layers.Conv2dSame`.

    Returns
    -------
    None.

    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

    def layers(self):
        """Sequence of convolution, batchnorm and relu layers.

        Returns
        -------
        layers : `torch.nn.Sequential` [`torch.nn.Module`]
            An instance of `torch.nn.Sequential` containing the sequence
            of convolution, batchnorm and relu layer (`torch.nn.Module`)
            instances.

        """
        return conv_bn_relu(self.in_channels, self.out_channels, **self.kwargs)

    def upsample(self, x, feature, indices):
        """2x2 max unpooling layer.

        Parameters
        ----------
        x : `torch.Tensor`
            Input tensor.
        feature : `torch.Tensor`, shape=(batch, channel, height, width)
            Intermediate output of a layer in the encoder. Used to determine
            the output shape of the upsampling operation.
        indices : `torch.Tensor`
            The indices of the maxima of the max pooling operation
            (as returned by `torch.nn.functional.max_pool2d`).

        Returns
        -------
        x : `torch.Tensor`
            The 2x2 max unpooled tensor.

        """
        return F.max_unpool2d(x, indices, kernel_size=2,
                              output_size=feature.shape[2:])


class ConvBnReluUpsample(DecoderBlock):
    """Block of convolution, batchnorm, relu and nearest neighbor upsampling.

    Parameters
    ----------
    in_channels : `int`
        Number of input channels.
    out_channels : `int`
        Number of output channels
    **kwargs: 'dict' [`str`]
        Additional arguments passed to `pysegcnn.core.layers.Conv2dSame`.

    Returns
    -------
    None.

    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

    def layers(self):
        """Sequence of convolution, batchnorm and relu layers.

        Returns
        -------
        layers : `torch.nn.Sequential` [`torch.nn.Module`]
            An instance of `torch.nn.Sequential` containing the sequence
            of convolution, batchnorm and relu layer (`torch.nn.Module`)
            instances.

        """
        return conv_bn_relu(self.in_channels, self.out_channels, **self.kwargs)

    def upsample(self, x, feature, indices=None):
        """Nearest neighbor upsampling.

        Parameters
        ----------
        x : `torch.Tensor`
            Input tensor.
        feature : `torch.Tensor`, shape=(batch, channel, height, width)
            Intermediate output of a layer in the encoder. Used to determine
            the output shape of the upsampling operation.
        indices : `None`, optional
            The indices of the maxima of the max pooling operation
            (as returned by `torch.nn.functional.max_pool2d`). Not required by
            this upsampling method.

        Returns
        -------
        x : `torch.Tensor`
            The 2x2 max unpooled tensor.

        """
        return F.interpolate(x, size=feature.shape[2:], mode='nearest')
