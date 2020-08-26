.. api:

.. currentmodule:: pysegcnn

API Reference
=============

This page lists the available functions and classes of ``pysegcnn``.


Dataset
-------

Custom dataset classes compliant to the PyTorch `standard <https://pytorch.org/docs/stable/data.html>`_.

Image Dataset
^^^^^^^^^^^^^
Generic class to implement `custom datasets <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>`_.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    core.dataset.ImageDataset

Supported datasets
^^^^^^^^^^^^^^^^^^

The following open-source spaceborne multispectral image datasets are supported
out-of-the-box:

.. autosummary::
    :toctree: generated/
    :nosignatures:

    core.dataset.SparcsDataset
    core.dataset.Cloud95Dataset


Models
------

Layers
^^^^^^

Convolutional neural network layers.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    core.layers.Block
    core.layers.Conv2dSame
    core.layers.ConvBnReluMaxPool
    core.layers.ConvBnReluMaxUnpool
    core.layers.ConvBnReluUpsample

Encoder-Decoder architechture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generic ``Encoder`` and ``Decoder`` classes to build an encoder-decoder
architecture.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    core.layers.EncoderBlock
    core.layers.DecoderBlock
    core.layers.Encoder
    core.layers.Decoder

Neural Networks
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    core.models.Network
    core.models.UNet


.. _Sparcs:
    https://www.usgs.gov/land-resources/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation

.. _Cloud-95:
    https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset

.. _Mohajerani & Saeedi (2020):
    https://arxiv.org/abs/2001.08768

.. _U-Net:
    https://arxiv.org/abs/1505.04597
