.. api:

.. currentmodule:: pysegcnn

API Reference
=============

This page lists functions and classes of the ``pysegcnn`` package, which are
relevant at the API level. If you need to dig deeper into the source files, go
to the git `repository <https://gitlab.inf.unibz.it/REMSEN/ccisnow/pysegcnn>`_.

Dataset
-------

The ``pysegcnn`` package offers support for custom datasets, which are compliant
to the PyTorch `standard <https://pytorch.org/docs/stable/data.html>`_.
Currently, image datasets from the `Landsat <https://landsat.usgs.gov/>`_ and
`Sentinel-2`_ satellites are supported.

Generic classes
^^^^^^^^^^^^^^^

The following two generic classes can be inherited to implement a
`custom dataset <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>`_.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    core.dataset.ImageDataset
    core.dataset.StandardEoDataset


Specific classes
^^^^^^^^^^^^^^^^

Specific classes for some open-source image datasets. Currently, the following
spaceborne multispectral image datasets are supported out-of-the-box:

.. autosummary::
    :toctree: generated/
    :nosignatures:

    core.dataset.SparcsDataset
    core.dataset.Cloud95Dataset

Models
------

The ``pysegcnn`` package ships with a customizable interface to build
convolutional neural networks for image segmentation.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    core.layers.Conv2dSame
    core.layers.Block
    core.layers.ConvBnReluMaxPool
    core.layers.ConvBnReluMaxUnpool
    core.layers.ConvBnReluUpsample
    core.layers.EncoderBlock
    core.layers.DecoderBlock
    core.layers.Encoder
    core.layers.Decoder

Implemented models.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    core.models.Network
    core.models.UNet


Training, Validation and Test set
---------------------------------

Classes to split a dataset into training, validation and test set.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    core.split.DateSplit
    core.split.RandomTileSplit
    core.split.RandomSceneSplit

..
    Links:

.. _Sparcs:
    https://www.usgs.gov/land-resources/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation

.. _Cloud-95:
    https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset

.. _Mohajerani & Saeedi (2020):
    https://arxiv.org/abs/2001.08768

.. _U-Net:
    https://arxiv.org/abs/1505.04597

.. _Landsat-8:
    https://www.usgs.gov/land-resources/nli/landsat/landsat-8?qt-science_support_page_related_con=0#qt-science_support_page_related_con

.. _Sentinel-2:
    https://sentinel.esa.int/web/sentinel/missions/sentinel-2

.. _Hughes & Hayes (2014):
    https://www.mdpi.com/2072-4292/6/6/4907

.. _Early Stopping:
    https://en.wikipedia.org/wiki/Early_stopping

.. _equations:
    https://www.usgs.gov/land-resources/nli/landsat/using-usgs-landsat-level-1-data-product
