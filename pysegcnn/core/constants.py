"""A collection of constants.

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


class MultiSpectralSensor(enum.Enum):
    """A generic class for a multispectral sensor."""

    @property
    def number(self):
        """Return the number of a band."""
        return self.value[0]

    @property
    def range(self):
        """Return the spectral range of a band in microns."""
        return self.value[1]

    @classmethod
    def band_dict(cls):
        return {band.number: band.name for band in cls}

    @classmethod
    def spectral_range(cls):
        return {band.name: band.range for band in cls}


class Landsat8(MultiSpectralSensor):
    """The spectral bands of the `Landsat-8`_ sensors.

    sensors:
        - Operational Land Imager (OLI), (bands 1-9)
        - Thermal Infrared Sensor (TIRS), (bands 10, 11)

    .. _Landsat-8:
        https://www.usgs.gov/land-resources/nli/landsat/landsat-8?qt-science_support_page_related_con=0#qt-science_support_page_related_con

    """

    ublue = 1, (0.43, 0.45)
    blue = 2, (0.45, 0.51)
    green = 3, (0.53, 0.59)
    red = 4, (0.64, 0.67)
    nir = 5, (0.85, 0.88)
    swir1 = 6, (1.57, 1.65)
    swir2 = 7, (2.11, 2.29)
    pan = 8, (0.5, 0.68)
    cirrus = 9, (1.36, 1.38)
    tir1 = 10, (10.60, 11.19)
    tir2 = 11, (11.50, 12.51)


class Sentinel2(MultiSpectralSensor):
    """The spectral bands of the `Sentinel-2`_ MultiSpectral Instrument (MSI).

    .. _Sentinel-2:
        https://sentinel.esa.int/web/sentinel/missions/sentinel-2

    """

    ublue = 1, (0.43, 0.45)
    blue = 2, (0.46, 0.53)
    green = 3, (0.54, 0.58)
    red = 4, (0.65, 0.68)
    vnir1 = 5, (0.69, 0.71)
    vnir2 = 6, (0.73, 0.75)
    vnir3 = 7, (0.77, 0.79)
    nnir = 8, (0.78, 0.89)
    nir = '8A', (0.85, 0.88)
    vapor = 9, (0.94, 0.96)
    cirrus = 10, (1.36, 1.39)
    swir1 = 11, (1.57, 1.66)
    swir2 = 12, (2.11, 2.29)


class Label(enum.Enum):
    """Generic enumeration for class labels."""

    @property
    def id(self):
        """Return the value of a class in the ground truth."""
        return self.value[0]

    @property
    def color(self):
        """Return the color to plot a class."""
        return self.value[1]

    @classmethod
    def label_dict(cls):
        """Return the enumeration as a nested dictionary."""
        return {label.id: {'label': label.name.replace('_', ' '),
                           'color': label.color} for label in cls}


class LabelMapping(dict):
    """Generic class for mapping class labels."""

    def to_numpy(self):
        """Return the label mapping dictionary as :py:class:`numpy.ndarray`."""
        return np.array(list(self.items()))


class SparcsLabels(Label):
    """Class labels of the `Sparcs`_ dataset.

    .. _Sparcs:
        https://www.usgs.gov/land-resources/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation

    """

    Shadow = 0, 'grey'
    Shadow_over_water = 1, 'darkblue'
    Water = 2, 'mediumblue'
    Snow = 3, 'mediumturquoise'
    Land = 4, 'seagreen'
    Cloud = 5, 'lightgrey'
    Flooded = 6, 'yellow'
    No_data = 7, 'black'


class Cloud95Labels(Label):
    """Class labels of the `Cloud-95`_ dataset.

    .. _Cloud-95:
        https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset

    """

    Clear = 0, 'skyblue'
    Cloud = 1, 'lightgrey'


class AlcdLabels(Label):
    """Class labels of the `Alcd`_ dataset.

    .. _Alcd:
        https://zenodo.org/record/1460961#.XYCTRzYzaHt

    """

    Shadow = 4, 'grey'
    Water = 6, 'mediumblue'
    Snow = 7, 'mediumturquoise'
    Land = 5, 'seagreen'
    Cloud = 2, 'lightgrey'
    Cirrus = 3, 'lightgrey'
    No_data = 0, 'black'
    Not_used = 1, 'black'


class Gdal2Numpy(enum.Enum):
    """Data type mapping from gdal to numpy."""

    Byte = np.uint8
    UInt8 = np.uint8
    Int8 = np.int8
    UInt16 = np.uint16
    Int16 = np.int16
    UInt32 = np.uint32
    Int32 = np.int32
    Float32 = np.float32
    Float64 = np.float64
    CInt16 = np.complex64
    CInt32 = np.complex64
    CFloat32 = np.complex64
    CFloat64 = np.complex64


# class label mapping from the Sparcs dataset to the Alcd dataset
Sparcs2Alcd = LabelMapping({

    SparcsLabels.Shadow.id: AlcdLabels.Shadow.id,
    SparcsLabels.Shadow_over_water.id: AlcdLabels.Shadow.id,
    SparcsLabels.Water.id: AlcdLabels.Water.id,
    SparcsLabels.Snow.id: AlcdLabels.Snow.id,
    SparcsLabels.Land.id: AlcdLabels.Land.id,
    SparcsLabels.Cloud.id: AlcdLabels.Cloud.id,
    SparcsLabels.Flooded.id: AlcdLabels.Land.id,
    SparcsLabels.No_data.id: AlcdLabels.No_data.id

    })

# class label mapping from the Alcd dataset to the Sparcs dataset
Alcd2Sparcs = LabelMapping({

    AlcdLabels.No_data.id: SparcsLabels.No_data.id,
    AlcdLabels.Not_used.id: SparcsLabels.No_data.id,
    AlcdLabels.Cloud.id: SparcsLabels.Cloud.id,
    AlcdLabels.Cirrus.id: SparcsLabels.Cloud.id,
    AlcdLabels.Shadow.id: SparcsLabels.Shadow.id,
    AlcdLabels.Land.id: SparcsLabels.Land.id,
    AlcdLabels.Water.id: SparcsLabels.Water.id,
    AlcdLabels.Snow.id: SparcsLabels.Snow.id

    })


def map_labels(source, target):
    """Map labels from a source domain to a target domain.

    Parameters
    ----------
    source : :py:class:`pysegcnn.core.constants.Label`
        The source domain labels, i.e. the labels a model is trained with.
    target : :py:class:`pysegcnn.core.constants.Label`
        The target domain labels, i.e. the labels of the dataset to apply the
        model to.

    Raises
    ------
    ValueError
        Raised if no label mapping from ``source`` to ``target`` is defined.

    Returns
    -------
    label_map : :py:class:`pysegcnn.core.constants.LabelMapping` [`int`, `int`]
        Dictionary with source labels as keys and corresponding target labels
        as values.

    """
    # if source and target labels are equal, the label mapping is the
    # identity
    if source is target:
        label_map = None

    # mapping from Sparcs to Alcd
    elif source is SparcsLabels and target is AlcdLabels:
        label_map = Sparcs2Alcd

    # mapping from Alcd to Sparcs
    elif source is AlcdLabels and target is SparcsLabels:
        label_map = Alcd2Sparcs

    else:
        raise ValueError('Unknown label mapping from {} to {}'.
                         format(source.__name__, target.__name__))

    return label_map
