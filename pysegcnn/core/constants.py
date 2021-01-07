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

    @classmethod
    def band_dict(cls):
        return {band.value: band.name for band in cls}


class Landsat8(MultiSpectralSensor):
    """The spectral bands of the `Landsat-8`_ sensors.

    sensors:
        - Operational Land Imager (OLI), (bands 1-9)
        - Thermal Infrared Sensor (TIRS), (bands 10, 11)

    .. _Landsat-8:
        https://www.usgs.gov/land-resources/nli/landsat/landsat-8?qt-science_support_page_related_con=0#qt-science_support_page_related_con

    """

    violet = 1
    blue = 2
    green = 3
    red = 4
    nir = 5
    swir1 = 6
    swir2 = 7
    pan = 8
    cirrus = 9
    tir1 = 10
    tir2 = 11


class Sentinel2(MultiSpectralSensor):
    """The spectral bands of the `Sentinel-2`_ MultiSpectral Instrument (MSI).

    .. _Sentinel-2:
        https://sentinel.esa.int/web/sentinel/missions/sentinel-2

    """

    aerosol = 1
    blue = 2
    green = 3
    red = 4
    vnir1 = 5
    vnir2 = 6
    vnir3 = 7
    nir = 8
    nnir = '8A'
    vapor = 9
    cirrus = 10
    swir1 = 11
    swir2 = 12


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
    Water = 2, 'blue'
    Snow = 3, 'lightblue'
    Land = 4, 'forestgreen'
    Cloud = 5, 'white'
    Flooded = 6, 'yellow'
    No_data = 7, 'black'


class Cloud95Labels(Label):
    """Class labels of the `Cloud-95`_ dataset.

    .. _Cloud-95:
        https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset

    """

    Clear = 0, 'skyblue'
    Cloud = 1, 'white'


class AlcdLabels(Label):
    """Class labels of the `Alcd`_ dataset.

    .. _Alcd:
        https://zenodo.org/record/1460961#.XYCTRzYzaHt

    """

    No_data = 0, 'black'
    Not_used = 1, 'black'
    Cloud = 2, 'white'
    Cirrus = 3, 'white'
    Shadow = 4, 'grey'
    Land = 5, 'forestgreen'
    Water = 6, 'blue'
    Snow = 7, 'lightblue'


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
