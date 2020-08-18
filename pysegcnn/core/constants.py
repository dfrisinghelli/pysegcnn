"""A collection of enumerations of constant values."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import enum


# Landsat 8 bands
class Landsat8(enum.Enum):
    """The spectral bands of the Landsat 8 sensors.

    sensors:
        - Operational Land Imager (OLI), (bands 1-9)
        - Thermal Infrared Sensor (TIRS), (bands 10, 11)

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
    # tir2 = 11


# Sentinel 2 bands
class Sentinel2(enum.Enum):
    """The spectral bands of the Sentinel-2 MultiSpectral Instrument (MSI)."""

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


# generic class label enumeration class
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


# labels of the Sparcs dataset
class SparcsLabels(Label):
    """Class labels of the `Sparcs`_ dataset.

    .. _Sparcs:
        https://www.usgs.gov/land-resources/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation

    """

    Shadow = 0, 'grey'
    Shadow_over_water = 1, 'darkblue'
    Water = 2, 'blue'
    Snow = 3, 'lightblue'
    Land = 4, 'sienna'
    Cloud = 5, 'white'
    Flooded = 6, 'yellow'


# labels of the Cloud95 dataset
class Cloud95Labels(Label):
    """Class labels of the `Cloud-95`_ dataset.

    .. _Cloud-95:
        https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset

    """

    Clear = 0, 'skyblue'
    Cloud = 1, 'white'


# labels of the ProSnow dataset
class ProSnowLabels(Label):
    """Class labels of the ProSnow datasets."""

    Cloud = 0, 'white'
    Snow = 1, 'lightblue'
    Snow_free = 2, 'sienna'
