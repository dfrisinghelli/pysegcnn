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


# Landsat 8 bands
class Landsat8(enum.Enum):
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


# Sentinel 2 bands
class Sentinel2(enum.Enum):
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
    No_data = 7, 'black'


# labels of the Cloud95 dataset
class Cloud95Labels(Label):
    """Class labels of the `Cloud-95`_ dataset.

    .. _Cloud-95:
        https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset

    """

    Clear = 0, 'skyblue'
    Cloud = 1, 'white'


# labels of the ProSnow datasets
class ProSnowLabels(Label):
    """Class labels of the ProSnow datasets."""

    Cloud = 0, 'white'
    Snow = 1, 'lightblue'
    Snow_free = 2, 'sienna'
    No_data = 3, 'black'


def map_labels(source, target):
    """Map labels from one dataset to another.

    Parameters
    ----------
    source : :py:class:`enum.Enum`
        The source domain labels, i.e. the labels a model is trained with.
    target : :py:class:`enum.Enum`
        The target domain labels, i.e. the labels of the dataset to apply the
        model to.

    Returns
    -------
    label_map : `dict` [`int`, `int`]
        Dictionary with source labels as keys and corresponding target labels
        as values.

    """
    # if source and target labels are equal, the label mapping is the identity
    if source is target:
        label_map = None

    # mapping from Sparcs to ProSnow
    if source is SparcsLabels and target is ProSnowLabels:
        # label transformation mapping
        label_map = {
            # Shadow = Snow Free
            SparcsLabels.Shadow.id: ProSnowLabels.Snow_free.id,
            # Shadow ow = Snow Free
            SparcsLabels.Shadow_over_water: ProSnowLabels.Snow_free.id,
            # Water = Snow Free
            SparcsLabels.Water.id: ProSnowLabels.Snow_free.id,
            # Snow = Snow
            SparcsLabels.Snow.id: ProSnowLabels.Snow.id,
            # Land = Snow Free
            SparcsLabels.Land.id: ProSnowLabels.Snow_free.id,
            # Cloud = Cloud
            SparcsLabels.Cloud.id: ProSnowLabels.Cloud.id,
            # Flooded = Snow Free
            SparcsLabels.Flooded.id: ProSnowLabels.Snow_free.id,
            # No data = No data
            SparcsLabels.No_data.id: ProSnowLabels.No_data.id
            }

    return label_map
