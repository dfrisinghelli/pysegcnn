"""Threshold methods for snow detection.

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
import torch
import torch.nn as nn
import numpy as np


class ThresholdMethod(nn.Module):
    """Generic class for snow detection methods based on thresholds.

    Attributes
    ----------
    x : :py:class:`numpy.ndarray` or :py:class:`torch.Tensor`
        The spectral input data.
    bands : `list` [`str`]
        List describing the order of the spectral bands in ``x``.

    """

    def __init__(self, x, bands):
        """Initialize.

        Parameters
        ----------
        x : :py:class:`numpy.ndarray` or :py:class:`torch.Tensor`
            The spectral input data. If ``x`` is three-dimensional, the shape
            is expected to be (bands, height, width), whereas if ``x`` is
            four-dimensional the shape is expected to be
            (batch, bands, height, width).
        bands : `list` [`str`]
            List describing the order of the spectral bands in ``x``, e.g. if
            bands = ['red', 'green'], the red band is expected to be
            ``x[:, 0, ...]``, the green band ``x[:, 1, ...]`` etc.

        """
        # the input array to classify
        self.x = np.asarray(x)

        # get the spectral bands of x
        self.bands = bands
        for band in self.bands:
            # check input dimensionality
            if x.ndim > 3:
                setattr(self, band, x[:, bands.index(band), ...])
            else:
                setattr(self, band, x[bands.index(band), ...])

    @property
    def ndsi(self):
        """Compute the normalized difference snow index.

        Returns
        -------
        ndsi : :py:class:`numpy.ndarray`
            Normalized Difference Snow Index.

        """
        return (self.green - self.swir1) / (self.green + self.swir1)

    @property
    def ndvi(self):
        """Compute the normalized difference vegetation index.

        Returns
        -------
        ndvi : :py:class:`numpy.ndarray`
            Normalized Difference Vegetation Index.

        """
        return (self.nir - self.red) / (self.nir + self.red)


class Snowmap(ThresholdMethod):
    """SNOWMAP algorithm by `Hall (1995)`_, based on `Dozier (1989)`_.

    .. _Hall (1995):
        https://www.sciencedirect.com/science/article/pii/003442579500137P

    .. _Dozier (1989):
        https://www.sciencedirect.com/science/article/pii/0034425789901016

    Attributes
    ----------
    snow : `int`
        Class label of snow covered pixels.
    snow_free : `int`
        Class labels of snow free pixels.

    """

    def __init__(self, x, bands, snow=1, snow_free=0):
        """Initialize.

        Parameters
        ----------
        x : :py:class:`numpy.ndarray` or :py:class:`torch.Tensor`
            The spectral input data.
        bands : `list` [`str`]
            List describing the order of the spectral bands in ``x``.
        snow : `int`, optional
            Class label of snow covered pixels.. The default is 1.
        snow_free : `int`, optional
            Class labels of snow free pixels. The default is 0.

        """
        super().__init__(x, bands)

        # class labels defining snow covered/free pixel
        self.snow = snow
        self.snow_free = snow_free

    def forward(self):
        """Classify.

        Returns
        -------
        x : :py:class:`torch.Tensor`
            The binary snow cover classification after SNOWMAP.

        """
        # binary classification map
        x = torch.ones(size=self.ndsi.shape) * self.snow_free
        x[torch.where((self.ndsi > 0.4) & (self.nir > 0.11))] = self.snow

        return x


class SupportedThresholdMethods(enum.Enum):
    """Names and corresponding classes of the implemented threshold models."""

    snowmap = Snowmap
