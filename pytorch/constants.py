# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:58:20 2020

@author: Daniel
"""
# builtins
from __future__ import absolute_import
import enum


# Landsat 8 bands
class Landsat8(enum.Enum):
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


# labels of the Sparcs dataset
class SparcsLabels(enum.Enum):
    Shadow = 0, 'black'
    Shadow_over_water = 1, 'darkblue'
    Water = 2, 'blue'
    Snow = 3, 'lightblue'
    Land = 4, 'grey'
    Cloud = 5, 'white'
    Flooded = 6, 'yellow'


# labels of the Cloud95 dataset
class Cloud95Labels(enum.Enum):
    Clear = 0, 'skyblue'
    Cloud = 1, 'white'


# labels of the ProSnow dataset
class ProSnowLabels(enum.Enum):
    Land = 0, 'grey'
    Cloud = 1, 'white'
    Snow = 2, 'lightblue'
