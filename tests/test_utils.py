# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 09:06:45 2020

@author: Daniel
"""
# builtins
from __future__ import absolute_import
import os

# externals
import gdal
import pytest
import numpy as np

# locals
from pytorch.utils import img2np, is_divisible, tile_topleft_corner
from main.config import dataset_path


def test_img2np():

    # define images to test function
    images = []

    # iterate over the selected dataset and get all tif files
    for dirpath, dirname, files in os.walk(dataset_path):
        if files:
            images.extend([os.path.join(dirpath, f) for f in files
                           if f.endswith('tif')])

    # randomly select an image from the list of files
    path = images[np.random.randint(len(images), size=1).item()]
    print('Processing image: {}'.format(path))

    # read the image to a numpy array
    img = gdal.Open(path)
    img_data = img.GetRasterBand(1).ReadAsArray()
    nbands = img.RasterCount

    # test whether img2np raises a ValueError if the image is not evenly
    # divisible into ntiles of size (tile_size, tile_size) and pad=False
    ntiles = img.RasterXSize / 4
    tile_size = int(ntiles if not ntiles.is_integer() else ntiles + 1)
    with pytest.raises(ValueError) as error:
        img_blocks = img2np(path, tile_size, pad=False)

    print('Testing image tiling with different tile sizes ...')

    # test whether img2np correctly divides the image into ntiles
    # for a range of different tile sizes
    for size in [64, 100, 128, 256]:
        ntiles, padding = is_divisible((img.RasterYSize, img.RasterXSize),
                                       size, pad=True)

        # number of tiles along columns and rows
        ntiles_columns = int((img.RasterXSize + padding[1] + padding[3]) /
                             size)
        ntiles_rows = int((img.RasterYSize + padding[0] + padding[2]) /
                          size)
        img_blocks = img2np(path, size, pad=True, verbose=True)

        # check whether the shape of the image is correct
        if nbands > 1:
            assert img_blocks.shape == (ntiles, nbands, size, size)
        else:
            assert img_blocks.shape == (ntiles, size, size)

        # add the padding to the image
        img_padded = np.pad(img_data, ((padding[2], padding[0]),
                                       (padding[1], padding[3])),
                            'constant', constant_values=0)

        print('Checking image tiling ...')
        # check whether the tiles contain the correct values
        k = 0
        for i in range(ntiles_rows):
            for j in range(ntiles_columns):
                # tile from the tiled array returned by img2np
                tile = img_blocks[k, ...]

                # same tile returned as single array from img2np
                stile = img2np(path, size, k, pad=True, verbose=True)

                # current tile in the padded image
                fr = (i * size)
                tr = fr + size
                fc = (j * size)
                tc = fc + size
                assert np.array_equal(tile, img_padded[fr:tr, fc:tc])
                assert np.array_equal(tile, stile)
                assert np.array_equal(stile, img_padded[fr:tr, fc:tc])
                print('Tile {} with topleft corner {} successfully created ...'
                      .format(k, (fr, fc)))
                k += 1
