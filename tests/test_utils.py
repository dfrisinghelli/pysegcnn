# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 09:06:45 2020

@author: Daniel
"""
# externals
import pytest
import numpy as np

# locals
from pysegcnn.core.utils import img2np, is_divisible, reconstruct_scene


def test_img2np():

    # generate a random image of arbitrary size

    # number of bands
    bands = np.random.randint(low=1, high=5, size=1).item()

    # spatial size: (height, width)
    height = np.random.randint(low=500, high=1000, size=1).item()
    width = np.random.randint(low=500, high=1000, size=1).item()

    # random image to test function
    image = np.random.randint(255, size=(bands, height, width))

    # test whether img2np raises a ValueError if the image is not evenly
    # divisible into ntiles of size (tile_size, tile_size) and pad=False
    ntiles = image.shape[1] / 4
    tile_size = int(ntiles if not ntiles.is_integer() else ntiles + 1)
    with pytest.raises(ValueError):
        img_blocks = img2np(image, tile_size, pad=False)

    print('Testing image tiling with different tile sizes ...')

    # test whether img2np correctly divides the image into ntiles
    # for a range of different tile sizes
    for size in [64, 100, 128, 256]:
        ntiles, padding = is_divisible((image.shape[1], image.shape[2]),
                                       size, pad=True)

        # number of tiles along columns and rows
        ntiles_columns = int((image.shape[2] + padding[1] + padding[3]) /
                             size)
        ntiles_rows = int((image.shape[1] + padding[0] + padding[2]) /
                          size)
        img_blocks = img2np(image, size, pad=True, verbose=True)

        # check whether the shape of the image is correct
        if bands > 1:
            assert img_blocks.shape == (ntiles, bands, size, size)
        else:
            assert img_blocks.shape == (ntiles, size, size)

        # add the padding to the image
        img_padded = np.pad(image, ((0, 0), (padding[2], padding[0]),
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
                stile = img2np(image, size, k, pad=True)

                # current tile in the padded image
                fr = (i * size)
                tr = fr + size
                fc = (j * size)
                tc = fc + size
                assert np.array_equal(tile,
                                      img_padded[:, fr:tr, fc:tc].squeeze())
                assert np.array_equal(tile, stile)
                assert np.array_equal(stile,
                                      img_padded[:, fr:tr, fc:tc].squeeze())
                print('Tile {} with topleft corner {} successfully created ...'
                      .format(k, (fr, fc)))
                k += 1


def test_reconstruct_scene():

    # generate a random image of arbitrary size

    # number of bands
    bands = np.random.randint(low=1, high=5, size=1).item()

    # spatial size: (height, width)
    height = np.random.randint(low=500, high=1000, size=1).item()
    width = np.random.randint(low=500, high=1000, size=1).item()

    # random image to test function
    image = np.random.randint(255, size=(bands, height, width))

    # test reconstruction for different tile sizes
    for size in [64, 100, 128, 256]:

        # get the amount of padding
        ntiles, padding = is_divisible((image.shape[1], image.shape[2]),
                                       size, pad=True)

        # add the padding to the image
        img_padded = np.pad(image, ((0, 0), (padding[2], padding[0]),
                                    (padding[1], padding[3])),
                            'constant', constant_values=0)

        # calculate resulting image size
        height = img_padded.shape[1]
        width = img_padded.shape[2]

        # split image into blocks
        img_blocks = img2np(image, size, pad=True)

        # reconstruct the blocks to the original image
        rec_img = reconstruct_scene(img_blocks, (height, width),
                                    nbands=image.shape[0])

        assert np.array_equal(img_padded.squeeze(), rec_img)
