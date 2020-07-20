# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:02:23 2020

@author: Daniel
"""
# builtins
import re
import datetime

# externals
import gdal
import numpy as np

# the following functions are utility functions for common image
# manipulation operations


# this function reads an image to a numpy array
def img2np(path, tile_size=None, tile=None):

    # open the tif file
    if path is None:
        print('Path is of NoneType, returning.')
        return
    img = gdal.Open(path)

    # check whether to read the image in tiles
    if tile_size is None:

        # create empty numpy array to store whole image
        image = np.empty(shape=(img.RasterCount, img.RasterYSize,
                                img.RasterXSize))

        # iterate over the bands of the image
        for b in range(img.RasterCount):

            # read the data of band b
            band = img.GetRasterBand(b+1)
            data = band.ReadAsArray()

            # append band b to numpy image array
            image[b, :, :] = data

    else:

        # check whether the image is evenly divisible in square tiles
        # of size (tile_size x tile_size)
        ntiles = is_divisible((img.RasterXSize, img.RasterYSize), tile_size)

        # get the indices of the top left corner for each tile
        topleft = tile_offsets((img.RasterYSize, img.RasterXSize), tile_size)

        # check whether to read all tiles or a single tile
        if tile is None:

            # create empty numpy array to store all tiles
            image = np.empty(shape=(ntiles, img.RasterCount,
                                    tile_size, tile_size))

            # iterate over the tiles
            for k, v in topleft.items():

                # iterate over the bands of the image
                for b in range(img.RasterCount):

                    # read the data of band b
                    band = img.GetRasterBand(b+1)
                    data = band.ReadAsArray(v[1], v[0],
                                            tile_size, tile_size)

                    # append band b to numpy image array
                    image[k, b, :, :] = data

        else:

            # create empty numpy array to store a single tile
            image = np.empty(shape=(img.RasterCount, tile_size, tile_size))

            # the tile of interest
            tile = topleft[tile]

            # iterate over the bands of the image
            for b in range(img.RasterCount):

                # read the data of band b
                band = img.GetRasterBand(b+1)
                data = band.ReadAsArray(tile[1], tile[0],
                                        tile_size, tile_size)

                # append band b to numpy image array
                image[b, :, :] = data

    # check if there are more than 1 band
    if not img.RasterCount > 1:
        image = image.squeeze()

    # close tif file
    del img

    # return the image
    return image


# this function checks whether an image is evenly divisible
# in square tiles of defined size tile_size
def is_divisible(img_size, tile_size):
    # calculate number of pixels per tile
    pixels_per_tile = tile_size ** 2

    # check whether the image is evenly divisible in square tiles of size
    # (tile_size x tile_size)
    ntiles = ((img_size[0] * img_size[1]) / pixels_per_tile)
    assert ntiles.is_integer(), ('Image not evenly divisible in '
                                 ' {} x {} tiles.').format(tile_size,
                                                           tile_size)

    return int(ntiles)


# this function returns the top-left corners for each tile
# if the image is evenly divisible in square tiles of
# defined size tile_size
def tile_offsets(img_size, tile_size):

    # check if divisible
    _ = is_divisible(img_size, tile_size)

    # number of tiles along the width (columns) of the image
    ntiles_columns = int(img_size[1] / tile_size)

    # number of tiles along the height (rows) of the image
    ntiles_rows = int(img_size[0] / tile_size)

    # get the indices of the top left corner for each tile
    indices = {}
    k = 0
    for i in range(ntiles_rows):
        for j in range(ntiles_columns):
            indices[k] = (i * tile_size, j * tile_size)
            k += 1

    return indices


def parse_landsat8_date(scene):

    # Landsat Collection 1 naming convention in regular expression
    sensor = 'L[COTEM]0[1-8]_'
    level = 'L[0-9][A-Z][A-Z]_'
    swath = '[0-2][0-9]{2}[0-1][0-9]{2}_'
    date = '[0-9]{4}[0-1][0-9][0-3][0-9]_'
    doy = '[0-9]{4}[0-3][0-9]{2}'
    collection = '0[0-9]_'
    category = '[A-Z]([A-Z]|[0-9])'

    # Landsat Collection 1 naming
    C1 = (sensor + level + swath + date + date + collection + category)
    Landsat_C1 = re.compile(C1)

    # Landsat naming convention before Collections
    C0 = (sensor.replace('_', '').replace('0', '') + swath.replace('_', '') +
          doy + '[A-Z]{3}' + '[0-9]{2}')
    Landsat_C0 = re.compile(C0)

    # whether a scene identifier matches the Landsat naming convention
    # if True, get the date of
    date = None
    if Landsat_C0.match(scene):
        date = doy2date(scene[9:13], scene[13:16])
    if Landsat_C1.match(scene):
        date = datetime.datetime.strptime(scene.split('_')[3], '%Y%m%d')

    return date


def parse_sentinel2_date(scene):

    # Sentinel 2 Level-1C products naming convention after 6th December 2016
    mission = 'S2[A-B]_'
    level = 'MSIL1C_'
    date = '[0-9]{4}[0-1][0-9][0-3][0-9]'
    time = 'T[0-2][0-9][0-5][0-9][0-5][0-9]_'
    processing = 'N[0-9]{4}_'
    orbit = 'R[0-1][0-9]{2}_'
    tile = 'T[0-9]{2}[A-Z]{3}_'
    level_1C = (mission + level + date + time + processing +
                orbit + tile + date + time.replace('_', ''))
    S2_L1C_New = re.compile(level_1C)

    # Sentinel 2 Level-1C products naming convention before 6th December 2016
    file_class = '[A-Z]{4}_'
    file_category = '[A-Z]{3}_'
    file_semantic = 'L[0-1]([ABC]|_)_[A-Z]{2}_'
    site = '[A-Z_]{4}_'
    aorbit = 'A[0-9]{6}_'
    S2_L1C_Old = re.compile(mission + file_class + file_category +
                            file_semantic + site + 'V' + date + time + aorbit +
                            tile.replace('_', ''))

    # ProSnow project naming convention
    ProSnow = re.compile(tile + date + time.replace('_', ''))

    # whether a scene identifier matches the Landsat naming convention
    # if True, get the date of
    date = None
    if S2_L1C_Old.match(scene):
        date = datetime.datetime.strptime(
            scene.split('_')[7].split('T')[0].replace('V', ''), '%Y%m%d')
    if S2_L1C_New.match(scene):
        date = datetime.datetime.strptime(scene.split('_')[2].split('T')[0],
                                          '%Y%m%d')
    if ProSnow.match(scene):
        date = datetime.datetime.strptime(scene.split('_')[1].split('T')[0],
                                          '%Y%m%d')

    return date


def doy2date(year, doy):
    """Converts the (year, day of the year) date format to a datetime object.

    Parameters
    ----------
    year : int
        the year
    doy : int
        the day of the year

    Returns
    -------
    date : datetime.datetime
        the converted date as datetime object
    """

    # convert year/day of year to a datetime object
    date = (datetime.datetime(int(year), 1, 1) +
            datetime.timedelta(days=(int(doy) - 1)))

    return date
