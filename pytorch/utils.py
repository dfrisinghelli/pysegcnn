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
def img2np(path, tile_size=None, tile=None, pad=False, cval=0, verbose=False):

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
        ntiles, padding = is_divisible((img.RasterYSize, img.RasterXSize),
                                       tile_size, pad)

        # image size after padding
        y_size = img.RasterYSize + padding[0] + padding[2]
        x_size = img.RasterXSize + padding[1] + padding[3]

        if verbose:
            print('Image size: {}'.format((img.RasterYSize, img.RasterXSize)))
            print('Dividing image into {} tiles of size {} ...'
                  .format(ntiles, (tile_size, tile_size)))
            print('Padding image: bottom = {}, left = {}, top = {}, right = {}'
                  ' ...'.format(*list(padding)))
            print('Padded image size: {}'.format((y_size, x_size)))

        # get the indices of the top left corner for each tile
        topleft = tile_topleft_corner((y_size, x_size), tile_size)

        # whether to read all tiles or a single tile
        if tile is not None:
            ntiles = 1

        # create empty numpy array to store the tiles
        image = np.ones((ntiles, img.RasterCount, tile_size, tile_size)) * cval

        # iterate over the topleft corners of the tiles
        for k, corner in topleft.items():

            if verbose:
                print('Creating tile {} with top-left corner {} ...'
                      .format(k, corner))

            # in case a single tile is required, skip the rest of the tiles
            if tile is not None:
                if k != tile:
                    if verbose:
                        print('Skipping tile {} ...'.format(k))
                    continue
                # set the key to 0 for correct array indexing when reading
                # a single tile from the image
                if verbose:
                    print('Processing tile {} ...'.format(k))
                k = 0

            # calculate shift between padded and original image
            row = corner[0] - padding[2] if corner[0] > 0 else corner[0]
            col = corner[1] - padding[1] if corner[1] > 0 else corner[1]
            y_tl = row + padding[2] if row == 0 else 0
            x_tl = col + padding[1] if col == 0 else 0

            # iterate over the bands of the image
            for b in range(img.RasterCount):

                # read the data of band b
                band = img.GetRasterBand(b+1)

                # check if the current tile extend exists in the image
                nrows, ncols = check_tile_extend(
                    (img.RasterYSize, img.RasterXSize), (row, col), tile_size)

                # read the current tile from the image
                data = band.ReadAsArray(col, row, ncols, nrows)

                # append band b to numpy image array
                image[k, b, y_tl:nrows, x_tl:ncols] = data[0:(nrows - y_tl),
                                                           0:(ncols - x_tl)]

    # check if there are more than 1 band
    if not img.RasterCount > 1:
        image = image.squeeze(axis=1)

    # check if there are more than 1 tile
    if not ntiles > 1:
        image = image.squeeze(axis=0)

    # close tif file
    del img

    # return the image
    return image


# this function checks whether an image is evenly divisible
# in square tiles of defined size tile_size
# if pad=True, a padding is returned to increase the image to the nearest size
# evenly fiting ntiles of size (tile_size, tile_size)
def is_divisible(img_size, tile_size, pad=False):
    # calculate number of pixels per tile
    pixels_per_tile = tile_size ** 2

    # check whether the image is evenly divisible in square tiles of size
    # (tile_size, tile_size)
    ntiles = ((img_size[0] * img_size[1]) / pixels_per_tile)

    # if it is evenly divisible, no padding is required
    if ntiles.is_integer():
        pad = 4 * (0,)

    if not ntiles.is_integer() and not pad:
        raise ValueError('Image of size {} not evenly divisible in ({}, {}) '
                         'tiles.'.format(img_size, tile_size, tile_size))

    if not ntiles.is_integer() and pad:

        # calculate the desired image size, i.e. the smallest size that is
        # evenly divisible into square tiles of size (tile_size, tile_size)
        h_new = int(np.ceil(img_size[0] / tile_size) * tile_size)
        w_new = int(np.ceil(img_size[1] / tile_size) * tile_size)

        # calculate center offset
        dh = h_new - img_size[0]
        dw = w_new - img_size[1]

        # check whether the center offsets are even or odd

        # in case both offsets are even, the padding is symmetric on both the
        # bottom/top and left/right
        if not dh % 2 and not dw % 2:
            pad = (dh // 2, dw // 2, dh // 2, dw // 2)

        # in case only one offset is even, the padding is symmetric along the
        # even offset and asymmetric along the odd offset
        if not dh % 2 and dw % 2:
            pad = (dh // 2, dw // 2, dh // 2, dw // 2 + 1)
        if dh % 2 and not dw % 2:
            pad = (dh // 2, dw // 2, dh // 2 + 1, dw // 2)

        # in case of offsets are odd, the padding is asymmetric on both the
        # bottom/top and left/right
        if dh % 2 and dw % 2:
            pad = (dh // 2, dw // 2, dh // 2 + 1, dw // 2 + 1)

        # calculate number of tiles on padded image
        ntiles = (h_new * w_new) / (tile_size ** 2)

    return int(ntiles), pad


# check whether a tile of size (tile_size, tile_size) with topleft corner at
# topleft exists in an image of size img_size
def check_tile_extend(img_size, topleft, tile_size):

    # check if the tile is within both the rows and the columns of the image
    if (topleft[0] + tile_size < img_size[0] and
            topleft[1] + tile_size < img_size[1]):

        # both rows and columns can be equal to the tile size
        nrows, ncols = tile_size, tile_size

    # check if the tile exceeds one of rows or columns of the image
    if (topleft[0] + tile_size < img_size[0] and not
            topleft[1] + tile_size < img_size[1]):

        # adjust columns to remaining columns in the original image
        nrows, ncols = tile_size, img_size[1] - topleft[1]

    if (topleft[1] + tile_size < img_size[1] and not
            topleft[0] + tile_size < img_size[0]):

        # adjust rows to remaining rows in the original image
        nrows, ncols = img_size[0] - topleft[0], tile_size

    # check if the tile exceeds both the rows and the columns of the image
    if (not topleft[0] + tile_size < img_size[0] and not
            topleft[1] + tile_size < img_size[1]):

        # adjust both rows and columns to the remaining ones
        nrows, ncols = img_size[0] - topleft[0], img_size[1] - topleft[1]

    return nrows, ncols

# this function returns the top-left corners for each tile
# if the image is evenly divisible in square tiles of
# defined size tile_size
def tile_topleft_corner(img_size, tile_size):

    # check if the image is divisible into square tiles of size
    # (tile_size, tile_size)
    _, _ = is_divisible(img_size, tile_size, pad=False)

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
