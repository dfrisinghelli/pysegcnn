"""Utility functions mainly for image IO and reshaping.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import os
import re
import shutil
import logging
import pathlib
import tarfile
import zipfile
import datetime
import warnings
import platform
import subprocess
import xml.etree.ElementTree as ET

# externals
import torch
import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal, ogr, osr

# locals
from pysegcnn.core.constants import Gdal2Numpy

# module level logger
LOGGER = logging.getLogger(__name__)

# suffixes for radiometrically calibrated scenes
SUFFIXES = ['toa_ref', 'toa_rad', 'toa_brt']

# maximum number of filename characters on Windows
MAX_FILENAME_CHARS_WINDOWS = 260

# file suffixes for hierarchical data format
HIERARCHICAL_DATA_FORMAT = ['.h4', '.hdf', '.hdf4', '.hdf5', '.he2', '.h5',
                            '.he5', 'nc']


def img2np(path, tile_size=None, tile=None, pad=False, cval=0):
    r"""Read an image to a :py:class:`numpy.ndarray`.

    If ``tile_size`` is not `None`, the input image is divided into square
    tiles of size ``(tile_size, tile_size)``. If the image is not evenly
    divisible and ``pad=False``, a ``ValueError`` is raised. However, if
    ``pad=True``, center padding with constant value ``cval`` is applied.

    The tiling works as follows:

        +-----------+-----------+-----------+-----------+
        |           |           |           |           |
        |  tile_00  |  tile_01  |    ...    |  tile_0n  |
        |           |           |           |           |
        +-----------+-----------+-----------+-----------+
        |           |           |           |           |
        |  tile_10  |  tile_11  |    ...    |  tile_1n  |
        |           |           |           |           |
        +-----------+-----------+-----------+-----------+
        |           |           |           |           |
        |    ...    |    ...    |    ...    |    ...    |
        |           |           |           |           |
        +-----------+-----------+-----------+-----------+
        |           |           |           |           |
        |  tile_m0  |  tile_m1  |    ...    |  tile_mn  |
        |           |           |           |           |
        +-----------+-----------+-----------+-----------+

    where :math:`m = n`. Each tile has its id, which starts at `0` in the
    topleft corner of the input image, i.e. `tile_00` has :math:`id=0`, and
    increases along the width axis, i.e. `tile_0n` has :math:`id=n`, `tile_10`
    has :math:`id=n+1`, ..., `tile_mn` has :math:`id=(m \\cdot n) - 1`.

    If ``tile`` is an integer, only the tile with ``id=tile`` is returned.

    Parameters
    ----------
    path : `str` or :py:class:`pathlib.Path` or :py:class:`numpy.ndarray`
        The image to read.
    tile_size : `None` or `int`, optional
        The size of a tile. The default is `None`.
    tile : `int`, optional
        The tile id. The default is `None`.
    pad : `bool`, optional
        Whether to center pad the input image. The default is `False`.
    cval : `float`, optional
        The constant padding value. The default is `0`.

    Raises
    ------
    FileNotFoundError
        Raised if ``path`` is a path that does not exist.
    TypeError
        Raised if ``path`` is not `str` or `None` or :py:class:`numpy.ndarray`.

    Returns
    -------
    image : :py:class:`numpy.ndarray`
        The image array. The output shape is:
            - `(tiles, bands, tile_size, tile_size)` if ``tile_size`` is not
            `None`. If the image does only have one band,
            `(tiles, tile_size, tile_size)`

            - `(bands, height, width)` if ``tile_size=None``. If the image does
            only have one band, `(height, width)`.

    """
    # check the type of path
    if isinstance(path, str) or isinstance(path, pathlib.Path):

        # check if the path is a url
        if str(path).startswith('http'):
            # gdal virtual file system for url paths
            img = gdal.Open('/vsicurl/{}'.format(str(path)))
        else:
            # image is stored in a file system
            img = gdal.Open(str(path))

        # number of bands
        bands = img.RasterCount

        # spatial size
        height = img.RasterYSize
        width = img.RasterXSize

        # data type
        dtype = getattr(Gdal2Numpy,
                        gdal.GetDataTypeName(img.GetRasterBand(1).DataType))
        dtype = dtype.value

    elif path is None:
        LOGGER.warning('Path is of NoneType, returning.')
        return

    # accept numpy arrays as input
    elif isinstance(path, np.ndarray):
        # input array
        img = path

        # check the dimensions of the input array
        if img.ndim > 2:
            bands = img.shape[0]
            height = img.shape[1]
            width = img.shape[2]
        else:
            bands = 1
            height = img.shape[0]
            width = img.shape[1]

            # expand input array to fit band dimension
            img = np.expand_dims(img, axis=0)

        # input array data type
        dtype = img.dtype

    else:
        raise TypeError('Input of type {} not supported'.format(type(path)))

    # check whether to read the image in tiles
    if tile_size is None:

        # number of tiles
        ntiles = 1

        # create empty numpy array to store whole image
        image = np.empty(shape=(ntiles, bands, height, width), dtype=dtype)

        # iterate over the bands of the image
        for b in range(bands):

            # read the data of band b
            if isinstance(img, np.ndarray):
                data = img[b, ...]
            else:
                band = img.GetRasterBand(b+1)
                data = band.ReadAsArray()

            # append band b to numpy image array
            image[0, b, :, :] = data

    else:

        # check whether the image is evenly divisible in square tiles
        # of size (tile_size x tile_size)
        ntiles, padding = is_divisible((height, width), tile_size, pad)

        # image size after padding
        y_size = height + padding[0] + padding[2]
        x_size = width + padding[1] + padding[3]

        # print progress
        LOGGER.debug('Image size: {}'.format((height, width)))
        LOGGER.debug('Dividing image into {} tiles of size {} ...'
                     .format(ntiles, (tile_size, tile_size)))
        LOGGER.debug('Padding image (b, l, t, r): {}'.format(tuple(padding)))
        LOGGER.debug('Padded image size: {}'.format((y_size, x_size)))

        # get the indices of the top left corner for each tile
        topleft = tile_topleft_corner((y_size, x_size), tile_size)

        # whether to read all tiles or a single tile
        if tile is not None:
            ntiles = 1

        # create empty numpy array to store the tiles
        image = np.ones((ntiles, bands, tile_size, tile_size),
                        dtype=dtype) * cval

        # iterate over the topleft corners of the tiles
        for k, corner in topleft.items():

            # in case a single tile is required, skip the rest of the tiles
            if tile is not None:
                if k != tile:
                    continue

                # set the key to 0 for correct array indexing when reading
                # a single tile from the image
                LOGGER.debug('Processing tile {} ...'.format(k))
                k = 0
            else:
                LOGGER.debug('Creating tile {} with top-left corner {} ...'
                             .format(k, corner))

            # calculate shift between padded and original image
            row = corner[0] - padding[2] if corner[0] > 0 else corner[0]
            col = corner[1] - padding[1] if corner[1] > 0 else corner[1]
            y_tl = row + padding[2] if row == 0 else 0
            x_tl = col + padding[1] if col == 0 else 0

            # iterate over the bands of the image
            for b in range(bands):

                # check if the current tile extend exists in the image
                nrows, ncols = check_tile_extend(
                    (height, width), (row, col), tile_size)

                # read the current tile from band b
                if isinstance(img, np.ndarray):
                    data = img[b, row:row+nrows, col:col+ncols]
                else:
                    band = img.GetRasterBand(b+1)
                    data = band.ReadAsArray(col, row, ncols, nrows)

                # append band b to numpy image array
                image[k, b, y_tl:nrows, x_tl:ncols] = data[0:(nrows - y_tl),
                                                           0:(ncols - x_tl)]

    # check if there are more than 1 band
    if not bands > 1:
        image = image.squeeze(axis=1)

    # check if there are more than 1 tile
    if not ntiles > 1:
        image = image.squeeze(axis=0)

    # close tif file
    del img

    # return the image
    return image


def np2tif(array, filename, no_data=None, names=None, src_ds=None, epsg=None,
           geotransform=None, overwrite=False):
    """Save a :py:class`numpy.ndarray` as a GeoTIFF.

    The spatial coordinate reference system can be specified in two ways:
        - by providing a source dataset (``src_ds``) from which the spatial
        reference is inherited
        - by providing the `EPSG`_ code (``epsg``) of the target coordinate
        reference system together with a tuple (``geotransform``) describing
        the spatial extent of the ``array``.

    Parameters
    ----------
    array : :py:class:`numpy.ndarray`
        The array to save as GeoTIFF. Two-dimensional arrays with shape
        (height, width) and three-dimensional arrays with shape
        (bands, height, width) are supported.
    filename : `str` or :py:class:`pathlib.Path`
        The filename of the GeoTIFF.
    no_data : `None` or `int` or `float`
        The NoData value for each band in the output raster. The default is
        `None`, which means no NoData value is specified.
    names : `list` [`str`], optional
        The names of the bands in ``array`` in order. The default is `None`.
        If `None`, no band description is added.
    src_ds : :py:class:`osgeo.gdal.Dataset`, optional
        The source dataset from which the spatial reference is inherited. The
        default is `None`.
    epsg : `int`, optional
        The EPSG code of the target coordinate reference system. The default is
        `None`.
    geotransform : `tuple`, optional
        A tuple with six elements of the form,
        (x_top_left, x_res, x_shift, y_top_left, -y_res, y_shift), describing
        the spatial reference.
    overwrite : `bool`, optional
        Whether to overwrite ``filename`` if it exists. The default is `False`.

    .. _EPSG:
        https://epsg.io/

    Raises
    ------
    ValueError
        Raised if ``filename`` is does not end with a file suffix, e.g. ".tif".

        Raised if not both ``epsg`` and ``geotransform`` are specified when
        ``src_ds=None``.

    """
    # create the GeoTIFF driver
    driver = gdal.GetDriverByName('GTiff')

    # shape of the input array
    if array.ndim > 2:
        # three-dimensional array
        bands, height, width = array.shape
    else:
        # two-dimensional array: expand to three-dimensions
        bands, height, width = (1,) + array.shape
        array = np.expand_dims(array, 0)

    # data type
    dtype = gdal.GetDataTypeByName(array.dtype.name)

    # check output filename
    filename = pathlib.Path(filename)
    if not filename.suffix:
        raise ValueError('{} is not a file.'.format(filename))
    filename = pathlib.Path(str(filename).replace(filename.suffix, '.tif'))

    # check if file exists
    if filename.exists() and not overwrite:
        LOGGER.info('{} already exists.'.format(filename))
        return

    # create temporary file
    tmp_path = _tmp_path(filename)

    # create output GeoTIFF
    tmp_ds = driver.Create(str(tmp_path), width, height, bands, dtype)

    # iterate over the number of bands and write to output file
    for b in range(bands):
        trg_band = tmp_ds.GetRasterBand(b + 1)
        trg_band.WriteArray(array[b, ...])

        # set the band description, if specified
        if names is not None:
            trg_band.SetDescription(names[b])

        # set the NoData value, if specified
        if no_data is not None:
            trg_band.SetNoDataValue(no_data)
        trg_band.FlushCache()

    # set spatial reference
    if src_ds is not None:
        # inherit spatial reference from source dataset
        tmp_ds.SetProjection(src_ds.GetProjection())
        tmp_ds.SetGeoTransform(src_ds.GetGeoTransform())
    else:
        # check whether both the epsg code and the geotransform tuple are
        # specified
        if epsg is not None and geotransform is not None:
            # create the spatial reference from the epsg code
            sr = osr.SpatialReference()
            sr.ImportFromEPSG(epsg)

            # set spatial reference from epsg
            tmp_ds.SetProjection(sr.ExportToWkt())
            tmp_ds.SetGeoTransform(geotransform)
        else:
            raise ValueError('Both "epsg" and "geotransform" required to set '
                             'spatial reference if "src_ds" is None.')

    # clear dataset
    del trg_band, tmp_ds

    # compress raster
    compress_raster(tmp_path, filename)


def read_hdf4(path):
    """Read a file in Hierarchical Data Format 4 (HDF4).

    Parameters
    ----------
    path : `str` or py:class:`pathlib.Path`
        The path to the hdf4 file to read.

    Raises
    ------
    ValueError
        Raised if ``path`` is not an hdf file.

    Returns
    -------
    hdf_ds : :py:class:`xarray.Dataset`
        The HDF4 file as :py:class:`xarray.Dataset`.

    """
    # check if the path points to an hdf file
    path = pathlib.Path(path)
    if path.suffix not in HIERARCHICAL_DATA_FORMAT:
        raise ValueError('{} is not an hdf file.'.format(path))

    # read the hdf dataset: get the different subdatasets
    sub_datasets = gdal.Open(str(path)).GetSubDatasets()

    # iterate over the different subsets
    subsets = []
    for filename in sub_datasets:
        ds_name = filename[0].split(':')[-1]
        subsets.append(xr.open_rasterio(filename[0]).to_dataset(name=ds_name))

    # merge subsets to single xarray dataset
    hdf_ds = xr.merge(subsets)

    return hdf_ds.squeeze()


def hdf2tifs(path, outpath, overwrite=False, create_stack=True, **kwargs):
    """Convert a file in Hierarchical Data Format (HDF) to GeoTIFFs.

    The GeoTIFFs share the same filename as ``path``, appended by the name of
    the respective subdatasets.

    The default (``outpath=None``) is to save the GeoTIFFs in a directory named
    after the filename of ``path``, within the parent directory of ``path``.

    The output GeoTIFFs are compressed by default.

    Parameters
    ----------
    path : `str` or py:class:`pathlib.Path`
        The path to the hdf file to convert.
    outpath : `str` or py:class:`pathlib.Path`
        Path to save the GeoTIFF files.
    overwrite : `bool`, optional
        Whether to overwrite existing GeoTIFF files in ``outpath``. The default
        is `False`.
    create_stack : `bool`, optional
        Whether to create a GeoTIFF stack of all the subdatasets in ``path``.
        The default is `True`.
    **kwargs :
        Additional keyword arguments passed to :py:func:`gdal.Translate`.

    """
    # check if the path points to an hdf file
    path = pathlib.Path(path)
    if path.suffix not in HIERARCHICAL_DATA_FORMAT:
        raise ValueError('{} is not an hdf file.'.format(path))

    # create the output directory for the GeoTiffs
    outpath = pathlib.Path(outpath)
    outpath = outpath.joinpath(path.stem.replace('.', '_'))

    # check whether the output path exists
    if not outpath.exists():
        LOGGER.info('mkdir {}'.format(outpath))
        outpath.mkdir(parents=True, exist_ok=True)

    # check whether the output path contains GeoTIFF files
    tifs = [f for f in outpath.iterdir() if f.suffix in ['.tif', '.TIF']]

    # check whether to overwrite existing files
    if tifs:
        LOGGER.info('The following files already exist in {}'
                    .format(str(outpath)))
        LOGGER.info(('\n ' + (len(__name__) + 1) * ' ').join(
            ['{}'.format(str(tif.name)) for tif in tifs]))
        if not overwrite:
            # return if not overwriting
            LOGGER.info('Aborting...')
            return

        # remove existing files and prepare to overwrite
        LOGGER.info('Overwrite {}'.format(str(outpath)))
        for tif in tifs:
            tif.unlink()

    # header file
    hdr = pathlib.Path('.'.join([str(path), 'hdr']))

    # check if header file exists and contains projection
    wkt = None
    if hdr.exists():
        LOGGER.info('Found header file: {}'.format(hdr))
        with open(hdr, 'r') as file:
            # search for WKT-projection string
            content = file.read()
            wkt = re.search('PROJCS[^}]*', content)
        if wkt is not None:
            wkt = wkt[0]

    # read the hdf dataset
    hdf = gdal.Open(str(path)).GetSubDatasets()

    # check if the dataset is not empty
    if hdf:

        # iterate over the different subdatasets in the hdf
        for ds in hdf:

            # name of the current subdataset
            hdf_ds = gdal.Open(ds[0])
            name = ds[0].split(':')[-1].lower()

            # filename of the GeoTIFF
            tif_name = outpath.joinpath(
                path.name.replace(path.suffix, '_{}.tif'.format(name)))

            # convert hdf subdataset to GeoTIFF
            LOGGER.info('Converting: {}'.format(tif_name.name))
            gdal.Translate(str(tif_name), hdf_ds, outputSRS=wkt,
                           creationOptions=[
                               'COMPRESS=DEFLATE', 'PREDICTOR=1', 'TILED=YES'],
                           **kwargs)

            # set metadata field
            tif_ds = gdal.Open(str(tif_name))
            tif_ds.SetMetadata(hdf_ds.GetMetadata())

            del tif_ds

        # check whether to create a GeoTIFF stack
        if create_stack:
            # filename for the GeoTIFF stack
            stk = tif_name.parent.joinpath(
                path.name.replace(path.suffix, '.tif'))
            LOGGER.info('Creating stack: {}'.format(stk))

            # generated GeoTIFF files
            tifs = sorted([str(f) for f in outpath.iterdir() if f.suffix in
                           ['.tif', '.TIF']])

            # create stacked GeoTIFF
            stack_tifs(str(stk), tifs)

    return


def stack_tifs(filename, tifs, **kwargs):
    """Create a stacked GeoTIFF from a list of single-band GeoTIFFs.

    The output GeoTIFF stack is compressed by default.

    Parameters
    ----------
    filename : `str` or py:class:`pathlib.Path`
        The filename of the stacked GeoTIFF ending with `.tif`.
    tifs : `list` [`str`]
        The list of the paths to the GeoTIFF files to stack.
    **kwargs :
        Additional keyword arguments passed to :py:func:`gdal.Translate`.

    """
    # build virtual raster dataset
    vrt = str(filename).replace('.tif', '.vrt')
    vrt_ds = gdal.BuildVRT(str(vrt), tifs, separate=True)

    # set band descriptions
    for i, tif in enumerate(tifs):

        # get the description of the band in the tif file
        tif_ds = gdal.Open(str(tif))
        band_name = tif_ds.GetRasterBand(1).GetDescription()

        # set the description of the band in the vrt file
        vrt_ds.GetRasterBand(i + 1).SetDescription(band_name)

    # create GeoTIFF stack
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.Translate(str(filename), vrt_ds, creationOptions=[
        'COMPRESS=DEFLATE', 'PREDICTOR=1', 'TILED=YES'], **kwargs)

    del vrt_ds

    return


def is_divisible(img_size, tile_size, pad=False):
    """Check whether an image is evenly divisible into square tiles.

    Parameters
    ----------
    img_size : `tuple`
        The image size (height, width).
    tile_size : `int`
        The size of the tile.
    pad : `bool`, optional
        Whether to center pad the input image. The default is `False`.

    Raises
    ------
    ValueError
        Raised if the image is not evenly divisible and ``pad=False``.

    Returns
    -------
    ntiles : `int`
        The number of tiles fitting ``img_size``.
    padding : `tuple`
        The amount of padding (bottom, left, top, right).

    """
    # calculate number of pixels per tile
    pixels_per_tile = tile_size ** 2

    # check whether the image is evenly divisible in square tiles of size
    # (tile_size, tile_size)
    ntiles = ((img_size[0] * img_size[1]) / pixels_per_tile)

    # if it is evenly divisible, no padding is required
    if ntiles.is_integer():
        padding = 4 * (0,)

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
            padding = (dh // 2, dw // 2, dh // 2, dw // 2)

        # in case only one offset is even, the padding is symmetric along the
        # even offset and asymmetric along the odd offset
        if not dh % 2 and dw % 2:
            padding = (dh // 2, dw // 2, dh // 2, dw // 2 + 1)
        if dh % 2 and not dw % 2:
            padding = (dh // 2, dw // 2, dh // 2 + 1, dw // 2)

        # in case of offsets are odd, the padding is asymmetric on both the
        # bottom/top and left/right
        if dh % 2 and dw % 2:
            padding = (dh // 2, dw // 2, dh // 2 + 1, dw // 2 + 1)

        # calculate number of tiles on padded image
        ntiles = (h_new * w_new) / (tile_size ** 2)

    return int(ntiles), padding


def check_tile_extend(img_size, topleft, tile_size):
    """Check if a tile exceeds the image size.

    Parameters
    ----------
    img_size : `tuple`
        The image size (height, width).
    topleft : `tuple`
        The topleft corner of the tile (y, x).
    tile_size : `int`
        The size of the tile.

    Returns
    -------
    nrows : `int`
        Number of rows of the tile within the image.
    ncols : `int`
        Number of columns of the tile within the image.

    """
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


def tile_topleft_corner(img_size, tile_size):
    """Return the topleft corners of the tiles in the image.

    Parameters
    ----------
    img_size : `tuple`
        The image size (height, width).
    tile_size : `int`
        The size of the tile.

    Returns
    -------
    indices : `dict`
        The keys of ``indices`` are the tile ids (`int`) and the values are the
        topleft corners (y, x) of the tiles.

    """
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


def reconstruct_scene(tiles):
    """Reconstruct a tiled image.

    Parameters
    ----------
    tiles : :py:class:`numpy.ndarray`
        The tiled image, shape: `(tiles, bands, tile_size, tile_size)` or
        `(tiles, tile_size, tile_size)`.

    Returns
    -------
    image : :py:class:`numpy.ndarray`
        The reconstructed image, shape: `(bands, height, width)`.

    """
    # check the dimensions of the input array
    if tiles.ndim > 3:
        nbands = tiles.shape[1]
        tile_size = tiles.shape[2]
    else:
        nbands = 1
        tile_size = tiles.shape[1]

    # calculate image size
    img_size = 2 * (int(np.sqrt(tiles.shape[0]) * tile_size),)

    # calculate the topleft corners of the tiles
    topleft = tile_topleft_corner(img_size, tile_size)

    # iterate over the tiles
    scene = np.zeros(shape=(nbands,) + img_size)
    for t in range(tiles.shape[0]):
        scene[...,
              topleft[t][0]: topleft[t][0] + tile_size,
              topleft[t][1]: topleft[t][1] + tile_size] = tiles[t, ...]

    return scene.squeeze()


def accuracy_function(outputs, labels):
    """Calculate prediction accuracy.

    Parameters
    ----------
    outputs : :py:class:`torch.Tensor` or `array_like`
        The model prediction.
    labels : :py:class:`torch.Tensor` or `array_like`
        The ground truth.

    Returns
    -------
    accuracy : `float`
        Mean prediction accuracy.

    """
    if isinstance(outputs, torch.Tensor):
        return (outputs == labels).float().mean().item()
    else:
        return (np.asarray(outputs) == np.asarray(labels)).mean().item()


def parse_landsat_scene(scene_id):
    """Parse a Landsat scene identifier.

    Parameters
    ----------
    scene_id : `str`
        A Landsat scene identifier.

    Returns
    -------
    scene : `dict` or `None`
        A dictionary containing scene metadata. If `None`, ``scene_id`` is not
        a valid Landsat scene identifier.

    """
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

    # mapping from sensor id to sensors
    lsensors = {'E': 'Enhanced Thematic Mapper Plus',
                'T': 'Thematic Mapper',
                'M': 'Multispectral Scanner'}
    l8sensors = {'C': 'Operational Land Imager (OLI) & Thermal Infrared Sensor'
                      ' (TIRS)',
                 'O': 'Operational Land Imager (OLI)',
                 'T': 'Thermal Infrared Sensor (TIRS)',
                 }

    # whether a scene identifier matches the Landsat naming convention
    scene = {}
    if Landsat_C0.search(scene_id):

        # the match of the regular expression
        match = Landsat_C0.search(scene_id)[0]

        # the metadata of the scene identifier
        scene['id'] = match
        scene['satellite'] = 'Landsat {}'.format(match[2])
        if int(match[2]) > 7:
            scene['sensor'] = l8sensors[match[1]]
        else:
            scene['sensor'] = lsensors[match[1]]
        scene['path'] = match[3:6]
        scene['row'] = match[6:9]
        scene['date'] = doy2date(match[9:13], match[13:16])
        scene['gsi'] = match[16:19]
        scene['version'] = match[19:]

    elif Landsat_C1.search(scene_id):

        # the match of the regular expression
        match = Landsat_C1.search(scene_id)[0]

        # split scene into respective parts
        parts = match.split('_')

        # the metadata of the scene identifier
        scene['id'] = match
        scene['satellite'] = 'Landsat {}'.format(parts[0][2:])
        if int(parts[0][3]) > 7:
            scene['sensor'] = l8sensors[parts[0][1]]
        else:
            scene['sensor'] = lsensors[parts[0][1]]
        scene['path'] = parts[2][0:3]
        scene['row'] = parts[2][3:]
        scene['date'] = datetime.datetime.strptime(parts[3], '%Y%m%d')
        scene['collection'] = int(parts[5])
        scene['version'] = parts[6]

    else:
        scene = None

    return scene


def parse_sentinel2_scene(scene_id):
    """Parse a Sentinel-2 scene identifier.

    Parameters
    ----------
    scene_id : `str`
        A Sentinel-2 scene identifier.

    Returns
    -------
    scene : `dict` or `None`
        A dictionary containing scene metadata. If `None`, ``scene_id`` is not
        a valid Sentinel-2 scene identifier.

    """
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
    file_semantic = '(MSI(L[0-1]([ABCP]|_)|CTI)|USER2A)_'
    site = '[A-Z_]{4}_'
    S2_L1C_Old = re.compile(mission + file_class + file_category +
                            file_semantic + site + date + time + orbit + 'V' +
                            date + time + date + time.replace('_', ''))

    # Sentinel 2 granule naming convention before 6th December 2016
    granule_semantic = 'L[0-1][ABC_]_(GR|DS|TL|TC|CO)_'
    det_or_tile = '(D[0-1][1-2]|T[0-9]{2}[A-Z]{3})(_)?'
    aorbit = '(A[0-9]{6}_' + '{}'.format('|S' + date + time + ')')
    baseline = '(N[0-9]{2})?'
    S2_L1C_Granule = re.compile(mission + file_class + file_category +
                                granule_semantic + site + 'V' + date + time +
                                aorbit + det_or_tile + baseline)

    # Sentitel 2 granule naming convention
    S2_L1C_Granule_Only = re.compile('L[0-1][ABC]_' + tile + 'A[0-9]{6}_' +
                                     date + time.replace('_', ''))

    # Sentinel 2 tile naming convetion
    S2_L1C_Tile = re.compile(tile + date + time.replace('_', ''))

    # whether a scene identifier matches the Sentinel naming convention
    scene = {}
    if S2_L1C_Old.search(scene_id):

        # the match of the regular expression
        match = S2_L1C_Old.search(scene_id)[0]

        # split scene into respective part
        parts = match.split('_')

        # the metadata of the scene identifier
        scene['id'] = match
        scene['satellite'] = parts[0]
        scene['file class'] = parts[1]
        scene['file category'] = parts[2]
        scene['file semantic'] = parts[3]
        scene['site'] = parts[4]
        scene['orbit'] = parts[6]
        scene['date'] = datetime.datetime.strptime(
           parts[7].split('T')[0].replace('V', ''), '%Y%m%d')
        scene['tile'] = None

    elif S2_L1C_New.search(scene_id):

        # the match of the regular expression
        match = S2_L1C_New.search(scene_id)[0]

        # split scene into respective parts
        parts = match.split('_')

        # the metadata of the scene identifier
        scene['id'] = match
        scene['satellite'] = parts[0]
        scene['product'] = parts[1]
        scene['date'] = datetime.datetime.strptime(parts[2].split('T')[0],
                                                   '%Y%m%d')
        scene['baseline'] = parts[3]
        scene['orbit'] = parts[4]
        scene['tile'] = parts[5]

    elif S2_L1C_Granule.search(scene_id):

        # the match of the regular expression
        match = S2_L1C_Granule.search(scene_id)[0]

        # split scene into respective parts
        parts = match.split('_')

        # the metadata of the scene identifier
        scene['id'] = match
        scene['satellite'] = parts[0]
        scene['file class'] = parts[1]
        scene['file category'] = parts[2]
        scene['file semantic'] = parts[3] + parts[4]
        scene['site'] = parts[5]
        scene['orbit'] = parts[8]
        scene['date'] = datetime.datetime.strptime(
           parts[7].split('T')[0].replace('V', ''), '%Y%m%d')
        scene['tile'] = parts[9]

    elif S2_L1C_Granule_Only.search(scene_id):

        # the match of the regular expression
        match = S2_L1C_Granule_Only.search(scene_id)[0]

        # split scene into respective parts
        parts = match.split('_')

        # the metadata of the scene identifier
        scene['id'] = match
        scene['satellite'] = 'Sentinel 2'
        scene['file class'] = parts[0]
        scene['orbit'] = parts[2]
        scene['date'] = datetime.datetime.strptime(
           parts[3].split('T')[0].replace('V', ''), '%Y%m%d')
        scene['tile'] = parts[1]

    elif S2_L1C_Tile.search(scene_id):

        # the match of the regular expression
        match = S2_L1C_Tile.search(scene_id)[0]

        # split scene into respective parts
        parts = match.split('_')

        # the metadata of the scene identifier
        scene['id'] = match
        scene['satellite'] = 'Sentinel 2'
        scene['file class'] = 'L1C'
        scene['date'] = datetime.datetime.strptime(
           parts[1].split('T')[0].replace('V', ''), '%Y%m%d')
        scene['tile'] = parts[0]

    return scene


def doy2date(year, doy):
    """Convert the (year, day of the year) date format to a datetime object.

    Parameters
    ----------
    year : `int`
        The year.
    doy : `int`
        The day of the year.

    Returns
    -------
    date : :py:class:`datetime.datetime`
        The converted date.
    """
    # convert year/day of year to a datetime object
    date = (datetime.datetime(int(year), 1, 1) +
            datetime.timedelta(days=(int(doy) - 1)))

    return date


def item_in_enum(name, enum):
    """Check if an item exists in an enumeration.

    Parameters
    ----------
    name : `str`
        Name of the item.
    enum : :py:class:`enum.Enum`
        An instance of :py:class:`enum.Enum`.

    Raises
    ------
    ValueError
        Raised if ``name`` is not in ``enum``.

    Returns
    -------
    value
        The value of ``name`` in ``enum``.

    """
    # check whether the input name exists in the enumeration
    if name not in enum.__members__:
        raise ValueError('{} is not in {} enumeration. Valid names are: \n {}'
                         .format(name, enum.__name__,
                                 '\n'.join('- {}'.format(n) for n in
                                           enum.__members__)))
    else:
        return enum.__members__[name].value


def destack_tiff(image, outpath=None, overwrite=False, remove=False,
                 suffix=''):
    """Destack a TIFF with more than one band into a TIFF file for each band.

    Each band in ``image`` is saved to ``outpath`` as distinct TIFF file.
    The default filenames are: `"filename(``image``) + _B(i).tif"`, where `i`
    is the respective number of each band in ``image``.

    Parameters
    ----------
    image : `str` or :py:class:`pathlib.Path`
        The TIFF to destack.
    outpath : `str`, optional
        Path to save the output TIFF files. The default is `None`. If `None`,
        ``outpath`` is the path to ``image``.
    remove : `bool`, optional
        Whether to remove ``image`` from disk after destacking. The default is
        `False`.
    overwrite : `bool`, optional
        Whether to overwrite existing TIFF files. The default is `False`.
    suffix : `str`, optional
        String to append to the filename of ``image``. If specified, the TIFF
        filenames for each band in ``image`` are, `"filename(``image``) +
        + _B(i)_ + ``suffix``.tif"`. The default is `''`.

    Raises
    ------
    FileNotFoundError
        Raised if ``image`` does not exist.

    Returns
    -------
    None.

    """
    # stop gdal printing warnings and errors to STDERR
    gdal.PushErrorHandler('CPLQuietErrorHandler')

    # raise Python exceptions for gdal errors
    gdal.UseExceptions()

    # convert to pathlib.Path
    image = pathlib.Path(image)
    if not image.exists():
        raise FileNotFoundError('{} does not exist.'.format(image))

    # name of the TIFF
    imgname = image.stem

    # default: output directory is equal to the input directory
    if outpath is None:
        outpath = image.parent
    else:
        outpath = pathlib.Path(outpath)
        # check if output directory exists
        if not outpath.exists():
            outpath.mkdir(parents=True, exist_ok=True)

    # open the TIFF
    img = gdal.Open(str(image))

    # check whether the file was already processed
    processed = list(outpath.glob(imgname + '_B*'))
    if len(processed) >= img.RasterCount and not overwrite:
        LOGGER.info('{} already processed.'.format(imgname))

    # destack the TIFF
    else:
        # image driver
        driver = gdal.GetDriverByName('GTiff')
        driver.Register()

        # image size and tiles
        cols = img.RasterXSize
        rows = img.RasterYSize
        bands = img.RasterCount

        # print progress
        LOGGER.info('Processing: {}'.format(imgname))

        # iterate the bands of the raster
        for b in range(1, bands + 1):

            # read the data of band b
            band = img.GetRasterBand(b)
            data = band.ReadAsArray()

            # output file: replace for band name
            fname = str(outpath.joinpath(imgname + '_B{:d}.tif'.format(b)))
            if suffix:
                fname = fname.replace('.tif', '_{}.tif'.format(suffix))
            outDs = driver.Create(fname, cols, rows, 1, band.DataType)

            # define output band
            outband = outDs.GetRasterBand(1)

            # write array to output band
            outband.WriteArray(data)
            outband.FlushCache()

            # Set the geographic information
            outDs.SetProjection(img.GetProjection())
            outDs.SetGeoTransform(img.GetGeoTransform())

            # clear memory
            del outband, band, data, outDs

    # remove old stacked GeoTIFF
    del img
    if remove:
        os.unlink(image)
    return


def standard_eo_structure(source_path, target_path, overwrite=False,
                          move=False, parser=parse_landsat_scene, skip=True):
    """Modify the directory structure of a remote sensing dataset.

    This function assumes that ``source_path`` points to a directory containing
    remote sensing data, where each file in ``source_path`` and its sub-folders
    should contain a scene identifier in its filename. The scene identifier is
    used to restructure the dataset.

    Currently, Landsat and Sentinel-2 datasets are supported.

    The directory tree in ``source_path`` is modified to the following
    structure in ``target_path``:

        - target_path/
            - scene_id_1/
                - files matching scene_id_1
            - scene_id_2/
                - files matching scene_id_2
            - ...
            - scene_id_n/
                - files matching scene_id_n


    Parameters
    ----------
    source_path : `str` or :py:class:`pathlib.Path`
        Path to the remote sensing dataset.
    target_path : `str` or :py:class:`pathlib.Path`
        Path to save the restructured dataset.
    overwrite : `bool`, optional
        Whether to overwrite existing files in ``target_path``.
        The default is `False`.
    move : `bool`, optional
        Whether to move the files from ``source_path`` to ``target_path``. If
        `True`, files in ``source_path`` are moved to ``target_path``, if
        `False`, files in ``source_path`` are copied to ``target_path``. The
        default is `False`.
    parser : `function`, optional
        The scene identifier parsing function. Depends on the sensor of the
        dataset. See e.g., :py:func:`pysegcnn.core.utils.parse_landsat_scene`.
    skip : `bool`, optional
        Whether to skip files not matching a scene identifier after ``parser``.
        The default is `True`.

    """
    # create a directory for each scene
    for dirpath, dirnames, filenames in os.walk(source_path):

        # check if there are files in the current folder
        if not filenames:
            continue

        # iterate over the files to modify
        for file in filenames:
            # get the path to the file
            old_path = os.path.join(dirpath, file)

            # get name of the scene
            scene = parser(file)
            if not scene:

                # check whether to skip file not matching a scene identifier
                if skip:
                    continue

                # path to copy files not matching a scene identifier
                new_path = pathlib.Path(target_path).joinpath('misc', file)

                # file a warning if the file does not match a scene identifier
                LOGGER.warning('{} does not match a scene identifier. Copying '
                               'to {}.'.format(os.path.basename(old_path),
                                               new_path.parent))
            else:

                # the name of the scene
                scene_name = scene['id']

                # define the new path to the file
                new_path = pathlib.Path(target_path).joinpath(scene_name, file)

            # move files to new directory
            if new_path.is_file() and not overwrite:
                LOGGER.info('{} already exists.'.format(new_path))
                continue
            else:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                if move:
                    shutil.move(old_path, new_path)
                    LOGGER.info('mv {}'.format(new_path))
                else:
                    LOGGER.info('cp {}'.format(new_path))
                    shutil.copy(old_path, new_path)

    # remove old file location
    if move:
        shutil.rmtree(source_path)


def extract_archive(inpath, outpath, overwrite=False):
    """Extract files from an archive.

    Parameters
    ----------
    inpath : `str` or :py:class:`pathlib.Path`
        Path to an archive.
    outpath : `str` or :py:class:`pathlib.Path`
        Path to save extracted files.
    overwrite : `bool`, optional
        Whether to overwrite existing extracted files. The default is `False`.

    Returns
    -------
    target : :py:class:`pathlib.Path`
        Path to the extracted files.

    """
    inpath = pathlib.Path(inpath)

    # create the output directory
    outpath = pathlib.Path(outpath)
    if not outpath.exists():
        outpath.mkdir(parents=True)

    # name of the archive
    archive = inpath.stem

    # path to the extracted files
    target = outpath.joinpath(archive)
    if target.exists():
        if overwrite:
            LOGGER.info('Overwriting: {}'.format(target))
            shutil.rmtree(target)
        else:
            LOGGER.info('Extracted files are located in: {}'.format(target))
            return target

    # create output directory
    target.mkdir(parents=True)

    # read the archive
    if inpath.name.endswith('.tgz') or inpath.name.endswith('.tar.gz'):
        tar = tarfile.open(inpath, "r:gz")
    elif inpath.name.endswith('.zip'):
        tar = zipfile.ZipFile(inpath, 'r')

    # extract files to the output directory
    LOGGER.info('Extracting: {}'.format(archive))
    tar.extractall(target)

    return target


def search_files(directory, pattern):
    """Recursively searches files matching a pattern.

    Parameters
    ----------
    directory : `str` or :py:class:`pathlib.Path`
        The directory to recursively search.
    pattern : `str`
        The pattern to match. Regular expressions are supported.

    Returns
    -------
    matches : `list` [:py:class:`pathlib.Path`]
        List of files in ``directory`` matching ``pattern``.

    """
    # create regular expression
    pattern = re.compile(pattern)

    # recursively search for files matching the pattern
    matches = []
    for dirpath, _, files in os.walk(directory):
        matches.extend([pathlib.Path(dirpath).joinpath(file) for file in files
                        if pattern.search(file)])

    return matches


def read_landsat_metadata(file):
    """Parse the Landsat metadata *_MTL.txt file.

    Parameters
    ----------
    file : `str` or :py:class:`pathlib.Path`
        Path to a Landsat *_MTL.txt file.

    Raises
    ------
    FileNotFoundError
        Raised if ``file`` does not exist.

    Returns
    -------
    metadata : `dict`
        The metadata text file as dictionary, where each line is a (key, value)
        pair.

    """
    file = pathlib.Path(file)
    # check if the metadata file exists
    if not file.exists():
        raise FileNotFoundError('{} does not exist'.format(file))

    # read metadata file
    metadata = {}
    LOGGER.info('Parsing metadata file: {}'.format(file.name))
    with open(file, 'r') as metafile:
        # iterate over the lines of the metadata file
        for line in metafile:
            try:
                # the current line as (key, value pair)
                (key, value) = line.split('=')

                # store current line in dictionary: skip lines defining the
                # parameter groups
                if 'GROUP' not in key:
                    metadata[key.strip()] = value.strip()

            # catch value error of line.split('='), i.e. if there is no '='
            # sign in the current line
            except ValueError:
                continue

    return metadata


def get_radiometric_constants(metadata):
    """Retrieve the radiometric calibration constants.

    Parameters
    ----------
    metadata : `dict`
        The dictionary returned by
        :py:func:`pysegcnn.core.utils.read_landsat_metadata`.

    Returns
    -------
    oli : `dict`
        Radiometric rescaling factors of the OLI sensor.
    tir : `dict`
        Thermal conversion constants of the TIRS sensor.

    """
    # regular expression patterns matching the radiometric rescaling factors
    oli_pattern = re.compile('(RADIANCE|REFLECTANCE)_(MULT|ADD)_BAND_\\d{1,2}')
    tir_pattern = re.compile('K(1|2)_CONSTANT_BAND_\\d{2}')

    # rescaling factors to calculate top of atmosphere radiance and reflectance
    oli = {key: float(value) for key, value in metadata.items() if
           oli_pattern.search(key) is not None}

    # rescaling factors to calculate at-satellite temperatures in Kelvin
    tir = {key: float(value) for key, value in metadata.items() if
           tir_pattern.search(key) is not None}

    return oli, tir


def landsat_radiometric_calibration(scene, outpath=None, exclude=[],
                                    radiance=False, overwrite=False,
                                    remove_raw=True):
    """Radiometric calibration of Landsat Collection Level 1 scenes.

    Convert the Landsat OLI bands to top of atmosphere radiance or reflectance
    and the TIRS bands to top of atmosphere brightness temperature.

    .. important::

        Conversion is performed following the `equations`_ provided by the
        USGS.

    The filename of each band is extended by one of the following suffixes,
    depending on the type of the radiometric calibration:

        - `'toa_ref'`: top of atmosphere reflectance
        - `'toa_rad'`: top of atmopshere radiance
        - `'toa_brt'`: top of atmosphere brightness temperature

    Parameters
    ----------
    scene : `str` or :py:class:`pathlib.Path`
        Path to a Landsat scene in digital number format.
    outpath : `str` or :py:class:`pathlib.Path`, optional
        Path to save the calibrated images. The default is `None`, which means
        saving in the same directory ``scene``.
    exclude : `list` [`str`], optional
        Bands to exclude from the radiometric calibration. The default is `[]`.
    radiance : `bool`, optional
        Whether to calculate top of atmosphere radiance. The default is `False`
        , which means calculating top of atmopshere reflectance.
    overwrite : `bool`, optional
        Whether to overwrite the calibrated images. The default is `False`.
    remove_raw : `bool`, optional
        Whether to remove the raw digitial number images. The default is `True`
        .

    Raises
    ------
    FileNotFoundError
        Raised if ``scene`` does not exist.

    .. _equations:
        https://www.usgs.gov/land-resources/nli/landsat/using-usgs-landsat-level-1-data-product

    """
    scene = pathlib.Path(scene)
    # check if the input scene exists
    if not scene.exists():
        raise FileNotFoundError('{} does not exist.'.format(scene))

    # search the scene metadata file
    try:
        metafile = search_files(scene, '[mMtTlL].txt').pop()
    except IndexError:
        LOGGER.error('Can not calibrate scene: metadata file matching pattern '
                     '"*_[mMtTlL].txt" does not exist.')
        return

    # parse metadata file
    metadata = read_landsat_metadata(metafile)

    # get the name of the scene
    name = metadata['LANDSAT_SCENE_ID']

    # default: output directory is equal to the input directory
    if outpath is None:
        outpath = scene
    else:
        outpath = pathlib.Path(outpath).joinpath(name)
        # check if output directory exists
        if not outpath.exists():
            outpath.mkdir(parents=True, exist_ok=True)

    # radiometric calibration constants
    oli, tir = get_radiometric_constants(metadata)

    # log current Landsat scene ID
    LOGGER.info('Landsat scene id: {}'.format(name))

    # images to process
    ipattern = 'B\\d{1,2}(.*)\\.[tT][iI][fF]'
    images = search_files(scene, ipattern)

    # check if any images were already processe
    processed = search_files(
        outpath, '({}|{}|{}).[tT][iI][fF]'.format(*SUFFIXES))
    if any(processed):
        LOGGER.info('The following images have already been processed:')

        # images that were already processed
        LOGGER.info(('\n ' + (len(__name__) + 1) * ' ').join(
            [str(file.name) for file in processed]))

        # overwrite: remove processed images and redo calibration
        if overwrite:
            LOGGER.info('Preparing to overwrite ...')

            # remove processed images
            for file in processed:
                # remove from disk
                file.unlink()
                LOGGER.info('rm {}'.format(file))

                # remove from list to process
                try:
                    images.remove(file)
                except ValueError:
                    # catch ValueError raised by remove if the item is not
                    # present in the list
                    pass

        # not overwriting, terminate calibration
        else:
            return

    # exclude defined bands from the calibration procedure
    for i in images:
        current_band = re.compile(ipattern).search(str(i))[0].split('.')[0]
        if current_band in exclude:
            images.remove(i)

    # iterate over the different bands
    for image in images:
        LOGGER.info('Processing: {}'.format(image.name))

        # read the image
        src_ds = gdal.Open(str(image))
        src_band = src_ds.GetRasterBand(1)

        # read data as array
        data = src_band.ReadAsArray()

        # mask of erroneous values, i.e. mask of values < 0
        mask = data < 0

        # output filename
        fname = outpath.joinpath(image.stem)

        # get the current band
        band = re.search('B\\d{1,2}', str(image))[0].replace('B', 'BAND_')

        # check if the current band is a thermal band
        if band in ['BAND_10', 'BAND_11']:

            # output file name for TIRS bands
            fname = pathlib.Path(str(fname) + '_toa_brt.tif')

            # calculate top of atmosphere brightness temperature
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                # top of atmosphere radiance
                rad = (oli['RADIANCE_MULT_{}'.format(band)] * data +
                       oli['RADIANCE_ADD_{}'.format(band)])

                # top of atmosphere brightness temperature
                den = np.log(tir['K1_CONSTANT_{}'.format(band)] / rad + 1)
                toa = tir['K2_CONSTANT_{}'.format(band)] / den

                # clear memory
                del den, data, rad
        else:

            # whether to calculate top of atmosphere radiance or reflectance
            if radiance:

                # output file name for OLI bands: radiance
                fname = pathlib.Path(str(fname) + '_toa_rad.tif')

                # calculate top of atmosphere radiance
                toa = (oli['RADIANCE_MULT_{}'.format(band)] * data +
                       oli['RADIANCE_ADD_{}'.format(band)])

                # clear memory
                del data

            else:

                # output file name for OLI bands: reflectance
                fname = pathlib.Path(str(fname) + '_toa_ref.tif')

                # solar zenith angle in radians
                zenith = np.radians(90 - float(metadata['SUN_ELEVATION']))

                # calculate top of the atmosphere reflectance
                ref = (oli['REFLECTANCE_MULT_{}'.format(band)] * data +
                       oli['REFLECTANCE_ADD_{}'.format(band)])
                toa = ref / np.cos(zenith)

                # clear memory
                del ref, data

        # mask erroneous values
        toa[mask] = np.nan

        # write to output file
        np2tif(toa, filename=fname, src_ds=src_ds)

        # clear memory
        del src_band, src_ds, toa, mask

    # check whether to remove raw archive data
    if remove_raw:

        # remove raw digital number images
        for image in images:
            # remove from disk
            image.unlink()
            LOGGER.info('rm {}'.format(str(image)))
    return


def s2_l1c_toa_ref(scene, outpath=None, exclude=[], overwrite=False,
                   remove_raw=False):
    """Convert a `Sentinel-2 L1C`_ product to top of atmosphere reflectance.

    Parameters
    ----------
    scene : `str` or :py:class:`pathlib.Path`
        Path to a Sentinel-2 Level 1C scene.
    outpath : `str` or :py:class:`pathlib.Path`, optional
        Path to save the calibrated images. The default is `None`, which means
        saving in the same directory ``scene``. If specified, ``outpath`` is
        appended with the Sentinel-2 Level 1C scene name.
    exclude : `list` [`str`], optional
        Bands to exclude from the radiometric calibration. The default is `[]`.
    overwrite : `bool`, optional
        Whether to overwrite the calibrated images. The default is `False`.
    remove_raw : `bool`, optional
        Whether to remove the raw digitial number images. The default is
        `False`.

    Raises
    ------
    FileNotFoundError
        Raised if ``scene`` does not exist.

    .. _Sentinel-2 L1C:
        https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-1c/algorithm

    """
    scene = pathlib.Path(scene)
    # check if the input scene exists
    if not scene.exists():
        raise FileNotFoundError('{} does not exist.'.format(scene))

    # search the scene metadata file
    try:
        metafile = search_files(scene, 'MTD_MSIL1C.xml').pop()
    except IndexError:
        LOGGER.error('Can not calibrate scene: metadata file matching pattern '
                     '"*_[mMtTlL].txt" does not exist.')
        return

    # parse the metadata file
    tree = ET.parse(metafile)
    root = tree.getroot()

    # get the name of the scene
    name = root.find('.//PRODUCT_URI').text.replace('.SAFE', '')

    # default: output directory is equal to the input directory
    if outpath is None:
        outpath = scene
    else:
        outpath = pathlib.Path(outpath).joinpath(name)
        # check if output directory exists
        if not outpath.exists():
            outpath.mkdir(parents=True, exist_ok=True)

    # get quantification value to convert to top of atmosphere reflectance
    Q = np.float(root.find('.//QUANTIFICATION_VALUE').text)

    # print current scene id
    LOGGER.info('Sentinel 2 scene id: {}'.format(name))

    # images to process
    ipattern = 'B[0-9](?:[0-9]|A)\.jp2'
    images = search_files(scene, ipattern)

    # check if any images were already processed
    processed = search_files(
        outpath, '({}|{}|{}).[tT][iI][fF]'.format(*SUFFIXES))
    if any(processed):
        LOGGER.info('The following images have already been processed:')

        # images that were already processed
        LOGGER.info(('\n ' + (len(__name__) + 1) * ' ').join(
            [str(file.name) for file in processed]))

        # overwrite: remove processed images and redo calibration
        if overwrite:
            LOGGER.info('Preparing to overwrite ...')

            # remove processed images
            for file in processed:
                # remove from disk
                file.unlink()
                LOGGER.info('rm {}'.format(file))

                # remove from list to process
                try:
                    images.remove(file)
                except ValueError:
                    # catch ValueError raised by remove if the item is not
                    # present in the list
                    pass

        # not overwriting, terminate calibration
        else:
            return

    # exclude defined bands from the calibration procedure
    for i in images:
        current_band = re.compile(ipattern).search(str(i))[0].split('.')[0]
        if current_band in exclude:
            images.remove(i)

    # iterate over the different bands
    for image in images:
        LOGGER.info('Processing: {}'.format(image.name))

        # read the image
        src_ds = gdal.Open(str(image))
        src_band = src_ds.GetRasterBand(1)

        # read data as array
        data = src_band.ReadAsArray()

        # mask of erroneous values, i.e. mask of values < 0
        mask = data < 0

        # output filename
        fname = outpath.joinpath(image.stem + '_toa_ref.tif')

        # convert to top of atmosphere reflectance
        toa = data.astype(np.float32) / Q

        # mask erroneous values
        toa[mask] = np.nan

        # write to output file
        np2tif(toa, filename=fname, src_ds=src_ds)

        # clear memory
        del src_band, src_ds, toa, mask

    # check whether to remove raw archive data
    if remove_raw:

        # remove raw digital number images
        for image in images:
            # remove from disk
            image.unlink()
            LOGGER.info('rm {}'.format(str(image)))
    return


def check_filename_length(filename):
    """Extended-length paths on Windows.

    See the official Microsoft `documentation`_ for more details.

    Parameters
    ----------
    filename : `str` or :py:class:`pathlib.Path`
        Absolute path to a file.

    Returns
    -------
    filename : :py:class:`pathlib.Path`
        The extended length path by using the prefix `//?/` in case
        ``filename`` has more than 260 characters.

    .. _documentation:
        https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation

    """
    # check if we are on Windows
    if platform.system() == 'Windows':
        filename = (pathlib.Path('\\\\?\\' + str(filename)) if
                    len(str(filename)) >= MAX_FILENAME_CHARS_WINDOWS else
                    filename)
    return filename


def get_epsg(ras_ds):
    """Get the `EPSG`_ code of a raster coordinate reference system.

    Parameters
    ----------
    ras_ds : :py:class:`osgeo.gdal.Dataset`
        A georeferenced raster dataset.

    Returns
    -------
    ras_epsg : `str`
        The EPSG code of the raster coordinate reference system.

    .. _EPSG:
        https://epsg.io/

    """
    # input raster spatial reference
    ras_sr = gdal.osr.SpatialReference(wkt=str(ras_ds.GetProjection()))
    ras_epsg = ras_sr.GetAttrValue('AUTHORITY', 1)

    return ras_epsg


def reproject_raster(src_ds, trg_ds, ref_ds=None, epsg=None, resample='near',
                     pixel_size=(None, None), no_data=0, overwrite=False):
    """Reproject a raster to a defined coordinate reference system.

    Reproject ``src_ds`` to ``trg_ds`` using either:
        - a defined `EPSG`_ code
        - a reference raster dataset ``ref_ds``, whose coordinate reference
        system is used for the reprojection

    If neither a reference dataset (``ref_ds``) nor an epsg code (``epsg``) is
    provided, the target raster inherits the spatial reference from the source
    raster.

    The spatial resolution of ``trg_ds`` can be specified with the parameter
    ``pixel_size``. If not specified, ``trg_ds`` shares the same spatial
    resolution as ``src_ds``. However, if ``pixel_size`` is specified and
    differs from the original spatial resolution of ``src_ds``, the spatial
    resampling algorithm defined by ``resample`` is used to up/downsample
    ``trg_ds``.

    Parameters
    ----------
    src_ds : `str` or :py:class:`pathlib.Path`
        The raster dataset to reproject.
    trg_ds : `str` or :py:class:`pathlib.Path`
        The target raster dataset.
    ref_ds : :py:class:`pathlib.Path` or :py:class:`osgeo.osr.SpatialReference`
        Either a reference raster dataset, whose coordinate reference system is
        used for reprojection or a valid spatial reference object. The default
        is `None`.
    epsg : `int`, optional
        The EPSG code of the target coordinate reference system. The default is
        `None`.
    resample : `str`, optional
        Spatial resampling algorithm to use, if the target resolution differs
        from the source resolution. The default is `'near'`.
    pixel_size : `tuple` [`int`, `int`], optional
        The pixel size of the target dataset, (height, width). The default is
        `(None, None)`.
    no_data : `int` or `float`
        The value to assign to NoData values in ``src_ds``. The default is `0`.
    overwrite : `bool`, optional
        Whether to overwrite ``trg_ds``, if it exists. The default is `False`.

    .. _EPSG:
        https://epsg.io/

    """
    # convert path to source dataset to pathlib.Path object
    src_path = pathlib.Path(src_ds)

    # check whether the source dataset exists
    if not src_path.exists():
        LOGGER.info('{} does not exist.'.format(str(src_path)))
        return

    # check whether the output datasets exists
    trg_path = pathlib.Path(trg_ds)
    if not trg_path.exists():
        trg_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # check whether to overwrite existing files
        if not overwrite:
            LOGGER.info('{} already exists.'.format(trg_path))
            return

        # overwrite
        LOGGER.info('Overwrite {}'.format(str(trg_path)))
        trg_path.unlink()

    # read the source dataset
    src_ds = gdal.Open(str(src_path))

    # encode the NoData value to the output data type
    out_type = src_ds.GetRasterBand(1).DataType
    if no_data is not None:
        no_data = getattr(Gdal2Numpy,
                          gdal.GetDataTypeName(out_type)).value(no_data)

    # get the projection of the source dataset
    src_sr = src_ds.GetSpatialRef()

    # check whether a reference raster is provided
    if ref_ds is not None:

        if isinstance(ref_ds, osr.SpatialReference):
            ref_sr = ref_ds

            # the specified spatial resolution
            ref_xres = pixel_size[0]
            ref_yres = pixel_size[1]

        else:
            # read reference dataset
            ref_path = pathlib.Path(ref_ds)
            ref_ds = gdal.Open(str(ref_path))

            # get the projection of the reference dataset
            ref_sr = ref_ds.GetSpatialRef()

            # get the spatial resolution of the reference dataset
            ref_gt = ref_ds.GetGeoTransform()
            ref_xres = ref_gt[1]
            ref_yres = ref_gt[-1]

    else:
        # check whether an epsg code is provided
        if epsg is None:
            # if no epsg code is specified, use the source dataset spatial
            # reference
            ref_sr = src_sr
            LOGGER.info('Neither a reference dataset nor an epsg code provided'
                        '. Using the source spatial reference.')
        else:
            # the specified projection
            ref_sr = osr.SpatialReference()
            ref_sr.ImportFromEPSG(epsg)

        # the specified spatial resolution
        ref_xres = pixel_size[0]
        ref_yres = pixel_size[1]

    # create a temporary path
    tmp_path = _tmp_path(trg_path)

    # reproject source dataset to target projection
    LOGGER.info('Reproject: {}, Coordinate reference system: {}'.format(
        src_path.name, ref_sr.GetName()))
    gdal.Warp(str(tmp_path), str(src_path),
              srcSRS=src_sr.ExportToWkt(),
              dstSRS=ref_sr.ExportToWkt(),
              outputType=out_type,
              dstNodata=no_data,
              xRes=ref_xres,
              yRes=ref_yres,
              resampleAlg=resample)

    # compress raster
    compress_raster(tmp_path, trg_path)

    # clear gdal cache
    del src_ds, ref_ds


def reproject_vector(src_ds, trg_ds, ref_ds=None, epsg=None, overwrite=False):
    """Reproject a shapefile to a defined coordinate reference system.

    Reproject ``src_ds`` to ``trg_ds`` using either:
        - a defined `EPSG`_ code
        - a reference raster dataset ``ref_ds``, whose coordinate reference
        system is used for the reprojection

    Parameters
    ----------
    src_ds : `str` or :py:class:`pathlib.Path`
        The shapefile to reproject.
    trg_ds : `str` or :py:class:`pathlib.Path`
        The target shapefile.
    ref_ds : `str` or :py:class:`pathlib.Path`, optional
        A reference raster dataset, whose coordinate reference system is used
        for reprojection. The default is `None`.
    epsg : `int`, optional
        The EPSG code of the target coordinate reference system. The default is
        `None`.
    overwrite : `bool`, optional
        Whether to overwrite ``trg_ds``, if it exists. The default is `False`.

    """
    # convert path to source dataset to pathlib.Path object
    src_path = pathlib.Path(src_ds)

    # check whether the source dataset exists
    if not src_path.exists():
        LOGGER.info('{} does not exist.'.format(str(src_path)))
        return

    # check whether the output datasets exists
    trg_path = pathlib.Path(trg_ds)
    if not trg_path.exists():
        trg_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # check whether to overwrite existing files
        if not overwrite:
            LOGGER.info('{} already exists.'.format(trg_path))
            return

        # overwrite
        LOGGER.info('Overwrite {}'.format(str(trg_path)))
        trg_path.unlink()

    # read the source vector dataset
    src_ds = ogr.Open(str(src_path))
    src_lr = src_ds.GetLayer()

    # source spatial reference
    src_sr = src_lr.GetSpatialRef()

    # target spatial reference
    trg_sr = osr.SpatialReference()

    # check whether a reference raster is provided
    if ref_ds is not None:
        # read reference dataset
        ref_path = pathlib.Path(ref_ds)
        ref_ds = gdal.Open(str(ref_path))

        # get the projection of the reference dataset
        ref_sr = ref_ds.GetSpatialRef().ExportToWkt()

        # set target projection
        trg_sr.ImportFromWkt(ref_sr)

    else:
        # check whether an epsg code is provided
        if epsg is None:
            LOGGER.warning('Specify a reference raster or a valid epsg code. '
                           'Aborting...')
            return

        # set target projection
        trg_sr.ImportFromEPSG(epsg)

    # coordinate transformation
    crs_tr = osr.CoordinateTransformation(src_sr, trg_sr)

    # create target dataset
    driver = ogr.GetDriverByName('ESRI Shapefile')
    trg_ds = driver.CreateDataSource(str(trg_path))
    trg_lr = trg_ds.CreateLayer(src_lr.GetName(), trg_sr, ogr.wkbPolygon)

    # add source fields to target dataset
    src_ld = src_lr.GetLayerDefn()
    for i in range(src_ld.GetFieldCount()):
        field = src_ld.GetFieldDefn(i)
        trg_lr.CreateField(field)

    # output layer feature definition
    trg_ld = trg_lr.GetLayerDefn()

    # iterate over the source features
    src_ft = src_lr.GetNextFeature()
    while src_ft:
        # feature geometry type
        geom = src_ft.GetGeometryRef()

        # reproject geometry
        geom.Transform(crs_tr)

        # create target feature
        trg_ft = ogr.Feature(trg_ld)
        trg_ft.SetGeometry(geom)
        for i in range(trg_ld.GetFieldCount()):
            trg_ft.SetField(trg_ld.GetFieldDefn(i).GetNameRef(),
                            src_ft.GetField(i))

        # add feature to shapefile
        trg_lr.CreateFeature(trg_ft)

        # clear target feature
        del trg_ft

        # next input feature
        src_ft = src_lr.GetNextFeature()

    # clear source and target dataset
    del src_ds, trg_ds


def utm_spatial_reference(utm_id, north=True):
    """Define a spatial reference of the projected UTM coordinate system.

    Parameters
    ----------
    utm_id : `int`
        UTM zone identifier.
    north : `bool`, optional
        Whether to UTM zone ``utm_id`` is located on the northern or southern
        hemisphere. The default is `True`.

    Returns
    -------
    crs : :py:class:`osgeo.osr.SpatialReference`
        Spatial reference system of the UTM zone.

    """

    # define a coordinate reference system
    crs = osr.SpatialReference()

    # define UTM zone and spheroid
    crs.SetProjCS('WGS84 / UTM zone {}'
                  .format(''.join([str(utm_id), 'N' if north else 'S'])))
    crs.SetWellKnownGeogCS('WGS84')
    crs.SetUTM(utm_id)

    # identify epsg code
    crs.AutoIdentifyEPSG()

    return crs


def mgrs_tile_extent(mgrs_grid, tile_names):
    """Extracts the extent of Sentinel-2 tiles from the `MGRS tiling grid`_.

    Parameters
    ----------
    mgrs_grid : `str` or :py:class:`pathlib.Path`
        Path to the Sentinel-2 tiling grid kml file.
    tile_names : `list` [`int`]
        Name of the tiles to extract.

    Returns
    -------
    tile_coordinates : `dict` [`str`, `tuple`]
        Extent of the tiles. The keys are the name of the tiles and the values
        are a tuple of the bounding box coordinates in clockwise order,
        starting from the top-left corner.

    .. _MGRS tiling grid:
        https://sentinel.esa.int/web/sentinel/missions/sentinel-2/data-products

    """

    # parse mgrs kml file using xml
    LOGGER.info('Parsing MGRS grid ...')
    tree = ET.parse(str(mgrs_grid))
    root = tree.getroot()

    # kml namespace: required to search for correct tag names
    ns = re.match(r'{.*}', root.tag).group(0)

    # get the tags and names of the different tiles
    tiles = [item for item in tree.iter('{}Placemark'.format(ns))]
    names = [item.find('{}name'.format(ns)).text for item in tiles]

    # get the tiles of interest
    tile_coordinates = {}
    for tile in tile_names:
        # tag of current tile
        tile_tag = tiles[names.index(tile.lstrip('T'))]

        # get the coordinates of the bounding box
        bbox = list(tile_tag.iter('{}LinearRing'.format(ns))).pop()
        coordinates = bbox.find('{}coordinates'.format(ns)).text.strip()

        # read the coordinates to tuples defining tile corners
        # tl: top left, tr: top right, br: bottom right, bl: bottom left
        tl, tr, br, bl = [tuple([np.float(c) for c in point.split(',')])
                          for point in coordinates.split(',0')[:-2]]

        # store tile extent in dictionary
        tile_coordinates[tile.lstrip('T')] = (tl, tr, br, bl)

    LOGGER.info('Found coordinates for tiles: {}'.format(
        ', '.join(tile_coordinates.keys())))
    return tile_coordinates


def raster2mgrs(src_ds, mgrs_grid, tiles, trg_path, overwrite=False, **kwargs):

    # convert source and target paths to pathlib.Path objects
    src_ds = pathlib.Path(src_ds)
    trg_path = pathlib.Path(trg_path)
    LOGGER.info('Initializing tiling of: {}'.format(src_ds.name))

    # get the coordinate reference system of the source dataset
    ds = gdal.Open(str(src_ds))
    src_crs = ds.GetSpatialRef()
    LOGGER.info('Coordinate reference system: {}'.format(src_crs.GetName()))

    # get the extent of the tiles
    extent = mgrs_tile_extent(mgrs_grid, tiles)

    # source spatial reference system of the tiles: WGS84
    tl_crs = osr.SpatialReference()
    tl_crs.SetWellKnownGeogCS('WGS84')

    # iterate over the tiles of interest
    for tile in tiles:
        LOGGER.info('Clipping to tile: {}'.format(tile))

        # target spatial reference system of the tile
        utm_id = int(re.search('(T|)[0-6][0-9]', tile)[0].lstrip('T'))
        trg_crs = utm_spatial_reference(utm_id)

        # extent of the tile
        bbox = extent[tile.lstrip('T')]

        # coordinate transformation: from tile crs (WGS84) to src crs
        crs_tr = osr.CoordinateTransformation(tl_crs, src_crs)

        # transform extent of tile to source coordinate system

        # TransfromPoint expects input:
        #   - gdal >= 3.0: x, y, z = TransformPoint(y, x)
        #   - gdal < 3.0 : x, y, z = TransformPoint(x, y)
        x_tl, y_tl, _ = crs_tr.TransformPoint(bbox[0][1], bbox[0][0])
        x_br, y_br, _ = crs_tr.TransformPoint(bbox[2][1], bbox[2][0])

        # calculate a pixel buffer: extend the tile extent by 20% to ensure
        # that the whole tile is clipped
        pixel_buffer = (x_br - x_tl) / 5

        # buffered extent of the tile in the source coordinate reference system
        tile_extent = (x_tl - pixel_buffer, y_br - pixel_buffer,
                       x_br + pixel_buffer, y_tl + pixel_buffer)

        # extract tile extent from source dataset
        clip_ds = trg_path.joinpath(src_ds.stem + '_{}_clip.tif'.format(tile))
        clip_raster(src_ds, tile_extent, clip_ds, overwrite=overwrite)

        # reproject to target coordinate reference system
        repr_ds = trg_path.joinpath(src_ds.stem + '_{}_repr.tif'.format(tile))
        reproject_raster(clip_ds, repr_ds, ref_ds=trg_crs, overwrite=overwrite,
                         **kwargs)

        # remove the clipped dataset from disk
        clip_ds.unlink()

        # transform extent of tile to target coordinate system
        crs_tr = osr.CoordinateTransformation(src_crs, trg_crs)
        x_tl, y_tl, _ = np.round(crs_tr.TransformPoint(x_tl, y_tl), decimals=2)
        x_br, y_br, _ = np.round(crs_tr.TransformPoint(x_br, y_br), decimals=2)

        # extent of the tile in the target coordinate reference system
        tile_extent = (x_tl, y_br, x_br, y_tl)

        # clip reprojected raster to exact tile extent
        trg_ds = trg_path.joinpath(src_ds.stem + '_{}.tif'.format(tile))
        clip_raster(repr_ds, tile_extent, trg_ds, overwrite=overwrite)

        # remove reprojected dataset from disk
        repr_ds.unlink()


def vector2raster(src_ds, trg_ds, pixel_size, out_type, attribute=None,
                  burn_value=255, no_data=0, overwrite=False):
    """Convert a shapefile to a GeoTIFF.

    The vector data in the shapefile is converted to a GeoTIFF with a spatial
    resolution defined by ``pixel_size``. The values of the GeoTIFF correspond
    to the values of ``attribute``, which should be an existing attribute in
    ``src_ds``.

    Parameters
    ----------
    src_ds : `str` or :py:class:`pathlib.Path`
        The shapefile to convert.
    trg_ds : `str` or :py:class:`pathlib.Path`
        The target raster dataset.
    pixel_size : `tuple` [`int`, `int`]
        The pixel size of the target dataset, (height, width). The default is
        `(None, None)`.
    out_type : `int`
        An integer describing the data type of the target raster dataset. See
        :py:func:`gdal.GetDataTypeName` for an enumeration of the data types
        corresponding to the different integers.
    attribute : `str`, optional
        The shapefile attribute to use for the GeoTIFF values. Note that an
        error is raised if ``attribute`` does not exist in ``src_ds``. The
        default is `None`, which means that a constant burn value of 255 is
        used.
    burn_value : `int`, optional
        A fixed value to burn into each band for all objets in ``src_ds``. Used
        if ``attribute=None``. The default is `255`.
    no_data : `int` or `float`
        The value to assign to NoData values in ``src_ds``. The default is `0`.
    overwrite : `bool`, optional
        Whether to overwrite ``trg_ds``, if it exists. The default is `False`.

    Raises
    ------
    ValueError
        Raised if ``attribute`` does not exist in ``src_ds``.

    """
    # convert path to source dataset to pathlib.Path object
    src_path = pathlib.Path(src_ds)

    # check whether the source dataset exists
    if not src_path.exists():
        LOGGER.info('{} does not exist.'.format(str(src_path)))
        return

    # check whether the output datasets exists
    trg_path = pathlib.Path(trg_ds)
    if not trg_path.exists():
        trg_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # check whether to overwrite existing files
        if not overwrite:
            LOGGER.info('{} already exists.'.format(trg_path))
            return

        # overwrite
        LOGGER.info('Overwrite {}'.format(str(trg_path)))
        trg_path.unlink()

    # read the source vector dataset
    src_ds = ogr.Open(str(src_path))
    src_lr = src_ds.GetLayer()

    # the field names of the source vector dataset
    field_names = [field.name for field in src_lr.schema]

    # encode the NoData value to the output data type
    nodata = getattr(Gdal2Numpy, gdal.GetDataTypeName(out_type)).value(no_data)

    # check whether to use only a single attribute
    if attribute is not None:

        # check whether the attribute is valid
        if attribute not in field_names:
            raise ValueError('"{}" is not a valid attribute. {} has the '
                             'following attributes: \n{}'.format(
                                 attribute, src_path.name,
                                 '\n'.join(field_names)))
        else:
            # do not use a constant burn value when retrieving a specific
            # attribute
            burn_value = None

    # create a temporary path
    tmp_path = _tmp_path(trg_path)

    # rasterize vector dataset to defined spatial resolution
    LOGGER.info('Rasterizing: {}'.format(src_path))
    gdal.Rasterize(str(tmp_path), str(src_path),
                   xRes=pixel_size[1], yRes=pixel_size[0],
                   noData=nodata, outputType=out_type, attribute=attribute,
                   outputSRS=src_lr.GetSpatialRef(), burnValues=burn_value)

    # compress raster dataset
    compress_raster(tmp_path, trg_path)

    # clear source dataset
    del src_ds


def array_replace(array, lookup):
    """Replace all values in an array using a lookup table.

    Note that this function only works for replacing each value in an array and
    is primarily intended for use with class labels.

    Parameters
    ----------
    array : :py:class:`numpy.ndarray`
        The array of values to replace.
    lookup : :py:class:`numpy.ndarray`, shape=(nvals, 2)
        The lookup table. The first column of ``lookup`` should contain all the
        `nvals` unique values in ``array`` to replace and the second column the
        values to use as replacement.

    Returns
    -------
    :py:class:`numpy.ndarray`
        The array with the replaced values.

    """
    # create an index array to replace the values in the lookup table
    indices = np.arange(np.int(lookup[:, 0].max()) + 1)
    indices[lookup[:, 0].astype(np.int)] = lookup[:, 1]

    # the array with the replaced values
    return indices[array].astype(array.dtype)


def dec2bin(number, nbits=8):
    """Convert a decimal number to binary.

    Parameters
    ----------
    number : `int`
        The decimal number to convert.
    nbits : `int`, optional
        The number of bits to encode ``number``. The default is `8`.

    Returns
    -------
    binary : `str`
        The binary representation of ``number``.

    """
    # the binary number as string
    binary = ''

    # iterate over the number of bits
    for bit in range(nbits - 1, -1, -1):
        pos = number >> bit
        # append bit value to binary number
        if (pos & 1):
            binary += '1'
        else:
            binary += '0'

    return binary


def extract_by_mask(src_ds, mask_ds, trg_ds, overwrite=False,
                    src_no_data=None, trg_no_data=None):
    """Extract raster values by a shapefile.

    Extract the extent of ``mask_ds`` from ``src_ds``. The masked values of
    ``src_ds`` are saved in ``trg_ds``.

    If ``mask_ds`` is a shapefile, it is expected to be in the same coordinate
    reference system as ``src_ds``.

    Parameters
    ----------
    src_ds : `str` or :py:class:`pathlib.Path`
        The input raster to extract values from.
    mask_ds : `str` or :py:class:`pathlib.Path`
        A shapefile defining the area of interest.
    trg_ds : `str` or :py:class:`pathlib.Path`
        The masked target dataset.
    overwrite : `bool`, optional
        Whether to overwrite ``trg_ds``, if it exists. The default is `False`.
    src_no_data : `int` or `float`, optional
        The value of NoData values in ``src_ds``. The default is `None`, which
        means the value is read from ``src_ds``. If specified, values equal to
        ``src_no_data`` are masked as ``trg_no_data`` in ``trg_ds``.
    trg_no_data : `int` or `float`, optional
        The value to assign to NoData values in ``trg_ds``. The default is
        `None`, which means conserving the NoData value of ``src_ds``.

    """
    # convert path to source dataset and mask dataset to pathlib.Path object
    src_path = pathlib.Path(src_ds)
    mask_path = pathlib.Path(mask_ds)

    # check whether the source dataset exists
    if not src_path.exists():
        LOGGER.info('{} does not exist.'.format(str(src_path)))
        return

    # check whether the mask exists
    if not mask_path.exists():
        LOGGER.info('{} does not exist.'.format(str(mask_path)))
        return

    # check whether the output datasets exists
    trg_path = pathlib.Path(trg_ds)
    if not trg_path.exists():
        trg_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # check whether to overwrite existing files
        if not overwrite:
            LOGGER.info('{} already exists.'.format(trg_path))
            return

        # overwrite
        LOGGER.info('Overwrite {}'.format(str(trg_path)))
        trg_path.unlink()

    # create a temporary file
    tmp_path = _tmp_path(trg_path)

    # read the source dataset
    src_ds = gdal.Open(str(src_path))
    LOGGER.info('Extract: {}, {}'.format(src_path.name, mask_path.name))

    # extract raster values by shapefile
    gdal.Warp(str(tmp_path), str(src_path),
              cutlineDSName=str(mask_path),
              cropToCutline=True,
              xRes=src_ds.GetGeoTransform()[1],
              yRes=src_ds.GetGeoTransform()[5],
              srcNodata=src_no_data,
              dstNodata=trg_no_data)

    # compress raster
    compress_raster(tmp_path, trg_path)

    # clear source dataset
    del src_ds


def clip_raster(src_ds, mask_ds, trg_ds, overwrite=False, src_no_data=None,
                trg_no_data=None):
    """Clip raster to extent of another raster.

    Clip the extent of ``src_ds`` to the extent of ``mask_ds``. The clipped
    raster is saved in ``trg_ds``.

    Parameters
    ----------
    src_ds : `str` or :py:class:`pathlib.Path`
        The input raster to clip.
    mask_ds : `str` or :py:class:`pathlib.Path` or `tuple`
        A raster, shapefile or a `tuple`  defining the extent of interest. If a
        `tuple`, it is assumed to define the extent in the coordinate system of
        ``src_ds`` as: (x_min, y_min, x_max, y_max).
    trg_ds : `str` or :py:class:`pathlib.Path`
        The clipped raster dataset.
    overwrite : `bool`, optional
        Whether to overwrite ``trg_ds``, if it exists. The default is `False`.
    src_no_data : `int` or `float`, optional
        The value of NoData values in ``src_ds``. The default is `None`, which
        means the value is read from ``src_ds``. If specified, values equal to
        ``src_no_data`` are masked as ``trg_no_data`` in ``trg_ds``.
    trg_no_data : `int` or `float`, optional
        The value to assign to NoData values in ``trg_ds``. The default is
        `None`, which means conserving the NoData value of ``src_ds``.

    """
    # convert path to source dataset and mask dataset to pathlib.Path object
    src_path = pathlib.Path(src_ds)

    # check whether the source dataset exists
    if not src_path.exists():
        LOGGER.info('{} does not exist.'.format(str(src_path)))
        return

    # source dataset spatial reference
    src_ds = gdal.Open(str(src_path))
    src_sr = osr.SpatialReference()
    src_sr.ImportFromWkt(src_ds.GetProjection())

    # check whether the output datasets exists
    trg_path = pathlib.Path(trg_ds)
    if not trg_path.exists():
        trg_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # check whether to overwrite existing files
        if not overwrite:
            LOGGER.info('{} already exists.'.format(trg_path))
            return

        # overwrite
        LOGGER.info('Overwrite {}'.format(str(trg_path)))
        trg_path.unlink()

    # mask is defined as extent
    if isinstance(mask_ds, tuple):
        extent = mask_ds
    else:
        # check whether the mask exists
        mask_path = pathlib.Path(mask_ds)
        if not mask_path.exists():
            LOGGER.info('{} does not exist.'.format(str(mask_path)))
            return

        # check whether the mask is a shapefile or a raster
        if mask_path.suffix == '.shp':
            # mask is a shapefile
            mask_ds = ogr.Open(str(mask_path))
            mask_lr = mask_ds.GetLayer()

            # spatial reference of the shapefile
            mask_sr = mask_lr.GetSpatialRef()

            # extent of the shapefile: (x_min, x_max, y_min, y_max)
            extent = mask_lr.GetExtent()
        else:
            # mask is a raster dataset
            mask_ds = gdal.Open(str(mask_path))

            # spatial extent of the mask: (x_min, x_max, y_min, y_max)
            gt = mask_ds.GetGeoTransform()
            extent = [gt[0], gt[0] + gt[1] * mask_ds.RasterXSize,
                      gt[3] + gt[5] * mask_ds.RasterYSize, gt[3]]

            # mask dataset spatial reference
            mask_sr = mask_ds.GetSpatialRef()

        # transform extent of mask to source coordinate system
        crs_tr = osr.CoordinateTransformation(mask_sr, src_sr)

        # TransfromPoint expects input:
        #   - gdal >= 3.0: x, y, z = TransformPoint(y, x)
        #   - gdal < 3.0 : x, y, z = TransformPoint(x, y)
        x_tl, y_tl, _ = crs_tr.TransformPoint(extent[0], extent[-1])
        x_br, y_br, _ = crs_tr.TransformPoint(extent[1], extent[2])

        # extent of the mask in the source reference coordinate system:
        # (x_min, y_min, x_max, y_max)
        extent = [x_tl, y_br, x_br, y_tl]

    # create a temporary file
    tmp_path = _tmp_path(trg_path)

    # clip raster extent
    LOGGER.info('Clipping: {}, Extent: (x_tl={:.2f}, y_br={:.2f}, x_br={:.2f},'
                ' y_tl={:.2f})'.format(src_path.name, *extent))
    gdal.Warp(str(tmp_path), str(src_path),
              outputBounds=extent,
              outputBoundsSRS=src_ds.GetSpatialRef(),
              xRes=src_ds.GetGeoTransform()[1],
              yRes=src_ds.GetGeoTransform()[5],
              srcNodata=src_no_data,
              dstNodata=trg_no_data,
              targetAlignedPixels=True)

    # compress raster dataset
    compress_raster(tmp_path, trg_path)

    # clear source and mask dataset
    del src_ds, mask_ds


def pixels_within_lowres(top_left, res_prop):
    """Calculate the position of high-res pixels within a low-res pixel.

    .. important:

        This function assumes that the low-resolution and high-resolution image
        have the same georeferenced origin, i.e. the same top left corner in
        physical coordinates.

    Parameters
    ----------
    top_left : `tuple`
        The top left corner indices of the low-resolution pixel in the
        low-resolution image.
    res_prop : `int` or `float`
        The proportion of the low-resolution to the high-resolution.

    Returns
    -------
    indices : `tuple` [:py:class:`numpy.ndarray`]
        The indices of the high-resolution pixels within the high-resolution
        image, which are fully-covered by the extent of the low-resolution
        pixel ``top_left``.

    Example
    -------

        The resolution of the low-resolution image is 250m, the one of the
        high-resolution image is 30m, thus ``res_prop=250/30``. Assume that the
        top left corner of the pixel of interest in the low-resolution image is
        given by the position ``top_left=(0, 1)``. In the high-resolution
        image, the top left corner of the pixel of interest is hence located
        at ``top_left * res_prop``. This function returns the indices of all
        the pixels in the high-resolution image, which are fully-covered by
        the low-resolution pixel with position ``top-left``.

    """
    # the top left corner of a pixel in the low resolution image
    xtl_lres, ytl_lres = top_left

    # Note: only high-resolution pixel which are fully-covered by the
    #       low-resolution pixel are considered.

    # the top left corner of the first pixel in the high resolution image,
    # which lies within the extent of the low-resolution pixel
    xtl_hres, ytl_hres = (np.ceil(xtl_lres * res_prop).astype(np.int),
                          np.ceil(ytl_lres * res_prop).astype(np.int))

    # the bottom right corner of last pixel in the high resolution image,
    # which lies within the extent of the low-resolution pixel
    xbl_hres, ybl_hres = (np.floor((xtl_lres + 1) * res_prop).astype(np.int),
                          np.floor((ytl_lres + 1) * res_prop).astype(np.int))

    # the range of the indices in the high-resolution image
    xhres = np.arange(xtl_hres, xbl_hres, 1)
    yhres = np.arange(ytl_hres, ybl_hres, 1)

    # the indices of all the high-resolution pixels, which fully lie within
    # the extent of the low-resolution pixel
    indices = (np.repeat(xhres, repeats=len(yhres)),
               np.tile(yhres, reps=len(xhres)))

    return indices


def gdb2shp(src_ds, feature=''):
    """Convert an Esri Geodatabase to a shapefile.

    The shapefiles are stored in the same directory as ``src_ds``.

    Parameters
    ----------
    src_ds : `str` or :py:class:`pathlib.Path`
        Path to the Esri Geodatabase.
    feature : `str`
        The name of a feature class in ``src_ds``. If specified, only
        ``feature`` is converted to a shapefile. The default is `''`.

    """
    # call the osgeo ogr2ogr system utility
    subprocess.run('ogr2ogr -f "ESRI Shapefile" {} {} {}'.format(
        src_ds, src_ds.parent, feature))


def merge_tifs(trg_ds, tifs, **kwargs):
    """Mosaic a set of images.

    Parameters
    ----------
    trg_ds : `str` or :py:class:`pathlib.Path`
        Path to the output GeoTiff file.
    tifs : `list` [`str` or :py:class:`pathlib.Path`]
        List of paths to the GeoTiffs to mosaic.
    **kwargs : `dict`, optional
        Optional keyword arguments passed to :py:func:`osgeo.gdal.Warp`.

    """
    # check if target path exists
    trg_ds = pathlib.Path(trg_ds)
    if not trg_ds.parent.exists():
        LOGGER.info('mkdir {}'.format(trg_ds.parent))
        trg_ds.parent.mkdir(parents=True, exist_ok=True)

    # create mosaic
    tmp_path = _tmp_path(trg_ds)
    LOGGER.info('Create mosaic: {}'.format(trg_ds))
    LOGGER.info('Using rasters:')
    LOGGER.info(('\n ' + (len(__name__) + 1) * ' ').join(
            ['{}'.format(str(tif)) for tif in tifs]))
    gdal.Warp(str(tmp_path), [str(tif) for tif in tifs], **kwargs)

    # compress raster
    compress_raster(tmp_path, trg_ds)


def compress_raster(src_ds, trg_ds):
    """Compress a raster dataset using :py:func:`gdal.Translate`.

    Parameters
    ----------
    src_ds : `str` or :py:class:`pathlib.Path`
        Path to the raster dataset to compress.
    trg_ds : `str` or :py:class:`pathlib.Path`
        Path to save the compressed raster dataset.

    """
    # check if the raster dataset exists
    src_ds = pathlib.Path(src_ds)
    if not src_ds.exists():
        LOGGER.info('{} does not exist'.format(src_ds))
        return

    # compress raster dataset
    LOGGER.info('Compressing: {}'.format(trg_ds))
    gdal.Translate(str(trg_ds), str(src_ds), creationOptions=[
        'COMPRESS=DEFLATE', 'PREDICTOR=1', 'TILED=YES'])

    # remove uncompressed raster dataset
    src_ds.unlink()


def _tmp_path(path):
    """Create a temporary filename from a given path.

    Parameters
    ----------
    path : `str` or :py:class:`pathlib.Path`
        Path to a file for which a temporary path is required.

    Returns
    -------
    tmp_path : `str` or :py:class:`pathlib.Path`
        Path to the temporary file.

    """
    path = pathlib.Path(path)
    return pathlib.Path(str(path).replace(path.suffix,
                                          '_tmp{}'.format(path.suffix)))


def report2df(report, labels=None):
    """Convert :py:func:`sklearn.metrics.classification_reports` to a
    :py:class:`pandas.DataFrame`.

    Parameters
    ----------
    report : `dict`
        The dictionary returned by
        :py:class:`sklearn.metrics.classification_report`.
    labels : `list` [`str`]
        List of class labels. If specified, the number of samples of each class
        is normalized. The default is `None`.

    """
    # convert to pandas DataFrame
    df = pd.DataFrame(report).transpose()

    # add errors of commission and omission
    df.insert(loc=3, column='commission', value=1 - df.precision)
    df.insert(loc=4, column='omission', value=1 - df.recall)

    # normalize support values to relative values
    if labels is not None:
        df.support = df.support / df.loc[labels].support.sum()

    return df
