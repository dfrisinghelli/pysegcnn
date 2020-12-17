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
import xml.etree.ElementTree as ET

# externals
import gdal
import torch
import numpy as np
from osgeo import ogr, osr

# locals
from pysegcnn.core.constants import Gdal2Numpy

# module level logger
LOGGER = logging.getLogger(__name__)

# suffixes for radiometrically calibrated scenes
SUFFIXES = ['toa_ref', 'toa_rad', 'toa_brt']

# maximum number of filename characters on Windows
MAX_FILENAME_CHARS_WINDOWS = 260


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

        # the image dataset as returned by gdal
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
        img = path
        bands = img.shape[0]
        height = img.shape[1]
        width = img.shape[2]
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


def np2tif(array, filename, src_ds=None, epsg=None, geotransform=None):
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
    src_ds : :py:class:`osgeo.gdal.Dataset`
        The source dataset from which the spatial reference is inherited. The
        default is `None`.
    epsg : `int`, optional
        The EPSG code of the target coordinate reference system. The default is
        `None`.
    geotransform : `tuple`
        A tuple with six elements of the form,
        (x_top_left, x_res, x_shift, y_top_left, -y_res, y_shift), describing
        the spatial reference.

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

    # create output GeoTIFF
    trg_ds = driver.Create(str(filename), width, height, bands, dtype)

    # iterate over the number of bands and write to output file
    for b in range(bands):
        trg_band = trg_ds.GetRasterBand(b + 1)
        trg_band.WriteArray(array[b, ...])
        trg_band.FlushCache()

    # set spatial reference
    if src_ds is not None:
        # inherit spatial reference from source dataset
        trg_ds.SetProjection(src_ds.GetProjection())
        trg_ds.SetGeoTransform(src_ds.GetGeoTransform())
    else:
        # check whether both the epsg code and the geotransform tuple are
        # specified
        if epsg is not None and geotransform is not None:
            # create the spatial reference from the epsg code
            sr = osr.SpatialReference().ImportFromEPSG(epsg).ExporttoWkt()

            # set spatial reference from epsg
            trg_ds.SetProjection(sr)
            trg_ds.SetGeoTransform(geotransform)
        else:
            raise ValueError('Both "epsg" and "geotransform" required to set '
                             'spatial reference if "src_ds" is None.')

    # clear dataset
    del trg_band, trg_ds

    return


def read_hdf(path, **kwargs):
    """Read a file in Hierarchical Data Format (HDF) to a dictionary.

    Parameters
    ----------
    path : `str` or py:class:`pathlib.Path`
        The path to the hdf file to read.
    **kwargs:
        Additional keyword arguments passed to
        :py:func:`pysegcnn.core.utils.img2np`.

    Raises
    ------
    ValueError
        Raised if ``path`` is not an hdf file.

    Returns
    -------
    hdf_ds : `dict` [`str`, :py:class:`numpy.ndarray`]
        The hdf file as a dictionary. The keys are the names of the
        different datasets in the hdf file and the values are the corresponding
        data as a :py:class:`numpy.ndarray`.

    """
    # check if the path points to an hdf file
    if not str(path).endswith(('.h4', '.hdf', '.hdf4', '.hdf5', '.he2', '.h5',
                               '.he5')):
        raise ValueError('{} is not an hdf file.'.format(path))

    # read the hdf dataset
    hdf = gdal.Open(str(path)).GetSubDatasets()

    # iterate over the different subsets and store arrays in dictionary
    hdf_ds = {}
    for ds in hdf:
        # name of the current dataset
        name = ds[0].split(':')[-1]

        # read the dataset to a numpy array
        data = img2np(ds[0], **kwargs)

        # store in dictionary
        hdf_ds[name] = data

    return hdf_ds


def hdf2tifs(path, outpath=None, overwrite=False, create_stack=True, **kwargs):
    """Convert a file in Hierarchical Data Format (HDF) to GeoTIFFs.

    The GeoTIFFs share the same filename as ``path``, appended by the name of
    the respective subdatasets.

    The default (``outpath=None``) is to save the GeoTIFFs in a directory named
    after the filename of ``path``, within the parent directory of ``path``.

    Parameters
    ----------
    path : `str` or py:class:`pathlib.Path`
        The path to the hdf file to convert.
    outpath : `str` or py:class:`pathlib.Path`, optional
        Path to save the GeoTIFF files. The default is `None`.
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
    if not str(path).endswith(('.h4', '.hdf', '.hdf4', '.hdf5', '.he2', '.h5',
                               '.he5')):
        raise ValueError('{} is not an hdf file.'.format(path))

    # check whether an output path is defined
    if outpath is None:
        outpath = path.parent
    else:
        outpath = pathlib.Path(outpath)

    # create the output directory
    outpath = outpath.joinpath(path.stem.replace('.', '_'))

    # check whether the output path exists
    if not outpath.exists():
        LOGGER.info('mkdir {}'.format(str(outpath)))
        outpath.mkdir(parents=True, exist_ok=True)

    # check whether the output path contains GeoTIFF files
    tifs = [f for f in outpath.iterdir() if f.suffix in ['.tif', '.TIF']]

    # check whether to overwrite existing files
    if tifs:
        LOGGER.info('The following files already exist in {}'
                    .format(str(outpath)))
        LOGGER.info('\n'.join(['{}'.format(str(tif.name)) for tif in tifs])
                    )
        if not overwrite:
            # return if not overwriting
            LOGGER.info('Aborting...')
            return

        # remove existing files and prepare to overwrite
        LOGGER.info('Overwrite {}'.format(str(outpath)))
        for tif in tifs:
            tif.unlink()

    # read the hdf dataset
    hdf = gdal.Open(str(path)).GetSubDatasets()

    # check if the dataset is not empty
    if hdf:

        # iterate over the different subdatasets in the hdf
        for ds in hdf:

            # name of the current subdataset
            name = ds[0].split(':')[-1]

            # filename of the GeoTIFF
            tif_name = outpath.joinpath(
                path.name.replace(path.suffix, '_{}.tif'.format(name)))

            # convert hdf subdataset to GeoTIFF
            gdal.Translate(str(tif_name), gdal.Open(ds[0]), **kwargs)

        # check whether to create a GeoTIFF stack
        if create_stack:
            # filename for the GeoTIFF stack
            stk = tif_name.parent.joinpath(
                path.name.replace(path.suffix, '.tif'))
            LOGGER.info('Creating GeoTIFF stack: {}'.format(stk))

            # generated GeoTIFF files
            tifs = [str(f) for f in outpath.iterdir() if f.suffix in
                    ['.tif', '.TIF']]

            # create stacked GeoTIFF
            stack_tifs(str(stk), tifs)

    return


def stack_tifs(filename, tifs, **kwargs):
    """Create a stacked GeoTIFF from a list of single-band GeoTIFFs.

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

    # create GeoTIFF stack
    gdal.Translate(str(filename), vrt_ds, **kwargs)

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
    tiles : :py:class:`torch.Tensor` or :py:class:`numpy.ndarray`
        The tiled image, shape: `(tiles, bands, tile_size, tile_size)` or
        `(tiles, tile_size, tile_size)`.

    Returns
    -------
    image : :py:class:`numpy.ndarray`
        The reconstructed image, shape: `(bands, height, width)`.

    """
    # convert to numpy array
    tiles = np.asarray(tiles)

    # check the size
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
    ref_ds : `str` or :py:class:`pathlib.Path`, optional
        A reference raster dataset, whose coordinate reference system is used
        for reprojection. The default is `None`.
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
        LOGGER.warning('{} does not exist.'.format(str(src_path)))

    # check whether the output datasets exists
    trg_path = pathlib.Path(trg_ds)
    if not trg_path.exists():
        LOGGER.info('mkdir {}'.format(str(trg_path.parent)))
        trg_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # check whether to overwrite existing files
        if overwrite:
            LOGGER.info('Overwrite {}'.format(str(trg_path)))
            trg_path.unlink()

    # read the source dataset
    src_ds = gdal.Open(str(src_path))

    # encode the NoData value to the output data type
    out_type = src_ds.GetRasterBand(1).DataType
    nodata = getattr(Gdal2Numpy, gdal.GetDataTypeName(out_type)).value(no_data)

    # get the projection of the source dataset
    src_epsg = get_epsg(src_ds)
    src_proj = 'epsg:{}'.format(src_epsg)

    # check whether a reference raster is provided
    if ref_ds is not None:
        # read reference dataset
        ref_path = pathlib.Path(ref_ds)
        ref_ds = gdal.Open(str(ref_path))

        # get the projection of the reference dataset
        ref_epsg = get_epsg(ref_ds)
        ref_proj = 'epsg:{}'.format(ref_epsg)

        # get the spatial resolution of the reference dataset
        ref_gt = ref_ds.GetGeoTransform()
        ref_xres = ref_gt[1]
        ref_yres = ref_gt[-1]

    else:
        # check whether an epsg code is provided
        if epsg is None:
            # if no epsg code is specified, use the source dataset spatial
            # reference
            epsg = src_epsg
            LOGGER.info('Neither a reference dataset nor an epsg code provided'
                        '. Using the source spatial reference.')

        # the specified projection
        ref_proj = 'epsg:{}'.format(epsg)

        # the specified spatial resolution
        ref_xres = pixel_size[0]
        ref_yres = pixel_size[1]

    # reproject source dataset to target projection
    LOGGER.info(
        'Reproject {}: {} to {}'.format(src_path.name, src_proj, ref_proj))
    gdal.Warp(str(trg_path), str(src_path),
              srcSRS=src_proj,
              dstSRS=ref_proj,
              outputType=out_type,
              dstNodata=nodata,
              xRes=ref_xres,
              yRes=ref_yres,
              resampleAlg=resample)

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
        LOGGER.warning('{} does not exist.'.format(str(src_path)))

    # check whether the output datasets exists
    trg_path = pathlib.Path(trg_ds)
    if not trg_path.exists():
        LOGGER.info('mkdir {}'.format(str(trg_path.parent)))
        trg_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # check whether to overwrite existing files
        if overwrite:
            LOGGER.info('Overwrite {}'.format(str(trg_path)))
            trg_path.unlink()

    # read the source vector dataset
    src_ds = ogr.Open(str(src_path))
    src_lr = src_ds.GetLayer()

    # source spatial reference
    src_sr = src_lr.GetSpatialRef()

    # check whether a reference raster is provided
    if ref_ds is not None:
        # read reference dataset
        ref_path = pathlib.Path(ref_ds)
        ref_ds = gdal.Open(str(ref_path))

        # get the projection of the reference dataset
        epsg = np.int(get_epsg(ref_ds))

    else:
        # check whether an epsg code is provided
        if epsg is None:
            LOGGER.warning('Specify a reference raster or a valid epsg code. '
                           'Aborting...')
            return

    # target spatial reference
    trg_sr = osr.SpatialReference()
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


def vector2raster(src_ds, trg_ds, attribute, pixel_size, out_type, no_data=0,
                  overwrite=False):
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
    attribute : `str`
        The shapefile attribute to use for the GeoTIFF values. Note that an
        error is raised if ``attribute`` does not exist in ``src_ds``.
    pixel_size : `tuple` [`int`, `int`]
        The pixel size of the target dataset, (height, width). The default is
        `(None, None)`.
    out_type : `int`
        An integer describing the data type of the target raster dataset. See
        :py:func:`gdal.GetDataTypeName` for an enumeration of the data types
        corresponding to the different integers.
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
        LOGGER.warning('{} does not exist.'.format(str(src_path)))

    # check whether the output datasets exists
    trg_path = pathlib.Path(trg_ds)
    if not trg_path.exists():
        LOGGER.info('mkdir {}'.format(str(trg_path.parent)))
        trg_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # check whether to overwrite existing files
        if overwrite:
            LOGGER.info('Overwrite {}'.format(str(trg_path)))
            trg_path.unlink()

    # read the source vector dataset
    src_ds = ogr.Open(str(src_path))
    src_lr = src_ds.GetLayer()

    # the field names of the source vector dataset
    field_names = [field.name for field in src_lr.schema]

    # check if the defined attribute name is in the field names
    if attribute not in field_names:
        raise ValueError('"{}" is not a valid attribute. {} has the following '
                         'attributes: \n{}'.format(
                             attribute, src_path.name, '\n'.join(field_names))
                         )

    # get the source spatial extent
    x_min, x_max, y_min, y_max = src_lr.GetExtent()

    # encode the NoData value to the output data type
    nodata = getattr(Gdal2Numpy, gdal.GetDataTypeName(out_type)).value(no_data)

    # rasterize vector dataset to defined spatial resolution
    gdal.Rasterize(str(trg_path), str(src_path),
                   xRes=pixel_size[1], yRes=pixel_size[0], noData=nodata,
                   outputType=out_type, attribute=attribute)

    # clear source dataset
    del src_ds
