"""Functions to preprocess the Sparcs dataset to work with pylandsat."""

# !/usr/bin/env python
# coding: utf-8

# builtins
import os
import sys
import glob
import shutil

# externals
import gdal
import numpy as np

# local packages
from pylandsat.core.untar import extract_data
from pylandsat.core.calibration import landsat_radiometric_calibration

# append path to local files to the python search path
sys.path.append('..')

# local
from sparcs.sparcs_00_config import sparcs_archive, sparcs_path


def sparcs2pylandsat(source_path, target_path, overwrite=True):
    """Convert the Sparcs dataset structure to the pylandsat standard.

    Parameters
    ----------
    source_path : string
        path to the Sparcs archive downloaded `here`_
    target_path : string
        path to save the preprocessed sparcs dataset
    overwrite : bool
        whether to overwrite existing files

    Returns
    -------
    None.

    .. _here:
        https://www.usgs.gov/land-resources/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation

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
            fname = file.split('_')[0]

            # define the new path to the file
            new_path = os.path.join(target_path, fname)

            # check if file is the metadata file
            if file.endswith('_mtl.txt'):

                # add the collection number to the metadata file
                with open(old_path, 'a') as mfile:
                    mfile.write('COLLECTION_NUMBER = 1')

                # replace file ending
                file = file.replace('mtl', 'MTL')

            # move files to new directory
            if os.path.isfile(new_path + os.sep + file) and not overwrite:
                print('{} already exists.'.format(new_path + os.sep + file))
                continue
            else:
                os.makedirs(new_path, exist_ok=True)
                shutil.move(old_path, new_path + os.sep + file)

    # remove old file location
    shutil.rmtree(source_path)


def destack_sparcs_raster(inpath, outpath=None, suffix='*_toa.tif'):
    """Destack a TIFF with more than one band into a TIFF file for each band.

    Parameters
    ----------
    inpath : string
        path to a directory containing the TIFF file to destack
    outpath : string, optional
        path to save the output TIFF files. The default is None. If None,
        ``outpath`` = ``inpath``.

    Returns
    -------
    None.

    """
    # default: output directory is equal to the input directory
    if outpath is None:
        outpath = inpath

    # check if output directory exists
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # get the TIFF to destack
    tif = glob.glob(inpath + os.sep + '*data.tif').pop()

    # open the raster
    img = gdal.Open(tif)

    # check whether the current scene was already processed
    processed = glob.glob(inpath + os.sep + suffix)
    if len(processed) == img.RasterCount:
        print('Scene: {} already processed.'.format(os.path.basename(inpath)))
        img = None
        os.unlink(tif)
        return

    # image driver
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    # output image type: digital numbers unsigned integer 16bit
    codage = gdal.GDT_UInt16
    nptype = np.uint16

    # image size and tiles
    cols = img.RasterXSize
    rows = img.RasterYSize
    bands = img.RasterCount

    # print progress
    imgname = os.path.basename(tif)
    print('Processing: {}'.format(imgname))

    # iterate the bands of the raster
    for b in range(1, bands + 1):
        # output file: replace for band name
        fname = os.path.join(outpath, imgname.replace('data', 'B' + str(b)))
        outDs = driver.Create(fname, cols, rows, 1, codage)

        # read the data of band b
        band = img.GetRasterBand(b)
        data = band.ReadAsArray().astype(nptype)

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
    img = None
    os.unlink(tif)


if __name__ == '__main__':

    # extract the raw archive to the output path
    location = extract_data(sparcs_archive, sparcs_path)

    # transform SPARCS directory structure to pylandsat standard
    sparcs2pylandsat(source_path=location, target_path=sparcs_path,
                     overwrite=False)

    # destack the TIFF rasterstack to a single TIFF for each band and perform
    # radiometric calibration
    for scene in os.listdir(sparcs_path):
        # path to the current scene
        scene_path = os.path.join(sparcs_path, scene)

        # build the GeoTIFFs for each band
        destack_sparcs_raster(scene_path, suffix='*_toa.tif')

        # convert the digital number format to top of atmosphere reflectance
        landsat_radiometric_calibration(scene_path, exclude=[], suffix='_toa',
                                        overwrite=False, remove_raw=True)
