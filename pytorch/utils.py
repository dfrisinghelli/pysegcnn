# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:02:23 2020

@author: Daniel
"""
# builtins
import re
import datetime


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
