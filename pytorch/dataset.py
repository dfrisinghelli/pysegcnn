"""Custom dataset classes compliant to the PyTorch standard.

Each custom dataset should inherit from torch.utils.data.Dataset to benefit
from the torch.utils.data.DataLoader class, which implements helpful utilities
during model training.

For any kind of image-like dataset, inherit the ImageDataset class to create
your custom dataset.
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import os
import re
import sys
import csv
import glob
import itertools

# externals
import gdal
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# append path to local files to the python search path
sys.path.append('..')

# locals
from pytorch.constants import (Landsat8, Sentinel2, SparcsLabels,
                               Cloud95Labels, ProSnowLabels)
from pytorch.graphics import plot_sample
from pytorch.utils import parse_landsat8_date, parse_sentinel2_date

# generic image dataset class
class ImageDataset(Dataset):

    def __init__(self, root_dir, use_bands, tile_size):
        super().__init__()

        # the root directory: path to the image dataset
        self.root = root_dir

        # the size of a scene/patch in the dataset
        self.size = self.get_size()

        # the available spectral bands in the dataset
        self.bands = self.get_bands()

        # the class labels
        self.labels = self.get_labels()

        # check which bands to use
        self.use_bands = (use_bands if use_bands else [*self.bands.values()])

        # each scene is divided into (tile_size x tile_size) blocks
        # each of these blocks is treated as a single sample
        self.tile_size = tile_size

        # calculate number of resulting tiles and check whether the images are
        # evenly divisible in square tiles of size (tile_size x tile_size)
        if self.tile_size is None:
            self.tiles = 1
        else:
            self.tiles = self.is_divisible(self.size, self.tile_size)

        # the samples of the dataset
        self.scenes = []

    # the __len__() method returns the number of samples in the dataset
    def __len__(self):
        # number of (tiles x channels x height x width) patches after each
        # scene is decomposed to tiles blocks
        return len(self.scenes)

    # the __getitem__() method returns a single sample of the dataset given an
    # index, i.e. an array/tensor of shape (channels x height x width)
    def __getitem__(self, idx):

        # select a scene
        scene = self.read_scene(idx)

        # get samples: (tiles x channels x height x width)
        data, gt = self.build_samples(scene)

        # preprocess input and return torch tensors of shape:
        # x : (bands, height, width)
        # y : (height, width)
        x, y = self.preprocess(data, gt)

        return x, y

    # the compose_scenes() method has to be implemented by the class inheriting
    # the ImageDataset class
    # compose_scenes() should return a list of dictionaries, where each
    # dictionary represent one sample of the dataset, a scene or a tile
    # of a scene, etc.
    # the dictionaries should have the following (key, value) pairs:
    #   - (band_1, path_to_band_1.tif)
    #   - (band_2, path_to_band_2.tif)
    #   - ...
    #   - (band_n, path_to_band_n.tif)
    #   - (gt, path_to_ground_truth.tif)
    #   - (tile, None or int)
    def compose_scenes(self, *args, **kwargs):
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    # the get_size() method has to be implemented by the class inheriting
    # the ImageDataset class
    # get_size() method should return the image size as tuple, (height, width)
    def get_size(self, *args, **kwargs):
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    # the get_bands() method has to be implemented by the class inheriting
    # the ImageDataset class
    # get_bands() should return a dictionary with the following
    # (key: int, value: str) pairs:
    #    - (1, band_1_name)
    #    - (2, band_2_name)
    #    - ...
    #    - (n, band_n_name)
    def get_bands(self, *args, **kwargs):
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    # the get_labels() method has to be implemented by the class inheriting
    # the ImageDataset class
    # get_labels() should return a dictionary with the following
    # (key: int, value: str) pairs:
    #    - (0, label_1_name)
    #    - (1, label_2_name)
    #    - ...
    #    - (n, label_n_name)
    # where the keys should be the values representing the values of the
    # corresponding label in the ground truth mask
    # the labels in the dictionary determine the classes to be segmented
    def get_labels(self, *args, **kwargs):
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    # the preprocess() method has to be implemented by the class inheriting
    # the ImageDataset class
    # preprocess() should return two torch.tensors:
    #    - input data: tensor of shape (bands, height, width)
    #    - ground truth: tensor of shape (height, width)
    def preprocess(self, data, gt):
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    # _read_scene() reads all the bands and the ground truth mask in a
    # scene/tile to a numpy array and returns a dictionary with
    # (key, value) = ('band_name', np.ndarray(band_data))
    def read_scene(self, idx):

        # select a scene from the root directory
        scene = self.scenes[idx]

        # read each band of the scene into a numpy array
        scene_data = {key: (self.img2np(value, tile_size=self.tile_size,
                                        tile=scene['tile'])
                            if key != 'tile' else value)
                      for key, value in scene.items()}

        return scene_data

    # _build_samples() stacks all bands of a scene/tile into a
    # numpy array of shape (bands x height x width)
    def build_samples(self, scene):

        # iterate over the channels to stack
        stack = np.stack([scene[band] for band in self.use_bands], axis=0)
        gt = scene['gt']

        return stack, gt

    # the following functions are utility functions for common image
    # manipulation operations

    # this function reads an image to a numpy array
    def img2np(self, path, tile_size=None, tile=None):

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
            ntiles = self.is_divisible((img.RasterXSize, img.RasterYSize),
                                       tile_size)

            # get the indices of the top left corner for each tile
            topleft = self.tile_offsets((img.RasterYSize, img.RasterXSize),
                                        tile_size)

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
    def is_divisible(self, img_size, tile_size):
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
    def tile_offsets(self, img_size, tile_size):

        # check if divisible
        _ = self.is_divisible(img_size, tile_size)

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


class StandardEoDataset(ImageDataset):

    def __init__(self, root_dir, use_bands, tile_size, sort):
        # initialize super class ImageDataset
        super().__init__(root_dir, use_bands, tile_size)

        # function that parses the date from a Landsat 8 scene id
        self.date_parser = None

        # whether to sort the list of samples:
        # for time series data, set sort=True to obtain the scenes in
        # chronological order
        self.sort = sort

    # returns the band number of a Landsat8 or Sentinel2 tif file
    # x: path to a tif file
    def get_band_number(self, path):

        # check whether the path leads to a tif file
        if not path.endswith(('tif', 'TIF')):
            raise ValueError('Expected a path to a tif file.')

        # filename
        fname = os.path.basename(path)

        # search for numbers following a "B" in the filename
        band = re.search('B\dA|B\d{1,2}', fname)[0].replace('B', '')

        # try converting to an integer:
        # raises a ValueError for Sentinel2 8A band
        try:
            band = int(band)
        except ValueError:
            pass

        return band

    # _store_bands() writes the paths to the data of each scene to a dictionary
    # only the bands of interest are stored
    def store_bands(self, bands, gt):

        # store the bands of interest in a dictionary
        scene_data = {}
        for i, b in enumerate(bands):
            band = self.bands[self.get_band_number(b)]
            if band in self.use_bands:
                scene_data[band] = b

        # store ground truth
        scene_data['gt'] = gt

        return scene_data

    # compose_scenes() creates a list of dictionaries containing the paths
    # to the tif files of each scene
    # if the scenes are divided into tiles, each tile has its own entry
    # with corresponding tile id
    def compose_scenes(self):

        # list of all samples in the dataset
        scenes = []
        for scene in os.listdir(self.root):

            # get the date of the current scene
            date = self.date_parser(scene)

            # list the spectral bands of the scene
            bands = glob.glob(os.path.join(self.root, scene, '*B*.tif'))

            # get the ground truth mask
            try:
                gt = glob.glob(
                    os.path.join(self.root, scene, '*mask.png')).pop()
            except IndexError:
                gt = None

            # iterate over the tiles
            for tile in range(self.tiles):

                # store the bands and the ground truth mask of the tile
                data = self.store_bands(bands, gt)

                # store tile number
                data['tile'] = tile

                # store date
                data['date'] = date

                # append to list
                scenes.append(data)

        # sort list of scenes in chronological order
        if self.sort:
            scenes.sort(key=lambda k: k['date'])

        return scenes

# SparcsDataset class: inherits from the generic ImageDataset class
class SparcsDataset(StandardEoDataset):

    def __init__(self, root_dir, use_bands=['red', 'green', 'blue'],
                 tile_size=None, sort=False):
        # initialize super class ImageDataset
        super().__init__(root_dir, use_bands, tile_size, sort)

        # function that parses the date from a Landsat 8 scene id
        self.date_parser = parse_landsat8_date

        # list of all scenes in the root directory
        # each scene is divided into tiles blocks
        self.scenes = self.compose_scenes()

    # image size of the Sparcs dataset: (height, width)
    def get_size(self):
        return (1000, 1000)

    # Landsat 8 bands of the Sparcs dataset
    def get_bands(self):
        return {band.value: band.name for band in Landsat8}

    # class labels of the Sparcs dataset
    def get_labels(self):
        return {band.value[0]: {'label': band.name.replace('_', ' '),
                                'color': band.value[1]}
                for band in SparcsLabels}

    # preprocessing of the Sparcs dataset
    def preprocess(self, data, gt):

        # if the preprocessing is not done externally, implement it here

        # convert to torch tensors
        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(gt, dtype=torch.uint8) if gt is not None else gt
        return x, y


class ProSnowDataset(StandardEoDataset):

    def __init__(self, root_dir, use_bands, tile_size, sort):
        super().__init__(root_dir, use_bands, tile_size, sort)

        # function that parses the date from a Sentinel 2 scene id
        self.date_parser = parse_sentinel2_date

        # list of samples in the dataset
        self.scenes = self.compose_scenes()

    # Sentinel 2 bands
    def get_bands(self):
        return {band.value: band.name for band in Sentinel2}

    # class labels of the ProSnow dataset
    def get_labels(self):
        return {band.value[0]: {'label': band.name, 'color': band.value[1]}
               for band in ProSnowLabels}

    # preprocessing of the ProSnow dataset
    def preprocess(self, data, gt):

        # if the preprocessing is not done externally, implement it here

        # convert to torch tensors
        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(gt, dtype=torch.uint8) if gt is not None else gt
        return x, y


class ProSnowGarmisch(ProSnowDataset):

    def __init__(self, root_dir, use_bands=[], tile_size=None, sort=True):
        super().__init__(root_dir, use_bands, tile_size, sort)

    def get_size(self):
        return (615, 543)


class ProSnowObergurgl(ProSnowDataset):

    def __init__(self, root_dir, use_bands=[], tile_size=None, sort=True):
        super().__init__(root_dir, use_bands, tile_size, sort)

    def get_size(self):
        return (310, 270)


class Cloud95Dataset(ImageDataset):

    def __init__(self, root_dir, use_bands=[], tile_size=None, exclude=None):

        # initialize super class ImageDataset
        super().__init__(root_dir, use_bands, tile_size)

        # whether to exclude patches with more than 80% black pixels, i.e.
        # patches resulting from the black margins around a Landsat 8 scene
        self.exclude = exclude

        # list of all scenes in the root directory
        # each scene is divided into tiles blocks
        self.scenes = self.compose_scenes()

    # image size of the Cloud-95 dataset: (height, width)
    def get_size(self):
        return (384, 384)

    # Landsat 8 bands in the Cloud-95 dataset
    def get_bands(self):
        return {band.value: band.name for band in Landsat8}

    # class labels of the Cloud-95 dataset
    def get_labels(self):
        return {band.value[0]: {'label': band.name, 'color': band.value[1]}
                for band in Cloud95Labels}

    # preprocess Cloud-95 dataset
    def preprocess(self, data, gt):

        # normalize the data
        # here, we use the normalization of the authors of Cloud-95, i.e.
        # Mohajerani and Saeedi (2019, 2020)
        x = torch.tensor(data / 65535, dtype=torch.float32)
        y = torch.tensor(gt / 255, dtype=torch.uint8)

        return x, y

    def compose_scenes(self):

        # whether to exclude patches with more than 80% black pixels
        ipatches = []
        if self.exclude is not None:
            with open(os.path.join(self.root, self.exclude), newline='') as f:
                reader = csv.reader(f)
                # list of informative patches
                ipatches = list(itertools.chain.from_iterable(reader))

        # get the names of the directories containing the tif files of
        # the bands of interest
        band_dirs = {}
        for dirpath, dirname, files in os.walk(self.root):
            # check if the current directory path includes the name of a band
            # or the name of the ground truth mask
            cband = [band for band in self.use_bands + ['gt'] if
                     dirpath.endswith(band) and os.path.isdir(dirpath)]

            # add path to current band files to dictionary
            if cband:
                band_dirs[cband.pop()] = dirpath

        # create empty list to store all patches to
        scenes = []

        # iterate over all the patches of the following band
        biter = [*band_dirs.keys()][0]
        for file in os.listdir(band_dirs[biter]):

            # get name of the current patch
            patchname = file.split('.')[0].replace(biter + '_', '')

            # check whether the current file is an informative patch
            if ipatches and patchname not in ipatches:
                continue

            # iterate over the tiles
            for tile in range(self.tiles):

                # initialize dictionary to store bands of current patch
                scene = {}

                # iterate over the bands of interest
                for band in band_dirs.keys():
                    # save path to current band tif file to dictionary
                    scene[band] = os.path.join(band_dirs[band],
                                               file.replace(biter, band))

                # store tile number
                scene['tile'] = tile

                # append patch to list of all patches
                scenes.append(scene)

        return scenes


if __name__ == '__main__':

    # define path to working directory
    # wd = '//projectdata.eurac.edu/projects/cci_snow/dfrisinghelli/'
    wd = '/mnt/CEPH_PROJECTS/cci_snow/dfrisinghelli'
    # wd = 'C:/Eurac/2020/'

    # path to the preprocessed sparcs dataset
    sparcs_path = os.path.join(wd, '_Datasets/Sparcs')

    # path to the Cloud-95 dataset
    cloud_path = os.path.join(wd, '_Datasets/Cloud95/Training')

    # path to the ProSnow dataset
    prosnow_path = os.path.join(wd, '_Datasets/ProSnow/')

    # the csv file containing the names of the informative patches
    # patches = 'training_patches_95-cloud_nonempty.csv'

    # instanciate the Cloud-95 dataset
    # cloud_dataset = Cloud95Dataset(cloud_path,
    #                                tile_size=192,
    #                                exclude=patches)

    # instanciate the SparcsDataset class
    sparcs_dataset = SparcsDataset(sparcs_path,
                                   tile_size=None,
                                   use_bands=['nir', 'red', 'green'],
                                   sort=False)

    # instanciate the ProSnow datasets
    garmisch = ProSnowGarmisch(os.path.join(prosnow_path, 'Garmisch'),
                               tile_size=None,
                               use_bands=['nir', 'red', 'green'],
                               sort=True)
    obergurgl = ProSnowObergurgl(os.path.join(prosnow_path, 'Obergurgl'),
                                 tile_size=None,
                                 use_bands=['nir', 'red', 'green'],
                                 sort=True)
