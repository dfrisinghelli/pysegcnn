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
import csv
import glob
import enum
import itertools

# externals
import numpy as np
import torch
from torch.utils.data import Dataset

# locals
from pysegcnn.core.constants import (Landsat8, Sentinel2, Label, SparcsLabels,
                                     Cloud95Labels, ProSnowLabels)
from pysegcnn.core.utils import (img2np, is_divisible, tile_topleft_corner,
                                 parse_landsat_scene, parse_sentinel2_scene)


# generic image dataset class
class ImageDataset(Dataset):

    # allowed keyword arguments and default values
    default_kwargs = {

        # which bands to use, if use_bands=[], use all available bands
        'use_bands': [],

        # each scene is divided into (tile_size x tile_size) blocks
        # each of these blocks is treated as a single sample
        'tile_size': None,

        # a pattern to match the ground truth file naming convention
        'gt_pattern': '*gt.tif',

        # whether to chronologically sort the samples
        'sort': False,

        # the transformations to apply to the original image
        # artificially increases the training data size
        'transforms': [],

        # whether to pad the image to be evenly divisible in square tiles
        # of size (tile_size x tile_size)
        'pad': False,

        # the value to pad the samples
        'cval': 0,

        }

    def __init__(self, root_dir, **kwargs):
        super().__init__()

        # the root directory: path to the image dataset
        self.root = root_dir

        # the size of a scene/patch in the dataset
        self.size = self.get_size()
        self._assert_get_size()

        # the available spectral bands in the dataset
        self.sensor = self.get_sensor()
        self._assert_get_sensor()
        self.bands = {band.value: band.name for band in self.sensor}

        # the class labels
        self._label_class = self.get_labels()
        self._assert_get_labels()
        self.labels = self._build_labels()

        # initialize keyword arguments
        self._init_kwargs(**kwargs)

        # the samples of the dataset
        self.scenes = self.compose_scenes()
        self._assert_compose_scenes()

    def _init_kwargs(self, **kwargs):

        # check if the keyword arguments are correctly specified
        if not set(self.default_kwargs.keys()).issubset(kwargs.keys()):
            raise TypeError('Valid keyword arguments are: \n' +
                            '\n'.join('- {}'.format(k) for k in
                                      self.default_kwargs.keys()))

        # update default arguments with specified keyword argument values
        self.default_kwargs.update(kwargs)
        for k, v in self.default_kwargs.items():
            setattr(self, k, v)

        # check which bands to use
        self.use_bands = (self.use_bands if self.use_bands else
                          [*self.bands.values()])

        # calculate number of resulting tiles and check whether the images are
        # evenly divisible in square tiles of size (tile_size x tile_size)
        self.tiles, self.padding = 1, (0, 0, 0, 0)
        if self.tile_size is not None:
            self.tiles, self.padding = is_divisible(self.size, self.tile_size,
                                                    self.pad)

        # the size of the padded scenes
        self.height = self.size[0] + self.padding[0] + self.padding[2]
        self.width = self.size[1] + self.padding[1] + self.padding[3]

        # the topleft corners of the tiles
        if self.tile_size is not None:
            self.topleft = tile_topleft_corner((self.height, self.width),
                                               self.tile_size)

        # always use the original dataset together with the augmentations
        self.transforms = [None] + self.transforms

        # check if the padding value is equal to any of the class
        # identifiers in the ground truth mask
        if self.pad and sum(self.padding) > 0:
            if self.cval in self.labels.keys():
                raise ValueError('Constant padding value cval={} is not '
                                 'allowed: class "{}" is represented as {} in '
                                 'the ground truth.'
                                 .format(self.cval,
                                         self.labels[self.cval]['label'],
                                         self.cval)
                                 )
            # add the "no data" label to the class labels of the ground truth
            else:
                if not 0 <= self.cval <= 255:
                    raise ValueError('Expecting 0 <= cval <= 255, got cval={}.'
                                     .format(self.cval))
                print('Adding label "No data" with value={} to ground truth.'
                      .format(self.cval))
                self.labels[self.cval] = {'label': 'No data', 'color': 'black'}

    def _build_labels(self):
        return {band.id: {'label': band.name.replace('_', ' '),
                          'color': band.color}
                for band in self._label_class}

    def _assert_compose_scenes(self):

        # list of required keys
        self.keys = self.use_bands + ['gt', 'date', 'tile', 'transform', 'id']

        # check if each scene is correctly composed
        for scene in self.scenes:
            # check the type of each scene
            if not isinstance(scene, dict):
                raise TypeError('{}.compose_scenes() should return a list of '
                                'dictionaries.'.format(self.__class__.__name__)
                                )

            # check if each scene dictionary has the correct keys
            if not all([k in scene for k in self.keys]):
                raise KeyError('Each scene dictionary should have keys {}.'
                               .format(self.keys))

    def _assert_get_size(self):
        if not isinstance(self.size, tuple) and len(self.size) == 2:
            raise TypeError('{}.get_size() should return the spatial size of '
                            'an image sample as tuple, i.e. (height, width).'
                            .format(self.__class__.__name__))

    def _assert_get_sensor(self):
        if not isinstance(self.sensor, enum.EnumMeta):
            raise TypeError('{}.get_sensor() should return an instance of '
                            'enum.Enum, containing an enumeration of the '
                            'spectral bands of the sensor the dataset is '
                            'derived from. Examples can be found in '
                            'pysegcnn.core.constants.py.'
                            .format(self.__class__.__name__))

    def _assert_get_labels(self):
        if not issubclass(self._label_class, Label):
            raise TypeError('{}.get_labels() should return an instance of '
                            'pysegcnn.core.constants.Label, '
                            'containing an enumeration of the '
                            'class labels, together with the corresponing id '
                            'in the ground truth mask and a color for '
                            'visualization. Examples can be found in '
                            'pysegcnn.core.constants.py.'
                            .format(self.__class__.__name__))

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

        # get samples
        # data: (tiles, bands, height, width)
        # gt: (height, width)
        data, gt = self.build_samples(scene)

        # preprocess samples
        x, y = self.preprocess(data, gt)

        # optional transformation
        if scene['transform'] is not None:
            x, y = scene['transform'](x, y)

        # convert to torch tensors
        x, y = self.to_tensor(x, y)

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
    #   - ...
    def compose_scenes(self, *args, **kwargs):
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    # the get_size() method has to be implemented by the class inheriting
    # the ImageDataset class
    # get_size() method should return the image size as tuple, (height, width)
    def get_size(self, *args, **kwargs):
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    # the get_sensor() method has to be implemented by the class inheriting
    # the ImageDataset class
    # get_sensor() should return an enum.Enum with the following
    # (name: str, value: int) tuples:
    #    - (red, 2)
    #    - (green, 3)
    #    - ...
    def get_sensor(self, *args, **kwargs):
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

    # the parse_scene_id() method has to be implemented by the class inheriting
    # the ImageDataset class
    # the input to the parse_scene_id() method is a string describing a scene
    # id, e.g. an id of a Landsat or a Sentinel scene
    # parse_scene_id() should return a dictionary containing the scene metadata
    def parse_scene_id(self, scene):
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    # _read_scene() reads all the bands and the ground truth mask in a
    # scene/tile to a numpy array and returns a dictionary with
    # (key, value) = ('band_name', np.ndarray(band_data))
    def read_scene(self, idx):

        # select a scene from the root directory
        scene = self.scenes[idx]

        # read each band of the scene into a numpy array
        scene_data = {}
        for key, value in scene.items():

            # pad zeros to each band if self.pad=True, but pad self.cval to
            # the ground truth
            npad = 1 if key == 'gt' else 0
            if isinstance(value, str) and key != 'id':
                scene_data[key] = img2np(value, self.tile_size, scene['tile'],
                                         self.pad, self.cval * npad)
            else:
                scene_data[key] = value

        return scene_data

    # _build_samples() stacks all bands of a scene/tile into a
    # numpy array of shape (bands x height x width)
    def build_samples(self, scene):

        # iterate over the channels to stack
        stack = np.stack([scene[band] for band in self.use_bands], axis=0)
        gt = scene['gt']

        return stack, gt

    def to_tensor(self, x, y):
        return (torch.tensor(x.copy(), dtype=torch.float32),
                torch.tensor(y.copy(), dtype=torch.uint8))

    def __repr__(self):

        # representation string to print
        fs = self.__class__.__name__ + '(\n'

        # sensor
        fs += '    (sensor):\n        - ' + self.sensor.__name__

        # bands used for the segmentation
        fs += '\n    (bands):\n        '
        fs += '\n        '.join('- Band {}: {}'.format(i, b) for i, b in
                                enumerate(self.use_bands))

        # scenes
        fs += '\n    (scene):\n        '
        fs += '- size (h, w): {}\n        '.format((self.height, self.width))
        fs += '- number of scenes: {}\n        '.format(
            len(np.unique([f['id'] for f in self.scenes])))
        fs += '- padding (bottom, left, top, right): {}'.format(self.padding)

        # tiles
        fs += '\n    (tiles):\n        '
        fs += '- number of tiles per scene: {}\n        '.format(self.tiles)
        fs += '- tile size: {}\n        '.format((self.tile_size,
                                                  self.tile_size))
        fs += '- number of tiles: {}'.format(len(self.scenes))

        # classes of interest
        fs += '\n    (classes):\n        '
        fs += '\n        '.join('- Class {}: {}'.format(k, v['label']) for
                                k, v in self.labels.items())
        fs += '\n)'
        return fs



class StandardEoDataset(ImageDataset):

    def __init__(self, root_dir, **kwargs):
        # initialize super class ImageDataset
        super().__init__(root_dir, **kwargs)

    # returns the band number of a Landsat8 or Sentinel2 tif file
    # path: path to a tif file
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

    # store_bands() writes the paths to the data of each scene to a dictionary
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

        # search the root directory
        scenes = []
        self.gt = []
        for dirpath, dirname, files in os.walk(self.root):

            # search for a ground truth in the current directory
            self.gt.extend(glob.glob(os.path.join(dirpath, self.gt_pattern)))

            # check if the current directory name matches a scene identifier
            scene = self.parse_scene_id(dirpath)

            if scene is not None:

                # get the date of the current scene
                date = scene['date']

                # list the spectral bands of the scene
                bands = glob.glob(os.path.join(dirpath, '*B*.tif'))

                # get the ground truth mask
                try:
                    gt = [truth for truth in self.gt if scene['id'] in truth]
                    gt = gt.pop()

                except IndexError:
                    print('Skipping scene {}: ground truth not available '
                          '(pattern = {}).'
                          .format(scene['id'], self.gt_pattern))
                    continue

                # iterate over the tiles
                for tile in range(self.tiles):

                    # iterate over the transformations to apply
                    for transf in self.transforms:

                        # store the bands and the ground truth mask of the tile
                        data = self.store_bands(bands, gt)

                        # the name of the scene
                        data['id'] = scene['id']

                        # store tile number
                        data['tile'] = tile

                        # store date
                        data['date'] = date

                        # store optional transformation
                        data['transform'] = transf

                        # append to list
                        scenes.append(data)

        # sort list of scenes and ground truths in chronological order
        if self.sort:
            scenes.sort(key=lambda k: k['date'])
            self.gt.sort(key=lambda k: self.parse_scene_id(k)['date'])

        return scenes


# SparcsDataset class: inherits from the generic ImageDataset class
class SparcsDataset(StandardEoDataset):

    def __init__(self, root_dir, **kwargs):
        # initialize super class StandardEoDataset
        super().__init__(root_dir, **kwargs)

    # image size of the Sparcs dataset: (height, width)
    def get_size(self):
        return (1000, 1000)

    # Landsat 8 bands of the Sparcs dataset
    def get_sensor(self):
        return Landsat8

    # class labels of the Sparcs dataset
    def get_labels(self):
        return SparcsLabels

    # preprocessing of the Sparcs dataset
    def preprocess(self, data, gt):

        # if the preprocessing is not done externally, implement it here
        return data, gt

    # function that parses the date from a Landsat 8 scene id
    def parse_scene_id(self, scene):
        return parse_landsat_scene(scene)


class ProSnowDataset(StandardEoDataset):

    def __init__(self, root_dir, **kwargs):
        # initialize super class StandardEoDataset
        super().__init__(root_dir, **kwargs)

    # Sentinel 2 bands
    def get_sensor(self):
        return Sentinel2

    # class labels of the ProSnow dataset
    def get_labels(self):
        return ProSnowLabels

    # preprocessing of the ProSnow dataset
    def preprocess(self, data, gt):

        # if the preprocessing is not done externally, implement it here
        return data, gt

    # function that parses the date from a Sentinel 2 scene id
    def parse_scene_id(self, scene):
        return parse_sentinel2_scene(scene)


class ProSnowGarmisch(ProSnowDataset):

    def __init__(self, root_dir, **kwargs):
        # initialize super class StandardEoDatasets
        super().__init__(root_dir, **kwargs)

    def get_size(self):
        return (615, 543)


class ProSnowObergurgl(ProSnowDataset):

    def __init__(self, root_dir, **kwargs):
        # initialize super class StandardEoDataset
        super().__init__(root_dir, **kwargs)

    def get_size(self):
        return (310, 270)


class Cloud95Dataset(ImageDataset):

    def __init__(self, root_dir, **kwargs):

        # the csv file containing the names of the informative patches
        # patches with more than 80% black pixels, i.e. patches resulting from
        # the black margins around a Landsat 8 scene are excluded
        self.exclude = 'training_patches_95-cloud_nonempty.csv'

        # initialize super class ImageDataset
        super().__init__(root_dir, **kwargs)

    # image size of the Cloud-95 dataset: (height, width)
    def get_size(self):
        return (384, 384)

    # Landsat 8 bands in the Cloud-95 dataset
    def get_sensor(self):
        return Landsat8

    # class labels of the Cloud-95 dataset
    def get_labels(self):
        return Cloud95Labels

    # preprocess Cloud-95 dataset
    def preprocess(self, data, gt):

        # normalize the data
        # here, we use the normalization of the authors of Cloud-95, i.e.
        # Mohajerani and Saeedi (2019, 2020)
        data /= 65535
        gt[gt != self.cval] /= 255

        return data, gt

    # function that parses the date from a Landsat 8 scene id
    def parse_scene_id(self, scene):
        return parse_landsat_scene(scene)

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

            # get the date of the current scene
            date = self.parse_scene_id(patchname)['date']

            # check whether the current file is an informative patch
            if ipatches and patchname not in ipatches:
                continue

            # iterate over the tiles
            for tile in range(self.tiles):

                # iterate over the transformations to apply
                for transf in self.transforms:

                    # initialize dictionary to store bands of current patch
                    scene = {}

                    # iterate over the bands of interest
                    for band in band_dirs.keys():
                        # save path to current band tif file to dictionary
                        scene[band] = os.path.join(band_dirs[band],
                                                   file.replace(biter, band))

                    # store tile number
                    scene['tile'] = tile

                    # store date
                    scene['date'] = date

                    # store optional transformation
                    scene['transform'] = transf

                    # append patch to list of all patches
                    scenes.append(scene)

        # sort list of scenes in chronological order
        if self.sort:
            scenes.sort(key=lambda k: k['date'])

        return scenes


class SupportedDatasets(enum.Enum):
    Sparcs = SparcsDataset
    Cloud95 = Cloud95Dataset
    Garmisch = ProSnowGarmisch
    Obergurgl = ProSnowObergurgl


if __name__ == '__main__':

    # define path to working directory
    # wd = '//projectdata.eurac.edu/projects/cci_snow/dfrisinghelli/'
    # wd = '/mnt/CEPH_PROJECTS/cci_snow/dfrisinghelli'
    wd = 'C:/Eurac/2020/'

    # path to the preprocessed sparcs dataset
    sparcs_path = os.path.join(wd, '_Datasets/Sparcs')

    # path to the Cloud-95 dataset
    # cloud_path = os.path.join(wd, '_Datasets/Cloud95/Training')

    # path to the ProSnow dataset
    # prosnow_path = os.path.join(wd, '_Datasets/ProSnow/')

    # instanciate the Cloud-95 dataset
    # cloud_dataset = Cloud95Dataset(cloud_path,
    #                                tile_size=192,
    #                                use_bands=[],
    #                                sort=False)

    # instanciate the SparcsDataset class
    sparcs_dataset = SparcsDataset(sparcs_path,
                                   tile_size=None,
                                   use_bands=['red', 'green', 'blue', 'nir'],
                                   sort=False,
                                   transforms=[],
                                   gt_pattern='*mask.png',
                                   pad=True,
                                   cval=99)

    # instanciate the ProSnow datasets
    # garmisch = ProSnowGarmisch(os.path.join(prosnow_path, 'Garmisch'),
    #                            tile_size=None,
    #                            use_bands=['nir', 'red', 'green', 'blue'],
    #                            sort=True,
    #                            transforms=[],
    #                            gt_pattern='*class.img')
