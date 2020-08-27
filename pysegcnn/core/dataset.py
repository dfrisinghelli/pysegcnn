"""Custom dataset classes compliant to the PyTorch standard.

Each custom dataset should inherit from torch.utils.data.Dataset to benefit
from the torch.utils.data.DataLoader class, which implements helpful utilities
during model training.

For any kind of image-like dataset, inherit the ImageDataset class to create
your custom dataset.

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
import csv
import enum
import itertools
import logging

# externals
import numpy as np
import torch
from torch.utils.data import Dataset

# locals
from pysegcnn.core.constants import (Landsat8, Sentinel2, Label, SparcsLabels,
                                     Cloud95Labels, ProSnowLabels)
from pysegcnn.core.utils import (img2np, is_divisible, tile_topleft_corner,
                                 parse_landsat_scene, parse_sentinel2_scene)

# module level logger
LOGGER = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Base class for multispectral image data.

    Inheriting from :py:class:`torch.utils.data.Dataset` to be compliant to the
    PyTorch standard. This enables the use of the handy
    :py:class:`torch.utils.data.DataLoader` class during model training.

    Attributes
    ----------
    root_dir : `str`
        The root directory, path to the dataset.
    use_bands : `list` [`str`]
        List of the spectral bands to use during model training.
    tile_size : `int` or `None`
        The size of the tiles.
    pad : `bool`
        Whether to center pad the input image.
    gt_pattern : `str`
        A regural expression to match the ground truth naming convention.
    sort : `bool`
        Whether to chronologically sort the samples.
    seed : `int`
        The random seed.
    transforms : `list`
        List of :py:class:`pysegcnn.core.transforms.Augment` instances.
    size : `tuple` [`int`]
        The size of an image of the dataset.
    sensor : :py:class:`enum.Enum`
        An enumeration of the bands of sensor the dataset is derived from,
        see e.g. :py:class:`pysegcnn.core.constants.Landsat8`.
    bands : `dict` [`int`, `str`]
        The spectral bands of ``sensor``. The keys are the number and the
        values are the name of the spectral bands.
    labels : `dict` [`int`, `dict`]
        The label dictionary. The keys are the values of the class labels
        in the ground truth. Each nested `dict` has keys:
            ``'color'``
                A named color (`str`).
            ``'label'``
                The name of the class label (`str`).
    tiles : `int`
        Number of tiles with size ``(tile_size, tile_size)`` within an image.
    padding : `tuple` [`int`]
        The amount of padding, (bottom, left, top, right).
    height : `int`
        The height of a padded image.
    width : `int`
        The width of a padded image.
    topleft : `dict` [`int`, `tuple`]
        The topleft corners of the tiles. The keys of are the tile ids (`int`)
        and the values are the topleft corners (y, x) of the tiles.
    cval : `int`
        When padding, ``cval`` is the value of the "no data" label in the
        ground truth. Otherwise, ``cval=0``.
    gt : `list` [`str` or :py:class:`pathlib.Path`]
        List of the ground truth images.
    keys : `list`
        List of required keys for each dictionary in ``scenes``.
    scenes : `list` [`dict`]
        List of dictionaries representing the samples of the dataset.

    """

    def __init__(self, root_dir, use_bands=[], tile_size=None, pad=False,
                 gt_pattern='(.*)gt\\.tif', sort=False, seed=0, transforms=[]):
        r"""Initialize.

        Parameters
        ----------
        root_dir : `str`
            The root directory, path to the dataset.
        use_bands : `list` [`str`], optional
            A list of the spectral bands to use. The default is `[]`.
        tile_size : `int` or `None`, optional
            The size of the tiles. If not `None`, each scene is divided into
            square tiles of shape ``(tile_size, tile_size)``. The default is
            `None`.
        pad : `bool`, optional
            Whether to center pad the input image. Set ``pad=True``, if the
            images are not evenly divisible by the ``tile_size``. The image
            data is padded with a constant padding value of zero. For each
            image, the corresponding ground truth image is padded with a
            "no data" label. The default is `False`.
        gt_pattern : `str`, optional
            A regural expression to match the ground truth naming convention.
            All directories and subdirectories in ``root_dir`` are searched for
            files matching ``gt_pattern``. The default is `(.*)gt\\.tif`.
        sort : `bool`, optional
            Whether to chronologically sort the samples. Useful for time series
            data. The default is `False`.
        seed : `int`, optional
            The random seed. Used to split the dataset into training,
            validation and test set. Useful for reproducibility. The default is
            `0`.
        transforms : `list`, optional
            List of :py:class:`pysegcnn.core.transforms.Augment` instances.
            Each item in ``transforms`` generates a distinct transformed
            version of the dataset. The total dataset is composed of the
            original untransformed dataset together with each transformed
            version of it. If ``transforms=[]``, only the original dataset is
            used. The default is `[]`.

        """
        super().__init__()

        # dataset configuration
        self.root = root_dir
        self.use_bands = use_bands
        self.tile_size = tile_size
        self.pad = pad
        self.gt_pattern = gt_pattern
        self.sort = sort
        self.seed = seed
        self.transforms = transforms

        # initialize instance attributes
        self._init_attributes()

        # the samples of the dataset
        self.scenes = self.compose_scenes()
        self._assert_compose_scenes()

    def _init_attributes(self):
        """Initialize the class instance attributes."""
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

        # when padding, add a new "no data" label to the ground truth
        self.cval = 0
        if self.pad and sum(self.padding) > 0:
            self.cval = max(self.labels) + 1
            self.labels[self.cval] = {'label': 'No data', 'color': 'black'}
            LOGGER.info('Adding label "No data" with value={} to ground truth.'
                        .format(self.cval))

        # list of ground truth images
        self.gt = []

    def _build_labels(self):
        """Build the label dictionary.

        Returns
        -------
        labels : `dict` [`int`, `dict`]
            The label dictionary. The keys are the values of the class labels
            in the ground truth. Each nested `dict` should have keys:
                ``'color'``
                    A named color (`str`).
                ``'label'``
                    The name of the class label (`str`).

        """
        return {band.id: {'label': band.name.replace('_', ' '),
                          'color': band.color}
                for band in self._label_class}

    def _assert_compose_scenes(self):
        """Check whether compose_scenes() is correctly implemented."""
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
        """Check whether get_size() is correctly implemented."""
        if not isinstance(self.size, tuple) and len(self.size) == 2:
            raise TypeError('{}.get_size() should return the spatial size of '
                            'an image sample as tuple, i.e. (height, width).'
                            .format(self.__class__.__name__))

    def _assert_get_sensor(self):
        """Check whether get_sensor() is correctly implemented."""
        if not isinstance(self.sensor, enum.EnumMeta):
            raise TypeError('{}.get_sensor() should return an instance of '
                            'enum.Enum, containing an enumeration of the '
                            'spectral bands of the sensor the dataset is '
                            'derived from. Examples can be found in '
                            'pysegcnn.core.constants.py.'
                            .format(self.__class__.__name__))

    def _assert_get_labels(self):
        """Check whether get_labels() is correctly implemented."""
        if not issubclass(self._label_class, Label):
            raise TypeError('{}.get_labels() should return an instance of '
                            'pysegcnn.core.constants.Label, '
                            'containing an enumeration of the '
                            'class labels, together with the corresponing id '
                            'in the ground truth mask and a color for '
                            'visualization. Examples can be found in '
                            'pysegcnn.core.constants.py.'
                            .format(self.__class__.__name__))

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns
        -------
        nsamples : `int`
            The number of samples in the dataset.
        """
        return len(self.scenes)

    def __getitem__(self, idx):
        """Return the data of a sample of the dataset given an index ``idx``.

        Parameters
        ----------
        idx : `int`
            The index of the sample.

        Returns
        -------
        x : `torch.Tensor`
            The sample input data.
        y : `torch.Tensor`
            The sample ground truth.

        """
        # select a scene
        scene = self.read_scene(idx)

        # get samples
        # data: (tiles, bands, height, width)
        # gt: (height, width)
        data, gt = self.build_samples(scene)

        # preprocess samples
        x, y = self.preprocess(data, gt)

        # apply transformation
        if scene['transform'] is not None:
            x, y = scene['transform'](x, y)

        # convert to torch tensors
        x = self.to_tensor(x, dtype=torch.float32)
        y = self.to_tensor(y, dtype=torch.uint8)

        return x, y

    def compose_scenes(self):
        """Build the list of samples of the dataset.

        Each sample is represented by a dictionary.

        Raises
        ------
        NotImplementedError
            Raised if the `pysegcnn.core.dataset.ImageDataset` class is not
            inherited.

        Returns
        -------
        samples : `list` [`dict`]
            Each dictionary representing a sample should have keys:
                ``'band_name_1'``
                    Path to the file of band_1.
                ``'band_name_2'``
                    Path to the file of band_2.
                ``'band_name_n'``
                    Path to the file of band_n.
                ``'gt'``
                    Path to the ground truth file.
                ``'date'``
                    The date of the sample.
                ``'tile'``
                    The tile id of the sample.
                ``'transform'``
                    The transformation to apply.
                ``'id'``
                    The scene identifier.

        """
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    def get_size(self):
        """Return the size of the images in the dataset.

        Raises
        ------
        NotImplementedError
            Raised if the `pysegcnn.core.dataset.ImageDataset` class is not
            inherited.

        Returns
        -------
        size : `tuple`
            The image size (height, width).

        """
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    def get_sensor(self):
        """Return an enumeration of the bands of the sensor of the dataset.

        Examples can be found in `pysegcnn.core.constants`.

        Raises
        ------
        NotImplementedError
            Raised if the `pysegcnn.core.dataset.ImageDataset` class is not
            inherited.

        Returns
        -------
        sensor : `enum.Enum`
            An enumeration of the bands of the sensor.

        """
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    def get_labels(self):
        """Return an enumeration of the class labels of the dataset.

        Examples can be found in `pysegcnn.core.constants`.

        Raises
        ------
        NotImplementedError
            Raised if the `pysegcnn.core.dataset.ImageDataset` class is not
            inherited.

        Returns
        -------
        labels : `enum.Enum`
            The class labels.

        """
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    def preprocess(self, data, gt):
        """Preprocess a sample before feeding it to a model.

        Parameters
        ----------
        data : `numpy.ndarray`
            The sample input data.
        gt : `numpy.ndarray`
            The sample ground truth.

        Raises
        ------
        NotImplementedError
            Raised if the `pysegcnn.core.dataset.ImageDataset` class is not
            inherited.

        Returns
        -------
        data : `numpy.ndarray`
            The preprocessed input data.
        gt : `numpy.ndarray`
            The preprocessed ground truth data.

        """
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    def parse_scene_id(self, scene_id):
        """Parse the scene identifier.

        Parameters
        ----------
        scene_id : `str`
            A scene identifier.

        Raises
        ------
        NotImplementedError
            Raised if the `pysegcnn.core.dataset.ImageDataset` class is not
            inherited.

        Returns
        -------
        scene : `dict` or `None`
            A dictionary containing scene metadata. If `None`, ``scene_id`` is
            not a valid scene identifier.

        """
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    def read_scene(self, idx):
        """Read the data of the sample with index ``idx``.

        Parameters
        ----------
        idx : `int`
            The index of the sample.

        Returns
        -------
        scene_data : `dict`
            The sample data dictionary with keys:
                ``'band_name_1'``
                    data of band_1 (`numpy.ndarray`).
                ``'band_name_2'``
                    data of band_2 (`numpy.ndarray`).
                ``'band_name_n'``
                    data of band_n (`numpy.ndarray`).
                ``'gt'``
                    data of the ground truth (`numpy.ndarray`).
                ``'date'``
                    The date of the sample.
                ``'tile'``
                    The tile id of the sample.
                ``'transform'``
                    The transformation to apply.
                ``'id'``
                    The scene identifier.

        """
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

    def build_samples(self, scene):
        """Stack the bands of a sample in a single array.

        Parameters
        ----------
        scene : `dict`
            The sample data dictionary with keys:
                ``'band_name_1'``
                    data of band_1 (`numpy.ndarray`).
                ``'band_name_2'``
                    data of band_2 (`numpy.ndarray`).
                ``'band_name_n'``
                    data of band_n (`numpy.ndarray`).
                ``'gt'``
                    data of the ground truth (`numpy.ndarray`).
                ``'date'``
                    The date of the sample.
                ``'tile'``
                    The tile id of the sample.
                ``'transform'``
                    The transformation to apply.
                ``'id'``
                    The scene identifier.

        Returns
        -------
        stack : `numpy.ndarray`
            The input data of the sample.
        gt : `numpy.ndarray`
            The ground truth of the sample.

        """
        # iterate over the channels to stack
        stack = np.stack([scene[band] for band in self.use_bands], axis=0)
        gt = scene['gt']

        return stack, gt

    def to_tensor(self, x, dtype):
        """Convert ``x`` to :py:class:`torch.Tensor`.

        Parameters
        ----------
        x : array_like
            The input data.
        dtype : `torch.dtype`
            The data type used to convert ``x``.

        Returns
        -------
        x : `torch.Tensor`
            The input data tensor.

        """
        return torch.tensor(np.asarray(x).copy(), dtype=dtype)

    def __repr__(self):
        """Dataset representation.

        Returns
        -------
        fs : `str`
            Representation string.

        """
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
    r"""Base class for standard Earth Observation style datasets.

    :py:class:`pysegcnn.core.dataset.StandardEoDataset` implements the
    :py:meth:`~pysegcnn.core.dataset.StandardEoDataset.compose_scenes` method
    for datasets with the following directory structure:

    root_dir/
        - scene_id_1/
             - scene_id_1_B1.tif
             - scene_id_1_B2.tif
             - ...
             - scene_id_1_BN.tif
        - scene_id_2/
             - scene_id_2_B1.tif
             - scene_id_2_B2.tif
             - ...
             - scene_id_2_BN.tif
        - ...
        - scene_id_N/
            - ...

    If your dataset shares this directory structure, you can directly inherit
    :py:class:`pysegcnn.core.dataset.StandardEoDataset` and implement the
    remaining methods. If not, you can use
    :py:func:`pysegcnn.core.utils.standard_eo_structure` to transfer your
    dataset to the above directory structure.

    See :py:class:`pysegcnn.core.dataset.SparcsDataset` for an example.

    """

    def __init__(self, root_dir, use_bands=[], tile_size=None, pad=False,
                 gt_pattern='(.*)gt\\.tif', sort=False, seed=0, transforms=[]):
        # initialize super class ImageDataset
        super().__init__(root_dir, use_bands, tile_size, pad, gt_pattern,
                         sort, seed, transforms)

    def _get_band_number(self, path):
        """Return the band number of a scene .tif file.

        Parameters
        ----------
        path : `str`
            The path to the .tif file.

        Returns
        -------
        band : `int` or `str`
            The band number.

        """
        # filename
        fname = os.path.basename(path)

        # search for numbers following a "B" in the filename
        band = re.search('B\\dA|B\\d{1,2}', fname)[0].replace('B', '')

        # try converting to an integer:
        # raises a ValueError for Sentinel2 8A band
        try:
            band = int(band)
        except ValueError:
            pass

        return band

    def _store_bands(self, bands, gt):
        """Write the bands of interest to a dictionary.

        Parameters
        ----------
        bands : `list` [`str`]
            Paths to the .tif files of the bands of the scene.
        gt : `str`
            Path to the ground truth of the scene.

        Returns
        -------
        scene_data : `dict`
            The scene data dictionary with keys:
                ``'band_name_1'``
                    Path to the .tif file of band_1.
                ``'band_name_2'``
                    Path to the .tif file of band_2.
                ``'band_name_n'``
                    Path to the .tif file of band_n.
                ``'gt'``
                    Path to the ground truth file.

        """
        # store the bands of interest in a dictionary
        scene_data = {}
        for i, b in enumerate(bands):
            band = self.bands[self._get_band_number(b)]
            if band in self.use_bands:
                scene_data[band] = b

        # store ground truth
        scene_data['gt'] = gt

        return scene_data

    def compose_scenes(self):
        """Build the list of samples of the dataset."""
        # search the root directory
        scenes = []
        for dirpath, dirname, files in os.walk(self.root):

            # search for a ground truth in the current directory
            self.gt.extend([os.path.join(dirpath, f) for f in files
                            if re.search(self.gt_pattern, f)])

            # check if the current directory name matches a scene identifier
            scene = self.parse_scene_id(dirpath)

            if scene is not None:

                # get the date of the current scene
                date = scene['date']

                # list the spectral bands of the scene
                bands = [os.path.join(dirpath, f) for f in files
                         if re.search('B\\dA|B\\d{1,2}(.*).tif$', f)]

                # get the ground truth mask
                try:
                    gt = [truth for truth in self.gt if scene['id'] in truth]
                    gt = gt.pop()

                except IndexError:
                    LOGGER.info('Skipping scene {}: ground truth not available'
                                ' (pattern = {}).'.format(scene['id'],
                                                          self.gt_pattern))
                    continue

                # iterate over the tiles
                for tile in range(self.tiles):

                    # iterate over the transformations to apply
                    for transf in self.transforms:

                        # store the bands and the ground truth mask of the tile
                        data = self._store_bands(bands, gt)

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


class SparcsDataset(StandardEoDataset):
    """Class for the `Sparcs`_ dataset by `Hughes & Hayes (2014)`_.

    .. _Sparcs:
        https://www.usgs.gov/land-resources/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation

    .. _Hughes & Hayes (2014):
        https://www.mdpi.com/2072-4292/6/6/4907

    """

    def __init__(self, root_dir, use_bands=[], tile_size=None, pad=False,
                 gt_pattern='(.*)gt\\.tif', sort=False, seed=0, transforms=[]):
        # initialize super class StandardEoDataset
        super().__init__(root_dir, use_bands, tile_size, pad, gt_pattern,
                         sort, seed, transforms)

    def get_size(self):
        """Image size of the Sparcs dataset.

        Returns
        -------
        size : `tuple`
            The image size (height, width).

        """
        return (1000, 1000)

    def get_sensor(self):
        """Landsat 8 bands of the Sparcs dataset.

        Returns
        -------
        sensor : `enum.Enum`
            An enumeration of the bands of the sensor.

        """
        return Landsat8

    def get_labels(self):
        """Class labels of the Sparcs dataset.

        Returns
        -------
        labels : `enum.Enum`
            The class labels.

        """
        return SparcsLabels

    def preprocess(self, data, gt):
        """Preprocess Sparcs dataset images.

        Parameters
        ----------
        data : `numpy.ndarray`
            The sample input data.
        gt : `numpy.ndarray`
            The sample ground truth.

        Returns
        -------
        data : `numpy.ndarray`
            The preprocessed input data.
        gt : `numpy.ndarray`
            The preprocessed ground truth data.

        """
        # if the preprocessing is not done externally, implement it here
        return data, gt

    def parse_scene_id(self, scene_id):
        """Parse Sparcs scene identifiers (Landsat 8).

        Parameters
        ----------
        scene_id : `str`
            A scene identifier.

        Returns
        -------
        scene : `dict` or `None`
            A dictionary containing scene metadata. If `None`, ``scene_id`` is
            not a valid Landsat scene identifier.

        """
        return parse_landsat_scene(scene_id)


class ProSnowDataset(StandardEoDataset):
    """Class for the ProSnow datasets."""

    def __init__(self, root_dir, use_bands=[], tile_size=None, pad=False,
                 gt_pattern='(.*)gt\\.tif', sort=False, seed=0, transforms=[]):
        # initialize super class StandardEoDataset
        super().__init__(root_dir, use_bands, tile_size, pad, gt_pattern,
                         sort, seed, transforms)

    def get_sensor(self):
        """Sentinel 2 bands of the ProSnow datasets.

        Returns
        -------
        sensor : `enum.Enum`
            An enumeration of the bands of the sensor.

        """
        return Sentinel2

    def get_labels(self):
        """Class labels of the ProSnow datasets.

        Returns
        -------
        labels : `enum.Enum`
            The class labels.

        """
        return ProSnowLabels

    def preprocess(self, data, gt):
        """Preprocess ProSnow dataset images.

        Parameters
        ----------
        data : `numpy.ndarray`
            The sample input data.
        gt : `numpy.ndarray`
            The sample ground truth.

        Returns
        -------
        data : `numpy.ndarray`
            The preprocessed input data.
        gt : `numpy.ndarray`
            The preprocessed ground truth data.

        """
        # if the preprocessing is not done externally, implement it here
        return data, gt

    def parse_scene_id(self, scene_id):
        """Parse ProSnow scene identifiers (Sentinel 2).

        Parameters
        ----------
        scene_id : `str`
            A scene identifier.

        Returns
        -------
        scene : `dict` or `None`
            A dictionary containing scene metadata. If `None`, ``scene_id`` is
            not a valid Sentinel-2 scene identifier.

        """
        return parse_sentinel2_scene(scene_id)


class ProSnowGarmisch(ProSnowDataset):
    """Class for the ProSnow Garmisch dataset."""

    def __init__(self, root_dir, use_bands=[], tile_size=None, pad=False,
                 gt_pattern='(.*)gt\\.tif', sort=False, seed=0, transforms=[]):
        # initialize super class StandardEoDataset
        super().__init__(root_dir, use_bands, tile_size, pad, gt_pattern,
                         sort, seed, transforms)

    def get_size(self):
        """Image size of the ProSnow Garmisch dataset.

        Returns
        -------
        size : `tuple`
            The image size (height, width).

        """
        return (615, 543)


class ProSnowObergurgl(ProSnowDataset):
    """Class for the ProSnow Obergurgl dataset."""

    def __init__(self, root_dir, use_bands=[], tile_size=None, pad=False,
                 gt_pattern='(.*)gt\\.tif', sort=False, seed=0, transforms=[]):
        # initialize super class StandardEoDataset
        super().__init__(root_dir, use_bands, tile_size, pad, gt_pattern,
                         sort, seed, transforms)

    def get_size(self):
        """Image size of the ProSnow Obergurgl dataset.

        Returns
        -------
        size : `tuple`
            The image size (height, width).

        """
        return (310, 270)


class Cloud95Dataset(ImageDataset):
    """Class for the `Cloud-95`_ dataset by `Mohajerani & Saeedi (2020)`_.

    .. _Cloud-95:
        https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset
    .. _Mohajerani & Saeedi (2020):
        https://arxiv.org/abs/2001.08768

    """

    def __init__(self, root_dir, use_bands=[], tile_size=None, pad=False,
                 gt_pattern='(.*)gt\\.tif', sort=False, seed=0, transforms=[]):
        # initialize super class StandardEoDataset
        super().__init__(root_dir, use_bands, tile_size, pad, gt_pattern,
                         sort, seed, transforms)

        # the csv file containing the names of the informative patches
        # patches with more than 80% black pixels, i.e. patches resulting from
        # the black margins around a Landsat 8 scene are excluded
        self.exclude = 'training_patches_95-cloud_nonempty.csv'

    def get_size(self):
        """Image size of the Cloud-95 dataset.

        Returns
        -------
        size : `tuple`
            The image size (height, width).

        """
        return (384, 384)

    def get_sensor(self):
        """Landsat 8 bands of the Cloud-95 dataset.

        Returns
        -------
        sensor : `enum.Enum`
            An enumeration of the bands of the sensor.

        """
        return Landsat8

    def get_labels(self):
        """Class labels of the Cloud-95 dataset.

        Returns
        -------
        labels : `enum.Enum`
            The class labels.

        """
        return Cloud95Labels

    def preprocess(self, data, gt):
        """Preprocess Cloud-95 dataset images.

        Parameters
        ----------
        data : `numpy.ndarray`
            The sample input data.
        gt : `numpy.ndarray`
            The sample ground truth.

        Returns
        -------
        data : `numpy.ndarray`
            The preprocessed input data.
        gt : `numpy.ndarray`
            The preprocessed ground truth data.

        """
        # normalize the data
        # here, we use the normalization of the authors of Cloud-95, i.e.
        # Mohajerani and Saeedi (2019, 2020)
        data /= 65535
        gt[gt != self.cval] /= 255
        return data, gt

    def parse_scene_id(self, scene_id):
        """Parse Sparcs scene identifiers (Landsat 8).

        Parameters
        ----------
        scene_id : `str`
            A scene identifier.

        Returns
        -------
        scene : `dict` or `None`
            A dictionary containing scene metadata. If `None`, ``scene_id`` is
            not a valid Landsat scene identifier.

        """
        return parse_landsat_scene(scene_id)

    def compose_scenes(self):
        """Build the list of samples of the dataset."""
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
            scene_meta = self.parse_scene_id(patchname)

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

                    # the name of the scene the patch was extracted from
                    scene['id'] = scene_meta['id']

                    # store tile number
                    scene['tile'] = tile

                    # store date
                    scene['date'] = scene_meta['date']

                    # store optional transformation
                    scene['transform'] = transf

                    # append patch to list of all patches
                    scenes.append(scene)

        # sort list of scenes in chronological order
        if self.sort:
            scenes.sort(key=lambda k: k['date'])

        return scenes


class SupportedDatasets(enum.Enum):
    """Names and corresponding classes of the implemented datasets."""

    Sparcs = SparcsDataset
    Cloud95 = Cloud95Dataset
    Garmisch = ProSnowGarmisch
    Obergurgl = ProSnowObergurgl
