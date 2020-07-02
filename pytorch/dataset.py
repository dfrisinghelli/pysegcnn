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
import glob

# externals
import gdal
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch.utils.data import Dataset


# generic image dataset class
class ImageDataset(Dataset):

    def __init__(self, root_dir):
        super().__init__()

        # the root directory: path to the image dataset
        self.root = root_dir

    # this function should return the length of the image dataset
    # __len__() is used by pytorch to determine the total number of samples in
    # the dataset, has to be implemented by a class inheriting from the
    # ImageDataset class
    def __len__(self):
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    # this function should return a single sample of the dataset given an
    # index, i.e. an array/tensor of shape (channels x height x width)
    # it has to be implemented by a class inheriting from the
    # ImageDataset class
    def __getitem__(self, idx):
        raise NotImplementedError('Inherit the ImageDataset class and '
                                  'implement the method.')

    # the following functions are utility functions for common image
    # manipulation operations

    # this function reads an image to a numpy array
    def img2np(self, path, tile_size=None, tile=None):

        # open the tif file
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

    # this function applies percentile stretching at the alpha level
    # can be used to increase constrast for visualization
    def contrast_stretching(self, image, alpha=2):

        # compute upper and lower percentiles defining the range of the stretch
        inf, sup = np.percentile(image, (alpha, 100 - alpha))

        # normalize image intensity distribution to
        # (alpha, 100 - alpha) percentiles
        norm = ((image - inf) * (image.max() - image.min()) /
                (sup - inf)) + image.min()

        # clip: values < inf = 0, values > sup = max
        norm[norm <= image.min()] = image.min()
        norm[norm >= image.max()] = image.max()

        return norm


# SparcsDataset class: inherits from the generic ImageDataset class
class SparcsDataset(ImageDataset):

    def __init__(self, root_dir, bands=['red', 'green', 'blue'],
                 tile_size=None):
        super().__init__(root_dir)

        # Landsat 8 bands in the SPARCS dataset
        self.sparcs_bands = {1: 'violet',
                             2: 'blue',
                             3: 'green',
                             4: 'red',
                             5: 'nir',
                             6: 'swir1',
                             7: 'swir2',
                             8: 'pan',
                             9: 'cirrus',
                             10: 'tir'}

        # class labels and corresponding color map
        self.labels = {0: 'Shadow',
                       1: 'Shadow over Water',
                       2: 'Water',
                       3: 'Snow',
                       4: 'Land',
                       5: 'Cloud',
                       6: 'Flooded'}
        self.colors = {0: 'black',
                       1: 'darkblue',
                       2: 'blue',
                       3: 'lightblue',
                       4: 'grey',
                       5: 'white',
                       6: 'yellow'}

        # image size of the SPARCS dataset: height x width
        self.size = (1000, 1000)

        # check which bands to use
        if bands == -1:
            # in case bands=-1, use all bands of the sparcs dataset
            self.bands = [*self.sparcs_bands.values()]
        else:
            self.bands = bands

        # each scene is divided into (tile_size x tile_size) blocks
        # each of these blocks is treated as a single sample
        self.tile_size = tile_size

        # calculate number of resulting tiles and check whether the images are
        # evenly divisible in square tiles of size (tile_size x tile_size)
        if self.tile_size is None:
            self.tiles = None
        else:
            self.tiles = self.is_divisible(self.size, self.tile_size)

        # list of all scenes in the root directory
        # each scene is divided into tiles blocks
        self.scenes = []
        for scene in os.listdir(root_dir):
            self.scenes += self._compose_scenes(os.path.join(root_dir, scene))

    # the __len__() method returns the number of samples in the Sparcs dataset
    def __len__(self):
        # number of (tiles x channels x height x width) patches after each
        # scene is decomposed to tiles blocks
        return len(self.scenes)

    # the __getitem__() method returns a sample of the Sparcs dataset
    # __getitem__() is implicitly used by pytorch to draw samples during
    # the training process
    def __getitem__(self, idx):

        # select a scene
        scene = self._read_scene(idx)

        # get samples: (tiles x channels x height x width)
        data, gt = self._build_samples(scene)

        # convert to torch tensors
        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(gt, dtype=torch.uint8)

        return x, y

    # returns the band number of the preprocessed Sparcs Tiff files
    def _get_band_number(self, x):
        return int(os.path.basename(x).split('_')[2].replace('B', ''))

    # _store_bands() writes the paths to the data of each scene to a dictionary
    # only the bands of interest are stored
    def _store_bands(self, bands, gt):

        # store the bands of interest in a dictionary
        scene_data = {}
        for i, b in enumerate(bands):
            band = self.sparcs_bands[self._get_band_number(b)]
            if band in self.bands:
                scene_data[band] = b

        # store ground truth
        scene_data['gt'] = gt

        return scene_data

    # _compose_scenes() creates a list of dictionaries containing the paths
    # to the files of each scene
    # if the scenes are divided into tiles, each tile has its own entry
    # with corresponding tile id
    def _compose_scenes(self, scene):

        # list the spectral bands of the scene
        bands = glob.glob(os.path.join(scene, '*B*.tif'))

        # sort the bands in ascending order
        bands.sort(key=self._get_band_number)

        # get the ground truth mask
        gt = glob.glob(os.path.join(scene, '*mask.png')).pop()

        # create an entry for each scene/tile
        scene_data = []

        # check whether the scenes are divided into tiles
        if self.tiles is None:

            # store the bands and the ground truth mask of the current schene
            data = self._store_bands(bands, gt)

            # indicate that no tiling was applied
            data['tile'] = None

            # append to list
            scene_data.append(data)

        else:

            # iterate over the tiles
            for tile in range(self.tiles):

                # store the bands and the ground truth mask of the tile
                data = self._store_bands(bands, gt)

                # store tile number
                data['tile'] = tile

                # append to list
                scene_data.append(data)

        return scene_data

    # _read_scene() reads all the bands and the ground truth mask in a
    # scene/tile to a numpy array and returns a dictionary with
    # (key, value) = ('band_name', np.ndarray(band_data))
    def _read_scene(self, idx):

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
    def _build_samples(self, scene):

        # iterate over the channels to stack
        stack = np.stack([scene[band] for band in self.bands], axis=0)
        gt = scene['gt']

        return stack, gt

    # plot_sample() plots a false color composite of the scene/tile together
    # with the model prediction and the corresponding ground truth
    def plot_sample(self, x, y, y_pred=None, figsize=(10, 10),
                    bands=['nir', 'red', 'green'], stretch=False, **kwargs):

        # check whether to apply constrast stretching
        func = self.contrast_stretching if stretch else lambda x: x

        # create an rgb stack
        rgb = np.dstack([func(x[self.bands.index(band)],
                              **kwargs) for band in bands])

        # create a ListedColormap
        cmap = ListedColormap(self.colors.values())
        boundaries = [*self.colors.keys(), cmap.N]
        norm = BoundaryNorm(boundaries, cmap.N)

        # create figure: check whether to plot model prediction
        if y_pred is not None:
            fig, ax = plt.subplots(1, 3, figsize=figsize)
            ax[2].imshow(y_pred, cmap=cmap, interpolation='nearest', norm=norm)
            ax[2].set_title('Prediction', pad=20)
        else:
            fig, ax = plt.subplots(1, 2, figsize=figsize)

        # plot false color composite
        ax[0].imshow(rgb)
        ax[0].set_title('R = {}, G = {}, B = {}'.format(*bands), pad=20)

        # plot ground thruth mask
        ax[1].imshow(y, cmap=cmap, interpolation='nearest', norm=norm)
        ax[1].set_title('Ground truth', pad=20)

        # create a patch (proxy artist) for every color
        patches = [mpatches.Patch(color=c, label=l) for c, l in
                   zip(self.colors.values(), self.labels.values())]

        # plot patches as legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2,
                   frameon=False)

        return fig, ax


if __name__ == '__main__':

    # path to the preprocessed sparcs dataset
    sparcs_path = "C:/Eurac/2020/Tutorial/Datasets/Sparcs"
    # sparcs_path = "/mnt/CEPH_PROJECTS/cci_snow/dfrisinghelli/Datasets/Sparcs"

    # instanciate the SparcsDataset class
    sparcs_dataset = SparcsDataset(sparcs_path, tile_size=None, bands=-1)

    # randomly sample an integer from [0, nsamples]
    sample = np.random.randint(len(sparcs_dataset), size=1).item()

    # a sample from the sparcs dataset
    sample_x, sample_y = sparcs_dataset[sample]

    # print shape of the sample
    print('A sample from the Sparcs dataset:')
    print('Shape of input data: {}'.format(sample_x.shape))
    print('Shape of ground truth: {}'.format(sample_y.shape))

    # plot the sample
    fig, ax = sparcs_dataset.plot_sample(sample_x, sample_y,
                                         bands=['nir', 'red', 'green'])
