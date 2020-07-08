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
import csv
import glob
import itertools

# externals
import gdal
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm as colormap
from torch.utils.data import Dataset


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

    # plot_sample() plots a false color composite of the scene/tile together
    # with the model prediction and the corresponding ground truth
    def plot_sample(self, x, y, y_pred=None, figsize=(10, 10),
                    bands=['red', 'green', 'blue'], stretch=False, state=None,
                    outpath=os.path.join(os.getcwd(), '_samples/'),  **kwargs):

        # check whether to apply constrast stretching
        stretch = True if kwargs else False
        func = self.contrast_stretching if stretch else lambda x: x

        # create an rgb stack
        rgb = np.dstack([func(x[self.use_bands.index(band)],
                              **kwargs) for band in bands])

        # get labels and corresponding colors
        labels = [label['label'] for label in self.labels.values()]
        colors = [label['color'] for label in self.labels.values()]

        # create a ListedColormap
        cmap = ListedColormap(colors)
        boundaries = [*self.labels.keys(), cmap.N]
        norm = BoundaryNorm(boundaries, cmap.N)

        # create figure: check whether to plot model prediction
        if y_pred is not None:

            # compute accuracy
            acc = (y_pred == y).float().mean()

            # plot model prediction
            fig, ax = plt.subplots(1, 3, figsize=figsize)
            ax[2].imshow(y_pred, cmap=cmap, interpolation='nearest', norm=norm)
            ax[2].set_title('Prediction ({:.2f}%)'.format(acc * 100), pad=15)

        else:
            fig, ax = plt.subplots(1, 2, figsize=figsize)

        # plot false color composite
        ax[0].imshow(rgb)
        ax[0].set_title('R = {}, G = {}, B = {}'.format(*bands), pad=15)

        # plot ground thruth mask
        ax[1].imshow(y, cmap=cmap, interpolation='nearest', norm=norm)
        ax[1].set_title('Ground truth', pad=15)

        # create a patch (proxy artist) for every color
        patches = [mpatches.Patch(color=c, label=l) for c, l in
                   zip(colors, labels)]

        # plot patches as legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2,
                   frameon=False)

        # save figure
        if state is not None:
            os.makedirs(outpath, exist_ok=True)
            fig.savefig(os.path.join(outpath, state.replace('.pt', '.png')),
                        dpi=300, bbox_inches='tight')

        return fig, ax

    # plot_confusion_matrix() plots the confusion matrix of the validation/test
    # set returned by the pytorch.predict function
    def plot_confusion_matrix(self, cm, labels=None, normalize=True,
                              figsize=(10, 10), cmap='Blues', state=None,
                              outpath=os.path.join(os.getcwd(), '_graphics/')):

        # check if labels are provided
        if labels is None:
            labels = [label['label'] for _, label in self.labels.items()]

        # number of classes
        nclasses = len(labels)

        # string format to plot values of confusion matrix
        fmt = '.0f'

        # minimum and maximum values of the colorbar
        vmin, vmax = 0, cm.max()

        # check whether to normalize the confusion matrix
        if normalize:
            # normalize
            cm = cm / cm.sum(axis=1, keepdims=True)

            # change string format to floating point
            fmt = '.2f'
            vmin, vmax= 0, 1

        # create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # get colormap
        cmap = colormap.get_cmap(cmap, 256)

        # plot confusion matrix
        im = ax.imshow(cm, cmap=cmap, vmin=vmin, vmax=vmax)

        # threshold determining the color of the values
        thresh = (cm.max() + cm.min()) / 2

        # brightest/darkest color of current colormap
        cmap_min, cmap_max = cmap(0), cmap(256)

        # plot values of confusion matrix
        for i, j in itertools.product(range(nclasses), range(nclasses)):
            ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                    color = cmap_max if cm[i, j] < thresh else cmap_min)

        # axes properties and labels
        ax.set(xticks=np.arange(nclasses),
               yticks=np.arange(nclasses),
               xticklabels=labels,
               yticklabels=labels,
               ylabel='True',
               xlabel='Predicted')

        # add colorbar axes
        cax = fig.add_axes([ax.get_position().x1 + 0.025, ax.get_position().y0,
                            0.05, ax.get_position().y1 - ax.get_position().y0])
        fig.colorbar(im, cax=cax)

        # save figure
        if state is not None:
            os.makedirs(outpath, exist_ok=True)
            fig.savefig(os.path.join(outpath, state.replace('.pt', '_cm.png')),
                        dpi=300, bbox_inches='tight')

        return fig, ax

    def plot_loss(self, state_file, figsize=(10, 10),
                  colors=['lightgreen', 'skyblue', 'darkgreen', 'steelblue'],
                  outpath=os.path.join(os.getcwd(), '_graphics/')):

        # load the model loss
        state = torch.load(state_file)

        # get all non-zero elements, i.e. get number of epochs trained before
        # early stop
        loss = {k: v[np.nonzero(v)].reshape(v.shape[0], -1) for k, v in
                state.items() if k != 'epoch'}

        # number of epochs trained
        epochs = np.arange(0, state['epoch'] + 1)

        # instanciate figure
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        # plot training and validation mean loss per epoch
        [ax1.plot(epochs, v.mean(axis=0),
                  label=k.capitalize().replace('_', ' '), color=c, lw=2)
         for (k, v), c in zip(loss.items(), colors) if 'loss' in k]

        # plot training loss per batch
        ax2 = ax1.twiny()
        [ax2.plot(v.flatten('F'), color=c, alpha=0.5)
         for (k, v), c in zip(loss.items(), colors) if 'loss' in k and
         'validation' not in k]

        # plot training and validation mean accuracy per epoch
        ax3 = ax1.twinx()
        [ax3.plot(epochs, v.mean(axis=0),
                  label=k.capitalize().replace('_', ' '), color=c, lw=2)
         for (k, v), c in zip(loss.items(), colors) if 'accuracy' in k]

        # plot training accuracy per batch
        ax4 = ax3.twiny()
        [ax4.plot(v.flatten('F'), color=c, alpha=0.5)
         for (k, v), c in zip(loss.items(), colors) if 'accuracy' in k and
         'validation' not in k]

        # axes properties and labels
        for ax in [ax2, ax4]:
            ax.set(xticks=[], xticklabels=[])
        ax1.set(xlabel='Epoch',
                ylabel='Loss',
                ylim=(0, 1))
        ax3.set(ylabel='Accuracy',
                ylim=(0, 1))

        # compute early stopping point
        esepoch = np.argmax(loss['validation_accuracy'].mean(axis=0))
        esacc = np.max(loss['validation_accuracy'].mean(axis=0))
        ax1.vlines(esepoch, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1],
                   ls='--', color='grey')
        ax1.text(esepoch - 1, ax1.get_ylim()[0] + 0.01,
                 'epoch = {}'.format(esepoch), ha='right', color='grey')
        ax1.text(esepoch + 1, ax1.get_ylim()[0] + 0.01,
                 'acc = {:.2f}%'.format(esacc * 100), ha='left', color='grey')

        # add legends
        ax1.legend(frameon=False, loc='lower left')
        ax3.legend(frameon=False, loc='upper left')

        # save figure
        os.makedirs(outpath, exist_ok=True)
        fig.savefig(os.path.join(
            outpath, os.path.basename(state_file).replace('.pt', '.png')),
                    dpi=300, bbox_inches='tight')

        return fig, ax


# SparcsDataset class: inherits from the generic ImageDataset class
class SparcsDataset(ImageDataset):

    def __init__(self, root_dir, use_bands=['red', 'green', 'blue'],
                 tile_size=None):
        # initialize super class ImageDataset
        super().__init__(root_dir, use_bands, tile_size)

        # list of all scenes in the root directory
        # each scene is divided into tiles blocks
        self.scenes = []
        for scene in os.listdir(self.root):
            self.scenes += self.compose_scenes(os.path.join(self.root, scene))

    # image size of the Sparcs dataset: (height, width)
    def get_size(self):
        return (1000, 1000)

    # Landsat 8 bands of the Sparcs dataset
    def get_bands(self):
        return {
            1: 'violet',
            2: 'blue',
            3: 'green',
            4: 'red',
            5: 'nir',
            6: 'swir1',
            7: 'swir2',
            8: 'pan',
            9: 'cirrus',
            10: 'tir'}

    # class labels of the Sparcs dataset
    def get_labels(self):
        labels = ['Shadow', 'Shadow over Water', 'Water', 'Snow', 'Land',
                  'Cloud', 'Flooded']
        colors = ['black', 'darkblue', 'blue', 'lightblue', 'grey', 'white',
                  'yellow']
        lc = {}
        for i, (l, c) in enumerate(zip(labels, colors)):
            lc[i] = {'label': l, 'color': c}
        return lc

    # preprocessing of the Sparcs dataset
    def preprocess(self, data, gt):

        # if the preprocessing is not done externally, implement it here

        # convert to torch tensors
        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(gt, dtype=torch.uint8)
        return x, y

    # _compose_scenes() creates a list of dictionaries containing the paths
    # to the files of each scene
    # if the scenes are divided into tiles, each tile has its own entry
    # with corresponding tile id
    def compose_scenes(self, scene):

        # list the spectral bands of the scene
        bands = glob.glob(os.path.join(scene, '*B*.tif'))

        # sort the bands in ascending order
        bands.sort(key=self._get_band_number)

        # get the ground truth mask
        gt = glob.glob(os.path.join(scene, '*mask.png')).pop()

        # create an entry for each scene/tile
        scene_data = []

        # iterate over the tiles
        for tile in range(self.tiles):

            # store the bands and the ground truth mask of the tile
            data = self._store_bands(bands, gt)

            # store tile number
            data['tile'] = tile

            # append to list
            scene_data.append(data)

        return scene_data

    # returns the band number of the preprocessed Sparcs Tiff files
    def _get_band_number(self, x):
        return int(os.path.basename(x).split('_')[2].replace('B', ''))

    # _store_bands() writes the paths to the data of each scene to a dictionary
    # only the bands of interest are stored
    def _store_bands(self, bands, gt):

        # store the bands of interest in a dictionary
        scene_data = {}
        for i, b in enumerate(bands):
            band = self.bands[self._get_band_number(b)]
            if band in self.use_bands:
                scene_data[band] = b

        # store ground truth
        scene_data['gt'] = gt

        return scene_data


class Cloud95Dataset(ImageDataset):

    def __init__(self, root_dir, use_bands=[], tile_size=None, exclude=None):

        # initialize super class ImageDataset
        super().__init__(root_dir, use_bands, tile_size)

        # whether to exclude patches with more than 80% black pixels, i.e.
        # patches resulting from the black margins around a Landsat 8 scene
        self.exclude = exclude

        # list of all scenes in the root directory
        # each scene is divided into tiles blocks
        self.scenes = self.compose_scenes(self.root)

    # image size of the Cloud-95 dataset: (height, width)
    def get_size(self):
        return (384, 384)

    # Landsat 8 bands in the Cloud-95 dataset
    def get_bands(self):
        return {1: 'red', 2: 'green', 3: 'blue', 4: 'nir'}

    # class labels of the Cloud-95 dataset
    def get_labels(self):
        return {0: {'label': 'Clear', 'color': 'skyblue'},
                1: {'label': 'Cloud', 'color': 'white'}}

    # preprocess Cloud-95 dataset
    def preprocess(self, data, gt):

        # normalize the data
        # here, we use the normalization of the authors of Cloud-95, i.e.
        # Mohajerani and Saeedi (2019, 2020)
        x = torch.tensor(data / 65535, dtype=torch.float32)
        y = torch.tensor(gt / 255, dtype=torch.uint8)

        return x, y


    def compose_scenes(self, root_dir):

        # whether to exclude patches with more than 80% black pixels
        ipatches = []
        if self.exclude is not None:
            with open(os.path.join(self.root, self.exclude), newline='') as f:
                reader = csv.reader(f)
                # list of informative patches
                ipatches = list(itertools.chain.from_iterable(reader))

        # get the names of the directories containing the TIFF files of
        # the bands of interest
        band_dirs = {}
        for dirpath, dirname, files in os.walk(root_dir):
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
        biter = self.bands[1]
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
                    # save path to current band TIFF file to dictionary
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

    # the csv file containing the names of the informative patches
    patches = 'training_patches_95-cloud_nonempty.csv'

    # instanciate the Cloud-95 dataset
    cloud_dataset = Cloud95Dataset(cloud_path, tile_size=192, exclude=patches)

    # instanciate the SparcsDataset class
    sparcs_dataset = SparcsDataset(sparcs_path, tile_size=None,
                                   use_bands=['nir', 'red', 'green'])

    # a sample from the sparcs dataset
    sample_s = np.random.randint(len(sparcs_dataset), size=1).item()
    s_x, s_y = sparcs_dataset[sample_s]
    fig, ax = sparcs_dataset.plot_sample(s_x, s_y,
                                         bands=['nir', 'red', 'green'])

    # a sample from the cloud dataset
    sample_c = np.random.randint(len(cloud_dataset), size=1).item()
    c_x, c_y = cloud_dataset[sample_c]
    fig, ax = cloud_dataset.plot_sample(c_x, c_y,
                                        bands=['nir', 'red', 'green'])

    # print shape of the sample
    for i, l, d in zip([s_x, c_x], [s_y, c_y],
                       [sparcs_dataset, cloud_dataset]):
        print('A sample from the {}:'.format(d.__class__.__name__))
        print('Shape of input data: {}'.format(i.shape))
        print('Shape of ground truth: {}'.format(l.shape))

    # show figures
    plt.show()
