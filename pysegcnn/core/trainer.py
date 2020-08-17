# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:24:34 2020

@author: Daniel
"""
# builtins
import dataclasses
import pathlib
import logging
import datetime

# externals
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer

# locals
from pysegcnn.core.dataset import SupportedDatasets, ImageDataset
from pysegcnn.core.transforms import Augment
from pysegcnn.core.utils import img2np, item_in_enum, accuracy_function
from pysegcnn.core.split import SupportedSplits
from pysegcnn.core.models import (SupportedModels, SupportedOptimizers,
                                  SupportedLossFunctions, Network)
from pysegcnn.core.layers import Conv2dSame
from pysegcnn.main.config import HERE

# module level logger
LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class BaseConfig:

    def __post_init__(self):
        # check input types
        for field in dataclasses.fields(self):
            # the value of the current field
            value = getattr(self, field.name)

            # check whether the value is of the correct type
            if not isinstance(value, field.type):
                # try to convert the value to the correct type
                try:
                    setattr(self, field.name, field.type(value))
                except TypeError:
                    # raise an exception if the conversion fails
                    raise TypeError('Expected {} to be {}, got {}.'
                                    .format(field.name, field.type,
                                            type(value)))


@dataclasses.dataclass
class DatasetConfig(BaseConfig):
    dataset_name: str
    root_dir: pathlib.Path
    bands: list
    tile_size: int
    gt_pattern: str
    seed: int
    sort: bool = False
    transforms: list = dataclasses.field(default_factory=list)
    pad: bool = False

    def __post_init__(self):
        # check input types
        super().__post_init__()

        # check whether the dataset is currently supported
        self.dataset_class = item_in_enum(self.dataset_name, SupportedDatasets)

        # check whether the root directory exists
        if not self.root_dir.exists():
            raise FileNotFoundError('{} does not exist.'.format(self.root_dir))

        # check whether the transformations inherit from the correct class
        if not all([isinstance(t, Augment) for t in self.transforms if
                    self.transforms]):
            raise TypeError('Each transformation is expected to be an instance'
                            ' of {}.'.format('.'.join([Augment.__module__,
                                                       Augment.__name__])))

    def init_dataset(self):

        # instanciate the dataset
        dataset = self.dataset_class(
                    root_dir=str(self.root_dir),
                    use_bands=self.bands,
                    tile_size=self.tile_size,
                    seed=self.seed,
                    sort=self.sort,
                    transforms=self.transforms,
                    pad=self.pad,
                    gt_pattern=self.gt_pattern
                    )

        return dataset


@dataclasses.dataclass
class SplitConfig(BaseConfig):
    split_mode: str
    ttratio: float
    tvratio: float
    date: str = 'yyyymmdd'
    dateformat: str = '%Y%m%d'
    drop: float = 0

    def __post_init__(self):
        # check input types
        super().__post_init__()

        # check if the split mode is valid
        self.split_class = item_in_enum(self.split_mode, SupportedSplits)

    # function to drop samples with a fraction of pixels equal to the constant
    # padding value self.cval >= self.drop
    @staticmethod
    def _drop_samples(ds, drop_threshold=1):

        # iterate over the scenes returned by self.compose_scenes()
        dropped = []
        for pos, i in enumerate(ds.indices):

            # the current scene
            s = ds.dataset.scenes[i]

            # the current tile in the ground truth
            tile_gt = img2np(s['gt'], ds.dataset.tile_size, s['tile'],
                             ds.dataset.pad, ds.dataset.cval)

            # percent of pixels equal to the constant padding value
            npixels = (tile_gt[tile_gt == ds.dataset.cval].size / tile_gt.size)

            # drop samples where npixels >= self.drop
            if npixels >= drop_threshold:
                LOGGER.info('Skipping scene {}, tile {}: {:.2f}% padded pixels'
                            ' ...'.format(s['id'], s['tile'], npixels * 100))
                dropped.append(s)
                _ = ds.indices.pop(pos)

        return dropped

    def train_val_test_split(self, ds):

        if not isinstance(ds, ImageDataset):
            raise TypeError('Expected "ds" to be {}.'
                            .format('.'.join([ImageDataset.__module__,
                                              ImageDataset.__name__])))

        if self.split_mode == 'random' or self.split_mode == 'scene':
            subset = self.split_class(ds,
                                      self.ttratio,
                                      self.tvratio,
                                      ds.seed)

        else:
            subset = self.split_class(ds, self.date, self.dateformat)

        # the training, validation and test dataset
        train_ds, valid_ds, test_ds = subset.split()

        # whether to drop training samples with a fraction of pixels equal to
        # the constant padding value cval >= drop
        if ds.pad and self.drop > 0:
            self.dropped = self._drop_samples(train_ds, self.drop)

        return train_ds, valid_ds, test_ds

    @staticmethod
    def dataloaders(*args, **kwargs):
        # check whether each dataset in args has the correct type
        loaders = []
        for ds in args:
            if not isinstance(ds, Dataset):
                raise TypeError('Expected {}, got {}.'
                                .format(repr(Dataset), type(ds)))

            # check if the dataset is not empty
            if len(ds) > 0:
                # build the dataloader
                loader = DataLoader(ds, **kwargs)
            else:
                loader = None
            loaders.append(loader)

        return loaders


@dataclasses.dataclass
class ModelConfig(BaseConfig):
    model_name: str
    filters: list
    torch_seed: int
    optim_name: str
    loss_name: str
    skip_connection: bool = True
    kwargs: dict = dataclasses.field(
        default_factory=lambda: {'kernel_size': 3, 'stride': 1, 'dilation': 1})
    batch_size: int = 64
    checkpoint: bool = False
    transfer: bool = False
    pretrained_model: str = ''
    lr: float = 0.001
    early_stop: bool = False
    mode: str = 'max'
    delta: float = 0
    patience: int = 10
    epochs: int = 50
    nthreads: int = torch.get_num_threads()
    save: bool = True

    def __post_init__(self):
        # check input types
        super().__post_init__()

        # check whether the model is currently supported
        self.model_class = item_in_enum(self.model_name, SupportedModels)

        # check whether the optimizer is currently supported
        self.optim_class = item_in_enum(self.optim_name, SupportedOptimizers)

        # check whether the loss function is currently supported
        self.loss_class = item_in_enum(self.loss_name, SupportedLossFunctions)

        # path to model states
        self.state_path = pathlib.Path(HERE).joinpath('_models/')

        # path to pretrained model
        self.pretrained_path = self.state_path.joinpath(self.pretrained_model)

    def init_optimizer(self, model):

        LOGGER.info('Optimizer: {}.'.format(repr(self.optim_class)))

        # initialize the optimizer for the specified model
        optimizer = self.optim_class(model.parameters(), self.lr)

        return optimizer

    def init_loss_function(self):

        LOGGER.info('Loss function: {}.'.format(repr(self.loss_class)))

        # instanciate the loss function
        loss_function = self.loss_class()

        return loss_function

    def init_model(self, ds, state_file):

        # write an initialization string to the log file
        # now = datetime.datetime.strftime(datetime.datetime.now(),
        #                                  '%Y-%m-%dT%H:%M:%S')
        # LOGGER.info(80 * '-')
        # LOGGER.info('{}: Initializing model run. '.format(now) + 35 * '-')
        # LOGGER.info(80 * '-')

        # case (1): build a new model
        if not self.transfer:

            # set the random seed for reproducibility
            torch.manual_seed(self.torch_seed)
            LOGGER.info('Initializing model: {}'.format(state_file.name))

            # instanciate the model
            model = self.model_class(
                in_channels=len(ds.use_bands),
                nclasses=len(ds.labels),
                filters=self.filters,
                skip=self.skip_connection,
                **self.kwargs)

        # case (2): load a pretrained model for transfer learning
        else:
            # load pretrained model
            LOGGER.info('Loading pretrained model for transfer learning from: '
                        '{}'.format(self.pretrained_path))
            model = self.transfer_model(self.pretrained_path, ds)

        # initialize the optimizer
        optimizer = self.init_optimizer(model)

        # whether to resume training from an existing model checkpoint
        checkpoint_state = {}
        if self.checkpoint:
            model, optimizer, checkpoint_state = self.load_checkpoint(
                model, optimizer, state_file)

        return model, optimizer, checkpoint_state

    @staticmethod
    def load_checkpoint(model, optimizer, state_file):

        # whether to resume training from an existing model checkpoint
        checkpoint_state = {}

        # if no checkpoint exists, file a warning and continue with a model
        # initialized from scratch
        if not state_file.exists():
            LOGGER.warning('Checkpoint for model {} does not exist. '
                           'Initializing new model.'
                           .format(state_file.name))
        else:
            # load model checkpoint
            model, optimizer, model_state = Network.load(state_file, optimizer)

            # load model loss and accuracy

            # get all non-zero elements, i.e. get number of epochs trained
            # before the early stop
            checkpoint_state = {k: v[np.nonzero(v)].reshape(v.shape[0], -1)
                                for k, v in model_state['state'].items()}

        return model, optimizer, checkpoint_state

    @staticmethod
    def transfer_model(state_file, ds):

        # check input type
        if not isinstance(ds, ImageDataset):
            raise TypeError('Expected "ds" to be {}.'
                            .format('.'.join([ImageDataset.__module__,
                                              ImageDataset.__name__])))

        # load the pretrained model
        model, _, model_state = Network.load(state_file)
        LOGGER.info('Configuring model for new dataset: {}.'.format(
            ds.__class__.__name__))

        # check whether the current dataset uses the correct spectral bands
        if new_ds.use_bands != model_state['bands']:
            raise ValueError('The pretrained network was trained with '
                             'bands {}, not with bands {}.'
                             .format(model_state['bands'], new_ds.use_bands))

        # get the number of convolutional filters
        filters = model_state['params']['filters']

        # reset model epoch to 0, since the model is trained on a different
        # dataset
        model.epoch = 0

        # adjust the number of classes in the model
        model.nclasses = len(ds.labels)
        LOGGER.info('Replacing classification layer to classes: {}.'
                    .format(', '.join('({}, {})'.format(k, v['label'])
                                      for k, v in ds.labels.items())))

        # adjust the classification layer to the number of classes of the
        # current dataset
        model.classifier = Conv2dSame(in_channels=filters[0],
                                      out_channels=model.nclasses,
                                      kernel_size=1)

        return model


@dataclasses.dataclass
class StateConfig(BaseConfig):
    ds: ImageDataset
    sc: SplitConfig
    mc: ModelConfig

    def __post_init__(self):
        super().__post_init__()

    def init_state(self):

        # file to save model state to:
        # network_dataset_optim_split_splitparams_tilesize_batchsize_bands.pt

        # model state filename
        state_file = '{}_{}_{}_{}Split_{}_t{}_b{}_{}.pt'

        # get the band numbers
        bformat = ''.join(band[0] +
                          str(self.ds.sensor.__members__[band].value) for
                              band in self.ds.use_bands)

        # check which split mode was used
        if self.sc.split_mode == 'date':
            # store the date that was used to split the dataset
            state_file = state_file.format(self.mc.model_name,
                                           self.ds.__class__.__name__,
                                           self.mc.optim_name,
                                           self.sc.split_mode.capitalize(),
                                           self.sc.date,
                                           self.ds.tile_size,
                                           self.mc.batch_size,
                                           bformat)
        else:
            # store the random split parameters
            split_params = 's{}_t{}v{}'.format(
                self.ds.seed, str(self.sc.ttratio).replace('.', ''),
                str(self.sc.tvratio).replace('.', ''))

            # model state filename
            state_file = state_file.format(self.mc.model_name,
                                           self.ds.__class__.__name__,
                                           self.mc.optim_name,
                                           self.sc.split_mode.capitalize(),
                                           split_params,
                                           self.ds.tile_size,
                                           self.mc.batch_size,
                                           bformat)

        # check whether a pretrained model was used and change state filename
        # accordingly
        if self.mc.transfer:
            # add the configuration of the pretrained model to the state name
            state_file = (state_file.replace('.pt', '_') +
                          'pretrained_' + self.mc.pretrained_model)

        # path to model state
        state = self.mc.state_path.joinpath(state_file)

        return state


@dataclasses.dataclass
class EvalConfig(BaseConfig):
    state_file: pathlib.Path
    test: object
    predict_scene: bool = False
    plot_samples: bool = False
    plot_scenes: bool = False
    plot_bands: list = dataclasses.field(
        default_factory=lambda: ['nir', 'red', 'green'])
    cm: bool = True
    figsize: tuple = (10, 10)
    alpha: int = 5

    def __post_init__(self):
        super().__post_init__()

        # check whether the test input parameter is correctly specified
        if self.test not in [None, False, True]:
            raise TypeError('Expected "test" to be None, True or False, got '
                            '{}.'.format(self.test))

        # the output paths for the different graphics
        self.base_path = pathlib.Path(HERE)
        self.sample_path = self.base_path.joinpath('_samples')
        self.scenes_path = self.base_path.joinpath('_scenes')
        self.models_path = self.base_path.joinpath('_graphics')

        # write initialization string to log file
        # LOGGER.info(80 * '-')
        # LOGGER.info('{}')
        # LOGGER.info(80 * '-')


@dataclasses.dataclass
class LogConfig(BaseConfig):
    state_file: pathlib.Path

    def __post_init__(self):
        super().__post_init__()

        # the path to store model logs
        self.log_path = pathlib.Path(HERE).joinpath('_logs')

        # the log file of the current model
        self.log_file = self.log_path.joinpath(
            self.state_file.name.replace('.pt', '.log'))


@dataclasses.dataclass
class NetworkTrainer(BaseConfig):
    model: Network
    optimizer: Optimizer
    loss_function: nn.Module
    train_dl: DataLoader
    valid_dl: DataLoader
    test_dl: DataLoader
    state_file: pathlib.Path
    epochs: int = 1
    nthreads: int = torch.get_num_threads()
    early_stop: bool = False
    mode: str = 'max'
    delta: float = 0
    patience: int = 10
    checkpoint_state: dict = dataclasses.field(default_factory=dict)
    save: bool = True

    def __post_init__(self):
        super().__post_init__()

        # whether to use the gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

        # maximum accuracy on the validation dataset
        self.max_accuracy = 0
        if self.checkpoint_state:
            self.max_accuracy = self.checkpoint_state['va'].mean(
                axis=0).max().item()

        # whether to use early stopping
        self.es = None
        if self.early_stop:
            self.es = EarlyStopping(self.mode, self.max_accuracy, self.delta,
                                    self.patience)

    def train(self):

        LOGGER.info(35 * '-' + ' Training ' + 35 * '-')

        # set the number of threads
        LOGGER.info('Device: {}'.format(self.device))
        LOGGER.info('Number of cpu threads: {}'.format(self.nthreads))
        torch.set_num_threads(self.nthreads)

        # create dictionary of the observed losses and accuracies on the
        # training and validation dataset
        tshape = (len(self.train_dl), self.epochs)
        vshape = (len(self.valid_dl), self.epochs)
        self.training_state = {'tl': np.zeros(shape=tshape),
                               'ta': np.zeros(shape=tshape),
                               'vl': np.zeros(shape=vshape),
                               'va': np.zeros(shape=vshape)
                               }

        # send the model to the gpu if available
        self.model = self.model.to(self.device)

        # initialize the training: iterate over the entire training data set
        for epoch in range(self.epochs):

            # set the model to training mode
            LOGGER.info('Setting model to training mode ...')
            self.model.train()

            # iterate over the dataloader object
            for batch, (inputs, labels) in enumerate(self.train_dl):

                # send the data to the gpu if available
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # reset the gradients
                self.optimizer.zero_grad()

                # perform forward pass
                outputs = self.model(inputs)

                # compute loss
                loss = self.loss_function(outputs, labels.long())
                observed_loss = loss.detach().numpy().item()
                self.training_state['tl'][batch, epoch] = observed_loss

                # compute the gradients of the loss function w.r.t.
                # the network weights
                loss.backward()

                # update the weights
                self.optimizer.step()

                # calculate predicted class labels
                ypred = F.softmax(outputs, dim=1).argmax(dim=1)

                # calculate accuracy on current batch
                observed_accuracy = accuracy_function(ypred, labels)
                self.training_state['ta'][batch, epoch] = observed_accuracy

                # print progress
                LOGGER.info('Epoch: {:d}/{:d}, Mini-batch: {:d}/{:d}, '
                            'Loss: {:.2f}, Accuracy: {:.2f}'.format(
                                epoch + 1,
                                self.epochs,
                                batch + 1,
                                len(self.train_dl),
                                observed_loss,
                                observed_accuracy))

            # update the number of epochs trained
            self.model.epoch += 1

            # whether to evaluate model performance on the validation set and
            # early stop the training process
            if self.early_stop:

                # model predictions on the validation set
                vacc, vloss = self.predict()

                # append observed accuracy and loss to arrays
                self.training_state['va'][:, epoch] = vacc.squeeze()
                self.training_state['vl'][:, epoch] = vloss.squeeze()

                # metric to assess model performance on the validation set
                epoch_acc = vacc.squeeze().mean()

                # whether the model improved with respect to the previous epoch
                if self.es.increased(epoch_acc, self.max_accuracy, self.delta):
                    self.max_accuracy = epoch_acc

                    # save model state if the model improved with
                    # respect to the previous epoch
                    self.save_state()

                # whether the early stopping criterion is met
                if self.es.stop(epoch_acc):
                    break

            else:
                # if no early stopping is required, the model state is
                # saved after each epoch
                self.save_state()


        return self.training_state

    def predict(self):

        # send the model to the gpu if available
        self.model = self.model.to(self.device)

        # set the model to evaluation mode
        LOGGER.info('Setting model to evaluation mode ...')
        self.model.eval()

        # create arrays of the observed losses and accuracies
        accuracies = np.zeros(shape=(len(self.valid_dl), 1))
        losses = np.zeros(shape=(len(self.valid_dl), 1))

        # iterate over the validation/test set
        LOGGER.info('Calculating accuracy on the validation set ...')
        for batch, (inputs, labels) in enumerate(self.valid_dl):

            # send the data to the gpu if available
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # calculate network outputs
            with torch.no_grad():
                outputs = self.model(inputs)

            # compute loss
            loss = self.loss_function(outputs, labels.long())
            losses[batch, 0] = loss.detach().numpy().item()

            # calculate predicted class labels
            pred = F.softmax(outputs, dim=1).argmax(dim=1)

            # calculate accuracy on current batch
            acc = accuracy_function(pred, labels)
            accuracies[batch, 0] = acc

            # print progress
            LOGGER.info('Mini-batch: {:d}/{:d}, Accuracy: {:.2f}'
                        .format(batch + 1, len(self.valid_dl), acc))

        # calculate overall accuracy on the validation/test set
        LOGGER.info('Epoch: {:d}, Mean accuracy: {:.2f}%.'
                    .format(self.model.epoch, accuracies.mean() * 100))

        return accuracies, losses

    def save_state(self):

        # whether to save the model state
        if self.save:

            # append the model performance before the checkpoint to the model
            # state, if a checkpoint is passed
            if self.checkpoint_state:

                # append values from checkpoint to current training state
                state = {k1: np.hstack([v1, v2]) for (k1, v1), (k2, v2) in
                         zip(self.checkpoint_state.items(),
                             self.training_state.items()) if k1 == k2}
            else:
                state = self.training_state

            # save model state
            _ = self.model.save(
                self.state_file,
                self.optimizer,
                bands=self.train_dl.dataset.dataset.use_bands,
                train_ds=self.train_dl.dataset,
                valid_ds=self.valid_dl.dataset,
                test_ds=self.test_dl.dataset,
                state=state,
                )


    def __repr__(self):

        # representation string to print
        fs = self.__class__.__name__ + '(\n'

        # dataset
        fs += '    (dataset):\n        '
        fs += ''.join(
            repr(self.train_dl.dataset.dataset)).replace('\n', '\n        ')

        # batch size
        fs += '\n    (batch):\n        '
        fs += '- batch size: {}\n        '.format(self.train_dl.batch_size)
        fs += '- mini-batch shape (b, c, h, w): {}'.format(
            (self.train_dl.batch_size,
             len(self.train_dl.dataset.dataset.use_bands),
             self.train_dl.dataset.dataset.tile_size,
             self.train_dl.dataset.dataset.tile_size)
            )

        # dataset split
        fs += '\n    (split):'
        fs += '\n        ' + repr(self.train_dl.dataset)
        fs += '\n        ' + repr(self.valid_dl.dataset)
        fs += '\n        ' + repr(self.test_dl.dataset)

        # model
        fs += '\n    (model):\n        '
        fs += ''.join(repr(self.model)).replace('\n', '\n        ')

        # optimizer
        fs += '\n    (optimizer):\n        '
        fs += ''.join(repr(self.optimizer)).replace('\n', '\n        ')

        # early stopping
        fs += '\n    (early stop):\n        '
        fs += ''.join(repr(self.es)).replace('\n', '\n        ')

        fs += '\n)'
        return fs


class EarlyStopping(object):

    def __init__(self, mode='max', best=0, min_delta=0, patience=10):

        # check if mode is correctly specified
        if mode not in ['min', 'max']:
            raise ValueError('Mode "{}" not supported. '
                             'Mode is either "min" (check whether the metric '
                             'decreased, e.g. loss) or "max" (check whether '
                             'the metric increased, e.g. accuracy).'
                             .format(mode))

        # mode to determine if metric improved
        self.mode = mode

        # whether to check for an increase or a decrease in a given metric
        self.is_better = self.decreased if mode == 'min' else self.increased

        # minimum change in metric to be classified as an improvement
        self.min_delta = min_delta

        # number of epochs to wait for improvement
        self.patience = patience

        # initialize best metric
        self.best = best

        # initialize early stopping flag
        self.early_stop = False

        # initialize the early stop counter
        self.counter = 0

    def stop(self, metric):

        # if the metric improved, reset the epochs counter, else, advance
        if self.is_better(metric, self.best, self.min_delta):
            self.counter = 0
            self.best = metric
        else:
            self.counter += 1
            LOGGER.info('Early stopping counter: {}/{}'.format(
                self.counter, self.patience))

        # if the metric did not improve over the last patience epochs,
        # the early stopping criterion is met
        if self.counter >= self.patience:
            LOGGER.info('Early stopping criterion met, stopping training.')
            self.early_stop = True

        return self.early_stop

    def decreased(self, metric, best, min_delta):
        return metric < best - min_delta

    def increased(self, metric, best, min_delta):
        return metric > best + min_delta

    def __repr__(self):
        fs = self.__class__.__name__
        fs += '(mode={}, best={:.2f}, delta={}, patience={})'.format(
            self.mode, self.best, self.min_delta, self.patience)
        return fs
