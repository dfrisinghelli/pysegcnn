# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:24:34 2020

@author: Daniel
"""
# builtins
import dataclasses
import pathlib

# externals
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# locals
from pysegcnn.core.dataset import SupportedDatasets, ImageDataset
from pysegcnn.core.transforms import Augment
from pysegcnn.core.utils import img2np, item_in_enum, accuracy_function
from pysegcnn.core.split import SupportedSplits
from pysegcnn.core.models import (SupportedModels, SupportedOptimizers,
                                  SupportedLossFunctions)
from pysegcnn.core.layers import Conv2dSame
from pysegcnn.main.config import HERE


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
    root_dir: pathlib.Path
    bands: list
    tile_size: int
    gt_pattern: str
    seed: int
    sort: bool = False
    transforms: list = dataclasses.field(default_factory=list)
    pad: bool = False
    cval: int = 99

    def __post_init__(self):
        # check input types
        super().__post_init__()

        # check whether the root directory exists
        if not self.root_dir.exists():
            raise FileNotFoundError('{} does not exist.'.format(self.root_dir))

        # check whether the transformations inherit from the correct class
        if not all([isinstance(t, Augment) for t in self.transforms if
                    self.transforms]):
            raise TypeError('Each transformation is expected to be an instance'
                            ' of {}.'.format('.'.join([Augment.__module__,
                                                       Augment.__name__])))

        # check whether the constant padding value is within the valid range
        if not 0 < self.cval < 255:
            raise ValueError('Expecting 0 <= cval <= 255, got cval={}.'
                             .format(self.cval))

        # the dataset name
        self.dataset_name = self.root_dir.name

        # check whether the dataset is currently supported
        self.dataset_class = item_in_enum(self.dataset_name, SupportedDatasets)

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
                    cval=self.cval,
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
    def _drop_samples(self, ds, drop_threshold=1):

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
                print('Skipping scene {}, tile {}: {:.2f}% padded pixels ...'
                      .format(s['id'], s['tile'], npixels * 100))
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


@dataclasses.dataclass
class ModelConfig(BaseConfig):
    model_name: str
    filters: list
    batch_size: int
    skip_connection: bool = True
    kwargs: dict = dataclasses.field(
        default_factory=lambda: {'kernel_size': 3, 'stride': 1, 'dilation': 1})
    state_path: pathlib.Path = pathlib.Path(HERE).joinpath('_models/')
    batch_size: int = 64
    checkpoint: bool = False
    pretrained: bool = False
    pretrained_model: str = ''

    def __post_init__(self):
        # check input types
        super().__post_init__()

        # check whether the model is currently supported
        self.model_class = item_in_enum(self.model_name, SupportedModels)

    def init_state(self, ds):

        # file to save model state to
        # format: network_dataset_seed_tilesize_batchsize_bands.pt

        # get the band numbers
        bformat = ''.join(band[0] +
                          str(ds.sensor.__members__[band].value) for
                          band in ds.use_bands)

        # model state filename
        state_file = ('{}_{}_s{}_t{}_b{}_{}.pt'
                      .format(self.model_class.__name__,
                              ds.__class__.__name__,
                              ds.seed,
                              ds.tile_size,
                              self.batch_size,
                              bformat))

        # check whether a pretrained model was used and change state filename
        # accordingly
        if self.pretrained:
            # add the configuration of the pretrained model to the state name
            state_file = (state_file.replace('.pt', '_') +
                          'pretrained_' + self.pretrained_model)

        # path to model state
        state = self.state_path.joinpath(state_file)

        # path to model loss/accuracy
        loss_state = pathlib.Path(str(state).replace('.pt', '_loss.pt'))

        return state, loss_state

    def init_model(self, ds):

        # case (1): build a new model
        if not self.pretrained:

            # instanciate the model
            model = self.model_class(
                in_channels=len(ds.use_bands),
                nclasses=len(ds.labels),
                filters=self.filters,
                skip=self.skip_connection,
                **self.kwargs)

        # case (2): load a pretrained model
        else:

            # load pretrained model
            model = self.load_pretrained()

        return model

    def load_checkpoint(self, state_file, loss_state, model, optimizer):

        # initial accuracy on the validation set
        max_accuracy = 0

        # set the model checkpoint to None, overwritten when resuming
        # training from an existing model checkpoint
        checkpoint_state = None

        # whether to resume training from an existing model
        if self.checkpoint:

            # check if a model checkpoint exists
            if not state_file.exists():
                raise FileNotFoundError('Model checkpoint {} does not exist.'
                                        .format(state_file))

            # load the model state
            state = model.load(state_file.name, optimizer, self.state_path)
            print('Resuming training from {} ...'.format(state))
            print('Model epoch: {:d}'.format(model.epoch))

            # load the model loss and accuracy
            checkpoint_state = torch.load(loss_state)

            # get all non-zero elements, i.e. get number of epochs trained
            # before the early stop
            checkpoint_state = {k: v[np.nonzero(v)].reshape(v.shape[0], -1)
                                for k, v in checkpoint_state.items()}

            # maximum accuracy on the validation set
            max_accuracy = checkpoint_state['va'][:, -1].mean().item()

        return checkpoint_state, max_accuracy

    def load_pretrained(self, ds):

        # load the pretrained model
        model_state = self.state_path.joinpath(self.pretrained_model)
        if not model_state.exists():
            raise FileNotFoundError('Pretrained model {} does not exist.'
                                    .format(model_state))

        # load the model state
        model_state = torch.load(model_state)

        # get the input bands of the pretrained model
        bands = model_state['bands']

        # get the number of convolutional filters
        filters = model_state['params']['filters']

        # check whether the current dataset uses the correct spectral bands
        if ds.use_bands != bands:
            raise ValueError('The bands of the pretrained network do not '
                             'match the specified bands: {}'
                             .format(bands))

        # instanciate pretrained model architecture
        model = self.model_class(**model_state['params'],
                                 **model_state['kwargs'])

        # load pretrained model weights
        model.load(self.pretrained_model, inpath=str(self.state_path))

        # reset model epoch to 0, since the model is trained on a different
        # dataset
        model.epoch = 0

        # adjust the number of classes in the model
        model.nclasses = len(ds.labels)

        # adjust the classification layer to the number of classes of the
        # current dataset
        model.classifier = Conv2dSame(in_channels=filters[0],
                                      out_channels=model.nclasses,
                                      kernel_size=1)

        return model


@dataclasses.dataclass
class TrainingConfig(BaseConfig):
    optim_name: str
    loss_name: str
    lr: float = 0.001
    early_stop: bool = False
    mode: str = 'max'
    delta: float = 0
    patience: int = 10
    epochs: int = 50
    nthreads: int = torch.get_num_threads()

    def __post_init__(self):

        # check whether the optimizer is currently supported
        self.optim_class = item_in_enum(self.optim_name, SupportedOptimizers)

        # check whether the loss function is currently supported
        self.loss_class = item_in_enum(self.loss_name, SupportedLossFunctions)

    def init_optimizer(self, model):

        # initialize the optimizer for the specified model
        optimizer = self.optim_class(model.parameters(), self.lr)

        return optimizer

    def init_loss_function(self):

        loss_function = self.loss_class()

        return loss_function


@dataclasses.dataclass
class NetworkTrainer(BaseConfig):
    dconfig: dict = dataclasses.field(default_factory=dict)
    sconfig: dict = dataclasses.field(default_factory=dict)
    mconfig: dict = dataclasses.field(default_factory=dict)
    tconfig: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        # whether to use the gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else
                                   "cpu")

        # instanciate the configurations
        self.dc = DatasetConfig(**self.dconfig)
        self.sc = SplitConfig(**self.sconfig)
        self.mc = ModelConfig(**self.mconfig)
        self.tc = TrainingConfig(**self.tconfig)

        # initialize the dataset to train the model on
        self.dataset = self.dc.init_dataset()

        # inialize the training, validation and test dataset
        (self.train_ds, self.valid_ds,
         self.test_ds) = self.sc.train_val_test_split(self.dataset)

        # create the dataloaders
        self._build_dataloaders()

        # initialize the model state files
        self.state_file, self.loss_state = self.mc.init_state(self.dataset)

        # initialize the model
        self.model = self.mc.init_model(self.dataset)

        # initialize the optimizer
        self.optimizer = self.tc.init_optimizer(self.model)

        # initialize the loss function
        self.loss_function = self.tc.init_loss_function()

        # whether to resume training from an existing model
        self.checkpoint_state, self.max_accuracy = self.mc.load_checkpoint(
            self.state_file, self.loss_state, self.model, self.optimizer)

    def train(self):

        print('------------------------- Training ---------------------------')

        # set the number of threads
        torch.set_num_threads(self.tc.nthreads)

        # instanciate early stopping class
        if self.tc.early_stop:
            es = EarlyStopping(self.tc.mode, self.tc.delta, self.tc.patience)
            print('Initializing early stopping ...')
            print('mode = {}, delta = {}, patience = {} epochs ...'
                  .format(self.tc.mode, self.tc.delta, self.tc.patience))

        # create dictionary of the observed losses and accuracies on the
        # training and validation dataset
        tshape = (len(self.train_dl), self.tc.epochs)
        vshape = (len(self.valid_dl), self.tc.epochs)
        training_state = {'tl': np.zeros(shape=tshape),
                          'ta': np.zeros(shape=tshape),
                          'vl': np.zeros(shape=vshape),
                          'va': np.zeros(shape=vshape)
                          }

        # send the model to the gpu if available
        self.model = self.model.to(self.device)

        # initialize the training: iterate over the entire training data set
        for epoch in range(self.tc.epochs):

            # set the model to training mode
            print('Setting model to training mode ...')
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
                training_state['tl'][batch, epoch] = observed_loss

                # compute the gradients of the loss function w.r.t.
                # the network weights
                loss.backward()

                # update the weights
                self.optimizer.step()

                # calculate predicted class labels
                ypred = F.softmax(outputs, dim=1).argmax(dim=1)

                # calculate accuracy on current batch
                observed_accuracy = accuracy_function(ypred, labels)
                training_state['ta'][batch, epoch] = observed_accuracy

                # print progress
                print('Epoch: {:d}/{:d}, Mini-batch: {:d}/{:d}, Loss: {:.2f}, '
                      'Accuracy: {:.2f}'.format(epoch + 1,
                                                self.tc.epochs,
                                                batch + 1,
                                                len(self.train_dl),
                                                observed_loss,
                                                observed_accuracy))

            # update the number of epochs trained
            self.model.epoch += 1

            # whether to evaluate model performance on the validation set and
            # early stop the training process
            if self.tc.early_stop:

                # model predictions on the validation set
                vacc, vloss = self.predict()

                # append observed accuracy and loss to arrays
                training_state['va'][:, epoch] = vacc.squeeze()
                training_state['vl'][:, epoch] = vloss.squeeze()

                # metric to assess model performance on the validation set
                epoch_acc = vacc.squeeze().mean()

                # whether the model improved with respect to the previous epoch
                if es.increased(epoch_acc, self.max_accuracy, self.tc.delta):
                    self.max_accuracy = epoch_acc
                    # save model state if the model improved with
                    # respect to the previous epoch
                    _ = self.model.save(self.state_file,
                                        self.optimizer,
                                        self.dataset.use_bands,
                                        self.mc.state_path)

                    # save losses and accuracy
                    self._save_loss(training_state)

                # whether the early stopping criterion is met
                if es.stop(epoch_acc):
                    break

            else:
                # if no early stopping is required, the model state is saved
                # after each epoch
                _ = self.model.save(self.state_file,
                                    self.optimizer,
                                    self.dataset.use_bands,
                                    self.mc.state_path)

                # save losses and accuracy after each epoch
                self._save_loss(training_state)

        return training_state

    def predict(self):

        print('------------------------ Predicting --------------------------')

        # send the model to the gpu if available
        self.model = self.model.to(self.device)

        # set the model to evaluation mode
        print('Setting model to evaluation mode ...')
        self.model.eval()

        # create arrays of the observed losses and accuracies
        accuracies = np.zeros(shape=(len(self.valid_dl), 1))
        losses = np.zeros(shape=(len(self.valid_dl), 1))

        # iterate over the validation/test set
        print('Calculating accuracy on the validation set ...')
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
            print('Mini-batch: {:d}/{:d}, Accuracy: {:.2f}'
                  .format(batch + 1, len(self.valid_dl), acc))

        # calculate overall accuracy on the validation/test set
        print('After training for {:d} epochs, we achieved an overall '
              'accuracy of {:.2f}%  on the validation set!'
              .format(self.model.epoch, accuracies.mean() * 100))

        return accuracies, losses

    def _build_dataloaders(self):

        # the shape of a single tile
        self.tile_shape = (len(self.dataset.use_bands),
                           self.dataset.tile_size,
                           self.dataset.tile_size)

        # the training dataloader
        self.train_dl = None
        if len(self.train_ds) > 0:
            self.train_dl = DataLoader(self.train_ds,
                                       self.mc.batch_size,
                                       shuffle=True,
                                       drop_last=False)
        # the validation dataloader
        self.valid_dl = None
        if len(self.valid_ds) > 0:
            self.valid_dl = DataLoader(self.valid_ds,
                                       self.mc.batch_size,
                                       shuffle=True,
                                       drop_last=False)

        # the test dataloader
        self.test_dl = None
        if len(self.test_ds) > 0:
            self.test_dl = DataLoader(self.test_ds,
                                      self.mc.batch_size,
                                      shuffle=True,
                                      drop_last=False)

    def _save_loss(self, training_state):

        # save losses and accuracy
        if self.mc.checkpoint and self.checkpoint_state is not None:

            # append values from checkpoint to current training
            # state
            torch.save({
                k1: np.hstack([v1, v2]) for (k1, v1), (k2, v2) in
                zip(self.checkpoint_state.items(), training_state.items())
                if k1 == k2},
                self.loss_state)
        else:
            torch.save(training_state, self.loss_state)



    def __repr__(self):

        # representation string to print
        fs = self.__class__.__name__ + '(\n'

        # dataset
        fs += '    (dataset):\n        '
        fs += ''.join(repr(self.dataset)).replace('\n', '\n        ')

        # batch size
        fs += '\n    (batch):\n        '
        fs += '- batch size: {}\n        '.format(self.mc.batch_size)
        fs += '- tile shape (c, h, w): {}\n        '.format(self.tile_shape)
        fs += '- mini-batch shape (b, c, h, w): {}'.format(
            (self.mc.batch_size,) + self.tile_shape)

        # dataset split
        fs += '\n    (split):'
        fs += '\n        ' + repr(self.train_ds)
        fs += '\n        ' + repr(self.valid_ds)
        fs += '\n        ' + repr(self.test_ds)

        # model
        fs += '\n    (model):\n        '
        fs += ''.join(repr(self.model)).replace('\n', '\n        ')

        # optimizer
        fs += '\n    (optimizer):\n        '
        fs += ''.join(repr(self.optimizer)).replace('\n', '\n        ')
        fs += '\n)'

        return fs


class EarlyStopping(object):

    def __init__(self, mode='max', min_delta=0, patience=10):

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
        self.best = None

        # initialize early stopping flag
        self.early_stop = False

    def stop(self, metric):

        if self.best is not None:

            # if the metric improved, reset the epochs counter, else, advance
            if self.is_better(metric, self.best, self.min_delta):
                self.counter = 0
                self.best = metric
            else:
                self.counter += 1
                print('Early stopping counter: {}/{}'.format(self.counter,
                                                             self.patience))

            # if the metric did not improve over the last patience epochs,
            # the early stopping criterion is met
            if self.counter >= self.patience:
                print('Early stopping criterion met, exiting training ...')
                self.early_stop = True

        else:
            self.best = metric

        return self.early_stop

    def decreased(self, metric, best, min_delta):
        return metric < best - min_delta

    def increased(self, metric, best, min_delta):
        return metric > best + min_delta
