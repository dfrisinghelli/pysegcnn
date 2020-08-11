# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:31:36 2020

@author: Daniel
"""
# builtins
import os

# externals
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# locals
from pysegcnn.core.dataset import SupportedDatasets
from pysegcnn.core.layers import Conv2dSame
from pysegcnn.core.utils import img2np, accuracy_function
from pysegcnn.core.split import (RandomTileSplit, RandomSceneSplit, DateSplit,
                                 VALID_SPLIT_MODES)


class NetworkTrainer(object):

    def __init__(self, config):

        # the configuration file as defined in pysegcnn.main.config.py
        for k, v in config.items():
            setattr(self, k, v)

        # whether to use the gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else
                                   "cpu")

        # initialize the dataset to train the model on
        self._init_dataset()

        # initialize the model state files
        self._init_state()

        # initialize the model
        self._init_model()

    def from_pretrained(self):

        # load the pretrained model
        model_state = os.path.join(self.state_path, self.pretrained_model)
        if not os.path.exists(model_state):
            raise FileNotFoundError('Pretrained model {} does not exist.'
                                    .format(model_state))

        # load the model state
        model_state = torch.load(model_state)

        # get the input bands of the pretrained model
        bands = model_state['bands']

        # get the number of convolutional filters
        filters = model_state['params']['filters']

        # check whether the current dataset uses the correct spectral bands
        if self.bands != bands:
            raise ValueError('The bands of the pretrained network do not '
                             'match the specified bands: {}'
                             .format(self.bands))

        # instanciate pretrained model architecture
        model = self.model(**model_state['params'], **model_state['kwargs'])

        # load pretrained model weights
        model.load(self.pretrained_model, inpath=self.state_path)

        # reset model epoch to 0, since the model is trained on a different
        # dataset
        model.epoch = 0

        # adjust the number of classes in the model
        model.nclasses = len(self.dataset.labels)

        # adjust the classification layer to the number of classes of the
        # current dataset
        model.classifier = Conv2dSame(in_channels=filters[0],
                                      out_channels=model.nclasses,
                                      kernel_size=1)


        return model

    def from_checkpoint(self):

        # whether to resume training from an existing model
        if not os.path.exists(self.state):
            raise FileNotFoundError('Model checkpoint {} does not exist.'
                                    .format(self.state))

        # load the model state
        state = self.model.load(self.state_file, self.optimizer,
                                self.state_path)
        print('Resuming training from {} ...'.format(state))
        print('Model epoch: {:d}'.format(self.model.epoch))

        # load the model loss and accuracy
        checkpoint_state = torch.load(self.loss_state)

        # get all non-zero elements, i.e. get number of epochs trained
        # before the early stop
        checkpoint_state = {k: v[np.nonzero(v)].reshape(v.shape[0], -1) for
                            k, v in checkpoint_state.items()}

        # maximum accuracy on the validation set
        max_accuracy = checkpoint_state['va'][:, -1].mean().item()

        return checkpoint_state, max_accuracy

    def train(self):

        print('------------------------- Training ---------------------------')

        # set the number of threads
        torch.set_num_threads(self.nthreads)

        # instanciate early stopping class
        if self.early_stop:
            es = EarlyStopping(self.mode, self.delta, self.patience)
            print('Initializing early stopping ...')
            print('mode = {}, delta = {}, patience = {} epochs ...'
                  .format(self.mode, self.delta, self.patience))

        # create dictionary of the observed losses and accuracies on the
        # training and validation dataset
        tshape = (len(self.train_dl), self.epochs)
        vshape = (len(self.valid_dl), self.epochs)
        training_state = {'tl': np.zeros(shape=tshape),
                          'ta': np.zeros(shape=tshape),
                          'vl': np.zeros(shape=vshape),
                          'va': np.zeros(shape=vshape)
                          }

        # send the model to the gpu if available
        self.model = self.model.to(self.device)

        # initialize the training: iterate over the entire training data set
        for epoch in range(self.epochs):

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
                training_state['va'][:, epoch] = vacc.squeeze()
                training_state['vl'][:, epoch] = vloss.squeeze()

                # metric to assess model performance on the validation set
                epoch_acc = vacc.squeeze().mean()

                # whether the model improved with respect to the previous epoch
                if es.increased(epoch_acc, self.max_accuracy, self.delta):
                    self.max_accuracy = epoch_acc
                    # save model state if the model improved with
                    # respect to the previous epoch
                    _ = self.model.save(self.state_file,
                                        self.optimizer,
                                        self.bands,
                                        self.state_path)

                    # save losses and accuracy
                    self._save_loss(training_state,
                                    self.checkpoint,
                                    self.checkpoint_state)

                # whether the early stopping criterion is met
                if es.stop(epoch_acc):
                    break

            else:
                # if no early stopping is required, the model state is saved
                # after each epoch
                _ = self.model.save(self.state_file,
                                    self.optimizer,
                                    self.bands,
                                    self.state_path)

                # save losses and accuracy after each epoch
                self._save_loss(training_state,
                                self.checkpoint,
                                self.checkpoint_state)

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

    def _init_state(self):

        # file to save model state to
        # format: networkname_datasetname_t(tilesize)_b(batchsize)_bands.pt
        bformat = ''.join([b[0] for b in self.bands]) if self.bands else 'all'
        self.state_file = ('{}_{}_t{}_b{}_{}.pt'
                           .format(self.model.__name__,
                                   self.dataset.__class__.__name__,
                                   self.tile_size,
                                   self.batch_size,
                                   bformat))

        # check whether a pretrained model was used and change state filename
        # accordingly
        if self.pretrained:
            # add the configuration of the pretrained model to the state name
            self.state_file = (self.state_file.replace('.pt', '_') +
                               'pretrained_' + self.pretrained_model)

        # path to model state
        self.state = os.path.join(self.state_path, self.state_file)

        # path to model loss/accuracy
        self.loss_state = self.state.replace('.pt', '_loss.pt')

    def _init_dataset(self):

        # the dataset name
        self.dataset_name = os.path.basename(self.root_dir)

        # check whether the dataset is currently supported
        if self.dataset_name not in SupportedDatasets.__members__:
            raise ValueError('{} is not a valid dataset. '
                             .format(self.dataset_name) +
                             'Available datasets are: \n' +
                             '\n'.join(name for name, _ in
                                       SupportedDatasets.__members__.items()))
        else:
            self.dataset_class = SupportedDatasets.__members__[
                self.dataset_name].value

        # instanciate the dataset
        self.dataset = self.dataset_class(
                    self.root_dir,
                    use_bands=self.bands,
                    tile_size=self.tile_size,
                    sort=self.sort,
                    transforms=self.transforms,
                    pad=self.pad,
                    cval=self.cval,
                    gt_pattern=self.gt_pattern
                    )

        # the mode to split
        if self.split_mode not in VALID_SPLIT_MODES:
            raise ValueError('{} is not supported. Valid modes are {}, see '
                             'pysegcnn.main.config.py for a description of '
                             'each mode.'.format(self.split_mode,
                                                 VALID_SPLIT_MODES))
        if self.split_mode == 'random':
            self.subset = RandomTileSplit(self.dataset,
                                          self.ttratio,
                                          self.tvratio,
                                          self.seed)
        if self.split_mode == 'scene':
            self.subset = RandomSceneSplit(self.dataset,
                                           self.ttratio,
                                           self.tvratio,
                                           self.seed)
        if self.split_mode == 'date':
            self.subset = DateSplit(self.dataset,
                                    self.date,
                                    self.dateformat)

        # the training, validation and test dataset
        self.train_ds, self.valid_ds, self.test_ds = self.subset.split()

        # whether to drop training samples with a fraction of pixels equal to
        # the constant padding value self.cval >= self.drop
        if self.pad and self.drop:
            self._drop(self.train_ds)

        # the shape of a single batch
        self.batch_shape = (len(self.bands), self.tile_size, self.tile_size)

        # the training dataloader
        self.train_dl = None
        if len(self.train_ds) > 0:
            self.train_dl = DataLoader(self.train_ds,
                                       self.batch_size,
                                       shuffle=True,
                                       drop_last=False)
        # the validation dataloader
        self.valid_dl = None
        if len(self.valid_ds) > 0:
            self.valid_dl = DataLoader(self.valid_ds,
                                       self.batch_size,
                                       shuffle=True,
                                       drop_last=False)

        # the test dataloader
        self.test_dl = None
        if len(self.test_ds) > 0:
            self.test_dl = DataLoader(self.test_ds,
                                      self.batch_size,
                                      shuffle=True,
                                      drop_last=False)

    def _init_model(self):

        # initial accuracy on the validation set
        self.max_accuracy = 0

        # set the model checkpoint to None, overwritten when resuming
        # training from an existing model checkpoint
        self.checkpoint_state = None

        # case (1): build a model for the specified dataset
        if not self.pretrained and not self.checkpoint:

            # instanciate the model
            self.model = self.model(in_channels=len(self.dataset.use_bands),
                                    nclasses=len(self.dataset.labels),
                                    filters=self.filters,
                                    skip=self.skip_connection,
                                    **self.kwargs)

            # the optimizer used to update the model weights
            self.optimizer = self.optimizer(self.model.parameters(), self.lr)

        # case (2): using a pretrained model withouth existing checkpoint on
        #           a new dataset, i.e. transfer learning
        if self.pretrained and not self.checkpoint:
            # load pretrained model
            self.model = self.from_pretrained()

            # the optimizer used to update the model weights
            self.optimizer = self.optimizer(self.model.parameters(), self.lr)

        # case (3): using a pretrained model with existing checkpoint on the
        #           same dataset the pretrained model was trained on
        elif self.checkpoint:

            # instanciate the model
            self.model = self.model(in_channels=len(self.dataset.use_bands),
                                    nclasses=len(self.dataset.labels),
                                    filters=self.filters,
                                    skip=self.skip_connection,
                                    **self.kwargs)

            # the optimizer used to update the model weights
            self.optimizer = self.optimizer(self.model.parameters(), self.lr)

            # whether to resume training from an existing model checkpoint
            if self.checkpoint:
                (self.checkpoint_state,
                 self.max_accuracy) = self.from_checkpoint()

    # function to drop samples with a fraction of pixels equal to the constant
    # padding value self.cval >= self.drop
    def _drop(self, ds):

        # iterate over the scenes returned by self.compose_scenes()
        self.dropped = []
        for pos, i in enumerate(ds.indices):

            # the current scene
            s = ds.dataset.scenes[i]

            # the current tile in the ground truth
            tile_gt = img2np(s['gt'], self.tile_size, s['tile'],
                             self.pad, self.cval)

            # percent of pixels equal to the constant padding value
            npixels = (tile_gt[tile_gt == self.cval].size / tile_gt.size)

            # drop samples where npixels >= self.drop
            if npixels >= self.drop:
                print('Skipping scene {}, tile {}: {:.2f}% padded pixels ...'
                      .format(s['id'], s['tile'], npixels * 100))
                self.dropped.append(s)
                _ = ds.indices.pop(pos)

    def _save_loss(self, training_state, checkpoint=False,
                   checkpoint_state=None):

        # save losses and accuracy
        if checkpoint and checkpoint_state is not None:

            # append values from checkpoint to current training
            # state
            torch.save({
                k1: np.hstack([v1, v2]) for (k1, v1), (k2, v2) in
                zip(checkpoint_state.items(), training_state.items())
                if k1 == k2},
                self.loss_state)
        else:
            torch.save(training_state, self.loss_state)

    def __repr__(self):

        # representation string to print
        fs = self.__class__.__name__ + '(\n'

        # dataset
        fs += '    (dataset):\n        '
        fs += ''.join(self.dataset.__repr__()).replace('\n', '\n        ')

        # batch size
        fs += '\n    (batch):\n        '
        fs += '- batch size: {}\n        '.format(self.batch_size)
        fs += '- batch shape (b, h, w): {}'.format(self.batch_shape)

        # dataset split
        fs += '\n    (split):\n        '
        fs += ''.join(self.subset.__repr__()).replace('\n', '\n        ')

        # model
        fs += '\n    (model):\n        '
        fs += ''.join(self.model.__repr__()).replace('\n', '\n        ')

        # optimizer
        fs += '\n    (optimizer):\n        '
        fs += ''.join(self.optimizer.__repr__()).replace('\n', '\n        ')
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
