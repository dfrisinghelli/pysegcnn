# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:31:36 2020

@author: Daniel
"""
# builtins
from __future__ import absolute_import
import os

# externals
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

# local modules
from pytorch.dataset import SparcsDataset, Cloud95Dataset, SupportedDatasets
from pytorch.layers import Conv2dSame


class NetworkTrainer(object):

    def __init__(self, config):

        # the configuration file as defined in main.config.py
        for k, v in config.items():
            setattr(self, k, v)

        # whether to use the gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else
                                   "cpu")

        # initialize the dataset to train the model on
        self._init_dataset()

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
        model = self.net(**model_state['params'], **model_state['kwargs'])

        # load pretrained model weights
        model.load(self.pretrained_model, inpath=self.state_path)

        # reset model epoch to 0, since the model is trained on a different
        # dataset
        model.epoch = 0

        # adjust the classification layer to the number of classes of the
        # current dataset
        model.classifier = Conv2dSame(in_channels=filters[0],
                                      out_channels=len(self.dataset.labels),
                                      kernel_size=1)

        # adjust the number of classes in the model
        model.nclasses = len(self.dataset.labels)

        return model

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

            # initial accuracy on the validation set
            max_accuracy = 0

        # create dictionary of the observed losses and accuracies on the
        # training and validation dataset
        tshape = (len(self.train_dl), self.epochs)
        vshape = (len(self.valid_dl), self.epochs)
        training_state = {'tl': np.zeros(shape=tshape),
                          'ta': np.zeros(shape=tshape),
                          'vl': np.zeros(shape=vshape),
                          'va': np.zeros(shape=vshape)
                          }

        # whether to resume training from an existing model
        checkpoint_state = None
        if os.path.exists(self.state) and self.checkpoint:
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
                      'Accuracy: {:.2f}'.format(epoch + 1, self.epochs,
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
                _, vacc, vloss = self.predict()

                # append observed accuracy and loss to arrays
                training_state['va'][:, epoch] = vacc.squeeze()
                training_state['vl'][:, epoch] = vloss.squeeze()

                # metric to assess model performance on the validation set
                epoch_acc = vacc.squeeze().mean()

                # whether the model improved with respect to the previous epoch
                if es.increased(epoch_acc, max_accuracy, self.delta):
                    max_accuracy = epoch_acc
                    # save model state if the model improved with
                    # respect to the previous epoch
                    _ = self.model.save(self.state_file,
                                        self.optimizer,
                                        self.bands,
                                        self.state_path)

                    # save losses and accuracy
                    self._save_loss(training_state,
                                    self.checkpoint,
                                    checkpoint_state)

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
                                checkpoint_state)

        return training_state

    def predict(self, pretrained=False, confusion=False, test=False):

        print('------------------------ Predicting --------------------------')

        # load the model state if evaluating a pretrained model is required
        if pretrained:
            state = self.model.load(self.state_file, self.optimizer,
                                    self.state_path)

        # send the model to the gpu if available
        self.model = self.model.to(self.device)

        # set the model to evaluation mode
        print('Setting model to evaluation mode ...')
        self.model.eval()

        # initialize confusion matrix
        cm = torch.zeros(self.model.nclasses, self.model.nclasses)

        # check which dataset to test the model on, either the validation or
        # the test set
        if self.valid_dl is None and self.test_dl is None:
            raise ValueError('Can not evaluate model performance: validation '
                             'and test set not specified.')
        dataloader = self.valid_dl
        set_name = 'validation'
        if test:
            # if the test set is empty, default to validation set
            if self.test_dl is not None:
                dataloader = self.test_dl
                set_name = 'test'
            else:
                print('You requested to evaluate the model on the test set, '
                      'but no test set is available. Falling back to evaluate '
                      'the model on the validation set ...')

        # create arrays of the observed losses and accuracies
        accuracies = np.zeros(shape=(len(dataloader), 1))
        losses = np.zeros(shape=(len(dataloader), 1))

        # iterate over the validation/test set
        print('Calculating accuracy on the {} set ...'.format(set_name))
        for batch, (inputs, labels) in enumerate(dataloader):

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
                  .format(batch + 1, len(dataloader), acc))

            # update confusion matrix
            if confusion:
                for ytrue, ypred in zip(labels.view(-1), pred.view(-1)):
                    cm[ytrue.long(), ypred.long()] += 1

        # calculate overall accuracy on the validation/test set
        acc = (cm.diag().sum() / cm.sum()).numpy().item()
        print('After training for {:d} epochs, we achieved an overall '
              'accuracy of {:.2f}%  on the {} set!'
              .format(self.model.epoch, accuracies.mean() * 100, set_name))

        # save confusion matrix and accuracies to file
        if pretrained and confusion:
            torch.save({'cm': cm}, state.replace('.pt',
                                                 '_cm_{}.pt'.format(set_name)))

        return cm, accuracies, losses

    def _init_dataset(self):

        # check whether the dataset is currently supported
        self.dataset = None
        for dataset in SupportedDatasets:
            if self.dataset_name == dataset.name:
                self.dataset = dataset.value['class'](
                    self.dataset_path,
                    use_bands=self.bands,
                    tile_size=self.tile_size,
                    sort=self.sort,
                    transforms=self.transforms,
                    pad=self.pad,
                    cval=self.cval,
                    gt_pattern=self.gt_pattern)

        if self.dataset is None:
            raise ValueError('{} is not a valid dataset. '
                             .format(self.dataset_name) +
                             'Available datasets are: \n' +
                             '\n'.join(name for name, _ in
                                       SupportedDatasets.__members__.items()))

        # the training, validation and dataset
        self.train_ds, self.valid_ds, self.test_ds = random_tvt_split(
            self.dataset, self.tvratio, self.ttratio, self.seed)

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

        # instanciate the segmentation network
        if self.pretrained:
            self.model = self.from_pretrained()
        else:
            self.model = self.net(in_channels=len(self.dataset.use_bands),
                                  nclasses=len(self.dataset.labels),
                                  filters=self.filters,
                                  skip=self.skip_connection,
                                  **self.kwargs)

        # the optimizer used to update the model weights
        self.optimizer = self.optimizer(self.model.parameters(), self.lr)

        # file to save model state to
        # format: networkname_datasetname_t(tilesize)_b(batchsize)_bands.pt
        bformat = ''.join([b[0] for b in self.bands]) if self.bands else 'all'
        self.state_file = ('{}_{}_t{}_b{}_{}.pt'
                           .format(self.model.__class__.__name__,
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
        fs += '    (bands):\n        '

        # bands used for the segmentation
        fs += '\n        '.join('- Band {}: {}'.format(i, b) for i, b in
                                enumerate(self.dataset.use_bands))

        # classes of interest
        fs += '\n    (classes):\n        '
        fs += '\n        '.join('- Class {}: {}'.format(k, v['label']) for
                                k, v in self.dataset.labels.items())

        # batch size
        fs += '\n    (batch):\n        '
        fs += '- batch size: {}\n        '.format(self.batch_size)
        fs += '- batch shape (b, h, w): {}'.format(self.batch_shape)

        # dataset split
        fs += '\n    (dataset):\n        '
        fs += '\n        '.join(
            '- {}: {:d} batches ({:.2f}%)'
            .format(k, v[0], v[1] * 100) for k, v in
            {'Training': (len(self.train_ds), self.ttratio * self.tvratio),
             'Validation': (len(self.valid_ds),
                            self.ttratio * (1 - self.tvratio)),
             'Test': (len(self.test_ds), 1 - self.ttratio)}.items())

        # model
        fs += '\n    (model):\n        '
        fs += ''.join(self.model.__repr__()).replace('\n', '\n        ')
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


# function calculating prediction accuracy
def accuracy_function(outputs, labels):
    return (outputs == labels).float().mean()


# function calculating number of samples in a dataset given a ratio
def _ds_len(ds, ratio):
    return int(np.round(len(ds) * ratio))


# function randomly splitting a dataset into training, validation and test set
def random_tvt_split(ds, tvratio, ttratio=1, seed=0):

    # set the random seed for reproducibility
    torch.manual_seed(seed)

    # length of the training and validation dataset
    trav_len = _ds_len(ds, ttratio)

    # length of the test dataset
    test_len = _ds_len(ds, 1 - ttratio)

    # split dataset into training and test set
    # (ttratio * 100) % will be used for training and validation
    train_val_ds, test_ds = random_split(ds, (trav_len, test_len))

    # length of the training set
    train_len = _ds_len(train_val_ds, tvratio)

    # length of the validation dataset
    valid_len = _ds_len(train_val_ds, 1 - tvratio)

    # split the training set into training and validation set
    # (ttratio * tvratio) * 100 % will be used as the training dataset
    # (1 - ttratio * tvratio) * 100 % will be used as the validation dataset
    train_ds, valid_ds = random_split(train_val_ds, (train_len, valid_len))

    return train_ds, valid_ds, test_ds
