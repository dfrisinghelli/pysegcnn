# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:31:36 2020

@author: Daniel
"""
# builtins
import os
import sys

# externals
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

# append path to local files to the python search path
sys.path.append('..')

# local modules
from pytorch.dataset import SparcsDataset, Cloud95Dataset
from pytorch.constants import SparcsLabels, Cloud95Labels


class NetworkTrainer(object):

    def __init__(self, config):

        # the configuration file as defined in main.config.py
        for k, v in config.items():
            setattr(self, k, v)

    def initialize(self):

        # check which dataset the model is trained on
        if self.dataset_name == 'Sparcs':
            # instanciate the SparcsDataset
            self.dataset = SparcsDataset(self.dataset_path,
                                         use_bands=self.bands,
                                         tile_size=self.tile_size)
        elif self.dataset_name == 'Cloud95':
            # instanciate the Cloud95Dataset
            self.dataset = Cloud95Dataset(self.dataset_path,
                                          use_bands=self.bands,
                                          tile_size=self.tile_size,
                                          exclude=self.patches)
        else:
            raise ValueError('{} is not a valid dataset. Available datasets '
                             'are "Sparcs" and "Cloud95".'
                             .format(self.dataset_name))

        # print the bands used for the segmentation
        print('------------------------ Input bands -------------------------')
        print(*['Band {}: {}'.format(i, b) for i, b in
                enumerate(self.dataset.use_bands)], sep='\n')
        print('--------------------------------------------------------------')

        # print the classes of interest
        print('-------------------------- Classes ---------------------------')
        print(*['Class {}: {}'.format(k, v['label']) for k, v in
                self.dataset.labels.items()], sep='\n')
        print('--------------------------------------------------------------')

        # instanciate the segmentation network
        print('------------------- Network architecture ---------------------')
        if self.pretrained:
            self.model = self.from_pretrained()
        else:
            self.model = self.net(in_channels=len(self.dataset.use_bands),
                                  nclasses=len(self.dataset.labels),
                                  filters=self.filters,
                                  skip=self.skip_connection,
                                  **self.kwargs)
        print(self.model)
        print('--------------------------------------------------------------')

        # the training and validation dataset
        print('------------------------ Dataset split -----------------------')
        self.train_ds, self.valid_ds, self.test_ds = self.train_val_test_split(
            self.dataset, self.tvratio, self.ttratio, self.seed)
        print('--------------------------------------------------------------')

        # the training and validation dataloaders
        self.train_dl = DataLoader(self.train_ds,
                                   self.batch_size,
                                   shuffle=True,
                                   drop_last=True)
        self.valid_dl = DataLoader(self.valid_ds,
                                   self.batch_size,
                                   shuffle=True,
                                   drop_last=True)

        # the optimizer used to update the model weights
        self.optimizer = self.optimizer(self.model.parameters(), self.lr)

        # whether to use the gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else
                                   "cpu")

        # file to save model state to
        # format: networkname_datasetname_t(tilesize)_b(batchsize)_bands.pt
        bformat = ''.join([b[0] for b in self.bands]) if self.bands else 'all'
        self.state_file = ('{}_{}_t{}_b{}_{}.pt'
                           .format(self.model.__class__.__name__,
                                   self.dataset.__class__.__name__,
                                   self.tile_size,
                                   self.batch_size,
                                   bformat))

        # path to model state
        self.state = os.path.join(self.state_path, self.state_file)

        # path to model loss/accuracy
        self.loss_state = self.state.replace('.pt', '_loss.pt')


    def from_pretrained(self):

        # name of the dataset the pretrained model was trained on
        dataset_name = self.pretrained_model.split('_')[1]

        # input bands of the pretrained model
        bands = self.pretrained_model.split('_')[-1].split('.')[0]

        if dataset_name == SparcsDataset.__name__:

            # number of input channels
            in_channels = len(bands) if bands != 'all' else 10

            # instanciate pretrained model architecture
            model = self.net(in_channels=in_channels,
                             nclasses=len(SparcsLabels),
                             filters=self.filters,
                             skip=self.skip_connection,
                             **self.kwargs)

        if dataset_name == Cloud95Dataset.__name__:

            # number of input channels
            in_channels = len(bands) if bands != 'all' else 4

            # instanciate pretrained model architecture
            model = self.net(in_channels=in_channels,
                             nclasses=len(Cloud95Labels),
                             filters=self.filters,
                             skip=self.skip_connection,
                             **self.kwargs)

        # load pretrained model weights
        model.load(self.pretrained_model, inpath=self.state_path)

        return model


    def ds_len(self, ds, ratio):
        return int(np.round(len(ds) * ratio))

    def train_val_test_split(self, ds, tvratio, ttratio=1, seed=0):

        # set the random seed for reproducibility
        torch.manual_seed(seed)

        # length of the training and validation dataset
        trav_len = self.ds_len(ds, ttratio)

        # length of the test dataset
        test_len = self.ds_len(ds, 1 - ttratio)

        # split dataset into training and test set
        # (ttratio * 100) % will be used for training and validation
        train_val_ds, test_ds = random_split(ds, (trav_len, test_len))

        # length of the training set
        train_len = self.ds_len(train_val_ds, tvratio)

        # length of the validation dataset
        valid_len = self.ds_len(train_val_ds, 1 - tvratio)

        # split the training set into training and validation set
        train_ds, valid_ds = random_split(train_val_ds, (train_len, valid_len))

        # print the dataset ratios
        print(*['{} set: {:.2f}%'.format(k, v * 100) for k, v in
                {'Training': ttratio * tvratio,
                 'Validation': ttratio * (1 - tvratio),
                 'Test': 1 - ttratio}.items()], sep='\n')

        return train_ds, valid_ds, test_ds

    def accuracy_function(self, outputs, labels):
        return (outputs == labels).float().mean()

    def train(self):

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

        # whether to resume training from an existing model
        if os.path.exists(self.state) and self.checkpoint:
            state = self.model.load(self.state_file, self.optimizer,
                                    self.state_path)
            print('Resuming training from {} ...'.format(state))
            print('Model epoch: {:d}'.format(self.model.epoch))

        # send the model to the gpu if available
        self.model = self.model.to(self.device)

        # number of batches in the validation set
        nvbatches = int(len(self.valid_ds) / self.batch_size)

        # number of batches in the training set
        nbatches = int(len(self.train_ds) / self.batch_size)

        # create arrays of the observed losses and accuracies on the
        # training set
        losses = np.zeros(shape=(nbatches, self.epochs))
        accuracies = np.zeros(shape=(nbatches, self.epochs))

        # create arrays of the observed losses and accuracies on the
        # validation set
        vlosses = np.zeros(shape=(nvbatches, self.epochs))
        vaccuracies = np.zeros(shape=(nvbatches, self.epochs))

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
                losses[batch, epoch] = loss.detach().numpy().item()

                # compute the gradients of the loss function w.r.t.
                # the network weights
                loss.backward()

                # update the weights
                self.optimizer.step()

                # calculate predicted class labels
                ypred = F.softmax(outputs, dim=1).argmax(dim=1)

                # calculate accuracy on current batch
                acc = self.accuracy_function(ypred, labels)
                accuracies[batch, epoch] = acc

                # print progress
                print('Epoch: {:d}/{:d}, Batch: {:d}/{:d}, Loss: {:.2f}, '
                      'Accuracy: {:.2f}'.format(epoch, self.epochs, batch,
                                                nbatches, losses[batch, epoch],
                                                acc))

            # update the number of epochs trained
            self.model.epoch += 1

            # whether to evaluate model performance on the validation set and
            # early stop the training process
            if self.early_stop:

                # model predictions on the validation set
                _, vacc, vloss = self.predict()

                # append observed accuracy and loss to arrays
                vaccuracies[:, epoch] = vacc.squeeze()
                vlosses[:, epoch] = vloss.squeeze()

                # metric to assess model performance on the validation set
                epoch_acc = vacc.squeeze().mean()

                # whether the model improved with respect to the previous epoch
                if es.increased(epoch_acc, max_accuracy, self.delta):
                    max_accuracy = epoch_acc
                    # save model state if the model improved with
                    # respect to the previous epoch
                    _ = self.model.save(self.optimizer, self.state_file,
                                        self.state_path)

                # whether the early stopping criterion is met
                if es.stop(epoch_acc):

                    # save losses and accuracy before exiting training
                    torch.save({'epoch': epoch,
                                'training_loss': losses,
                                'training_accuracy': accuracies,
                                'validation_loss': vlosses,
                                'validation_accuracy': vaccuracies},
                               self.loss_state)

                    break

            else:
                # if no early stopping is required, the model state is saved
                # after each epoch
                _ = self.model.save(self.optimizer, self.state_file,
                                    self.state_path)

            # save losses and accuracy after each epoch to file
            torch.save({'epoch': epoch,
                        'training_loss': losses,
                        'training_accuracy': accuracies,
                        'validation_loss': vlosses,
                        'validation_accuracy': vaccuracies},
                       self.loss_state)

        return losses, accuracies, vlosses, vaccuracies

    def predict(self, pretrained=False, confusion=False):

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

        # number of batches in the validation set
        nbatches = int(len(self.valid_ds) / self.batch_size)

        # create arrays of the observed losses and accuracies
        accuracies = np.zeros(shape=(nbatches, 1))
        losses = np.zeros(shape=(nbatches, 1))

        # iterate over the validation/test set
        print('Calculating accuracy on validation set ...')
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
            acc = self.accuracy_function(pred, labels)
            accuracies[batch, 0] = acc

            # print progress
            print('Batch: {:d}/{:d}, Accuracy: {:.2f}'.format(batch,
                                                              nbatches, acc))

            # update confusion matrix
            if confusion:
                for ytrue, ypred in zip(labels.view(-1), pred.view(-1)):
                    cm[ytrue.long(), ypred.long()] += 1

        # calculate overall accuracy on the validation set
        print('Current mean accuracy on the validation set: {:.2f}%'
              .format(accuracies.mean() * 100))

        # save confusion matrix and accuracies to file
        if pretrained and confusion:
            torch.save({'cm': cm}, state.replace('.pt', '_cm.pt'))

        return cm, accuracies, losses


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
