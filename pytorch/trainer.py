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
from torch.utils.data import random_split, DataLoader


class InitNetwork(object):
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)


class NetworkTrainer(object):

    def __init__(self, model, dataset, loss_function, optimizer, batch_size=32,
                 tvratio=0.8, ttratio=1, seed=0):

        # the model to train
        self.model = model

        # the dataset to train the model on
        self.dataset = dataset

        # the training and validation dataset
        self.train_ds, self.valid_ds, self.test_ds = self.train_val_test_split(
            self.dataset, tvratio, ttratio, seed)

        # the batch size
        self.batch_size = batch_size

        # the training and validation dataloaders
        self.train_dl = DataLoader(self.train_ds, batch_size, shuffle=True)
        self.valid_dl = DataLoader(self.valid_ds, batch_size, shuffle=True)

        # the loss function to compute the model error
        self.loss_function = loss_function

        # the optimizer used to update the model weights
        self.optimizer = optimizer

        # whether to use the gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else
                                   "cpu")

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

    def train(self, state_path, state_file, epochs=1, resume=True,
              early_stop=True, nthreads=os.cpu_count(), **kwargs):

        # set the number of threads
        torch.set_num_threads(nthreads)

        # store path to model state
        self.state = os.path.join(state_path, state_file)
        self.loss_state = self.state.replace('.pt', '_loss.pt')

        # instanciate early stopping class
        if early_stop:
            es = EarlyStopping(**kwargs)

            # initial accuracy on the validation set
            max_accuracy = 0

        # whether to resume training from an existing model
        if os.path.exists(self.state) and resume:
            state = self.model.load(self.optimizer, state_file, state_path)
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
        losses = np.zeros(shape=(nbatches, epochs))
        accuracies = np.zeros(shape=(nbatches, epochs))

        # create arrays of the observed losses and accuracies on the
        # validation set
        vlosses = np.zeros(shape=(nvbatches, epochs))
        vaccuracies = np.zeros(shape=(nvbatches, epochs))

        # initialize the training: iterate over the entire training data set
        for epoch in range(epochs):

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
                      'Accuracy: {:.2f}'.format(epoch, epochs, batch, nbatches,
                                                losses[batch, epoch], acc))

            # update the number of epochs trained
            self.model.epoch += 1

            # whether to evaluate model performance on the validation set and
            # early stop the training process
            if early_stop:

                # model predictions on the validation set
                _, vacc, vloss = self.predict()

                # append observed accuracy and loss to arrays
                vaccuracies[:, epoch] = vacc.squeeze()
                vlosses[:, epoch] = vloss.squeeze()

                # metric to assess model performance on the validation set
                epoch_acc = vacc.squeeze().mean()

                # whether the model improved with respect to the previous epoch
                if epoch_acc > max_accuracy:
                    max_accuracy = epoch_acc
                    # save model state if the model improved with
                    # respect to the previous epoch
                    _ = self.model.save(self.optimizer, state_file, state_path)

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
                _ = self.model.save(self.optimizer, state_file, state_path)

            # save losses and accuracy after each epoch to file
            torch.save({'epoch': epoch,
                        'training_loss': losses,
                        'training_accuracy': accuracies,
                        'validation_loss': vlosses,
                        'validation_accuracy': vaccuracies},
                       self.loss_state)

        return losses, accuracies, vlosses, vaccuracies

    def predict(self, state_path=None, state_file=None, confusion=False):

        # load the model state if provided
        if state_file is not None:
            state = self.model.load(self.optimizer, state_file, state_path)

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
        if state_file is not None and confusion:
            torch.save({'cm': cm}, state.replace('.pt', '_cm.pt'))

        return cm, accuracies, losses


class EarlyStopping(object):

    def __init__(self, mode='min', min_delta=0, patience=5):

        # check if mode is correctly specified
        if mode not in ['min', 'max']:
            raise ValueError('Mode "{}" not supported. '
                             'Mode is either "min" (check whether the metric '
                             'decreased, e.g. loss) or "max" (check whether '
                             'the metric increased, e.g. accuracy).'
                             .format(mode))

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
