# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:31:36 2020

@author: Daniel
"""

# externals
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split


def train(model, dataloader, loss_function, optimizer, accuracy, state_file,
          epochs=1, nthreads=1):

    # set the number of threads
    torch.set_num_threads(nthreads)

    # check whether a gpu is available for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # send the model to the gpu if available
    model = model.to(device)

    # set the model to training mode
    model.train()

    # number of batches per epoch
    nbatches = int(len(dataloader.dataset) / dataloader.batch_size)

    # initialize the training: iterate over the entire training data set
    for epoch in range(epochs):

        # create a list of the observed losses and accuracies
        losses = []
        accuracies = []

        # iterate over the dataloader object
        for batch, (inputs, labels) in enumerate(dataloader):

            # send the data to the gpu if available
            inputs = inputs.to(device)
            labels = labels.to(device)

            # reset the gradients
            optimizer.zero_grad()

            # perform forward pass
            outputs = model(inputs)

            # compute loss
            loss = loss_function(outputs, labels.long())
            losses.append(loss.detach().numpy().item())

            # compute the gradients of the loss function w.r.t.
            # the network weights
            loss.backward()

            # update the weights
            optimizer.step()

            # calculate predicted class labels
            ypred = F.softmax(outputs, dim=1).argmax(dim=1)

            # calculate accuracy
            acc = accuracy(ypred, labels)
            accuracies.append(acc)

            # print progress
            print('Epoch: {:d}/{:d}, Batch: {:d}/{:d}, Loss: {:.2f}, '
                  'Accuracy: {:.2f}'.format(epoch, epochs, batch, nbatches,
                                            loss.detach().numpy().item(), acc))

        # update the number of epochs trained
        model.epoch += 1

        # save model state to file
        state = model.save(optimizer, state_file)

        # save losses and accuracy to file
        torch.save({'loss': losses, 'accuracy': accuracies},
                   state.split('.pt')[0] + '_loss.pt')

    return losses, accuracies


def ds_len(ds, ratio):
    return int(np.round(len(ds) * ratio))


def train_test_split(ds, ratio, seed=0):

    # set the random seed for reproducibility
    torch.manual_seed(seed)

    # length of the training and validation data set
    train_len = ds_len(ds, ratio)

    # length of the test data set
    test_len = ds_len(ds, 1 - ratio)

    # split dataset into training and test set
    # (ratio * 100) % will be used for training and validation
    train_ds, test_ds = random_split(ds, (train_len, test_len))

    return train_ds, test_ds
