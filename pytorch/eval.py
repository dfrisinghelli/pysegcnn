# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:33:46 2020

@author: Daniel
"""

# externals
import torch
import torch.nn.functional as F


def predict(model, dataloader, optimizer, accuracy, state_file=None):

    # check whether a gpu is available for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the model state if provided
    if state_file is not None:
        model.load(optimizer, state_file)

    # set the model to evaluation mode
    model.eval()

    # initialize confusion matrix
    cm = torch.zeros(model.nclasses, model.nclasses)

    # number of batches in the validation set
    nbatches = int(len(dataloader.dataset) / dataloader.batch_size)

    # iterate over the validation/test set
    accuracies = []
    for batch, (inputs, labels) in enumerate(dataloader):

        # send the data to the gpu if available
        inputs = inputs.to(device)
        labels = labels.to(device)

        # calculate network outputs
        with torch.no_grad():
            outputs = model(inputs)

        # calculate predicted class labels
        pred = F.softmax(outputs, dim=1).argmax(dim=1)

        # calculate accuracy
        acc = accuracy(pred, labels)
        accuracies.append(acc)

        # print progress
        print('Batch: {:d}/{:d}, Accuracy: {:.2f}'.format(batch,
                                                          nbatches, acc))

        # update confusion matrix
        for ytrue, ypred in zip(labels.view(-1), pred.view(-1)):
            cm[ytrue.long(), ypred.long()] += 1

    # save confusion matrix and accuracies to file
    torch.save({'cm': cm, 'accuracy': accuracies},
               state.split('.pt')[0] + '_cm.pt')

    return cm


def accuracy_function(outputs, labels):
    return (outputs == labels).float().mean()
