"""Unsupervised domain adaptation.

This module provides functions and classes to implement algorithms for
unsupervised domain adaptation.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import enum
import logging

# externals
import torch
import torch.nn as nn

# module level logger
LOGGER = logging.getLogger(__name__)

# global variable used to determine where to compute the domain adaptation loss
# within the model
UDA_POSITIONS = ['enc', 'dec', 'cla']


def coral(source, target, device='cpu'):
    """Correlation Alignment (CORAL) loss function.

    An implementation of `DeepCORAL`_ for unsupervised domain adaptation.

    Parameters
    ----------
    source : :py:class:`torch.Tensor`
        The source domain features.
    target : :py:class:`torch.Tensor`
        The target domain features.
    device : `str`, optional
        The device to compute on. The default is 'cpu'.

    Returns
    -------
    coral_loss : :py:class:`torch.Tensor`
        The distance in covariance between ``source`` and ``target``.

    .. _DeepCORAL:
        https://arxiv.org/abs/1607.01719

    """
    # shape of source and target features
    src_s = source.shape
    trg_s = target.shape

    # compute distance between covariances for each band iteratively
    coral_loss = 0
    for band in range(src_s[1]):

        # the current bands: flatten spatial dimesions
        src_band = source[:, band, ...].view(src_s[0], src_s[2] * src_s[3])
        trg_band = target[:, band, ...].view(trg_s[0], trg_s[2] * trg_s[3])

        # move tensors to gpu, if available
        src_band = src_band.to(device)
        trg_band = trg_band.to(device)

        # source and target covariance matrices
        src_cov = cov_coral(src_band)
        trg_cov = cov_coral(trg_band)

        # Frobenius norm: distance between covariance matrices
        coral_loss += torch.pow(src_cov - trg_cov, 2).sum().sqrt()

    # scale by dimensionality
    coral_loss /= 4 * (src_s[1] ** 2)

    return coral_loss


def cov_coral(M):
    """Compute the covariance matrix.

    Parameters
    ----------
    M : :py:class:`torch.Tensor`
        The two-dimensional feature matrix of shape (m, n).

    Returns
    -------
    cov : :py:class:`torch.Tensor`
        The covariance matrix.

    """
    # number of batches
    bs = M.size(0)

    # right hand side term
    rt = torch.matmul(torch.ones(1, bs), M)
    rt = torch.matmul(rt.t(), rt)

    # squared input matrix
    M2 = torch.matmul(M.t(), M)

    # covariance matrix
    cov = (M2 - rt / bs) / (bs - 1)

    # clear cache
    del bs, rt, M, M2

    return cov


class CoralLoss(nn.Module):
    """Correlation Alignment (CORAL) loss function class.

    :py:class:`torch.nn.Module` wrapper for :py:func:`pysegcnn.core.uda.coral`.

    Attributes
    ----------
    uda_lambda : `float`
        The weight of the domain adaptation.
    device : `str`
        The device to compute on.

    """

    def __init__(self, uda_lambda, device='cpu'):
        """Initialize.

        Parameters
        ----------
        uda_lambda : `float`
            The weight of the domain adaptation.
        device : `str`
            The device to compute on. The default is 'cpu'.

        """
        super().__init__()
        self.uda_lambda = uda_lambda
        self.device = device

    def forward(self, source, target):
        """Compute CORAL loss.

        Parameters
        ----------
        source : :py:class:`torch.Tensor`
            The source domain features.
        target : :py:class:`torch.Tensor`
            The target domain features.

        Returns
        -------
        coral_loss : :py:class:`torch.Tensor`
            The distance in covariance between ``source`` and ``target``.

        """
        return coral(source, target, device=self.device)

    def __repr__(self):
        """Representation.

        Returns
        -------
        fs : `str`
            Representation string.

        """
        # representation string to print
        fs = self.__class__.__name__ + '(lambda={:.2f})'.format(
            self.uda_lambda)
        return fs


class SupportedUdaMethods(enum.Enum):
    """Supported unsupervised domain adaptation methods."""

    coral = CoralLoss
