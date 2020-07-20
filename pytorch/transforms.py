# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:28:18 2020

@author: Daniel
"""

# externals
import numpy as np
from scipy import ndimage


class Transform(object):

    def __init__(self):
        pass

    def __call__(self, image):
        raise NotImplementedError


class VariantTransform(Transform):

    def __init__(self):

        # requires transformation on the ground thruth
        self.invariant = False


class InvariantTransform(Transform):

    def __init__(self):

        # transformation on the ground truth not required
        self.invariant = True


class FlipLr(VariantTransform):

    def __init__(self):
        super().__init__()

    def __call__(self, image):
        return np.asarray(image)[..., ::-1]

    def __repr__(self):
        return self.__class__.__name__


class FlipUd(VariantTransform):

    def __init__(self):
        super().__init__()

    def __call__(self, image):
        return np.asarray(image)[..., ::-1, :]

    def __repr__(self):
        return self.__class__.__name__


class Rotate(VariantTransform):

    def __init__(self, angle):
        self.angle = angle
        super().__init__()

    def __call__(self, image):

        # check dimension of input image
        ndim = np.asarray(image).ndim

        # axes defining the rotational plane
        rot_axes = (0, 1)
        if ndim > 2:
            rot_axes = (ndim - 2, ndim - 1)

        return ndimage.rotate(image, self.angle, axes=rot_axes, reshape=False)

    def __repr__(self):
        return self.__class__.__name__ + '(angle = {})'.format(self.angle)


class Noise(InvariantTransform):

    def __init__(self, mode, mean=0, var=0.05):
        super().__init__()

        # check which kind of noise to apply
        modes = ['gaussian', 'speckle']
        if mode not in modes:
            raise ValueError('Supported noise types are: {}.'.format(modes))
        self.mode = mode

        # mean and variance of the gaussian distribution the noise signal is
        # sampled from
        self.mean = mean
        self.var = var

    def __call__(self, image):

        # generate gaussian noise
        noise = np.random.normal(self.mean, self.var, image.shape)

        if self.mode == 'gaussian':
            return (np.asarray(image) + noise).clip(0, 1)

        if self.mode == 'speckle':
            return (np.asarray(image) + np.asarray(image) * noise).clip(0, 1)

    def __repr__(self):
        return self.__class__.__name__ + ('(mode = {}, mean = {}, var = {})'
                                          .format(self.mode, self.mean,
                                                  self.var))


class Augment(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, gt):

        # apply transformations to the input image in specified order
        for t in self.transforms:
            image = t(image)

            # check whether the transformations are invariant and if not, apply
            # the transformation also to the ground truth
            if not t.invariant:
                gt = t(gt)

        return image, gt

    def __repr__(self):
        fstring = self.__class__.__name__ + '('
        for t in self.transforms:
            fstring += '\n'
            fstring += '    {0}'.format(t)
        fstring += '\n)'
        return fstring
