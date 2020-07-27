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

        # requires transformation on the ground truth
        self.invariant = False


class InvariantTransform(Transform):

    def __init__(self):

        # transformation on the ground truth not required
        self.invariant = True


class FlipLr(VariantTransform):

    def __init__(self, p=0.5):
        super().__init__()
        # the probability to apply the transformation
        self.p = p

    def __call__(self, image):
        if np.random.random(1) < self.p:
            # transformation applied
            self.applied = True
            return np.asarray(image)[..., ::-1]

        # transformation not applied
        self.applied = False
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__ + '(p = {})'.format(self.p)


class FlipUd(VariantTransform):

    def __init__(self, p=0.5):
        super().__init__()
        # the probability to apply the transformation
        self.p = p

    def __call__(self, image):
        if np.random.random(1) < self.p:
            # transformation applied
            self.applied = True
            return np.asarray(image)[..., ::-1, :]

        # transformation not applied
        self.applied = False
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__ + '(p = {})'.format(self.p)


class Rotate(VariantTransform):

    def __init__(self, angle, p=0.5):
        super().__init__()

        # the rotation angle
        self.angle = angle

        # the probability to apply the transformation
        self.p = p

    def __call__(self, image):

        if np.random.random(1) < self.p:

            # transformation applied
            self.applied = True

            # check dimension of input image
            ndim = np.asarray(image).ndim

            # axes defining the rotational plane
            rot_axes = (0, 1)
            if ndim > 2:
                rot_axes = (ndim - 2, ndim - 1)

            return ndimage.rotate(image, self.angle, axes=rot_axes,
                                  reshape=False)

        # transformation not applied
        self.applied = False
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__ + '(angle = {}, p = {})'.format(
            self.angle, self.p)


class Noise(InvariantTransform):

    def __init__(self, mode, mean=0, var=0.05, p=0.5, exclude=0):
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

        # the probability to apply the transformation
        self.p = p

        # the value in the image that should not be modified by the noise
        self.exclude = exclude

    def __call__(self, image):

        if np.random.random(1) < self.p:

            # transformation applied
            self.applied = True

            # generate gaussian noise
            noise = np.random.normal(self.mean, self.var, image.shape)

            # check which values should not be modified by adding noise
            noise[image == self.exclude] = 0

            if self.mode == 'gaussian':
                return (np.asarray(image) + noise).clip(0, 1)

            if self.mode == 'speckle':
                return (np.asarray(image) +
                        np.asarray(image) * noise).clip(0, 1)

        # transformation not applied
        self.applied = False
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__ + ('(mode = {}, mean = {}, var = {}, '
                                          'p = {})'
                                          .format(self.mode, self.mean,
                                                  self.var, self.p))


class Augment(object):

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, image, gt):

        # apply transformations to the input image in specified order
        for t in self.transforms:
            image = t(image)

            # check whether the transformations are invariant and if not, apply
            # the transformation also to the ground truth
            if not t.invariant and t.applied:
                # get the probability with which the transformation is applied
                p = t.p

                # overwrite the probability to 1, to ensure the same
                # transformation is applied on the ground truth
                t.p = 1

                # transform ground truth
                gt = t(gt)

                # reset probability to original value
                t.p = p

        return image, gt

    def __repr__(self):
        fstring = self.__class__.__name__ + '('
        for t in self.transforms:
            fstring += '\n'
            fstring += '    {0}'.format(t)
        fstring += '\n)'
        return fstring
