# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:28:18 2020

@author: Daniel
"""

# externals
import numpy as np
from scipy import ndimage


class Transformation(object):

    def __init__(self):
        pass

    def apply(self, image):
        raise NotImplementedError


class VariantTransformation(Transformation):

    def __init__(self):

        # requires transformation on the ground thruth
        self.invariant = False


class InvariantTransformation(Transformation):

    def __init__(self):

        # transformation on the ground truth not required
        self.invariant = True


class FlipLr(VariantTransformation):

    def __init__(self):
        super().__init__()

    def apply(self, image):
        return np.asarray(image)[..., ::-1]


class FlipUd(VariantTransformation):

    def __init__(self):
        super().__init__()

    def _transform(self, image):
        return np.asarray(image)[..., ::-1, :]


class Rotate(VariantTransformation):

    def __init__(self):
        super().__init__()

    def apply(self, image, angle):

        # check dimension of input image
        ndim = np.asarray(image).ndim

        # axes defining the rotational plane
        rot_axes = (0, 1)
        if ndim > 2:
            rot_axes = (ndim - 2, ndim - 1)

        return ndimage.rotate(image, angle, axes=rot_axes, reshape=False)
