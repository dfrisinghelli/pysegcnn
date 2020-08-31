"""Data augmentation.

This module provides classes implementing common image augmentation methods.

These methods may be used to artificially increase a dataset.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# externals
import numpy as np
from scipy import ndimage


class Transform(object):
    """Base class for an image transformation."""

    def __init__(self):
        pass

    def __call__(self, image):
        """Apply transformation.

        Parameters
        ----------
        image : :py:class:`numpy.ndarray`
            The image to transform.

        Raises
        ------
        NotImplementedError
            Raised if :py:class:`pysegcnn.core.transforms.Transform` is not
            inherited.

        """
        raise NotImplementedError


class VariantTransform(Transform):
    """Base class for a spatially variant transformation.

    Transformation on the ground truth required.

    Attributes
    ----------
    invariant : `bool`
        Whether the transformation is spatially invariant.

    """

    def __init__(self):

        # transformation on the ground truth required
        self.invariant = False


class InvariantTransform(Transform):
    """Base class for a spatially invariant transformation.

    Transformation on the ground truth not required.

    Attributes
    ----------
    invariant : `bool`
        Whether the transformation is spatially invariant.

    """

    def __init__(self):

        # transformation on the ground truth not required
        self.invariant = True


class FlipLr(VariantTransform):
    """Flip an image horizontally.

    Attributes
    ----------
    p : `float`
        The probability to apply the transformation.
    applied : `bool`
        Whether the transformation was applied.

    """

    def __init__(self, p=0.5):
        """Initialize.

        Parameters
        ----------
        p : `float`, optional
            The probability to apply the transformation. The default is `0.5`.

        """
        super().__init__()
        # the probability to apply the transformation
        self.p = p

    def __call__(self, image):
        """Apply transformation.

        Parameters
        ----------
        image : :py:class:`numpy.ndarray`
            The image to transform.

        Returns
        -------
        transform : :py:class:`numpy.ndarray`
            The transformed image.

        """
        if np.random.random(1) < self.p:
            # transformation applied
            self.applied = True
            return np.asarray(image)[..., ::-1]

        # transformation not applied
        self.applied = False
        return np.asarray(image)

    def __repr__(self):
        """Representation of the transformation.

        Returns
        -------
        repr : `str`
            Representation string.

        """
        return self.__class__.__name__ + '(p = {})'.format(self.p)


class FlipUd(VariantTransform):
    """Flip an image vertically.

    Attributes
    ----------
    p : `float`
        The probability to apply the transformation.
    applied : `bool`
        Whether the transformation was applied.

    """

    def __init__(self, p=0.5):
        """Initialize.

        Parameters
        ----------
        p : `float`, optional
            The probability to apply the transformation. The default is `0.5`.

        """
        super().__init__()
        # the probability to apply the transformation
        self.p = p

    def __call__(self, image):
        """Apply transformation.

        Parameters
        ----------
        image : :py:class:`numpy.ndarray`
            The image to transform.

        Returns
        -------
        transform : :py:class:`numpy.ndarray`
            The transformed image.

        """
        if np.random.random(1) < self.p:
            # transformation applied
            self.applied = True
            return np.asarray(image)[..., ::-1, :]

        # transformation not applied
        self.applied = False
        return np.asarray(image)

    def __repr__(self):
        """Representation of the transformation.

        Returns
        -------
        repr : `str`
            Representation string.

        """
        return self.__class__.__name__ + '(p = {})'.format(self.p)


class Rotate(VariantTransform):
    """Rotate an image in the spatial plane.

    .. important::

        If the input array has more then two dimensions, the spatial dimensions
        are assumed to be the last two dimensions of the array.

    Attributes
    ----------
    p : `float`
        The probability to apply the transformation.
    applied : `bool`
        Whether the transformation was applied.
    angle : `float`
        The rotation angle in degrees.

    """

    def __init__(self, angle, p=0.5):
        """Initialize.

        Parameters
        ----------
        angle : `float`
            The rotation angle in degrees.
        p : `float`, optional
            The probability to apply the transformation. The default is `0.5`.

        """
        super().__init__()

        # the rotation angle
        self.angle = angle

        # the probability to apply the transformation
        self.p = p

    def __call__(self, image):
        """Apply transformation.

        Parameters
        ----------
        image : :py:class:`numpy.ndarray`
            The image to transform.

        Returns
        -------
        transform : :py:class:`numpy.ndarray`
            The transformed image.

        """
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
        """Representation of the transformation.

        Returns
        -------
        repr : `str`
            Representation string.

        """
        return self.__class__.__name__ + '(angle = {}, p = {})'.format(
            self.angle, self.p)


class Noise(InvariantTransform):
    """Add gaussian noise to an image.

    Supported modes are:

        - 'add' : ``image = image + noise``
        - 'speckle' : ``image = image + image * noise``

    Attributes
    ----------
    modes : `list` [`str`]
        The supported modes.
    mode : `str`
        The mode to add the noise.
    mean : `float`
        The mean of the gaussian distribution.
    var : `float`
        The variance of the gaussian distribution.
    p : `float`
        The probability to apply the transformation.
    applied : `bool`
        Whether the transformation was applied.
    exclude : `list` [`float`]
        Values for which the noise is not added.

    """

    # supported modes
    modes = ['add', 'speckle']

    def __init__(self, mode, mean=0, var=0.05, p=0.5, exclude=[]):
        """Initialize.

        Parameters
        ----------
        mode : `str`
            The mode to add the noise.
        mean : `float`, optional
            The mean of the gaussian distribution from which the noise is
            sampled. The default is `0`.
        var : `float`, optional
            The variance of the gaussian distribution from which the noise is
            sampled. The default is `0.05`.
        p : `float`, optional
            The probability to apply the transformation. The default is `0.5`.
        exclude : `list` [`float`] or `list` [`int`], optional
            Values for which the noise is not added. Useful for pixels
            resulting from image padding. The default is `[]`.

        Raises
        ------
        ValueError
            Raised if ``mode`` is not supported.

        """
        super().__init__()

        # check which kind of noise to apply
        if mode not in self.modes:
            raise ValueError('Supported noise types are: {}.'
                             .format(self.modes))
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
        """Apply transformation.

        Parameters
        ----------
        image : :py:class:`numpy.ndarray`
            The image to transform.

        Returns
        -------
        transform : :py:class:`numpy.ndarray`
            The transformed image.

        """
        if np.random.random(1) < self.p:

            # transformation applied
            self.applied = True

            # generate gaussian noise
            noise = np.random.normal(self.mean, self.var, image.shape)

            # check which values should not be modified by adding noise
            for val in self.exclude:
                noise[image == val] = 0

            if self.mode == 'gaussian':
                return (np.asarray(image) + noise).clip(0, 1)

            if self.mode == 'speckle':
                return (np.asarray(image) +
                        np.asarray(image) * noise).clip(0, 1)

        # transformation not applied
        self.applied = False
        return np.asarray(image)

    def __repr__(self):
        """Representation of the transformation.

        Returns
        -------
        repr : `str`
            Representation string.

        """
        return self.__class__.__name__ + ('(mode = {}, mean = {}, var = {}, '
                                          'p = {})'
                                          .format(self.mode, self.mean,
                                                  self.var, self.p))


class Augment(object):
    """Apply a sequence of transformations.

    Container class applying each transformation in ``transforms`` in order.

    Attributes
    ----------
    transforms : `list` or `tuple`
        The transformations to apply.

    """

    def __init__(self, transforms):
        """Initialize.

        Parameters
        ----------
        transforms : `list` or `tuple`
            A list of instances of
            :py:class:`pysegcnn.core.transforms.VariantTransform`
            or :py:class:`pysegcnn.core.transforms.InvariantTransform`.

        """
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, image, gt):
        """Apply a sequence of transformations to ``image``.

        For each spatially variant transformation, the ground truth ``gt`` is
        transformed respectively.

        Parameters
        ----------
        image : :py:class:`numpy.ndarray`
            The input image.
        gt : :py:class:`numpy.ndarray`
            The corresponding ground truth of ``image``.

        Returns
        -------
        image : :py:class:`numpy.ndarray`
            The transformed image.
        gt : :py:class:`numpy.ndarray`
            The transformed ground truth.

        """
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
        """Representation.

        Returns
        -------
        fs : `str`
            Representation string.

        """
        fs = self.__class__.__name__ + '('
        for t in self.transforms:
            fs += '\n'
            fs += '    {0}'.format(t)
        fs += '\n)'
        return fs
