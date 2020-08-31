#########################################################################
PySegCNN: Image segmentation with convolutional neural networks in Python
#########################################################################

``pysegcnn`` is a Python package to build deep fully convolutional neural
networks for semantic image segmentation. The package is primarily designed to
work with multispectral satellite imagery. ``pysegcnn`` is based on the machine
learning framework `PyTorch <https://pytorch.org/>`_.

.. include:: docs/source/installation.rst

Datasets
========
Currently, the following publicly available satellite imagery datasets are
supported out-of-the-box:

- Spatial Procedures for Automated Removal of Cloud and Shadow `SPARCS`_
by `Hughes M.J. & Hayes D.J. (2014)`_
- `Cloud-38`_ and `Cloud-95`_ by Mohajerani S. & Saeedi P. (`2019`_, `2020`_)

Contact
=======
For further information or ideas for future development please contact:
daniel.frisinghelli@gmail.com.

License
=======
If not explicitly stated otherwise, this repository is licensed under the
**GNU GENERAL PUBLIC LICENSE v3.0**
(see `LICENSE <https://gitlab.inf.unibz.it/REMSEN/ccisnow/pysegcnn/-/blob/master/LICENSE>`_).

Acknowledgements
================
I wrote a part of the code base for the ``pysegcnn`` package while I was working
at the `Institute for Earth Observation <http://www.eurac.edu/en/research/mountains/remsen/Pages/default.aspx>`_ of
`Eurac Research <http://www.eurac.edu/en/Pages/default.aspx>`_, Bolzano.


..
    Links:

.. _SPARCS:
    https://www.usgs.gov/land-resources/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation)

.. _Hughes M.J. & Hayes D.J. 2014:
    https://www.mdpi.com/2072-4292/6/6/4907

.. _Cloud-38:
    https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset

.. _Cloud-95:
    https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset

.. _2019:
    https://arxiv.org/abs/1901.10077

.. _2020:
    https://arxiv.org/abs/2001.08768