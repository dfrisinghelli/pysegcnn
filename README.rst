#########################################################################
PySegCNN: Image segmentation with convolutional neural networks in Python
#########################################################################

``pysegcnn`` is a Python package to build deep fully convolutional neural
networks for semantic image segmentation. The package is primarily designed to
work with multispectral satellite imagery. ``pysegcnn`` is based on the machine
learning framework `PyTorch <https://pytorch.org/>`_.

Installation
============

Requirements
------------
``pysegcnn`` is a pure Python package that works on both Windows and Linux.

.. important::

    ``pysegcnn`` requires ``Python>=3.7``.

Here is a list of all dependencies of ``pysegcnn``.

    - numpy
    - scipy
    - matplotlib
    - pytorch
    - gdal

Download
---------
You can download ``pysegcnn`` from this repository's
`website <https://gitlab.inf.unibz.it/REMSEN/ccisnow/pysegcnn>`_
or alternatively use ``git`` from terminal:

.. code-block:: bash

    git clone https://gitlab.inf.unibz.it/REMSEN/ccisnow/pysegcnn.git

This creates a copy of the repository in your current directory on the file
system.

Conda package manager
---------------------

To install ``pysegcnn``, I recommend to use the ``conda`` package manager.
You can download ``conda`` `here <https://docs.conda.io/en/latest/miniconda.html>`_.
Once successfully installed ``conda``, I recommend to add ``conda-forge`` as
your default channel:

.. code-block:: bash

    conda config --add channels conda-forge

Conda environment
-----------------

To install ``pysegcnn``, I recommend to create a specific ``conda``
`environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_,
by using the provided ``environment.yml`` file. In a terminal, navigate to the
**cloned git repositories root directory** and type:

.. code-block:: bash

    conda env create -f environment.yml

This may take a while. The above command creates a conda environment with all
required dependencies except the ``pytorch`` package. The first line in
``environment.yml`` defines the environment name, in this case ``pysegcnn``.
Activate your environment using:

.. code-block:: bash

    conda activate pysegcnn

Install PyTorch
---------------
The installation of ``pytorch`` is heavily dependent on the hardware of your
machine. Therefore, after activating your environment, install the version of
`PyTorch <https://pytorch.org/>`_ that your system supports by following the
official `instructions <https://pytorch.org/get-started/locally/>`_.

If you have to build ``pytorch`` from source, follow this
`guide <https://github.com/pytorch/pytorch#from-source>`_.

To verify the installation, run some sample PyTorch
`code <https://pytorch.org/get-started/locally/#linux-verification>`_.

Install PySegCNN
----------------
To finally install ``pysegcnn`` run the below command **from this repositories
root directory within the activated ``pysegcnn`` conda environment**:

.. code-block:: bash

    pip install -e .

If successful, you should be able to import ``pysegcnn`` from any Python
interpreter using:

.. code-block:: python

    import pysegcnn

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

.. _Hughes M.J. & Hayes D.J. (2014):
    https://www.mdpi.com/2072-4292/6/6/4907

.. _Cloud-38:
    https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset

.. _Cloud-95:
    https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset

.. _2019:
    https://arxiv.org/abs/1901.10077

.. _2020:
    https://arxiv.org/abs/2001.08768
