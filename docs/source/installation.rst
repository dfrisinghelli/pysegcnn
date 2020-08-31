.. Installation:

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
