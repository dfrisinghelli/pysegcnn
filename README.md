# PySegCNN: Image segmentation with convolutional neural networks in Python
This repository hosts a Python package called ``pysegcnn``. The package is
primarily designed to build deep fully convolutional neural networks for
semantic image segmentation of multispectral satellite imagery. ``pysegcnn``
is based on the machine learning framework [PyTorch](https://pytorch.org/).

## Requirements
``pysegcnn`` requires **Python 3.7** or greater.

## Installation
You can download ``pysegcnn`` from [this repository's website](https://gitlab.inf.unibz.it/REMSEN/ccisnow/pysegcnn)
or alternatively use ``git`` from terminal:

```bash
git clone https://gitlab.inf.unibz.it/REMSEN/ccisnow/pysegcnn.git
```
This creates a copy of the repository in your current directory on the file
system.

To install ``pysegcnn``, I recommend to use the ``conda`` package manager.
You can download ``conda`` [here](https://docs.conda.io/en/latest/miniconda.html).
Once successfully installed ``conda``, I recommend to add ``conda-forge`` as
your default channel:

```bash
conda config --add channels conda-forge
```
To finally install ``pysegcnn``, I recommend to create a specific ``conda``
[environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html),
by using the provided ``environment.yml`` file. In a terminal, navigate to the
**cloned git repositories root directory** (``/pysegcnn``) and type:

```bash
conda env create -f environment.yml
```
This may take a while. The first line in ``environment.yml`` defines the
environment name, in this case ``pysegcnn``. Activate your environment using:

```bash
conda activate pysegcnn
```
After activating your environment, install the version of PyTorch and CUDA
that your system supports by following this [guide](https://pytorch.org/get-started/locally/).
Having [successfully installed](https://pytorch.org/get-started/locally/#linux-verification)
PyTorch, type:

```bash
pip install -e .
```
Make sure you run the above command **from this repositories root directory
within the activated ``pysegcnn`` conda environment**. If successful,
you should be able to import ``pysegcnn`` from any Python interpreter using

```python
import pysegcnn
```

## Datasets
Currently, the following publicly available satellite imagery datasets are
supported out-of-the-box:

- Spatial Procedures for Automated Removal of Cloud and Shadow ([SPARCS](https://www.usgs.gov/land-resources/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation)) by Hughes M.J. & Hayes D.J. ([2014](https://www.mdpi.com/2072-4292/6/6/4907))
- [Cloud-38](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset)
and [Cloud-95](https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset)
by Mohajerani S. & Saeedi P. ([2019](https://arxiv.org/abs/1901.10077), [2020](https://arxiv.org/abs/2001.08768))

## Contributors & Contact
- Daniel Frisinghelli

For further information or ideas for future development please contact:
daniel.frisinghelli@gmail.com.

## License
If not explicitly stated otherwise, this repository is licensed under the
**GNU GENERAL PUBLIC LICENSE v3.0**
(see [LICENSE](https://gitlab.inf.unibz.it/REMSEN/ccisnow/pysegcnn/-/blob/master/LICENSE)).

## Acknowledgements
I wrote a part of the code base for the ``pysegcnn`` package while I was working
at the [Institute for Earth Observation](http://www.eurac.edu/en/research/mountains/remsen/Pages/default.aspx) of
[Eurac Research](http://www.eurac.edu/en/Pages/default.aspx), Bolzano.
