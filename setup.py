"""A setuptools based setup module adapted from PyPa's sample project.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

We reproduce their license terms here:

Copyright (c) 2016 The Python Packaging Authority (PyPA)
Copyright (c) 2018 Fabien Maussion
Copyright (c) 2018 Daniel Frisinghelli

Permission is hereby granted, free of charge, to any person obtaining a copy of
this file (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import os
import codecs
from setuptools import setup, find_packages

# SETUP: PACKAGE PROPERTIES

# Arguments marked as "Required" in the setup() function below must be included
# for upload to PyPI. Fields marked as "Optional" may be commented out.

# This is the name of your project. The first time you publish this
# package, this name will be registered for you. It will determine how
# users can install this project, e.g.:
#
# $ pip install myproject
#
# And where it will live on PyPI: https://pypi.org/project/myproject/
#
# There are some restrictions on what makes a valid project name
# specification here:
# https://packaging.python.org/specifications/core-metadata/#name
PACKAGE_NAME = 'pysegcnn'


# Versions should comply with PEP 440:
# https://www.python.org/dev/peps/pep-0440/
#
# For a discussion on single-sourcing the version across setup.py and the
# project code, see
# https://packaging.python.org/en/latest/single_source_version.html
VERSION = '1.0.0'


# This is a one-line description or tagline of what your project does. This
# corresponds to the "Summary" metadata field:
# https://packaging.python.org/specifications/core-metadata/#summary
DESCRIPTION = """A Python package for multispectral image segmentation using
deep convolutional neural networks."""


# This is an optional longer description of your project that represents
# the body of text which users will see when they visit PyPI.
#
# Often, this is the same as your README, so you can just read it in from
# that file directly:
HERE = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with codecs.open(os.path.join(HERE, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()
# This field corresponds to the "Description" metadata field:
# https://packaging.python.org/specifications/core-metadata/#description-optional


# Denotes that our long_description is in Markdown; valid values are
# text/plain, text/x-rst, and text/markdown
#
# Optional if long_description is written in reStructuredText (rst) but
# required for plain-text or Markdown; if unspecified, "applications should
# attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
# fall back to text/plain if it is not valid rst" (see link below)
#
# This field corresponds to the "Description-Content-Type" metadata field:
# https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
CONTENT_TYPE = 'text/markdown'


# This should be a valid link to your project's main homepage.
#
# This field corresponds to the "Home-Page" metadata field:
# https://packaging.python.org/specifications/core-metadata/#home-page-optional
URL = 'https://gitlab.inf.unibz.it/REMSEN/ccisnow/pysegcnn'


# This should be your name or the name of the organization which owns the
# project.
AUTHOR = 'Daniel Frisinghelli'


# This should be a valid email address corresponding to the author listed
# above.
AUTHOR_EMAIL = 'daniel.frisinghelli@gmail.com'


# Classifiers help users find your project by categorizing it.
#
# For a list of valid classifiers, see https://pypi.org/classifiers/
CLASSIFIERS = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',

    # License
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

    # Operating System
    'Operating System :: OS Independent',

    # Topic
    'Topic :: Scientific/Engineering :: Image Recognition',

    # Audience
    'Intended Audience :: Science/Research'
],


# This field adds keywords for your project which will appear on the
# project page. What does your project relate to?
#
# Note that this is a string of words separated by whitespace, not a list.
KEYWORDS = """Image Segmentation, Convolutional neural networks,
Deep Learning, Transfer Learning"""

# When your source code is in a subdirectory under the project root, e.g.
# `src/`, it is necessary to specify the `package_dir` argument.
# PACKAGE_DIR = {'': 'pysegcnn'},  # Optional

# You can just specify package directories manually here if your project is
# simple. Or you can use find_packages().
#
# Alternatively, if you just want to distribute a single Python file, use
# the `py_modules` argument instead as follows, which will expect a file
# called `my_module.py` to exist:
#
#   py_modules=["my_module"],
PACKAGES = find_packages(exclude=['tests'])


# This field lists other packages that your project depends on to run.
# Any package you put here will be installed by pip when your project is
# installed, so they must be valid existing projects.
#
# For an analysis of "install_requires" vs pip's requirements files see:
# https://packaging.python.org/en/latest/requirements.html
INSTALL_REQUIRES = ['numpy',
                    'pandas',
                    'xarray',
                    'scipy',
                    'scikit-learn',
                    'gdal',
                    'rasterio',
                    'netcdf4',
                    'h5py',
                    'h5netcdf',
                    'matplotlib',
                    'seaborn',
                    'ipython'
                    ]

# Specify which Python versions you support. In contrast to the
# 'Programming Language' classifiers above, 'pip install' will check this
# and refuse to install the project if the version does not match. See
# https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
PYTHON_REQUIRES = '>=3.7'


# List additional groups of dependencies here (e.g. development
# dependencies). Users will be able to install these using the "extras"
# syntax, for example:
#
#   $ pip install sampleproject[dev]
#
# Similar to `install_requires` above, these must be valid existing
# projects.
EXTRAS_REQUIRE = {
    'tests': 'pytest',
    }


# If there are data files included in your packages that need to be
# installed, specify them here.
PACKAGE_DATA = {}


# Although 'package_data' is the preferred approach, in some case you may
# need to place data files outside of your packages. See:
# http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
DATA_FILES = {}


# To provide executable scripts, use entry points in preference to the
# "scripts" keyword. Entry points provide cross-platform support and allow
# `pip` to create the appropriate form of executable for the target
# platform.
ENTRY_POINTS = {}


# List additional URLs that are relevant to your project as a dict.
#
# This field corresponds to the "Project-URL" metadata fields:
# https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
#
# Examples listed include a pattern for specifying where the package tracks
# issues, where the source is hosted, where to say thanks to the package
# maintainers, and where to support the project financially. The key is
# what's used to render the link text on PyPI.
PROJECT_URLS = {}


# SETUP YOUR PACKAGE
setup(
    name=PACKAGE_NAME,  # Required
    version=VERSION,  # Required
    description=DESCRIPTION,  # Required
    long_description=LONG_DESCRIPTION,  # Optional
    long_description_content_type=CONTENT_TYPE,  # Optional
    url=URL,  # Optional
    author=AUTHOR,  # Optional
    author_email=AUTHOR_EMAIL,  # Optional
    classifiers=CLASSIFIERS,  # Optional
    keywords=KEYWORDS,  # Optional
    packages=PACKAGES,  # Required
    # package_dir=PACKAGE_DIR,  # Optional
    install_requires=INSTALL_REQUIRES,  # Required
    extras_require=EXTRAS_REQUIRE,  # Optional
    package_data=PACKAGE_DATA,  # Optional
    data_files=DATA_FILES,  # Optional
    entry_points=ENTRY_POINTS,  # Optional
    project_urls=PROJECT_URLS,  # Optional
    python_requires=PYTHON_REQUIRES  # Optional
)
