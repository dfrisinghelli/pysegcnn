# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# builtins
import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath('./source'))
sys.path.insert(0, os.path.abspath('../pysegcnn'))


# -- Project information -----------------------------------------------------

project = 'PySegCNN'
copyright = '2020, Daniel Frisinghelli'
author = 'Daniel Frisinghelli'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting'
]

# numpydoc options
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_attributes_as_param_list = True
# numpydoc_xref_param_type = True
# numpydoc_xref_ignore = {'type', 'optional', 'default'}

# autosummary options
autosummary_generate = True
autodoc_default_options = {
    'inherited-members': False,
    'show-inheritance': True,
    }

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
