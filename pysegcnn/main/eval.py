"""Main script to evaluate a model.

Steps to run a model evaluation:

    1. Configure the model evaluation in :py:mod:`pysegcnn.main.eval_config.py`
    2. Save :py:mod:`pysegcnn.main.eval_config.py`
    3. In a terminal, navigate to the repository's root directory
    4. Run

    .. code-block:: bash

        python pysegcnn/main/eval.py


License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# locals
from pysegcnn.core.trainer import NetworkInference
from pysegcnn.main.eval_config import eval_config


if __name__ == '__main__':

    # instanciate the network inference class
    inference = NetworkInference(**eval_config)

    # evaluate model
    output = inference.evaluate()
