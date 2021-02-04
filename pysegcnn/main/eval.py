"""Main script to evaluate a model.

This is a command line script, which can be customized on the terminal.

Steps to run a model evaluation:

    3. In a terminal, navigate to the repository's root directory
    4. Run

    .. code-block:: bash

        python pysegcnn/main/eval.py

    This will print a list of options for the evaluation.


License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys

# locals
from pysegcnn.core.cli import evaluation_parser
from pysegcnn.core.trainer import NetworkInference
from pysegcnn.core.utils import search_files
from pysegcnn.main.eval_config import trg_ds, trg_ds_split


if __name__ == '__main__':

    # define command line argument parser
    parser = evaluation_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether the input path exists
    if args.source.exists():

        # get the model state files
        state_files = search_files(args.source, args.pattern)

        # check whether to evaluate on datasets defined at training time or
        # on explicitly defined datasets
        ds = ds_split = {}
        if not args.implicit:
            ds = trg_ds
            ds_split = trg_ds_split

        # instanciate the network inference class
        inference = NetworkInference(
            state_files=state_files,
            implicit=args.implicit,
            domain=args.domain,
            test=args.subset,
            aggregate=args.aggregate,
            ds=ds,
            ds_split=ds_split,
            map_labels=args.map_labels,
            predict_scene=args.predict_scene,
            plot_scenes=args.plot_scenes,
            cm=args.confusion_matrix)

        # evaluate models
        output = inference.evaluate()

    else:
        print('{} does not exist.'.format(str(args.source)))
        sys.exit()
