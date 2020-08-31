"""Command line script to preprocess the Sparcs dataset.

After downloading the Sparcs dataset, from the repository's root directory run:

.. code-block:: bash

    python pysegcnn/preprocessing/sparcs.py

The above command will print the usage of the script.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# coding: utf-8

# builtins
import sys
from logging.config import dictConfig

# locals
from pysegcnn.core.utils import (destack_tiff, landsat_radiometric_calibration,
                                 standard_eo_structure, extract_archive)
from pysegcnn.core.logging import log_conf
from pysegcnn.core.cli import structure_parser


if __name__ == '__main__':

    # configure logging
    dictConfig(log_conf(__file__.replace('.py', '.log')))

    # the argument parser
    parser = structure_parser()

    # parse the command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # extract the archive
    extracted = extract_archive(args.archive, args.target, args.overwrite)

    # transform SPARCS directory structure to standard structure
    standard_eo_structure(source_path=extracted, target_path=args.target,
                          overwrite=args.overwrite, move=args.remove)

    # destack the TIFF raster to a single TIFF for each band
    for scene in args.target.iterdir():
        # the TIFF file containing the bands
        try:
            data = list(scene.glob('*data.tif')).pop()
        except IndexError:
            continue

        # build the TIFFs for each band
        destack_tiff(data, overwrite=args.overwrite, remove=args.remove)

        # perform radiometric calibration
        landsat_radiometric_calibration(scene, remove_raw=args.remove,
                                        overwrite=args.overwrite)
