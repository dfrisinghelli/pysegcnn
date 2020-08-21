"""Logging configuration.

License
-------

    Copyright (c) 2020 Daniel Frisinghelli

    This source code is licensed under the GNU General Public License v3.

    See the LICENSE file in the repository's root directory.

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import pathlib


# the logging configuration dictionary
def log_conf(logfile):
    """Set basic logging configuration passed to `logging.config.dictConfig`.

    See the logging `docs`_ for a detailed description of the configuration
    dictionary.

    .. _docs:
        https://docs.python.org/3/library/logging.config.html#dictionary-schema-details

    Parameters
    ----------
    logfile : `str` or `pathlib.Path`
        The file to save the logs to.

    Returns
    -------
    LOGGING_CONFIG : `dict`
        The logging configuration.

    """
    # check if the parent directory of the log file exists
    logfile = pathlib.Path(logfile)
    if not logfile.parent.is_dir():
        logfile.parent.mkdir(parents=True, exist_ok=True)

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'brief': {
                'format': '%(name)s: %(message)s'
                },
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%dT%H:%M:%S'
                },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'brief',
                'level': 'INFO',
                'stream': 'ext://sys.stderr',
            },

            'file': {
                'class': 'logging.FileHandler',
                'formatter': 'standard',
                'level': 'INFO',
                'filename': logfile,
                'mode': 'a'
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
            },
        }
    }

    return LOGGING_CONFIG
