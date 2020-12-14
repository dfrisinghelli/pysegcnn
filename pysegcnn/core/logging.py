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


# logging configuration dictionary
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
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    }
}


# the logging configuration dictionary
def log_conf(logfile=None):
    """Set basic logging configuration.

    See the logging `docs`_ for a detailed description of the configuration
    dictionary.

    .. _docs:
        https://docs.python.org/3/library/logging.config.html#dictionary-schema-details

    Parameters
    ----------
    logfile : `str` or :py:class:`pathlib.Path`
        The file to save the logs to. The default is `None`, which means only
        log to stdout and not to file.

    Returns
    -------
    LOGGING_CONFIG : `dict`
        The logging configuration.

    """
    # check if the parent directory of the log file exists
    if logfile is not None:
        logfile = pathlib.Path(logfile)
        if not logfile.parent.is_dir():
            logfile.parent.mkdir(parents=True, exist_ok=True)

        # add log file to logging configuration
        LOGGING_CONFIG['handlers']['file'] = {
                'class': 'logging.FileHandler',
                'formatter': 'standard',
                'level': 'INFO',
                'filename': logfile,
                'mode': 'a'
            }
        LOGGING_CONFIG['loggers']['']['handlers'].append('file')

    return LOGGING_CONFIG
