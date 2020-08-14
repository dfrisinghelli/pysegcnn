# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:07:12 2020

@author: Daniel
"""


# the logging configuration dictionary
def log_conf(logfile):

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'brief': {
                'format': '%(name)s: %(message)s'
                },
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
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
                'mode': 'w'
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
