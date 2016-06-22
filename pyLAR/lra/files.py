# Library: pyLAR
#
# Copyright 2014 Kitware Inc. 28 Corporate Drive,
# Clifton Park, NY, 12065, USA.
#
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions convenient with pyLAR

Utility functions load files (configuration or text files), and verify their content.
"""

import ast
import logging
import os
import re

def _normalize_path(path):
    """
    Normalizes path on Windows:
    Replaces '\' with '/' because '\' creates issues when saved and read.
    """
    if os.name is not 'posix':
        return '/'.join(re.split(r'\\|/',path ))
    else:
        return path



def loadConfiguration(filename, _):
    log = logging.getLogger(__name__)
    log.info('Loading configuration file ' + filename)
    config = type('obj', (object,), {})
    with open(filename) as f:
        lines = f.read().splitlines()
        for line in lines:
            args = line.rstrip().lstrip().split("=")
            if len(args[0]) == 0 and len(args) == 1:  # empty line, jump to next line
                continue
            if len(args) != 0 and len(args) != 2:
                raise Exception('Error while loading %s. Incomplete line: %s. %d arguments'
                                % (filename, line, len(args)))
            args[0]=args[0].rstrip()
            args[1]=args[1].lstrip()
            current_value = ast.literal_eval(args[1])
            try:
                setattr(config, args[0], current_value)
            except:
                Exception('Argument of unknown type: %s' % args[1])
    return config


def saveConfiguration(filename, config):
    with open(filename, 'w') as f:
        log = logging.getLogger(__name__)
        log.info('Saving configuration file in ' + filename)
        for i in [i for i in dir(config) if not i.startswith('__')]:
            f.write(str(i) + " = ")
            if isinstance(getattr(config, i), basestring):
                f.write('\''+_normalize_path(getattr(config, i)) + '\'\n')
            else:
                f.write(str(getattr(config, i)) + '\n')


def readTxtIntoList(filename):
    log = logging.getLogger(__name__)
    log.info('Reading text file into list: ' + filename)
    with open(filename) as f:
        flist = f.read().splitlines()
    return flist


def writeTxtFromList(filename, content):
    log = logging.getLogger(__name__)
    log.info('Writing text file from list: ' + filename)
    with open(filename,'w') as f:
        for i in content:
            f.write(_normalize_path(i) + '\n')


def containsRequirements(config, requirements, configFileName=None):
    log = logging.getLogger(__name__)
    log.info('Checking requirements')
    log.info('Requirements are: ' + str(requirements))
    if configFileName:
        log.info('Configuration file was: ' + configFileName)
        configFileName = " in " + configFileName
    else:
        configFileName = ""
    for i in requirements:
        if not hasattr(config, i):
            error_message = "Requires "+i+" to be set"+configFileName
            raise Exception(error_message)
