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

import imp


def loadConfiguration(filename, type):
    with open(filename) as f:
        config = imp.load_source(type, '', f)
        print type + ': ' + filename
    return config


def saveConfiguration(filename, config):
    with open(filename, 'w') as f:
        for i in [i for i in dir(config) if not i.startswith('__')]:
            f.write(str(i) + " = ")
            if isinstance(getattr(config, i), str):
                f.write('\''+getattr(config, i) + '\'\n')
            else:
                f.write(str(getattr(config, i)) + '\n')


def readTxtIntoList(filename):
    with open(filename) as f:
        flist = f.read().splitlines()
    return flist


def writeTxtIntoList(filename, content):
    with open(filename,'w') as f:
        for i in content:
            f.write(i + '\n')


def containsRequirements(config, requirements, configFileName=None):
    if configFileName:
         configFileName = " in " + configFileName
    else:
        configFileName = ""
    for i in requirements:
        if not hasattr(config, i):
            print "Requires "+i+" to be set"+configFileName
            return False
    return True
