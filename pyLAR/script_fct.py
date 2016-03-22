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

"""
Convenience functions to initialize a logger that saves INFO and everything above to a file
and prints the same information to stdout if verbose is set to True in the configuration file.
"""

import logging
import os
import pyLAR
import sys

def run(algorithm, config, software, im_fns, result_dir,
        configFN=None, configSoftware=None, file_list_file_name=None):
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    config.data_dir = result_dir
    savedFileName = lambda name, default: os.path.basename(name) if name else default
    pyLAR.saveConfiguration(os.path.join(result_dir, savedFileName(configFN, 'Config.txt')), config)
    pyLAR.saveConfiguration(os.path.join(result_dir, savedFileName(configSoftware, 'Software.txt')),
                            software)
    pyLAR.writeTxtFromList(os.path.join(result_dir, savedFileName(file_list_file_name, 'listFiles.txt')), im_fns)
    # Set maximum number of threads used by ITK filters
    if hasattr(config, "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(config.ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS)
    if algorithm == "lr":
        pyLAR.lr.run(config, software, im_fns)
    elif algorithm == "uab":
        pyLAR.uab.run(config, software, im_fns)
    elif algorithm == "nglra":
        pyLAR.nglra.run(config, software, im_fns)
    else:
        raise Exception("Algorithm selected is not part of pyLAR")


def _get_formatter():
    return logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def close_handlers(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

def _stream_logger(config, logger):
    # configure logger
    formatter = _get_formatter()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if hasattr(config, "verbose") and config.verbose:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.ERROR)


def _file_logger(result_dir, logger):
    # Set log file
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    formatter = _get_formatter()
    log_file = os.path.join(result_dir, 'RUN.log')
    hdlr = logging.FileHandler(log_file, mode='w')
    hdlr.setLevel(logging.INFO)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)


def configure_logger(logger, config, config_file_name):
    _stream_logger(config, logger)
    pyLAR.containsRequirements(config, ['result_dir'], config_file_name)
    result_dir = config.result_dir
    _file_logger(result_dir, logger)
