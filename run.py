#!/usr/bin/env python

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

""" Unbiased low-rank atlas creation from a selection of images

Command line arguments (See command line help: -h):
---------------------------------------------------
    Required:
        --configFN (string): Parameter configuration file.
        --configSoftware (string): Software configuration file.
        --algorithm (string): Choose the algorithm to run:
            * 'lr': Low-Rank decomposition
            * 'uab': Unbiased Atlas Building
            * 'nglra': Non-Greedy Low-Rank Atlas

Configuration file must contain:
--------------------------------
    file_list_file_name (string): File containing input images paths.
    result_dir (string): output directory where outputs will be saved.

    Additional fields are required, depending on the chosen algorithm. For more information,
    check the docstring of the algorithm you want to run (e.g.in nglra.py).

Optional:
----------------------------------------------------
    clean (boolean): If specified, output directory (=result_dir) content is removed before processing starts.
"""

import sys
import pyLAR
import os
import argparse
import logging
import shutil

def main(argv=None):
    """Parsing command line arguments and reading input files."""
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(
            prog=argv[0],
            description=__doc__
    )
    parser.add_argument('-c', "--configFN", required=True, help="Parameter configuration file")
    parser.add_argument('-s', "--configSoftware", required=True, help="Software configuration file")
    parser.add_argument('-a', "--algorithm", required=True, choices=['lr', 'uab', 'nglra'],
                        help="Software configuration file")
    args = parser.parse_args(argv[1:])
    # Assign parameters from the input config txt file
    configFN = args.configFN
    config = pyLAR.loadConfiguration(configFN, 'config')
    # Load software paths from file
    configSoftware = args.configSoftware
    software = pyLAR.loadConfiguration(configSoftware, 'software')
    pyLAR.containsRequirements(config, ['file_list_file_name', 'result_dir'], configFN)
    result_dir = config.result_dir
    file_list_file_name = config.file_list_file_name
    im_fns = pyLAR.readTxtIntoList(file_list_file_name)
    # 'clean' needs to be done before configuring the logger that creates a file in the output directory
    if hasattr(config, "clean") and config.clean:
        shutil.rmtree(result_dir)
    # configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    pyLAR.configure_logger(logger, config, configFN)
    try:
        pyLAR.run(args.algorithm, config, software, im_fns, result_dir,
                  configFN, configSoftware, file_list_file_name)
        val = 0
    except:
        logger.exception('Error while processing', exc_info=True)
        val = 1
    finally:
        pyLAR.close_handlers(logger)
        return val


if __name__ == "__main__":
    sys.exit(main())
