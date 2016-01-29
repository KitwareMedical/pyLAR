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

"""Low rank decomposition of a set of images

Command line arguments (See command line help: -h):
---------------------------------------------------
    Required:
        --configFN (string): Parameter configuration file.
        --configSoftware (string): Software configuration file.
    Optional:
        --HistogramMatching (boolean) that overwrites configuration file histogram matching parameter.

Configuration file must contain:
--------------------------------
    lamda (float): the tuning parameter that weights between the low-rank component and the sparse component.
    sigma (float): blurring kernel size.
    fileListFN (string): File containing path to input images.
    data_dir (string): Folder containing the "fileListFN" file.
    result_dir (string): output directory where outputs will be saved.
    selection (list): select images that are processed in given list [must contain at least 1 value].
    reference_im_fn (string): reference image used for the registration.
    registration (string): 'affine' or 'rigid'

Optional for 'set_and_run'/required for 'run_low_rank':
----------------------------------------------------
    HistogramMatching (boolean): If not specified or set to False, no histogram matching performed.
    verbose (boolean): If not specified or set to False, outputs are written in a log file.

Configuration Software file must contain:
-----------------------------------------
    EXE_BRAINSFit (string): Path to BRAINSFit executable (BRAINSFit package)
"""

import sys
import os
import shutil
import pyLAR
import argparse


def setup_and_run(config, software, im_fns, configFN="", configSoftware="", fileListFN=""):
    """Setting up processing:

    -Setting up options.
    -Verifying that all options and software paths are set.
    -Saving parameters in output folders for reproducibility.
    """
    pyLAR.lr.check_requirements(config, software, configFN, configSoftware, True)
    print config.HistogramMatching
    result_dir = config.result_dir
    # For reproducibility: save all parameters into the result dir
    savedFileName = lambda name, default: os.path.basename(name) if name else default
    configFN = savedFileName(configFN, 'Config.txt')
    pyLAR.saveConfiguration(os.path.join(result_dir, configFN), config)
    configSoftware = savedFileName(configSoftware, 'Software.txt')
    pyLAR.saveConfiguration(os.path.join(result_dir, configSoftware), software)
    fileListFN = savedFileName(fileListFN, 'listFiles.txt')
    pyLAR.writeTxtIntoList(os.path.join(result_dir, fileListFN), im_fns)
    currentPyFile = os.path.realpath(__file__)
    shutil.copy(currentPyFile, result_dir)
    # Start processing
    if not(hasattr(config, "verbose") and config.verbose):
        sys.stdout = open(os.path.join(result_dir, 'RUN.log'), "w")
    pyLAR.lr.run(config, software, im_fns, False, True)


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
    parser.add_argument('-m', "--HistogramMatching", action='store_true',
                        help="overwrites configuration file histogram matching parameter")
    args = parser.parse_args(argv[1:])
    # Assign parameters from the input config txt file
    configFN = args.configFN
    config = pyLAR.loadConfiguration(configFN, 'config')

    # Load software paths from file
    configSoftware = args.configSoftware
    software = pyLAR.loadConfiguration(configSoftware, 'software')
    if args.HistogramMatching:
        config.HistogramMatching = True

    if not pyLAR.containsRequirements(config, ['data_dir', 'fileListFN'], configFN):
        return 1
    data_dir = config.data_dir
    fileListFN = config.fileListFN
    im_fns = pyLAR.readTxtIntoList(os.path.join(data_dir, fileListFN))

    setup_and_run(config, software, im_fns, configFN, configSoftware, fileListFN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
