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

""" Unbiased atlas creation from a selection of images

Command line arguments (See command line help: -h):
---------------------------------------------------
    Required:
        --configFN (string): Parameter configuration file.
        --configSoftware (string): Software configuration file.

Configuration file must contain:
--------------------------------
    fileListFN (string): File containing path to input images.
    data_dir (string): Folder containing the "fileListFN" file.
    result_dir (string): output directory where outputs will be saved.
    selection (list): select images that are processed in given list [must contain at least 2 values].
    reference_im_fn (string): reference image used for the registration.
    NUM_OF_ITERATIONS_PER_LEVEL (int): Number of iteration per level for the registration [>=0]
    NUM_OF_LEVELS (int): Number of levels (starting the registration at a down-sampled level) for the registration [>=1]
    antsParams (see example and ANTS documentation):
            antsParams = {'Convergence' : '[100x50x25,1e-6,10]',\
                  'Dimension': 3,\
                  'ShrinkFactors' : '4x2x1',\
                  'SmoothingSigmas' : '2x1x0vox',\
                  'Transform' :'SyN[0.5]',\
                  'Metric': 'Mattes[fixedIm,movingIm,1,50,Regular,0.95]'}

Optional for 'set_and_run'/required for 'run_low_rank':
----------------------------------------------------
    verbose (boolean): If not specified or set to False, outputs are written in a log file.

Configuration Software file must contain:
-----------------------------------------
    EXE_BRAINSFit (string): Path to BRAINSFit executable (BRAINSFit package)
    EXE_AverageImages (string): Path to AverageImages executable (ANTS package)
    EXE_ANTS (string): Path to ANTS executable (ANTS package)
    EXE_WarpImageMultiTransform (string): path to WarpImageMultiTransform (ANTS package)
"""

import sys
import pyLAR
import shutil
import os
import argparse


def setup_and_run(config, software, im_fns, configFN=None, configSoftware=None, fileListFN=None):
    """Setting up processing:

    -Verifying that all options and software paths are set.
    -Saving parameters in output folders for reproducibility.
    """
    result_dir = config.result_dir
    pyLAR.uab.check_requirements(config, software, configFN, configSoftware, True)
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
    if not(hasattr(config, "verbose") and config.verbose):
        sys.stdout = open(os.path.join(result_dir, 'RUN.log'), "w")
    pyLAR.uab.run(config, software, im_fns, False, True)


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
    args = parser.parse_args(argv[1:])
    # Assign parameters from the input config txt file
    configFN = args.configFN
    config = pyLAR.loadConfiguration(configFN, 'config')
    # Load software paths from file
    configSoftware = args.configSoftware
    software = pyLAR.loadConfiguration(configSoftware, 'software')
    if not pyLAR.containsRequirements(config, ['data_dir', 'fileListFN'], configFN):
        return 1
    data_dir = config.data_dir
    fileListFN = config.fileListFN
    im_fns = pyLAR.readTxtIntoList(os.path.join(data_dir, fileListFN))
    setup_and_run(config, software, im_fns, configFN, configSoftware, fileListFN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
