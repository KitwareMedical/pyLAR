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

Configuration file must contain:
--------------------------------
    lamda (float): the tuning parameter that weights between the low-rank component and the sparse component.
    sigma (float): blurring kernel size.
    file_list_file_name (string): File containing path to input images.
    data_dir (string): Folder containing the "file_list_file_name" file.
    result_dir (string): output directory where outputs will be saved.
    selection (list): select images that are processed in given list [must contain at least 2 values].
    reference_im_fn (string): reference image used for the registration.
    use_healthy_atlas (boolean): use a specified healthy atlas as reference image or compute a reference image from
                                 the average of all the low-ranked images computed from the selected input images.
    num_of_iterations_per_level (int): Number of iteration per level for the registration [>=0]
    num_of_levels (int): Number of levels (starting the registration at a down-sampled level) for the registration [>=1]
    registration_type (string): Type of registration performed, selected among [BSpline,ANTS,Demons]
    ants_params (see example and ANTS documentation): Only necessary if registration_type is set to ANTS.
            ants_params = {'Convergence' : '[100x50x25,1e-6,10]',\
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
    Required:
        EXE_BRAINSFit (string): Path to BRAINSFit executable (BRAINSTools package)

    If use_healthy_atlas is set to True:
        EXE_AverageImages (string): Path to AverageImages executable (ANTS package)

    If registration_type is set to 'BSpline':
        EXE_InvertDeformationField (string): Path to InvertDeformationField executable [1]
        EXE_BRAINSResample (string): Path to BRAINSResample executable (BRAINSTools package)
        EXE_BSplineToDeformationField (string): Path to BSplineDeformationField (Slicer module)
    Else if registration_type is set to 'Demons':
        EXE_BRAINSDemonWarp (string): Path to BRAINSDemonWarp executable (BRAINSTools package)
        EXE_BRAINSResample (string): Path to BRAINSResample executable (BRAINSTools package)
        EXE_InvertDeformationField (string): Path to InvertDeformationField executable [1]
    Else if registration_type is set to 'ANTS':
        EXE_ANTS (string): Path to ANTS executable (ANTS package)
        EXE_WarpImageMultiTransform (string): path to WarpImageMultiTransform (ANTS package)

[1] https://github.com/XiaoxiaoLiu/ITKUtils
"""

import sys
import pyLAR
import shutil
import os
import argparse


def setup_and_run(config, software, im_fns, configFN=None, configSoftware=None, file_list_file_name=None):
    """Setting up processing:

    -Verifying that all options and software paths are set.
    -Saving parameters in output folders for reproducibility.
    """
    pyLAR.nglra.check_requirements(config, software, configFN, configSoftware, True)
    result_dir = config.result_dir
    # For reproducibility: save all parameters into the result dir
    savedFileName = lambda name, default: os.path.basename(name) if name else default
    pyLAR.saveConfiguration(os.path.join(result_dir, savedFileName(configFN, 'Config.txt')), config)
    pyLAR.saveConfiguration(os.path.join(result_dir, savedFileName(configSoftware, 'Software.txt')),
                            software)
    pyLAR.writeTxtIntoList(os.path.join(result_dir, savedFileName(file_list_file_name, 'listFiles.txt')), im_fns)
    currentPyFile = os.path.realpath(__file__)
    shutil.copy(currentPyFile, result_dir)
    if not(hasattr(config, "verbose") and config.verbose):
        sys.stdout = open(os.path.join(result_dir, 'RUN.log'), "w")
    pyLAR.nglra.run(config, software, im_fns, False, True)


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
    pyLAR.containsRequirements(config, ['data_dir', 'file_list_file_name'], configFN)
    data_dir = config.data_dir
    file_list_file_name = config.file_list_file_name
    im_fns = pyLAR.readTxtIntoList(os.path.join(data_dir, file_list_file_name))
    setup_and_run(config, software, im_fns, configFN, configSoftware, file_list_file_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
