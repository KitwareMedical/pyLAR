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

Configuration file must contain:
--------------------------------
    result_dir (string): output directory where outputs will be saved.
    selection (list): select images that are processed in given list [must contain at least 2 values].
    reference_im_fn (string): reference image used for the registration.
    num_of_iterations_per_level (int): Number of iteration per level for the registration [>=0]
    num_of_levels (int): Number of levels (starting the registration at a down-sampled level) for the registration [>=1]
    ants_params (see example and ANTS documentation):
            ants_params = {'Convergence' : '[100x50x25,1e-6,10]',\
                  'Dimension': 3,\
                  'ShrinkFactors' : '4x2x1',\
                  'SmoothingSigmas' : '2x1x0vox',\
                  'Transform' :'SyN[0.5]',\
                  'Metric': 'Mattes[fixedIm,movingIm,1,50,Regular,0.95]'}

Optional:
--------
    number_of_cpu (integer): Number of diffeomorphic registration run in parallel. In not specified,
                             it will run as many processes as there are CPU available. Beware, the processes might
                             already be multithreaded.

Configuration Software file must contain:
-----------------------------------------
    EXE_BRAINSFit (string): Path to BRAINSFit executable (BRAINSTools package)
    EXE_AverageImages (string): Path to AverageImages executable (ANTS package)
    EXE_ANTS (string): Path to ANTS executable (ANTS package)
    EXE_WarpImageMultiTransform (string): path to WarpImageMultiTransform (ANTS package)
"""

import pyLAR
import shutil
import os
import gc
import subprocess
import time
import logging
import multiprocessing

def _runIteration(level, currentIter, ants_params, result_dir, selection, software, number_of_cpu):
    """Iterative Atlas-to-image registration"""
    log = logging.getLogger(__name__)
    EXE_AverageImages = software.EXE_AverageImages
    EXE_ANTS = software.EXE_ANTS
    EXE_WarpImageMultiTransform = software.EXE_WarpImageMultiTransform
    # average the images to produce the Atlas
    prefix = 'L' + str(level) + '_Iter'
    prev_prefix = prefix + str(currentIter-1)
    prev_iter_path = os.path.join(result_dir, prev_prefix)
    current_prefix = prefix + str(currentIter)
    current_prefix_path = os.path.join(result_dir, current_prefix)
    atlasIm = prev_iter_path + '_atlas.nrrd'
    listOfImages = []
    num_of_data = len(selection)
    for i in range(num_of_data):
        lrIm = prev_iter_path + '_' + str(i) + '.nrrd'
        listOfImages.append(lrIm)
    pyLAR.AverageImages(EXE_AverageImages, listOfImages, atlasIm)

    try:
        import matplotlib.pyplot as plt
        import SimpleITK as sitk
        im = sitk.ReadImage(atlasIm)
        im_array = sitk.GetArrayFromImage(im)
        z_dim, x_dim, y_dim = im_array.shape
        plt.figure()
        implot = plt.imshow(im_array[z_dim/2, :, :], plt.cm.gray)
        plt.title(prev_prefix+ ' atlas')
        plt.savefig(os.path.join(result_dir, 'atlas_' + prev_prefix + '.png'))
    except ImportError:
        pass
    reference_im_fn = atlasIm

    cmd_list = [] # to use multiple processors
    for i in range(num_of_data):
        cmd = ''
        initialInputImage= os.path.join(result_dir, prefix + '0_' + str(i) + '.nrrd')
        newInputImage = current_prefix_path + '_' + str(i) + '.nrrd'

        # Will generate a warp(DVF) file and an affine file
        outputTransformPrefix = current_prefix_path + '_' + str(i) + '_'
        fixedIm = atlasIm
        movingIm = initialInputImage
        cmd += pyLAR.ANTS(EXE_ANTS, fixedIm, movingIm, outputTransformPrefix, ants_params)
        cmd += ";" + pyLAR.ANTSWarpImage(EXE_WarpImageMultiTransform, initialInputImage,\
                                         newInputImage, reference_im_fn, outputTransformPrefix)
        log.info("Running: " + cmd)
        cmd_list.append(cmd)
    ps = []  # to use multiple processors
    while len(cmd_list) > 0 and len(ps) < number_of_cpu:
        run_command(cmd_list, log, ps)
    while len(ps) > 0:
        stdout, stderr = ps.pop(0).communicate()
        if stdout:
            log.info(stdout)
        if stderr:
            log.error(stderr)
        if len(cmd_list) > 0:
            run_command(cmd_list, log, ps)

def run_command(cmd_list, log, ps):
    cmd = cmd_list.pop(0)
    log.info(cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    ps.append(process)


def run(config, software, im_fns, check=True):
    """Unbiased atlas building - Atlas-to-image registration"""
    log = logging.getLogger(__name__)
    if check:
        check_requirements(config, software)
    reference_im_fn = config.reference_im_fn
    selection = config.selection
    result_dir = config.result_dir
    ants_params = config.ants_params
    num_of_iterations_per_level = config.num_of_iterations_per_level
    num_of_levels = config.num_of_levels  # multiscale bluring (coarse-to-fine)

    if hasattr(config, 'number_of_cpu'):
        number_of_cpu = config.number_of_cpu
    else:
        number_of_cpu = multiprocessing.cpu_count()

    s = time.time()

    pyLAR.affineRegistrationStep(software.EXE_BRAINSFit, im_fns, result_dir, selection, reference_im_fn)
    #cnormalizeIntensityStep()
    #histogramMatchingStep()

    num_of_data = len(selection)
    iterCount = 0
    for level in range(0, num_of_levels):
        for iterCount in range(1, num_of_iterations_per_level+1):
            log.info('Level: ' + str(level))
            log.info('Iteration ' + str(iterCount))
            _runIteration(level, iterCount, ants_params, result_dir, selection, software, number_of_cpu)
            gc.collect()  # garbage collection
        # We need to check if num_of_iterations_per_level is set to 0, which leads
        # to computing an average on the affine registration.
        if level != num_of_levels - 1:
            log.warning('No need for multiple levels! TO BE REMOVED!')
            for i in range(num_of_data):
                current_file_name = 'L' + str(level) + '_Iter' + str(iterCount) + '_' + str(i) + '.nrrd'
                current_file_path = os.path.join(result_dir, current_file_name)
                nextLevelInitIm = os.path.join(result_dir, 'L'+str(level+1)+'_Iter0_' + str(i) + '.nrrd')
                shutil.copyfile(current_file_path, nextLevelInitIm)
        # if num_of_levels > 1:
        #     print 'WARNING: No need for multiple levels! TO BE REMOVED!'
        #     for i in range(num_of_data):
        #         next_prefix = 'L' + str(level+1) + '_Iter0_'
        #         next_path = os.path.join(result_dir, next_prefix)
        #         newLevelInitIm = next_path + str(i) + '.nrrd'
    current_prefix = 'L' + str(num_of_levels-1) + '_Iter' + str(num_of_iterations_per_level)
    current_path = os.path.join(result_dir, current_prefix)
    atlasIm = current_path + '_atlas.nrrd'
    listOfImages = []
    num_of_data = len(selection)
    for i in range(num_of_data):
        lrIm = current_path + '_' + str(i) + '.nrrd'
        listOfImages.append(lrIm)
    pyLAR.AverageImages(software.EXE_AverageImages, listOfImages, atlasIm)
    try:
        import matplotlib.pyplot as plt
        import SimpleITK as sitk
        import numpy as np
        im = sitk.ReadImage(atlasIm)
        im_array = sitk.GetArrayFromImage(im)
        z_dim, x_dim, y_dim = im_array.shape
        plt.figure()
        plt.imshow(np.flipud(im_array[z_dim/2, :]), plt.cm.gray)
        plt.title(current_prefix + ' atlas')
        plt.savefig(current_path + '.png')
    except ImportError:
        pass

    e = time.time()
    l = e - s
    log.info('Total running time:  %f mins' % (l/60.0))

def check_requirements(config, software, configFileName=None, softwareFileName=None):
    """Verifying that all options and software paths are set."""
    log = logging.getLogger(__name__)
    result_dir = config.result_dir
    required_field = ['reference_im_fn', 'result_dir', 'selection',
                      'num_of_iterations_per_level', 'num_of_levels', 'ants_params']
    pyLAR.containsRequirements(config, required_field, configFileName)
    required_software = ['EXE_BRAINSFit', 'EXE_AverageImages', 'EXE_ANTS', 'EXE_WarpImageMultiTransform']
    pyLAR.containsRequirements(software, required_software, softwareFileName)

    if not config.num_of_iterations_per_level >= 0:
        raise Exception("'num_of_iterations_per_level' must be a positive integer (>=0).")
    if not config.num_of_levels >= 1:
        raise Exception("''num_of_levels' must be a strictly positive integer (>=1).")
    if len(config.selection) < 2:
        raise Exception("'selection' must contain at least two values.")
    log.info('Results will be stored in:' + result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
