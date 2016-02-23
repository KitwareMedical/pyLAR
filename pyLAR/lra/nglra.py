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

Configuration file must contain:
--------------------------------
    lamda (float): the tuning parameter that weights between the low-rank component and the sparse component.
    sigma (float): blurring kernel size.
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

Optional:
--------
    number_of_cpu (integer): Number of diffeomorphic registration run in parallel. In not specified,
                             it will run as many processes as there are CPU available. Beware, the processes might
                             already be multithreaded.

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


import os
import pyLAR
import time
import numpy as np
import SimpleITK as sitk
import subprocess
import shutil
import gc
import multiprocessing
import logging

def _runIteration(vector_length, level, currentIter, config, im_fns, sigma, gridSize, maxDisp, software, number_of_cpu):
    """Iterative unbiased low-rank atlas creation from a selection of images"""
    log = logging.getLogger(__name__)
    result_dir = config.result_dir
    selection = config.selection
    reference_im_fn = config.reference_im_fn
    use_healthy_atlas = config.use_healthy_atlas
    registration_type = config.registration_type
    lamda = config.lamda
    if registration_type == 'BSpline' or registration_type == 'Demons':
        EXE_BRAINSResample = software.EXE_BRAINSResample
        EXE_InvertDeformationField = software.EXE_InvertDeformationField
        if registration_type == 'BSpline':
            EXE_BRAINSFit = software.EXE_BRAINSFit
            EXE_BSplineToDeformationField = software.EXE_BSplineToDeformationField
        elif registration_type == 'Demons':
            EXE_BRAINSDemonWarp = software.EXE_BRAINSDemonWarp
    elif registration_type == 'ANTS':
        EXE_ANTS = software.EXE_ANTS
        EXE_WarpImageMultiTransform = software.EXE_WarpImageMultiTransform
        ants_params = config.ants_params
    # Prepares data matrix for low-rank decomposition
    num_of_data = len(selection)
    Y = np.zeros((vector_length, num_of_data))
    iter_prefix = 'L' + str(level) + '_Iter'
    iter_path = os.path.join(result_dir, iter_prefix)
    current_path_iter = iter_path + str(currentIter)
    prev_path_iter = iter_path + str(currentIter-1)
    for i in range(num_of_data):
        im_file = prev_path_iter + '_' + str(i) + '.nrrd'
        inIm = sitk.ReadImage(im_file)
        tmp = sitk.GetArrayFromImage(inIm)
        if sigma > 0:  # blurring
            log.info("Blurring: " + str(sigma))
            outIm = pyLAR.GaussianBlur(inIm, None, sigma)
            tmp = sitk.GetArrayFromImage(outIm)
        Y[:, i] = tmp.reshape(-1)
        del tmp

    # Low-rank and sparse decomposition
    low_rank, sparse, n_iter, rank, sparsity, sum_sparse = pyLAR.rpca(Y, lamda)
    pyLAR.saveImagesFromDM(low_rank, current_path_iter + '_LowRank_', reference_im_fn)
    pyLAR.saveImagesFromDM(sparse, current_path_iter + '_Sparse_', reference_im_fn)

    # Visualize and inspect
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 5))
        slice_prefix = 'L' + str(level) + '_' + str(currentIter)
        pyLAR.showSlice(Y, slice_prefix + ' Input', plt.cm.gray, 0, reference_im_fn)
        pyLAR.showSlice(low_rank, slice_prefix + ' low rank', plt.cm.gray, 1, reference_im_fn)
        pyLAR.showSlice(np.abs(sparse), slice_prefix + ' sparse', plt.cm.gray, 2, reference_im_fn)
        plt.savefig(current_path_iter + '.png')
        fig.clf()
        plt.close(fig)
    except ImportError:
        pass

    del low_rank, sparse, Y

    # Unbiased low-rank atlas building (ULAB)
    if not use_healthy_atlas:
        EXE_AverageImages = software.EXE_AverageImages
        # Average the low-rank images to produce the Atlas
        atlasIm = current_path_iter + '_atlas.nrrd'
        listOfImages = []
        num_of_data = len(selection)
        for i in range(num_of_data):
            lrIm = current_path_iter + '_LowRank_' + str(i) + '.nrrd'
            listOfImages.append(lrIm)
        pyLAR.AverageImages(EXE_AverageImages, listOfImages, atlasIm)

        im = sitk.ReadImage(atlasIm)
        im_array = sitk.GetArrayFromImage(im)
        z_dim, x_dim, y_dim = im_array.shape
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            implot = plt.imshow(np.flipud(im_array[z_dim / 2, :, :]), plt.cm.gray)
            plt.title(iter_prefix + str(currentIter) + ' atlas')
            plt.savefig(current_path_iter + '.png')
        except ImportError:
            pass
        reference_im_fn = atlasIm

    cmd_list = [] # to use multiple processors
    for i in range(num_of_data):
        # Pipes command lines sequencially
        cmd = ''
        # Warps the low-rank image back to the initial state (the non-greedy way)
        invWarpedlowRankIm = ''
        if currentIter == 1:
            invWarpedlowRankIm = current_path_iter + '_LowRank_' + str(i) + '.nrrd'
        else:
            lowRankIm = current_path_iter + '_LowRank_' + str(i) + '.nrrd'
            invWarpedlowRankIm = current_path_iter + '_InvWarped_LowRank_' + str(i) + '.nrrd'
            if registration_type == 'BSpline' or registration_type == 'Demons':
                previousIterDVF = prev_path_iter + '_DVF_' + str(i) + '.nrrd'
                inverseDVF = prev_path_iter + '_INV_DVF_' + str(i) + '.nrrd'
                pyLAR.genInverseDVF(EXE_InvertDeformationField, previousIterDVF, inverseDVF, True)
                pyLAR.updateInputImageWithDVF(EXE_BRAINSResample, lowRankIm, reference_im_fn,
                                              inverseDVF, invWarpedlowRankIm, True)
            if registration_type == 'ANTS':
                previousIterTransformPrefix = prev_path_iter + '_' + str(i) + '_'
                pyLAR.ANTSWarpImage(EXE_WarpImageMultiTransform, lowRankIm, invWarpedlowRankIm, reference_im_fn,
                                    previousIterTransformPrefix, True, True)

        # Registers each inversely-warped low-rank image to the Atlas image
        outputIm = current_path_iter + '_Deformed_LowRank' + str(i) + '.nrrd'
        # .tfm for BSpline only
        outputTransform = current_path_iter + '_Transform_' + str(i) + '.tfm'
        outputDVF = current_path_iter + '_DVF_' + str(i) + '.nrrd'

        movingIm = invWarpedlowRankIm
        fixedIm = reference_im_fn

        initial_prefix = 'L' + str(level) + '_Iter0_'
        initialInputImage = os.path.join(result_dir, initial_prefix + str(i) + '.nrrd')
        newInputImage = current_path_iter + '_' + str(i) + '.nrrd'

        if registration_type == 'BSpline':
            cmd += pyLAR.BSplineReg_BRAINSFit(EXE_BRAINSFit, fixedIm, movingIm, outputIm, outputTransform,
                                              gridSize, maxDisp)
            cmd += ';' + pyLAR.ConvertTransform(EXE_BSplineToDeformationField, reference_im_fn,
                                                outputTransform, outputDVF)
            cmd += ";" + pyLAR.updateInputImageWithDVF(EXE_BRAINSResample, initialInputImage, reference_im_fn,
                                                       outputDVF, newInputImage)
        elif registration_type == 'Demons':
            cmd += pyLAR.DemonsReg(EXE_BRAINSDemonWarp, fixedIm, movingIm, outputIm, outputDVF)
            cmd += ";" + pyLAR.updateInputImageWithDVF(EXE_BRAINSResample, initialInputImage, reference_im_fn,
                                                       outputDVF, newInputImage)
        elif registration_type == 'ANTS':
            # Generates a warp(DVF) file and an affine file
            outputTransformPrefix = current_path_iter + '_' + str(i) + '_'
            # if currentIter > 1:
            # initialTransform = os.path.join(result_dir, iter_prefix + str(currentIter-1) + '_' + str(i) + '_0Warp.nii.gz')
            # else:
            cmd += pyLAR.ANTS(EXE_ANTS, fixedIm, movingIm, outputTransformPrefix, ants_params)
            # Generates the warped input image with the specified file name
            cmd += ";" + pyLAR.ANTSWarpImage(EXE_WarpImageMultiTransform, initialInputImage, newInputImage,
                                             reference_im_fn, outputTransformPrefix)
        else:
            raise('Unrecognized registration type:', registration_type)
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

    return sparsity, sum_sparse


def run_command(cmd_list, log, ps):
    cmd = cmd_list.pop(0)
    log.info(cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    ps.append(process)


def run(config, software, im_fns, check=True):
    """unbiased low-rank atlas creation from a selection of images"""
    log = logging.getLogger(__name__)
    if check:
        check_requirements(config, software)

    reference_im_fn = config.reference_im_fn
    result_dir = config.result_dir
    selection = config.selection
    lamda = config.lamda
    sigma = config.sigma
    if hasattr(config, 'number_of_cpu'):
        number_of_cpu = config.number_of_cpu
    else:
        number_of_cpu = multiprocessing.cpu_count()
    num_of_iterations_per_level = config.num_of_iterations_per_level
    num_of_levels = config.num_of_levels  # Multi-scale blurring (coarse-to-fine)
    registration_type = config.registration_type
    gridSize = [0, 0, 0]
    if registration_type == 'BSpline':
        gridSize = config.gridSize

    s = time.time()
    pyLAR.showImageMidSlice(reference_im_fn)
    pyLAR.affineRegistrationStep(software.EXE_BRAINSFit, im_fns, result_dir,
                                 selection, reference_im_fn)
    # pyLAR.histogramMatchingStep()

    im_ref = sitk.ReadImage(reference_im_fn)
    im_ref_array = sitk.GetArrayFromImage(im_ref)
    z_dim, x_dim, y_dim = im_ref_array.shape
    vector_length = z_dim * x_dim * y_dim
    del im_ref, im_ref_array

    num_of_data = len(selection)
    factor = 0.5  # BSpline max displacement constrain, 0.5 refers to half of the grid size
    iterCount = 0
    for level in range(0, num_of_levels):
        for iterCount in range(1, num_of_iterations_per_level + 1):
            maxDisp = -1
            log.info('Level: ' + str(level))
            log.info('Iteration ' + str(iterCount) + ' lamda = %f' % lamda)
            log.info('Blurring Sigma: ' + str(sigma))

            if registration_type == 'BSpline':
                log.info('Grid size: ' + str(gridSize))
                maxDisp = z_dim / gridSize[2] * factor

            _runIteration(vector_length, level, iterCount, config, im_fns,
                          sigma, gridSize, maxDisp, software, number_of_cpu)

            # Adjust grid size for finner BSpline Registration
            if registration_type == 'BSpline' and gridSize[0] < 10:
                gridSize = np.add(gridSize, [1, 2, 1])

            # Reduce the amount of  blurring sizes gradually
            if sigma > 0:
                sigma = sigma - 0.5

            gc.collect()  # Garbage collection

        if level != num_of_levels - 1:
            log.warning('No need for multiple levels! TO BE REMOVED!')
            for i in range(num_of_data):
                current_file_name = 'L' + str(level) + '_Iter' + str(iterCount) + '_' + str(i) + '.nrrd'
                current_file_path = os.path.join(result_dir, current_file_name)
                nextLevelInitIm = os.path.join(result_dir, 'L' + str(level + 1) + '_Iter0_' + str(i) + '.nrrd')
                shutil.copyfile(current_file_path, nextLevelInitIm)

            if gridSize[0] < 10:
                gridSize = np.add(gridSize, [1, 2, 1])
            if sigma > 0:
                sigma = sigma - 1
            factor = factor * 0.5

            # a = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print 'Current memory usage :',a/1024.0/1024.0,'GB'
            # h = hpy()
            # print h.heap()

    e = time.time()
    l = e - s
    log.info('Total running time:  %f mins' % (l / 60.0))


def check_requirements(config, software, configFileName=None, softwareFileName=None):
    """Verifying that all options and software paths are set."""
    log = logging.getLogger(__name__)

    required_field = ['use_healthy_atlas', 'reference_im_fn', 'result_dir', 'selection', 'lamda', 'sigma',
                      'num_of_iterations_per_level', 'num_of_levels', 'registration_type']
    pyLAR.containsRequirements(config, required_field, configFileName)

    result_dir = config.result_dir
    registration_type = config.registration_type

    required_software = ['EXE_BRAINSFit']
    if not config.use_healthy_atlas:
        required_software.append('EXE_AverageImages')
    if registration_type == 'BSpline':
        required_software.extend(['EXE_InvertDeformationField', 'EXE_BRAINSResample', 'EXE_BSplineToDeformationField'])
        pyLAR.containsRequirements(config, ['gridSize'], configFileName)
    elif registration_type == 'Demons':
        required_software.extend(['EXE_BRAINSDemonWarp', 'EXE_BRAINSResample','EXE_InvertDeformationField'])
    elif registration_type == 'ANTS':
        required_software.extend(['EXE_ANTS', 'EXE_WarpImageMultiTransform'])
        pyLAR.containsRequirements(config, ['ants_params'], configFileName)
    if not config.num_of_iterations_per_level >= 0:
        raise Exception('Error in configuration file: "num_of_iterations_per_level"\
         must be a positive integer (>=0).')
    if not config.num_of_levels >= 1:
        raise Exception('Error in configuration file: "num_of_levels" must be a strictly positive integer (>=1).')
    pyLAR.containsRequirements(software, required_software, softwareFileName)
    if len(config.selection) < 2:
        raise Exception('Error in configuration file: "selection" must contain at least two values.')
    log.info('Results will be stored in:' + result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
