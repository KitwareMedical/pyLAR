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
    fileListFN (string): File containing path to input images.
    data_dir (string): Folder containing the "fileListFN" file.
    result_dir (string): output directory where outputs will be saved.
    selection (list): select images that are processed in given list [must contain at least 2 values].
    reference_im_fn (string): reference image used for the registration.
    USE_HEALTHY_ATLAS (boolean): use a specified healthy atlas as reference image or compute a reference image from
                                 the average of all the low-ranked images computed from the selected input images.
    NUM_OF_ITERATIONS_PER_LEVEL (int): Number of iteration per level for the registration [>=0]
    NUM_OF_LEVELS (int): Number of levels (starting the registration at a down-sampled level) for the registration [>=1]
    REGISTRATION_TYPE (string): Type of registration performed, selected among [BSpline,ANTS,Demons]
    antsParams (see example and ANTS documentation): Only necessary if REGISTRATION_TYPE is set to ANTS.
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
    Required:
        EXE_BRAINSFit (string): Path to BRAINSFit executable (BRAINSFit package)

    If USE_HEALTHY_ATLAS is set to True:
        EXE_AverageImages (string): Path to AverageImages executable (ANTS package)

    If REGISTRATION_TYPE is set to 'BSpline':
        EXE_InvertDeformationField (string): Path to InvertDeformationField executable [1]
        EXE_BRAINSResample (string): Path to BRAINSResample executable (BRAINSFit package)
        EXE_BSplineToDeformationField (string): Path to BSplineDeformationField (Slicer module)
    Else if REGISTRATION_TYPE is set to 'Demons':
        EXE_BRAINSDemonWarp (string): Path to BRAINSDemonWarp executable (BRAINSFit package)
        EXE_BRAINSResample (string): Path to BRAINSResample executable (BRAINSFit package)
        EXE_InvertDeformationField (string): Path to InvertDeformationField executable [1]
    Else if REGISTRATION_TYPE is set to 'ANTS':
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


def _runIteration(vector_length, level, currentIter, config, im_fns, sigma, gridSize, maxDisp, software, verbose):
    """Iterative unbiased low-rank atlas creation from a selection of images"""
    result_dir = config.result_dir
    selection = config.selection
    reference_im_fn = config.reference_im_fn
    USE_HEALTHY_ATLAS = config.USE_HEALTHY_ATLAS
    REGISTRATION_TYPE = config.REGISTRATION_TYPE
    lamda = config.lamda
    if REGISTRATION_TYPE == 'BSpline' or REGISTRATION_TYPE == 'Demons':
        EXE_BRAINSResample = software.EXE_BRAINSResample
        EXE_InvertDeformationField = software.EXE_InvertDeformationField
        if REGISTRATION_TYPE == 'BSpline':
            EXE_BRAINSFit = software.EXE_BRAINSFit
            EXE_BSplineToDeformationField = software.EXE_BSplineToDeformationField
        elif REGISTRATION_TYPE == 'Demons':
            EXE_BRAINSDemonWarp = software.EXE_BRAINSDemonWarp
    elif REGISTRATION_TYPE == 'ANTS':
        EXE_ANTS = software.EXE_ANTS
        EXE_WarpImageMultiTransform = software.EXE_WarpImageMultiTransform
        antsParams = config.antsParams
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
            if verbose:
                print "Blurring: " + str(sigma)
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
    if not USE_HEALTHY_ATLAS:
        EXE_AverageImages = software.EXE_AverageImages
        # Average the low-rank images to produce the Atlas
        atlasIm = current_path_iter + '_atlas.nrrd'
        listOfImages = []
        num_of_data = len(selection)
        for i in range(num_of_data):
            lrIm = current_path_iter + '_LowRank_' + str(i) + '.nrrd'
            listOfImages.append(lrIm)
        pyLAR.AverageImages(EXE_AverageImages, listOfImages, atlasIm, verbose=verbose)

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

    ps = []  # to use multiple processors
    for i in range(num_of_data):
        logFile = open(current_path_iter + '_RUN_' + str(i) + '.log', 'w')
        # Pipes command lines sequencially
        cmd = ''
        # Warps the low-rank image back to the initial state (the non-greedy way)
        invWarpedlowRankIm = ''
        if currentIter == 1:
            invWarpedlowRankIm = current_path_iter + '_LowRank_' + str(i) + '.nrrd'
        else:
            lowRankIm = current_path_iter + '_LowRank_' + str(i) + '.nrrd'
            invWarpedlowRankIm = current_path_iter + '_InvWarped_LowRank_' + str(i) + '.nrrd'
            if REGISTRATION_TYPE == 'BSpline' or REGISTRATION_TYPE == 'Demons':
                previousIterDVF = prev_path_iter + '_DVF_' + str(i) + '.nrrd'
                inverseDVF = prev_path_iter + '_INV_DVF_' + str(i) + '.nrrd'
                pyLAR.genInverseDVF(EXE_InvertDeformationField, previousIterDVF, inverseDVF, True, verbose=verbose)
                pyLAR.updateInputImageWithDVF(EXE_BRAINSResample, lowRankIm, reference_im_fn,
                                              inverseDVF, invWarpedlowRankIm, True, verbose=verbose)
            if REGISTRATION_TYPE == 'ANTS':
                previousIterTransformPrefix = prev_path_iter + '_' + str(i) + '_'
                pyLAR.ANTSWarpImage(EXE_WarpImageMultiTransform, lowRankIm, invWarpedlowRankIm, reference_im_fn,
                                    previousIterTransformPrefix, True, True, verbose=verbose)

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

        if REGISTRATION_TYPE == 'BSpline':
            cmd += pyLAR.BSplineReg_BRAINSFit(EXE_BRAINSFit, fixedIm, movingIm, outputIm, outputTransform,
                                              gridSize, maxDisp, verbose=verbose)
            cmd += ';' + pyLAR.ConvertTransform(EXE_BSplineToDeformationField, reference_im_fn,
                                                outputTransform, outputDVF, verbose=verbose)
            cmd += ";" + pyLAR.updateInputImageWithDVF(EXE_BRAINSResample, initialInputImage, reference_im_fn,
                                                       outputDVF, newInputImage, verbose=verbose)
        elif REGISTRATION_TYPE == 'Demons':
            cmd += pyLAR.DemonsReg(EXE_BRAINSDemonWarp, fixedIm, movingIm, outputIm, outputDVF, verbose=verbose)
            cmd += ";" + pyLAR.updateInputImageWithDVF(EXE_BRAINSResample, initialInputImage, reference_im_fn,
                                                       outputDVF, newInputImage, verbose=verbose)
        elif REGISTRATION_TYPE == 'ANTS':
            # Generates a warp(DVF) file and an affine file
            outputTransformPrefix = current_path_iter + '_' + str(i) + '_'
            # if currentIter > 1:
            # initialTransform = os.path.join(result_dir, iter_prefix + str(currentIter-1) + '_' + str(i) + '_0Warp.nii.gz')
            # else:
            cmd += pyLAR.ANTS(EXE_ANTS, fixedIm, movingIm, outputTransformPrefix, antsParams, verbose=verbose)
            # Generates the warped input image with the specified file name
            cmd += ";" + pyLAR.ANTSWarpImage(EXE_WarpImageMultiTransform, initialInputImage, newInputImage,
                                             reference_im_fn, outputTransformPrefix, verbose=verbose)
        else:
            raise('Unrecognized registration type:', REGISTRATION_TYPE)

        process = subprocess.Popen(cmd, stdout=logFile, shell=True)
        ps.append(process)
    for p in ps:
        p.wait()
    return sparsity, sum_sparse


def run(config, software, im_fns, check=True, verbose=True):
    """unbiased low-rank atlas creation from a selection of images"""
    if check:
        check_requirements(config, software, verbose=verbose)

    reference_im_fn = config.reference_im_fn
    result_dir = config.result_dir
    selection = config.selection
    lamda = config.lamda
    sigma = config.sigma

    NUM_OF_ITERATIONS_PER_LEVEL = config.NUM_OF_ITERATIONS_PER_LEVEL
    NUM_OF_LEVELS = config.NUM_OF_LEVELS  # Multi-scale blurring (coarse-to-fine)
    REGISTRATION_TYPE = config.REGISTRATION_TYPE
    gridSize = [0, 0, 0]
    if REGISTRATION_TYPE == 'BSpline':
        gridSize = config.gridSize

    s = time.time()
    pyLAR.showImageMidSlice(reference_im_fn)
    pyLAR.affineRegistrationStep(software.EXE_BRAINSFit, im_fns, result_dir,
                                 selection, reference_im_fn, verbose=verbose)
    # pyLAR.histogramMatchingStep()

    im_ref = sitk.ReadImage(reference_im_fn)
    im_ref_array = sitk.GetArrayFromImage(im_ref)
    z_dim, x_dim, y_dim = im_ref_array.shape
    vector_length = z_dim * x_dim * y_dim
    del im_ref, im_ref_array

    num_of_data = len(selection)
    factor = 0.5  # BSpline max displacement constrain, 0.5 refers to half of the grid size
    iterCount = 0
    for level in range(0, NUM_OF_LEVELS):
        for iterCount in range(1, NUM_OF_ITERATIONS_PER_LEVEL + 1):
            maxDisp = -1
            print 'Level: ', level
            print 'Iteration ' + str(iterCount) + ' lamda = %f' % lamda
            print 'Blurring Sigma: ', sigma

            if REGISTRATION_TYPE == 'BSpline':
                print 'Grid size: ', gridSize
                maxDisp = z_dim / gridSize[2] * factor

            _runIteration(vector_length, level, iterCount, config, im_fns, sigma, gridSize, maxDisp, software, verbose)

            # Adjust grid size for finner BSpline Registration
            if REGISTRATION_TYPE == 'BSpline' and gridSize[0] < 10:
                gridSize = np.add(gridSize, [1, 2, 1])

            # Reduce the amount of  blurring sizes gradually
            if sigma > 0:
                sigma = sigma - 0.5

            gc.collect()  # Garbage collection

        if level != NUM_OF_LEVELS - 1:
            print 'WARNING: No need for multiple levels! TO BE REMOVED!'
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
    if verbose:
        print 'Total running time:  %f mins' % (l / 60.0)


def check_requirements(config, software, configFileName=None, softwareFileName=None, verbose=True):
    """Verifying that all options and software paths are set."""

    required_field = ['USE_HEALTHY_ATLAS', 'reference_im_fn', 'data_dir',
                      'result_dir', 'fileListFN', 'selection', 'lamda', 'sigma',
                      'NUM_OF_ITERATIONS_PER_LEVEL', 'NUM_OF_LEVELS', 'REGISTRATION_TYPE']
    if not pyLAR.containsRequirements(config, required_field, configFileName):
        raise Exception('Error in configuration file')
    result_dir = config.result_dir
    REGISTRATION_TYPE = config.REGISTRATION_TYPE

    required_software = ['EXE_BRAINSFit']
    if not config.USE_HEALTHY_ATLAS:
        required_software.append('EXE_AverageImages')
    if REGISTRATION_TYPE == 'BSpline':
        required_software.extend(['EXE_InvertDeformationField', 'EXE_BRAINSResample', 'EXE_BSplineToDeformationField'])
        if not pyLAR.containsRequirements(config, ['gridSize'], configFileName):
            raise Exception('Error in configuration file')
    elif REGISTRATION_TYPE == 'Demons':
        required_software.extend(['EXE_BRAINSDemonWarp', 'EXE_BRAINSResample','EXE_InvertDeformationField'])
    elif REGISTRATION_TYPE == 'ANTS':
        required_software.extend(['EXE_ANTS', 'EXE_WarpImageMultiTransform'])
        if not pyLAR.containsRequirements(config, ['antsParams'], configFileName):
            raise Exception('Error in configuration file')
    if not config.NUM_OF_ITERATIONS_PER_LEVEL >= 0:
        if verbose:
            print '\'NUM_OF_ITERATIONS_PER_LEVEL\' must be a positive integer (>=0).'
        raise Exception('Error in configuration file')
    if not config.NUM_OF_LEVELS >= 1:
        if verbose:
            print '\'NUM_OF_LEVELS\' must be a strictly positive integer (>=1).'
        raise Exception('Error in configuration file')
    if not pyLAR.containsRequirements(software, required_software, softwareFileName):
        raise Exception('Error in configuration file')
    if len(config.selection) < 2:
        if verbose:
            print '\'selection\' must contain at least two values.'
        raise Exception('Error in configuration file')
    if verbose:
        print 'Results will be stored in:', result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
