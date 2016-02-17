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

"""Batch processing functions

Functions that conveniently process multiple images in a row, from a given selection in a list.
"""

import os
import pyLAR
import SimpleITK as sitk
import logging

def affineRegistrationStep(EXE_BRAINSFit, im_fns, result_dir, selection, reference_im_fn):
    """Affine registering each input image to the reference(healthy atlas) image."""
    num_of_data = len(selection)
    log = logging.getLogger(__name__)
    log.info('affineRegistrationStep')
    log.info('Selection: '+repr(selection))
    for i in range(num_of_data):
        outputIm = os.path.join(result_dir, 'L0_Iter0_' + str(i) + '.nrrd')
        pyLAR.AffineReg(EXE_BRAINSFit, reference_im_fn, im_fns[selection[i]], outputIm, None)
    return


def rigidRegistrationStep(EXE_BRAINSFit, im_fns, result_dir, selection, reference_im_fn):
    """Rigid registering each input image to the reference(healthy atlas) image."""
    num_of_data = len(selection)
    log = logging.getLogger(__name__)
    log.info('rigidRegistrationStep')
    log.info('Selection: '+repr(selection))
    for i in range(num_of_data):
        outputIm = os.path.join(result_dir, 'L0_' + str(i) + '.nrrd')
        pyLAR.RigidReg(EXE_BRAINSFit, reference_im_fn, im_fns[selection[i]], outputIm, None)
    return


def histogramMatchingStep(selection, result_dir ):
    """Histogram matching preprocessing."""
    num_of_data = len(selection)
    log = logging.getLogger(__name__)
    log.info('histogramMatchingStep')
    log.info('Selection: '+repr(selection))
    for i in range(0, num_of_data):
        inIm = os.path.join(result_dir, 'L0_Iter0_' + str(i) + '.nrrd')
        refIm = os.path.join(result_dir, 'L0_Iter0_' + str(0) + '.nrrd')
        outIm = os.path.join(result_dir, 'L0_Iter0_' + str(i) + '.nrrd')
        pyLAR.HistogramMatching(inIm, outIm, refIm)
    return


def normalizeIntensityStep(selection, result_dir):
    num_of_data = len(selection)
    log = logging.getLogger(__name__)
    log.info('normalizeIntensityStep')
    log.info('Selection: '+repr(selection))
    for i in range(num_of_data):
        inIm = os.path.join(result_dir, 'L0_Iter0_' + str(i) + '.nrrd')
        outIm = os.path.join(result_dir, 'L0_Iter0_' + str(i) + '.nrrd')
        normalizeFilter = sitk.NormalizeImageFilter()
        inputIm = sitk.ReadImage(inIm)
        outputIm = normalizeFilter.Execute(inputIm)
        sitk.WriteImage(outputIm, outIm, True)
    return
