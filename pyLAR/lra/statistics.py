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

""" Module containing pyLAR statistics functions
"""

import SimpleITK as sitk


def computeLabelStatistics(inputIm, labelmapIm, maskImage=None):
    """ Computes regional statistics of the given input image for each region in the given label map.
    A mask can be applied to the image beforehand.

    Parameters
    ----------
    inputIm: Grayscale image on which the statistics are computed
    labelmapIm: label map (=segmentation) used to compute regional statistics
    maskImage: mask image

    Returns
    -------
    stats: numpy array containing the following statistics for each region of the label map:
    [regionIndex, 0]: mean
    [regionIndex, 1]: computed standard deviation
    [regionIndex, 2]: computed variance
    [regionIndex, 3]: minimum
    [regionIndex, 4]: maximum
    """
    inIm = sitk.ReadImage(inputIm)
    labelIm = sitk.ReadImage(labelmapIm)
    labelIm.SetOrigin(inIm.GetOrigin())
    labelIm.SetDirection(inIm.GetDirection())
    maskedLabelIm = labelIm

    if maskImage:
        maskImage = sitk.ReadImage(maskImage)
        mask = sitk.MaskNegatedImageFilter()
        maskImage.SetOrigin(labelIm.GetOrigin())
        maskImage.SetDirection(labelIm.GetDirection())
        thre = sitk.BinaryThresholdImageFilter()
        thre.SetLowerThreshold(0.5)
        thMaskImage = thre.Execute(maskImage)
        maskedLabelIm = mask.Execute(labelIm, thMaskImage)

    statsFilter = sitk.LabelStatisticsImageFilter()
    statsFilter.Execute(inIm, maskedLabelIm)

    numOfLabels = len(statsFilter.GetValidLabels())
    stats = np.zeros((numOfLabels, 5))
    for i in range(numOfLabels):
        stats[i, 0] = statsFilter.GetMean(i)
        stats[i, 1] = statsFilter.GetSigma(i)
        stats[i, 2] = statsFilter.GetVariance(i)
        stats[i, 3] = statsFilter.GetMinimum(i)
        stats[i, 4] = statsFilter.GetMaximum(i)

    return stats
