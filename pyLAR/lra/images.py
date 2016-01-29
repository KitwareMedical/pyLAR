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

"""Sets of functions processing images

This module contains functions that process, visualize, or save images.
Images are either SimpleITK [1] images or numpy [2] arrays, depending on the function and the parameter.

[1] http://www.simpleitk.org/
[2] http://www.numpy.org/
"""

import SimpleITK as sitk
import numpy as np
import os


def CropImage(inImFile, outImFile, lowerCropSize, upperCropSize):
    """Crop input image using SimpleITK

    This function uses CropImageFilter implemented in SimpleITK/ITK.

    Parameters
    ----------
        inImFile (string): input image file name.
        outImFile (string): output image file name. If set to 'None', the output is not saved.
        lowerCropSize (vector<unsigned int>): lower corner.
        upperCropSize (vector<unsigned int>): upper corner.

    Returns
    -------
        outputIm (SimpleITK image): cropped image.

    More information
    ----------------
    http://www.itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1CropImageFilter.html
    """
    inIm = sitk.ReadImage(inImFile)
    crop = sitk.CropImageFilter()
    crop.SetLowerBoundaryCropSize(lowerCropSize)
    crop.SetUpperBoundaryCropSize(upperCropSize)
    outIm = crop.Execute(inIm)
    outIm.SetOrigin(inIm.GetOrigin())
    outIm.SetDirection(inIm.GetDirection())
    if outImFile is not None:
        sitk.WriteImage(outIm, outImFile, True)
    return outIm


def GaussianBlur(inImFile, outImFile, sigma):
    """Gaussian blur of the image using SimpleITK

    This function uses SmoothingRecursiveGaussianImageFilter implemented in SimpleITK/ITK.

    Parameters
    ----------
        inImFile (string): input image file name.
        outImFile (string): output image file name. If set to 'None', the output is not saved.
        sigma (float): blurring kernel size.

    Returns
    -------
        outputIm (SimpleITK image): gaussian blurred image.

    More information
    ----------------
    http://www.itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1SmoothingRecursiveGaussianImageFilter.html
    """
    inputIm = sitk.ReadImage(inImFile)
    srg = sitk.SmoothingRecursiveGaussianImageFilter()
    srg.SetSigma(sigma)
    outputIm = srg.Execute(inputIm)
    if outImFile is not None:
        sitk.WriteImage(outputIm, outImFile, True)
    return outputIm

def HistogramMatching(inImFile, outImFile, refImFile,
                      number_of_histogram_levels = 1024,
                      number_of_match_points = 7,
                      threshold_at_mean_intensity = False
                      ):
    """Histogram matching of the input image with the reference image using SimpleITK

    This function uses HistogramMatchingImageFilter implemented in SimpleITK/ITK.

    Parameters
    ----------
    inImFile (string): input image file name.
    outImFile (string): output image file name: If set to 'None', the output is not saved.
    refImFile (string): reference image file name.
    number_of_histogram_levels (int): Number of histogram levels.
    number_of_match_points (int): Number of match points.
    threshold_at_mean_intensity (boolean): Threshold at mean intensity or not.

    Returns
    -------
        outputIm (SimpleITK image): histogram matched image.

    More information
    ----------------
    http://www.itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1HistogramMatchingImageFilter.html
    """
    inputIm = sitk.ReadImage(inImFile)
    referenceIm = sitk.ReadImage(refImFile)
    histMatchingFilter = sitk.HistogramMatchingImageFilter()
    histMatchingFilter.SetNumberOfHistogramLevels(number_of_histogram_levels)
    histMatchingFilter.SetNumberOfMatchPoints(number_of_match_points)
    histMatchingFilter.SetThresholdAtMeanIntensity(threshold_at_mean_intensity)
    outputIm = histMatchingFilter.Execute(inputIm, referenceIm)
    if outImFile is not None:
        sitk.WriteImage(outputIm, outImFile, True)
    return outputIm


def showSlice(dataMatrix, title, color, subplotRow, referenceImName, slice_nr=-1):
    """Display a 2D slice within a given figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print "ShowSlice not supported - matplotlib not available"
        return
    im_ref = sitk.ReadImage(referenceImName)
    im_ref_array = sitk.GetArrayFromImage(im_ref)  # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape  # get 3D volume shape
    if slice_nr == -1:
        slice_nr = z_dim / 2
    num_of_data = dataMatrix.shape[1]
    for i in range(num_of_data):
        plt.subplot2grid((3, num_of_data), (subplotRow, i))
        im = np.array(dataMatrix[:, i]).reshape(z_dim, x_dim, y_dim)
        implot = plt.imshow(np.fliplr(np.flipud(im[slice_nr, :, :])), color)
        plt.axis('off')
        plt.title(title + ' ' + str(i))
        # plt.colorbar()
    del im_ref, im_ref_array
    return


def showImageMidSlice(reference_im_fn, size_x=15, size_y=5):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print "showImageMidSlice not supported - matplotlib not available"
        return
    im_ref = sitk.ReadImage(reference_im_fn)  # image in SITK format
    im_ref_array = sitk.GetArrayFromImage(im_ref)  # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape  # get 3D volume shape
    # display reference image
    fig = plt.figure(figsize=(size_x,size_y))
    plt.subplot(131)
    implot = plt.imshow(np.flipud(im_ref_array[z_dim/2,:,:]),plt.cm.gray)
    plt.subplot(132)
    implot = plt.imshow(np.flipud(im_ref_array[:,x_dim/2,:]),plt.cm.gray)
    plt.subplot(133)
    implot = plt.imshow(np.flipud(im_ref_array[:,:,y_dim/2]),plt.cm.gray)
    plt.axis('off')
    plt.title('healthy atlas')
    fig.clf()
    del im_ref, im_ref_array
    return


def saveImagesFromDM(dataMatrix, outputPrefix, referenceImName):
    """Save 3D images from data matrix."""
    im_ref = sitk.ReadImage(referenceImName)
    im_ref_array = sitk.GetArrayFromImage(im_ref)  # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape  # get 3D volume shape
    num_of_data = dataMatrix.shape[1]
    for i in range(num_of_data):
        im = np.array(dataMatrix[:, i]).reshape(z_dim, x_dim, y_dim)
        img = sitk.GetImageFromArray(im)
        img.SetOrigin(im_ref.GetOrigin())
        img.SetSpacing(im_ref.GetSpacing())
        img.SetDirection(im_ref.GetDirection())
        fn = outputPrefix + str(i) + '.nrrd'
        sitk.WriteImage(img, fn, True)
    del im_ref, im_ref_array
    return


def gridVisDVF(dvfImFileName, sliceNum=-1, titleString='DVF', saveFigPath='.', deformedImFileName=None, contourNum=40):
    """Visualize deformation fields on a mesh grid."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print "gridVisDVF not supported - matplotlib not available"
        return
    dvf = sitk.ReadImage(dvfImFileName)
    dvfIm = sitk.GetArrayFromImage(dvf)  # get numpy array
    z_dim, y_dim, x_dim, channels = dvfIm.shape  # get 3D volume shape
    if not (channels == 3):
        print "dvf image expected to have three scalar channels"

    if sliceNum == -1:
        sliceNum = z_dim / 2
    [gridX, gridY] = np.meshgrid(np.arange(1, x_dim + 1), np.arange(1, y_dim + 1))

    fig = plt.figure()
    if deformedImFileName:
        bgGray = sitk.ReadImage(deformedImFileName)
        bgGrayIm = sitk.GetArrayFromImage(bgGray)  # get numpy array
        plt.imshow(np.fliplr(np.flipud(bgGrayIm[sliceNum, :, :])), cmap=plt.cm.gray)

    idMap = np.zeros(dvfIm.shape)
    for i in range(z_dim):
        for j in range(y_dim):
            for k in range(x_dim):
                idMap[i, j, k, 0] = i
                idMap[i, j, k, 1] = j
                idMap[i, j, k, 2] = k
    mapIm = dvfIm + idMap

    CS = plt.contour(gridX, gridY, np.fliplr(np.flipud(mapIm[sliceNum, :, :, 1])), contourNum, hold='on', colors='red')
    CS = plt.contour(gridX, gridY, np.fliplr(np.flipud(mapIm[sliceNum, :, :, 2])), contourNum, hold='on', colors='red')
    plt.title(titleString)
    plt.savefig(os.path.join(saveFigPath, titleString))
    fig.clf()
    plt.close(fig)
    return
