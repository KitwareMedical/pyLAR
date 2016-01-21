import numpy as np  # Numpy for general purpose processing
import SimpleITK as sitk  # SimpleITK to load images
import sys
import subprocess
import os
import matplotlib.pyplot as plt
import time
import gc
import inspect

myfilepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
sys.path.insert(0, os.path.abspath(os.path.join(myfilepath, '../')))
import core.ialm as ialm

__status__ = "Development"


###################################################
# utility functions and simple image processing functions


def readTxtIntoList(filename):
    flist = []
    with open(filename) as f:
        flist = f.read().splitlines()
    return flist


def computeLabelStatistics(inputIm, labelmapIm, tumorMaskImage=None):
    inIm = sitk.ReadImage(inputIm)
    labelIm = sitk.ReadImage(labelmapIm)
    labelIm.SetOrigin(inIm.GetOrigin())
    labelIm.SetDirection(inIm.GetDirection())
    maskedLabelIm = labelIm

    if tumorMaskImage:
        maskImage = sitk.ReadImage(tumorMaskImage)
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


def CropImage(inIm_name, outputIm_name, lowerCropSize, upperCropSize):
    inIm = sitk.ReadImage(inIm_name)
    crop = sitk.CropImageFilter()
    crop.SetLowerBoundaryCropSize(lowerCropSize)
    crop.SetUpperBoundaryCropSize(upperCropSize)
    outIm = crop.Execute(inIm)
    outIm.SetOrigin(inIm.GetOrigin())
    outIm.SetDirection(inIm.GetDirection())
    sitk.WriteImage(outIm, outputIm_name, True)
    return


def GaussianBlur(inputIm, outputIm, sigma):
    inIm = sitk.ReadImage(inputIm)
    srg = sitk.SmoothingRecursiveGaussianImageFilter()
    srg.SetSigma(sigma)
    outIm = srg.Execute(inIm)

    sitk.WriteImage(outIm, outputIm, True)


####################################################
# run Robust PCA via ialm implementation in Core
def rpca(Y, lamda):
    t_begin = time.clock()

    gamma = lamda * np.sqrt(float(Y.shape[1]) / Y.shape[0])
    low_rank, sparse, n_iter, rank, sparsity, sumSparse = ialm.recover(Y, gamma)
    gc.collect()

    t_end = time.clock()
    t_elapsed = t_end - t_begin
    print 'RPCA takes:%f seconds' % t_elapsed

    return (low_rank, sparse, n_iter, rank, sparsity, sumSparse)


#####################################################
# display a 2D slice within a given figure
def showSlice(dataMatrix, title, color, subplotRow, referenceImName, slice_nr=-1):
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


#####################################################
# save 3D images from data matrix
def saveImagesFromDM(dataMatrix, outputPrefix, referenceImName):
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


#####################################################
# visualize deformation fields on a mesh grid
def gridVisDVF(dvfImFileName, sliceNum=-1, titleString='DVF', saveFigPath='.', deformedImFileName=None, contourNum=40):
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
    plt.savefig(saveFigPath + '/' + titleString)
    fig.clf()
    plt.close(fig)
    return


###################################################
###################################################
# Image Registration Related Functions

# Requires to build the following libraries first:
#     BRAINSTools: http://brainsia.github.io/BRAINSTools/
#     ANTS: http://stnava.github.io/ANTs/
# Please manually edit the corresponding binary paths for all
# executable paths: EXE_<executable name>

EXE_BRAINSFit = '/home/fbudin/Tools/Slicer-4.5.0-1-linux-amd64/Slicer --launch /home/fbudin/Tools/Slicer-4.5.0-1-linux-amd64/lib/Slicer-4.5/cli-modules/BRAINSFit'


def AffineReg(fixedIm, movingIm, outputIm, outputTransform=None):
    executable = EXE_BRAINSFit
    if not outputTransform:
        outputTransform = outputIm + '.tfm'

    result_folder = os.path.dirname(outputIm)
    arguments = ' --fixedVolume  ' + fixedIm \
                + ' --movingVolume ' + movingIm \
                + ' --outputVolume ' + outputIm \
                + ' --linearTransform ' + outputTransform \
                + ' --initializeTransformMode  useMomentsAlign --useAffine --samplingPercentage 0.1   \
                  --numberOfIterations 1500 --maskProcessingMode NOMASK --outputVolumePixelType float \
                  --backgroundFillValue 0 --maskInferiorCutOffFromCenter 1000 --interpolationMode Linear \
                  --minimumStepLength 0.005 --translationScale 1000 --reproportionScale 1 --skewScale 1 \
                  --numberOfHistogramBins 50 --numberOfMatchPoints 10 --fixedVolumeTimeIndex 0 \
                  --movingVolumeTimeIndex 0 --medianFilterSize 0,0,0 --removeIntensityOutliers 0 \
                  --ROIAutoDilateSize 0 --ROIAutoClosingSize 9 --relaxationFactor 0.5 --maximumStepLength 0.2 \
                  --failureExitCode -1 --numberOfThreads -1 --debugLevel 0 --costFunctionConvergenceFactor 1e+09 \
                  --projectedGradientTolerance 1e-05 --costMetric MMI'
    cmd = executable + ' ' + arguments
    print cmd
    tempFile = open(result_folder + '/affine_reg.log', 'w')
    process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
    process.wait()
    tempFile.close()
    return


def RigidReg(fixedIm, movingIm, outputIm, outputTransform=None):
    executable = EXE_BRAINSFit
    if not outputTransform:
        outputTransform = outputIm + '.tfm'

    result_folder = os.path.dirname(outputIm)
    arguments = ' --fixedVolume  ' + fixedIm \
                + ' --movingVolume ' + movingIm \
                + ' --outputVolume ' + outputIm \
                + ' --linearTransform ' + outputTransform \
                + ' --initializeTransformMode  useMomentsAlign --useRigid --samplingPercentage 0.1   \
                  --numberOfIterations 1500 --maskProcessingMode NOMASK --outputVolumePixelType float \
                  --backgroundFillValue 0 --maskInferiorCutOffFromCenter 1000 --interpolationMode Linear \
                  --minimumStepLength 0.005 --translationScale 1000 --reproportionScale 1 --skewScale 1 \
                  --numberOfHistogramBins 50 --numberOfMatchPoints 10 --fixedVolumeTimeIndex 0 \
                  --movingVolumeTimeIndex 0 --medianFilterSize 0,0,0 --removeIntensityOutliers 0 \
                  --ROIAutoDilateSize 0 --ROIAutoClosingSize 9 --relaxationFactor 0.5 --maximumStepLength 0.2 \
                  --failureExitCode -1 --numberOfThreads -1 --debugLevel 0 --costFunctionConvergenceFactor 1e+09 \
                  --projectedGradientTolerance 1e-05 --costMetric MMI'
    cmd = executable + ' ' + arguments
    tempFile = open(result_folder + '/affine_reg.log', 'w')
    print cmd
    process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
    process.wait()
    tempFile.close()
    return


EXE_ANTS = '/home/fbudin/Tools/ANTs.2.1.0.Debian-Ubuntu_X64/antsRegistration'


def ANTS(fixedIm, movingIm, outputTransformPrefix, params, initialTransform=None, EXECUTE=False):
    executable = EXE_ANTS

    dim = params['Dimension']
    CONVERGENCE = params['Convergence']  # "[100x70x50x20,1e-6,10]"
    SHRINKFACTORS = params['ShrinkFactors']  # "8x4x2x1"
    SMOOTHINGSIGMAS = params['SmoothingSigmas']  # "3x2x1x0vox"
    TRANSFORM = params['Transform']  # "SyN[0.25]"
    METRIC = params['Metric']  # "MI[%s,%s, 1,50]" %(fixedIm,movingIm)
    METRIC = METRIC.replace('fixedIm', fixedIm)
    METRIC = METRIC.replace('movingIm', movingIm)
    ##Parameter Notes From ANTS/Eaxmples/antsRegistration.cxx:
    # "CC[fixedImage,movingImage,metricWeight,radius,<samplingStrategy={None,Regular,Random}>,<samplingPercentage=[0,1]>]" );
    # "Mattes[fixedImage,movingImage,metricWeight,numberOfBins,<samplingStrategy={None,Regular,Random}>,<samplingPercentage=[0,1]>]" );
    # "Demons[fixedImage,movingImage,metricWeight,radius=NA,<samplingStrategy={None,Regular,Random}>,<samplingPercentage=[0,1]>]" );
    # option->SetUsageOption( 10, "SyN[gradientStep,updateFieldVarianceInVoxelSpace,totalFieldVarianceInVoxelSpace]" );
    arguments = ' --dimensionality ' + str(dim) \
                + ' --float 1' \
                + ' --interpolation Linear' \
                + ' --output [%s,%sWarped.nrrd]' % (outputTransformPrefix, outputTransformPrefix) \
                + ' --interpolation Linear' \
                + ' --transform ' + TRANSFORM \
                + ' -m ' + METRIC \
                + ' --convergence ' + CONVERGENCE \
                + ' --shrink-factors ' + SHRINKFACTORS \
                + ' --smoothing-sigmas ' + SMOOTHINGSIGMAS
    #            +' --use-histogram-match'
    if initialTransform:
        arguments += ' --initial-moving-transform  %s' % (initialTransform)

    cmd = executable + ' ' + arguments
    if EXECUTE:
        tempFile = open(outputTransformPrefix + 'ANTS.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


# Utility function to extract the integrated velocity field norm from the ANTS log file
def getANTSOutputVelocityNorm(logFile):
    # spatio-temporal velocity field norm : 2.8212e-02
    STRING = "    spatio-temporal velocity field norm : "
    vn = 0.0
    f = open(logFile, 'r')
    for line in f:
        if line.find(STRING) > -1:
            vn = float(line.split(STRING)[1].split()[0][0:-1])
    return vn


def geodesicDistance3D(inputImage, referenceImage, outputTransformPrefix):
    geodesicDis = -1
    affineParams = {
        'Dimension': 3,
        'Convergence': '[200,1e-6,10]',
        'ShrinkFactors': '1',
        'SmoothingSigmas': '0vox',
        'Transform': 'affine[0.1]',
        'Metric': 'Mattes[fixedIm,movingIm,1,50,Regular,0.95]'
    }
    antsParams = {
        'Dimension': 3,
        'Convergence': '[200x100x50,1e-6,10]',
        'ShrinkFactors': '4x2x1',
        'SmoothingSigmas': '2x1x0vox',
        'Transform': 'TimeVaryingVelocityField[1.0,4,8,0,0,0]',
        'Metric': 'Mattes[fixedIm,movingIm,1,50,Regular,0.95]'
    }
    ANTS(referenceImage, inputImage, outputTransformPrefix + '_affine', affineParams, None, True)
    outputIm = outputTransformPrefix + '_affineWarped.nrrd'
    if os.path.isfile(outputIm):
        ANTS(referenceImage, inputImage, outputTransformPrefix, antsParams, None, True)
        logFile = outputTransformPrefix + 'ANTS.log'
        geodesicDis = getANTSOutputVelocityNorm(logFile)
    else:
        print "affine registraion failed: no affine results are generated"
    return geodesicDis


EXE_WarpImageMultiTransform = '/home/fbudin/Tools/ANTs.2.1.0.Debian-Ubuntu_X64/WarpImageMultiTransform'


def ANTSWarpImage(inputIm, outputIm, referenceIm, transformPrefix, inverse=False, EXECUTE=False):
    dim = 3
    executable = EXE_WarpImageMultiTransform
    result_folder = os.path.dirname(outputIm)
    if not inverse:
        t = transformPrefix + '0Warp.nii.gz'
    else:
        t = transformPrefix + '0InverseWarp.nii.gz'
    arguments = str(dim) + ' %s  %s  -R %s %s ' % (inputIm, outputIm, referenceIm, t)
    cmd = executable + ' ' + arguments
    if EXECUTE:
        tempFile = open(result_folder + '/ANTSWarpImage.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


def ANTSWarp2DImage(inputIm, outputIm, referenceIm, transformPrefix, inverse=False, EXECUTE=False):
    dim = 2
    executable = EXE_WarpImageMultiTransform
    result_folder = os.path.dirname(outputIm)
    if not inverse:
        t = transformPrefix + '0Warp.nii.gz'
    else:
        t = transformPrefix + '0InverseWarp.nii.gz'
    arguments = str(dim) + ' %s  %s  -R %s %s ' % (inputIm, outputIm, referenceIm, t)
    cmd = executable + ' ' + arguments
    if EXECUTE:
        tempFile = open(result_folder + '/ANTSWarpImage.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


EXE_CreateJacobianDeterminantImage = '/home/fbudin/Tools/ANTs.2.1.0.Debian-Ubuntu_X64/CreateJacobianDeterminantImage'


def createJacobianDeterminantImage(imageDimension, dvfImage, outputIm, EXECUTE=False):
    executable = EXE_CreateJacobianDeterminantImage
    result_folder = os.path.dirname(outputIm)
    arguments = str(imageDimension) + ' %s  %s ' % (dvfImage, outputIm)
    cmd = executable + ' ' + arguments
    if EXECUTE:
        tempFile = open(result_folder + '/CreateJacobianDeterminantImage.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


EXE_BRAINSDemonWarp = '/home/fbudin/Tools/Slicer-4.5.0-1-linux-amd64/Slicer --launch /home/fbudin/Tools/Slicer-4.5.0-1-linux-amd64/lib/Slicer-4.5/cli-modules/BRAINSDemonWarp'


def DemonsReg(fixedIm, movingIm, outputIm, outputDVF, EXECUTE=False):
    executable = EXE_BRAINSDemonWarp
    result_folder = os.path.dirname(movingIm)
    arguments = '--movingVolume ' + movingIm \
                + ' --fixedVolume ' + fixedIm \
                + ' --inputPixelType float ' \
                + ' --outputVolume ' + outputIm \
                + ' --outputDisplacementFieldVolume ' + outputDVF \
                + ' --outputPixelType float ' \
                + ' --interpolationMode Linear --registrationFilterType Diffeomorphic \
       --smoothDisplacementFieldSigma 1 --numberOfPyramidLevels 3 \
       --minimumFixedPyramid 8,8,8 --minimumMovingPyramid 8,8,8 \
       --arrayOfPyramidLevelIterations 300,50,30,20,15 \
       --numberOfHistogramBins 256 --numberOfMatchPoints 2 --medianFilterSize 0,0,0 --maskProcessingMode NOMASK \
--lowerThresholdForBOBF 0 --upperThresholdForBOBF 70 --backgroundFillValue 0 --seedForBOBF 0,0,0 \
--neighborhoodForBOBF 1,1,1 --outputDisplacementFieldPrefix none --checkerboardPatternSubdivisions 4,4,4 \
--gradient_type 0 --upFieldSmoothing 0 --max_step_length 2 --numberOfBCHApproximationTerms 2 --numberOfThreads -1'
    cmd = executable + ' ' + arguments
    if EXECUTE:
        tempFile = open(result_folder + '/demons.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


EXE_BRAINSFit = '/home/fbudin/Tools/Slicer-4.5.0-1-linux-amd64/Slicer --launch /home/fbudin/Tools/Slicer-4.5.0-1-linux-amd64/lib/Slicer-4.5/cli-modules/BRAINSFit'


def BSplineReg_BRAINSFit(fixedIm, movingIm, outputIm, outputTransform, gridSize=[5, 5, 5], maxDisp=5.0, EXECUTE=False):
    result_folder = os.path.dirname(movingIm)
    string_gridSize = ','.join([str(gridSize[0]), str(gridSize[1]), str(gridSize[2])])
    executable = EXE_BRAINSFit
    arguments = ' --fixedVolume  ' + fixedIm \
                + ' --movingVolume ' + movingIm \
                + ' --outputVolume ' + outputIm \
                + ' --outputTransform ' + outputTransform \
                + ' --initializeTransformMode Off --useBSpline \
                  --samplingPercentage 0.1  --splineGridSize ' + string_gridSize \
                + ' --maxBSplineDisplacement  ' + str(maxDisp) \
                + ' --numberOfHistogramBins 50  --numberOfIterations 500 --maskProcessingMode NOMASK --outputVolumePixelType float --backgroundFillValue 0   --numberOfThreads -1 --costMetric MMI'

    cmd = executable + ' ' + arguments
    if EXECUTE:
        tempFile = open(result_folder + '/bspline.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


EXE_BSplineDeformableRegistration = '/Applications/Slicer.app/Contents/lib/Slicer-4.3/cli-modules/BSplineDeformableRegistration'


def BSplineReg_Legacy(fixedIm, movingIm, outputIm, outputDVF, gridSize=5, iterationNum=20, EXECUTE=False):
    result_folder = os.path.dirname(movingIm)
    executable = EXE_BSplineDeformableRegistration
    arguments = '  --iterations ' + str(iterationNum) \
                + ' --gridSize ' + str(gridSize) \
                + ' --histogrambins 100 --spatialsamples 50000 ' \
                + ' --outputwarp ' + outputDVF \
                + ' --resampledmovingfilename ' + outputIm \
                + ' ' + fixedIm \
                + ' ' + movingIm

    cmd = executable + ' ' + arguments
    if EXECUTE:
        tempFile = open(result_folder + '/bspline.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


EXE_BSplineToDeformationField = '/home/fbudin/Tools/Slicer-4.5.0-1-linux-amd64/Slicer --launch /home/fbudin/Tools/Slicer-4.5.0-1-linux-amd64/lib/Slicer-4.5/cli-modules/BSplineToDeformationField'


# '/Users/xiaoxiaoliu/work/bin/Slicer/Slicer-build/lib/Slicer-4.3/cli-modules/BSplineToDeformationField'
def ConvertTransform(fixedIm, outputTransform, outputDVF, EXECUTE=False):
    result_folder = os.path.dirname(outputDVF)
    cmd = EXE_BSplineToDeformationField \
        + ' --tfm ' + outputTransform \
        + ' --refImage ' + fixedIm \
        + ' --defImage ' + outputDVF
    if EXECUTE:
        tempFile = open(result_folder + '/convertTransform.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


EXE_WarpImageMultiTransform = '/home/fbudin/Tools/ANTs.2.1.0.Debian-Ubuntu_X64/WarpImageMultiTransform'


def WarpImageMultiDVF(movingImage, refImage, DVFImageList, outputImage, EXECUTE=False):
    result_folder = os.path.dirname(outputImage)
    string_DVFImageList = ' '.join(DVFImageList)

    cmd = EXE_WarpImageMultiTransform \
        + ' 3 ' \
        + '  ' + movingImage \
        + '  ' + outputImage \
        + ' -R  ' + refImage \
        + '  ' + string_DVFImageList

    if EXECUTE:
        tempFile = open(result_folder + '/warpImageMultiDVF.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


EXE_ComposeMultiTransform = '/home/fbudin/Tools/ANTs.2.1.0.Debian-Ubuntu_X64/ComposeMultiTransform'


def composeMultipleDVFs(refImage, DVFImageList, outputDVFImage, EXECUTE=False):
    result_folder = os.path.dirname(outputDVFImage)
    # T3(T2(T1(I))) = T3*T2*T1(I), need to reverse the sequence of the DVFs
    string_DVFImageList = ' '.join(DVFImageList[::-1])

    cmd = EXE_ComposeMultiTransform \
        + ' 3 ' \
        + '  ' + outputDVFImage \
        + ' -R  ' + refImage \
        + '  ' + string_DVFImageList

    if EXECUTE:
        tempFile = open(result_folder + '/composeDVF.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


EXE_BRAINSResample = '/home/fbudin/Tools/Slicer-4.5.0-1-linux-amd64/Slicer --launch /home/fbudin/Tools/Slicer-4.5.0-1-linux-amd64/lib/Slicer-4.5/cli-modules/BRAINSResample'


def applyLinearTransform(inputImage, refImage, transform, newInputImage, EXECUTE=False):
    result_folder = os.path.dirname(newInputImage)
    cmd = EXE_BRAINSResample \
        + ' --inputVolume ' + inputImage \
        + ' --referenceVolume ' + refImage \
        + ' --outputVolume ' + newInputImage \
        + ' --pixelType float ' \
        + ' --warpTransform ' + transform \
        + ' --defaultValue 0 --numberOfThreads -1 '
    if EXECUTE:
        tempFile = open(result_folder + '/applyLinearTransform.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


def updateInputImageWithDVF(inputImage, refImage, DVFImage, newInputImage, EXECUTE=False):
    result_folder = os.path.dirname(newInputImage)
    cmd = EXE_BRAINSResample \
        + ' --inputVolume ' + inputImage \
        + ' --referenceVolume ' + refImage \
        + ' --outputVolume ' + newInputImage \
        + ' --pixelType float ' \
        + ' --deformationVolume ' + DVFImage \
        + ' --defaultValue 0 --numberOfThreads -1 '
    if EXECUTE:
        tempFile = open(result_folder + '/applyDVF.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


def updateInputImageWithTFM(inputImage, refImage, transform, newInputImage, EXECUTE=False):
    result_folder = os.path.dirname(movingIm)
    cmd = EXE_BRAINSResample \
        + ' --inputVolume ' + inputImage \
        + ' --referenceVolume ' + refImage \
        + ' --outputVolume ' + newInputImage \
        + ' --pixelType float ' \
        + ' --warpTransform ' + transform \
        + ' --defaultValue 0 --numberOfThreads -1 '

    if EXECUTE:
        tempFile = open(result_folder + '/applyTransform.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


EXE_AverageImages = '/home/fbudin/Tools/ANTs.2.1.0.Debian-Ubuntu_X64/AverageImages'


def AverageImages(listOfImages, outputIm):
    arguments = ' 3 ' + outputIm + '  0  ' + ' '.join(listOfImages)
    cmd = EXE_AverageImages + ' ' + arguments
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    return


EXE_InvertDeformationField = '/home/fbudin/Tools/ITKUtils-build/InvertDeformationField '


def genInverseDVF(DVFImage, InverseDVFImage, EXECUTE=False):
    result_folder = os.path.dirname(InverseDVFImage)
    cmd = EXE_InvertDeformationField \
        + DVFImage \
        + ' ' \
        + InverseDVFImage
    if EXECUTE:
        tempFile = open(result_folder + '/InverseDVF.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd
