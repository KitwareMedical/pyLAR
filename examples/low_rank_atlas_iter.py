__status__  = "Development"

import numpy as np # Numpy for general purpose processing
import SimpleITK as sitk # SimpleITK to load images
import sys
import subprocess
import os
import matplotlib.pyplot as plt
sys.path.append('../')
import core.ialm as ialm
import time
import nrrd
import gc

###################################################
# preprocessing

def readTxtIntoList(filename):
   flist = []
   with open(filename) as f:
         flist = f.read().splitlines()
   return flist


def CropImage(inIm_name, outputIm_name, lowerCropSize, upperCropSize):
    inIm = sitk.ReadImage(inIm_name)
    crop = sitk.CropImageFilter()
    crop.SetLowerBoundaryCropSize(lowerCropSize)
    crop.SetUpperBoundaryCropSize(upperCropSize)
    outIm = crop.Execute(inIm)
    outIm.SetOrigin(inIm.GetOrigin())
    outIm.SetDirection(inIm.GetDirection())
    sitk.WriteImage(outIm,outputIm_name)
    return


####################################################
# RPCA
def rpca(Y,lamda):
    t_begin = time.clock()

    gamma = lamda* np.sqrt(float(Y.shape[1])/Y.shape[0])
    low_rank, sparse, n_iter,rank, sparsity, sumSparse= ialm.recover(Y,gamma)
    gc.collect()

    t_end = time.clock()
    t_elapsed = t_end- t_begin
    print 'RPCA takes:%f seconds'%t_elapsed

    return (low_rank, sparse, n_iter,rank, sparsity, sumSparse)


#####################################################
# show 2D slices in a subplot figure
def showSlice(dataMatrix,title,color,subplotRow, referenceImName, slice_nr = -1):
    im_ref = sitk.ReadImage(referenceImName)
    im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
    if slice_nr == -1:
        slice_nr = z_dim/2
    num_of_data = dataMatrix.shape[1]
    for i  in range(num_of_data):
        plt.subplot2grid((3,num_of_data),(subplotRow,i))
        im = np.array(dataMatrix[:,i]).reshape(z_dim,x_dim,y_dim)
        implot = plt.imshow(np.fliplr(np.flipud(im[slice_nr,:,:])),color)
        plt.axis('off')
        plt.title(title+' '+str(i))
        # plt.colorbar()
    del im_ref,im_ref_array
    return

# save 3D images from data matrix
def saveImagesFromDM(dataMatrix,outputPrefix,referenceImName):
    im_ref = sitk.ReadImage(referenceImName)
    im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
    num_of_data = dataMatrix.shape[1]
    for i in range(num_of_data):
        im = np.array(dataMatrix[:,i]).reshape(z_dim,x_dim,y_dim)
        img = sitk.GetImageFromArray(im)
        img.SetOrigin(im_ref.GetOrigin())
        img.SetSpacing(im_ref.GetSpacing())
        img.SetDirection(im_ref.GetDirection())
        fn = outputPrefix + str(i) + '.nrrd'
        sitk.WriteImage(img,fn)
    del im_ref,im_ref_array
    return


def gridVisDVF(dvfImFileName,sliceNum = -1,titleString = 'DVF',saveFigPath ='.',deformedImFileName = None, contourNum=40):
     dvfIm, options = nrrd.read(dvfImFileName)
     dim,x_dim, y_dim,z_dim = dvfIm.shape
     if sliceNum == -1:
            sliceNum = z_dim/2
     [gridX,gridY]=np.meshgrid(np.arange(1,x_dim+1),np.arange(1,y_dim+1))

     fig = plt.figure()
     if deformedImFileName :
         bgGrayIm,options = nrrd.read(deformedImFileName)
         plt.imshow(np.transpose(bgGrayIm[:,:,sliceNum]),cmap=plt.cm.gray)

     idMap = np.zeros((dvfIm.shape))
     for x in range(dvfIm.shape[1]):
        for y in range(dvfIm.shape[2]):
            for z in range(dvfIm.shape[3]):
                idMap[0,x,y,z] = x
                idMap[1,x,y,z] = y
                idMap[2,x,y,z] = z
    # for composed DVF, sometimes it get really big  values?
     overflow_values_indices = dvfIm > 10000
     dvfIm[overflow_values_indices] = 0
     mapIm = dvfIm + idMap

     CS = plt.contour(gridX,gridY,np.transpose(mapIm[0,:,:,sliceNum]), contourNum, hold='on', colors='red')
     CS = plt.contour(gridX,gridY,np.transpose(mapIm[1,:,:,sliceNum]), contourNum, hold='on', colors='red')
     plt.title(titleString)
     plt.savefig(saveFigPath + '/' + titleString)
     fig.clf()
     plt.close(fig)
     return

######################  REGISTRATIONs ##############################

# register to the reference image (normal control)
def AffineReg(fixedIm,movingIm,outputIm, outputTransform = None):
    if not outputTransform:
       outputTransform = outputIm+'.tfm'
    executable = '/home/xiaoxiao/work/bin/BRAINSTools/bin/BRAINSFit'
    result_folder = os.path.dirname(movingIm)
    arguments = ' --fixedVolume  ' + fixedIm \
               +' --movingVolume ' + movingIm \
               +' --outputVolume ' + outputIm \
               +' --linearTransform ' + outputTransform \
               +' --initializeTransformMode  useMomentsAlign --useAffine --numberOfSamples 100000   \
                  --numberOfIterations 1500 --maskProcessingMode NOMASK --outputVolumePixelType float \
                  --backgroundFillValue 0 --maskInferiorCutOffFromCenter 1000 --interpolationMode Linear \
                  --minimumStepLength 0.005 --translationScale 1000 --reproportionScale 1 --skewScale 1 \
                  --numberOfHistogramBins 50 --numberOfMatchPoints 10 --fixedVolumeTimeIndex 0 \
--movingVolumeTimeIndex 0 --medianFilterSize 0,0,0 --removeIntensityOutliers 0 --useCachingOfBSplineWeightsMode ON \
--useExplicitPDFDerivativesMode AUTO --ROIAutoDilateSize 0 --ROIAutoClosingSize 9 --relaxationFactor 0.5 --maximumStepLength 0.2 \
--failureExitCode -1 --numberOfThreads -1 --forceMINumberOfThreads -1 --debugLevel 0 --costFunctionConvergenceFactor 1e+09 \
--projectedGradientTolerance 1e-05 --costMetric MMI'
    cmd = executable + ' ' + arguments
    tempFile = open(result_folder+'/affine_reg.log', 'w')
    process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
    process.wait()
    tempFile.close()
    return

# deformable image registration
# call BrainsFit
def DemonsReg(fixedIm,movingIm,outputIm, outputDVF,EXECUTE = False):
    executable = '/home/xiaoxiao/work/bin/BRAINSTools/bin/BRAINSDemonWarp'
    result_folder = os.path.dirname(movingIm)
    arguments = '--movingVolume ' +movingIm \
    +' --fixedVolume ' + fixedIm \
    +' --inputPixelType float ' \
    +' --outputVolume ' + outputIm \
    +' --outputDisplacementFieldVolume ' + outputDVF \
    +' --outputPixelType float ' \
    +' --interpolationMode Linear --registrationFilterType Diffeomorphic \
       --smoothDisplacementFieldSigma 1 --numberOfPyramidLevels 3 \
       --minimumFixedPyramid 8,8,8 --minimumMovingPyramid 8,8,8 \
       --arrayOfPyramidLevelIterations 300,50,30,20,15 \
       --numberOfHistogramBins 256 --numberOfMatchPoints 2 --medianFilterSize 0,0,0 --maskProcessingMode NOMASK \
--lowerThresholdForBOBF 0 --upperThresholdForBOBF 70 --backgroundFillValue 0 --seedForBOBF 0,0,0 \
--neighborhoodForBOBF 1,1,1 --outputDisplacementFieldPrefix none --checkerboardPatternSubdivisions 4,4,4 \
--gradient_type 0 --upFieldSmoothing 0 --max_step_length 2 --numberOfBCHApproximationTerms 2 --numberOfThreads -1'
    cmd = executable + ' ' + arguments
    if (EXECUTE):
        tempFile = open(result_folder+'/demons.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd

# call BrainsFit
def BSplineReg_BRAINSFit(fixedIm,movingIm,outputIm, outputTransform,gridSize =[5,5,5] ,maxDisp = 5.0 , EXECUTE = False):
    result_folder = os.path.dirname(movingIm)
    string_gridSize = ','.join([str(gridSize[0]),str(gridSize[1]),str(gridSize[2])])
    executable = '/home/xiaoxiao/work/bin/BRAINSTools/bin/BRAINSFit'
    arguments = ' --fixedVolume  ' + fixedIm \
               +' --movingVolume ' + movingIm \
               +' --outputVolume ' + outputIm \
               +' --outputTransform ' + outputTransform \
               +' --initializeTransformMode Off --useBSpline \
                  --numberOfSamples 50000 --splineGridSize ' + string_gridSize \
               +' --maxBSplineDisplacement  ' +str(maxDisp)\
               +' --numberOfHistogramBins 50  --numberOfIterations 500 --maskProcessingMode NOMASK --outputVolumePixelType float --backgroundFillValue 0   --numberOfThreads -1 --costMetric MMI'

    cmd = executable + ' ' + arguments
    if (EXECUTE):
        tempFile = open(result_folder+'/bspline.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd


def BSplineReg_Legacy(fixedIm,movingIm,outputIm, outputDVF, gridSize=5, iterationNum=20, EXECUTE = False):
    result_folder = os.path.dirname(movingIm)
    executable = '/home/xiaoxiao/work/bin/Slicer/Slicer-build/lib/Slicer-4.3/cli-modules/BSplineDeformableRegistration'
    arguments = '  --iterations ' + str(iterationNum)\
                 +' --gridSize ' + str(gridSize)  \
                 +' --histogrambins 100 --spatialsamples 50000 '\
                 +' --outputwarp ' + outputDVF \
                 +' --resampledmovingfilename ' + outputIm \
                 +' ' + fixedIm \
                 +' ' + movingIm

    cmd = executable + ' ' + arguments
    if (EXECUTE):
        tempFile = open(result_folder+'/bspline.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd

def ConvertTransform(fixedIm, outputTransform,outputDVF,EXECUTE = False):
    result_folder = os.path.dirname(outputDVF)
    cmd ='/home/xiaoxiao/work/bin/Slicer/Slicer-build/lib/Slicer-4.3/cli-modules/BSplineToDeformationField' \
       + ' --tfm '      + outputTransform \
       + ' --refImage ' + fixedIm \
       + ' --defImage ' + outputDVF
    if (EXECUTE):
        tempFile = open(result_folder+'/convertTransform.log', 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()
    return cmd

def WarpImageMultiDVF(movingImage, refImage,DVFImageList, outputImage,EXECUTE = False):
    result_folder = os.path.dirname(outputImage)
    string_DVFImageList = ' '.join(DVFImageList)

    cmd ='/home/xiaoxiao/work/bin/BRAINSTools/bin/WarpImageMultiTransform' \
      +' 3 '    \
      +'  ' + movingImage  \
      +'  ' + outputImage \
      +' -R  '   +  refImage \
      +'  '  + string_DVFImageList


    if (EXECUTE):
        tempFile = open(result_folder+'/warpImageMultiDVF.log', 'w')
        process = subprocess.Popen(cmd, stdout = tempFile, shell = True)
        process.wait()
        tempFile.close()
    return cmd

def composeMultipleDVFs(refImage,DVFImageList, outputDVFImage,EXECUTE = False):
    result_folder = os.path.dirname(outputDVFImage)
    # T3(T2(T1(I))) = T3*T2*T1(I), need to reverse the sequence of the DVFs
    string_DVFImageList = ' '.join(DVFImageList[::-1])

    cmd ='/home/xiaoxiao/work/bin/BRAINSTools/bin/ComposeMultiTransform' \
      +' 3 '    \
      +'  ' + outputDVFImage  \
      +' -R  '   +  refImage \
      +'  '  + string_DVFImageList


    if (EXECUTE):
        tempFile = open(result_folder+'/composeDVF.log', 'w')
        process = subprocess.Popen(cmd, stdout = tempFile, shell = True)
        process.wait()
        tempFile.close()
    return cmd

def applyLinearTransform(inputImage,refImage,transform, newInputImage,EXECUTE = False):
    result_folder = os.path.dirname(newInputImage)
    cmd='/home/xiaoxiao/work/bin/BRAINSTools/bin/BRAINSResample' \
      +' --inputVolume '    +  inputImage \
      +' --referenceVolume '+  refImage   \
      +' --outputVolume '   +  newInputImage\
      +' --pixelType float ' \
      +' --warpTransform '  + transform \
      +' --defaultValue 0 --numberOfThreads -1 '
    if (EXECUTE):
        tempFile = open(result_folder+'/applyLinearTransform.log', 'w')
        process = subprocess.Popen(cmd, stdout = tempFile, shell = True)
        process.wait()
        tempFile.close()
    return cmd

def updateInputImageWithDVF(inputImage,refImage,DVFImage, newInputImage,EXECUTE = False):
    result_folder = os.path.dirname(newInputImage)
    cmd='/home/xiaoxiao/work/bin/BRAINSTools/bin/BRAINSResample' \
      +' --inputVolume '    +  inputImage \
      +' --referenceVolume '+  refImage   \
      +' --outputVolume '   +  newInputImage\
      +' --pixelType float ' \
      +' --deformationVolume '  + DVFImage \
      +' --defaultValue 0 --numberOfThreads -1 '
    if (EXECUTE):
        tempFile = open(result_folder+'/applyDVF.log', 'w')
        process = subprocess.Popen(cmd, stdout = tempFile, shell = True)
        process.wait()
        tempFile.close()
    return cmd

def updateInputImageWithTFM(inputImage,refImage, transform, newInputImage,EXECUTE = False):
    result_folder = os.path.dirname(movingIm)
    cmd='/home/xiaoxiao/work/bin/BRAINSTools/bin/BRAINSResample' \
      +' --inputVolume '    +  inputImage \
      +' --referenceVolume '+  refImage   \
      +' --outputVolume '   +  newInputImage\
      +' --pixelType float ' \
      +' --warpTransform '  + transform \
      +' --defaultValue 0 --numberOfThreads -1 '

    if (EXECUTE):
        tempFile = open(result_folder+'/applyTransform.log', 'w')
        process = subprocess.Popen(cmd, stdout = tempFile, shell = True)
        process.wait()
        tempFile.close()
    return cmd

def AverageImages(listOfImages,outputIm):
    executable = '/home/xiaoxiao/work/bin/BRAINSTools/bin/AverageImages'
    arguments = ' 3 ' + outputIm +'  0  ' +  ' '.join(listOfImages)
    cmd = executable + ' ' + arguments
    process = subprocess.Popen(cmd,  shell=True)
    process.wait()
    return

def genInverseDVF(DVFImage, InverseDVFImage, EXECUTE = False):
    result_folder = os.path.dirname(InverseDVFImage)
    cmd ='/home/xiaoxiao/work/bin/ITKUtils/bin/InvertDeformationField '\
        + DVFImage \
        +' ' \
        + InverseDVFImage
    if (EXECUTE):
        tempFile = open(result_folder+'/InverseDVF.log', 'w')
        process = subprocess.Popen(cmd, stdout = tempFile, shell = True)
        process.wait()
        tempFile.close()
    return cmd

def TBDapplyInverseDVFToTissue(DVFImage, inputTissueImage, outputTissueImage, EXECUTE=False):
    result_folder = os.path.dirname(outputTissueImage)
    reference_im_name = '/home/xiaoxiao/work/data/SRI24/T1_Crop.nii.gz'
    cmd ='/home/xiaoxiao/work/bin/BRAINSTools/bin/BRAINSResample ' \
      +' --inputVolume '    +  inputTissueImage \
      +' --outputVolume '   +  outputTissueImage \
      +' --referenceVolume '   +  reference_im_name\
      +' --pixelType short ' \
      +' --deformationVolume '  + DVFImage \
      +' --defaultValue 0 --numberOfThreads -1 '
    print cmd
    if (EXECUTE):
        tempFile = open(result_folder+'/applyDVFToTissue.log', 'w')
        process = subprocess.Popen(cmd, stdout = tempFile, shell = True)

        process.wait()
        tempFile.close()
    return cmd

def computeLabelStatistics(inputIm, labelmapIm, tumorMaskImage = None):

    inIm = sitk.ReadImage(inputIm)
    labelIm = sitk.ReadImage(labelmapIm)
    labelIm.SetOrigin(inIm.GetOrigin())
    labelIm.SetDirection(inIm.GetDirection())
    maskedLabelIm = labelIm

    if tumorMaskImage :
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
    stats = np.zeros((numOfLabels,5))
    for i in range(numOfLabels):
      stats[i,0] = statsFilter.GetMean(i)
      stats[i,1] = statsFilter.GetSigma(i)
      stats[i,2] = statsFilter.GetVariance(i)
      stats[i,3] = statsFilter.GetMinimum(i)
      stats[i,4] = statsFilter.GetMaximum(i)

    return stats
