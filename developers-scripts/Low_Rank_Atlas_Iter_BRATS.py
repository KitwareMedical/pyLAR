#!/usr/bin/env python
r"""BRATS 2012 low rank atlas building.

BRATS 2012 data can be downloaded from here: http://challenge-legacy.kitware.com/midas/folder/102
(or here https://www.smir.ch/BRATS/Start2012)
For more information: http://www2.imm.dtu.dk/projects/BRATS2012/data.html
"""
import sys
import os
import inspect
import shutil
import argparse
import numpy as np
import SimpleITK as sitk
myfilepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
sys.path.insert(0, os.path.abspath(os.path.join(myfilepath, '../')))
from low_rank_atlas_iter import *


def useData_BRATS2_FLAIR_midas():
    global im_names, data_dir
    im_names = [
        os.path.join(data_dir, 'BRATS_HG0026/BRATS_HG0026_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0012/BRATS_HG0012_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0009/BRATS_HG0009_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0024/BRATS_HG0024_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0007/BRATS_HG0007_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0014/BRATS_HG0014_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0005/BRATS_HG0005_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0001/BRATS_HG0001_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0003/BRATS_HG0003_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0027/BRATS_HG0027_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0002/BRATS_HG0002_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0004/BRATS_HG0004_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0013/BRATS_HG0013_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0008/BRATS_HG0008_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0011/BRATS_HG0011_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0006/BRATS_HG0006_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0022/BRATS_HG0022_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0015/BRATS_HG0015_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0025/BRATS_HG0025_FLAIR.mha'),
        os.path.join(data_dir, 'BRATS_HG0010/BRATS_HG0010_FLAIR.mha')
    ]
    return


def useData_BRATS2_FLAIR_dtu():
    global im_names, data_dir
    im_names = [
        os.path.join(data_dir, '0001/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.684.mha'),
        os.path.join(data_dir, '0002/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.691.mha'),
        os.path.join(data_dir, '0003/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.697.mha'),
        os.path.join(data_dir, '0004/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.703.mha'),
        os.path.join(data_dir, '0005/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.709.mha'),
        os.path.join(data_dir, '0006/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.715.mha'),
        os.path.join(data_dir, '0007/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.721.mha'),
        os.path.join(data_dir, '0008/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.727.mha'),
        os.path.join(data_dir, '0009/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.733.mha'),
        os.path.join(data_dir, '0010/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.739.mha'),
        os.path.join(data_dir, '0011/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.745.mha'),
        os.path.join(data_dir, '0012/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.751.mha'),
        os.path.join(data_dir, '0013/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.757.mha'),
        os.path.join(data_dir, '0014/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.763.mha'),
        os.path.join(data_dir, '0015/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.769.mha'),
        os.path.join(data_dir, '0022/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.775.mha'),
        os.path.join(data_dir, '0024/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.781.mha'),
        os.path.join(data_dir, '0025/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.787.mha'),
        os.path.join(data_dir, '0026/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.793.mha'),
        os.path.join(data_dir, '0027/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.799.mha')
    ]
    return


# Data info
def useData_BRATS2_Synthetic_midas():
    global im_names, data_dir
    im_names = [
        os.path.join(data_dir, 'SimBRATS_HG0022/SimBRATS_HG0022_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0016/SimBRATS_HG0016_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0024/SimBRATS_HG0024_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0008/SimBRATS_HG0008_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0009/SimBRATS_HG0009_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0020/SimBRATS_HG0020_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0025/SimBRATS_HG0025_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0001/SimBRATS_HG0001_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0010/SimBRATS_HG0010_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0021/SimBRATS_HG0021_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0004/SimBRATS_HG0004_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0005/SimBRATS_HG0005_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0014/SimBRATS_HG0014_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0017/SimBRATS_HG0017_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0007/SimBRATS_HG0007_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0015/SimBRATS_HG0015_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0023/SimBRATS_HG0023_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0019/SimBRATS_HG0019_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0003/SimBRATS_HG0003_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0011/SimBRATS_HG0011_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0018/SimBRATS_HG0018_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0006/SimBRATS_HG0006_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0012/SimBRATS_HG0012_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0002/SimBRATS_HG0002_T1.mha'),
        os.path.join(data_dir, 'SimBRATS_HG0013/SimBRATS_HG0013_T1.mha')
    ]
    return


def useData_BRATS2_Synthetic_dtu():
    global im_names, data_dir
    im_names = [
        os.path.join(data_dir, '0001/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.866.mha'),
        os.path.join(data_dir, '0002/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.872.mha'),
        os.path.join(data_dir, '0003/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.878.mha'),
        os.path.join(data_dir, '0004/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.884.mha'),
        os.path.join(data_dir, '0005/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.890.mha'),
        os.path.join(data_dir, '0006/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.896.mha'),
        os.path.join(data_dir, '0007/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.902.mha'),
        os.path.join(data_dir, '0008/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.908.mha'),
        os.path.join(data_dir, '0009/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.914.mha'),
        os.path.join(data_dir, '0010/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.920.mha'),
        os.path.join(data_dir, '0011/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.926.mha'),
        os.path.join(data_dir, '0012/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.932.mha'),
        os.path.join(data_dir, '0013/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.938.mha'),
        os.path.join(data_dir, '0014/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.944.mha'),
        os.path.join(data_dir, '0015/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.950.mha'),
        os.path.join(data_dir, '0016/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.956.mha'),
        os.path.join(data_dir, '0017/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.962.mha'),
        os.path.join(data_dir, '0018/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.968.mha'),
        os.path.join(data_dir, '0019/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.974.mha'),
        os.path.join(data_dir, '0020/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.980.mha'),
        os.path.join(data_dir, '0021/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.986.mha'),
        os.path.join(data_dir, '0022/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.992.mha'),
        os.path.join(data_dir, '0023/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.998.mha'),
        os.path.join(data_dir, '0024/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.1004.mha'),
        os.path.join(data_dir, '0025/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.1010.mha')
    ]
    return


def useData_BRATS2_midas():
    global im_names, data_dir
    im_names = [
        os.path.join(data_dir, 'BRATS_HG0026/BRATS_HG0026_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0012/BRATS_HG0012_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0009/BRATS_HG0009_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0024/BRATS_HG0024_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0007/BRATS_HG0007_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0014/BRATS_HG0014_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0005/BRATS_HG0005_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0001/BRATS_HG0001_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0003/BRATS_HG0003_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0027/BRATS_HG0027_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0002/BRATS_HG0002_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0004/BRATS_HG0004_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0013/BRATS_HG0013_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0008/BRATS_HG0008_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0011/BRATS_HG0011_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0006/BRATS_HG0006_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0022/BRATS_HG0022_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0015/BRATS_HG0015_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0025/BRATS_HG0025_T1.mha'),
        os.path.join(data_dir, 'BRATS_HG0010/BRATS_HG0010_T1.mha')
    ]
    return


def useData_BRATS2_dtu():
    global im_names, data_dir
    im_names = [
        os.path.join(data_dir, '/HG/0001/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.685.mha'),
        os.path.join(data_dir, '/HG/0002/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.692.mha'),
        os.path.join(data_dir, '/HG/0003/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.698.mha'),
        os.path.join(data_dir, '/HG/0004/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.704.mha'),
        os.path.join(data_dir, '/HG/0005/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.710.mha'),
        os.path.join(data_dir, '/HG/0006/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.716.mha'),
        os.path.join(data_dir, '/HG/0007/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.722.mha'),
        os.path.join(data_dir, '/HG/0008/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.728.mha'),
        os.path.join(data_dir, '/HG/0009/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.734.mha'),
        os.path.join(data_dir, '/HG/0010/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.740.mha'),
        os.path.join(data_dir, '/HG/0011/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.746.mha'),
        os.path.join(data_dir, '/HG/0012/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.752.mha'),
        os.path.join(data_dir, '/HG/0013/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.758.mha'),
        os.path.join(data_dir, '/HG/0014/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.764.mha'),
        os.path.join(data_dir, '/HG/0015/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.770.mha'),
        os.path.join(data_dir, '/HG/0022/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.776.mha'),
        os.path.join(data_dir, '/HG/0024/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.782.mha'),
        os.path.join(data_dir, '/HG/0025/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.788.mha'),
        os.path.join(data_dir, '/HG/0026/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.794.mha'),
        os.path.join(data_dir, '/HG/0027/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.800.mha')
    ]
    return


###############################  the main pipeline #############################
def runIteration(Y, currentIter, lamda, gridSize, maxDisp):
    low_rank, sparse, n_iter, rank, sparsity, sum_sparse = rpca(Y, lamda)
    saveImagesFromDM(low_rank, os.path.join(result_folder, 'Iter' + str(currentIter) + '_LowRank_'), reference_im_name)
    saveImagesFromDM(sparse, os.path.join(result_folder, 'Iter' + str(currentIter) + '_Sparse_'), reference_im_name)

    # Visualize and inspect
    fig = plt.figure(figsize=(15, 5))
    showSlice(Y, ' Input', plt.cm.gray, 0, reference_im_name)
    showSlice(low_rank, ' low rank', plt.cm.gray, 1, reference_im_name)
    showSlice(sparse, ' sparse', plt.cm.gray, 2, reference_im_name)
    plt.savefig(os.path.join(result_folder, 'Iter' + str(currentIter) + '_w_' + str(lamda) + '.png'))
    fig.clf()
    plt.close(fig)

    num_of_data = Y.shape[1]
    del low_rank, sparse, Y

    print 'start image registrations'
    # Register low-rank images to the reference (healthy) image,
    # and update the input images to the next iteration
    ps = []

    for i in range(num_of_data):
        movingIm = os.path.join(result_folder, 'Iter' + str(currentIter) + '_LowRank_' + str(i) + '.nrrd')
        outputIm = os.path.join(result_folder, 'Iter' + str(currentIter) + '_Deformed_LowRank' + str(i) + '.nrrd')
        outputTransform = os.path.join(result_folder, 'Iter' + str(currentIter) + '_Transform_' + str(i) + '.tfm')
        outputDVF = os.path.join(result_folder, 'Iter' + str(currentIter) + '_DVF_' + str(i) + '.nrrd')
        previousInputImage = os.path.join(result_folder, 'Iter' + str(currentIter - 1) + '_T1_' + str(i) + '.nrrd')
        logFile = open(os.path.join(result_folder, 'Iter' + str(currentIter) + '_RUN_' + str(i) + '.log'), 'w')

        # pipe steps sequencially
        cmd = BSplineReg_BRAINSFit(software.EXE_BRAINSFit,
                                   reference_im_name, movingIm, outputIm, outputTransform, gridSize, maxDisp)

        cmd += ';' + ConvertTransform(software.EXE_BSplineToDeformationField,
                                      reference_im_name, outputTransform, outputDVF)

        outputComposedDVFIm = os.path.join(result_folder,
                                           'Iter' + str(currentIter) + '_Composed_DVF_' + str(i) + '.nrrd')
        initialInputImage = os.path.join(result_folder, 'Iter0_T1_' + str(i) + '.nrrd')
        newInputImage = os.path.join(result_folder, 'Iter' + str(currentIter) + '_T1_' + str(i) + '.nrrd')

        # compose deformations
        COMPOSE_DVF = True
        if COMPOSE_DVF:
            DVFImageList = []
            for k in range(currentIter):
                DVFImageList.append(os.path.join(result_folder, 'Iter' + str(k + 1) + '_DVF_' + str(i) + '.nrrd'))

                cmd += ';' + composeMultipleDVFs(software.EXE_ComposeMultiTransform,
                                                 reference_im_name, DVFImageList, outputComposedDVFIm)

        cmd += ";" + updateInputImageWithDVF(software.EXE_BRAINSResample, initialInputImage, reference_im_name,
                                             outputComposedDVFIm, newInputImage)
        process = subprocess.Popen(cmd, stdout=logFile, shell=True)
        ps.append(process)

    for p in ps:
        p.wait()

    return sparsity, sum_sparse


def showReferenceImage(reference_im_name):
    im_ref = sitk.ReadImage(reference_im_name)  # image in SITK format
    im_ref_array = sitk.GetArrayFromImage(im_ref)  # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape  # get 3D volume shape
    vector_length = z_dim * x_dim * y_dim

    # display reference image
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(131)
    implot = plt.imshow(np.flipud(im_ref_array[z_dim / 2, :, :]), plt.cm.gray)
    plt.subplot(132)
    implot = plt.imshow(np.flipud(im_ref_array[:, x_dim / 2, :]), plt.cm.gray)
    plt.subplot(133)
    implot = plt.imshow(np.flipud(im_ref_array[:, :, y_dim / 2]), plt.cm.gray)
    plt.axis('off')
    plt.title('healthy atlas')
    fig.clf()
    del im_ref, im_ref_array
    return


# Affine registering each input image to the reference(healthy atlas)  image
def affineRegistrationStep():
    num_of_data = len(selection)
    for i in range(num_of_data):
        outputIm = os.path.join(result_folder, 'Iter0_T1_' + str(i) + '.nrrd')
        AffineReg(software.EXE_BRAINSFit, reference_im_name, im_names[selection[i]], outputIm)
    return


#######################################  manual setting  ##################################
parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description=__doc__
)
parser.add_argument("--result_folder", nargs=1, type=str, required=True, help='Directory to store the output images')
parser.add_argument("--reference_im_name", nargs=1, type=str, required=True, help='Reference Image')
parser.add_argument("--software", nargs=1, type=str, required=True, help='Software Configuration File')
parser.add_argument("--data_dir", nargs=1, type=str, required=True, help='Directory containing BRATS data')
args = parser.parse_args()
software = loadConfiguration(args.software[0], 'software')
required_software = ['EXE_BRAINSFit',
                     'EXE_BSplineToDeformationField',
                     'EXE_ComposeMultiTransform',
                     'EXE_BRAINSResample',
                     ]
for i in required_software:
    if not hasattr(software, i):
        print "Requires " + i + "to be set in the software configuration file."
        sys.exit(1)
# global variables
im_names = []
reference_im_name = args.reference_im_name[0]  # 'SRI24/T1_Crop.nii.gz'
######  set data folders
data_dir = args.data_dir[0]
lamda = 0.9

useData_BRATS2_midas()
selection = range(8)
result_folder = args.result_folder[0]  #'BRATS/BRATS-2/Image_Data/RegulateBspline_w' + str(lamda)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# useData_BRATS2_Synthetic()
# result_folder = 'BRATS/BRATS-2/Synthetic_Data/RegulateBSpline_w'+str(lamda)
selection = range(8)


# @profile
def main():
    import time
    global lamda

    s = time.clock()
    # save script to the result folder for paramter checkups
    shutil.copy(inspect.getfile(inspect.currentframe()), result_folder)

    # showReferenceImage(reference_im_name)
    affineRegistrationStep()

    # sys.stdout = open(result_folder+'/RUN.log', "w")
    im_ref = sitk.ReadImage(reference_im_name)  # image in SITK format
    im_ref_array = sitk.GetArrayFromImage(im_ref)  # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape  # get 3D volume shape
    vector_length = z_dim * x_dim * y_dim
    del im_ref, im_ref_array

    num_of_data = len(selection)

    NUM_OF_ITERATIONS = 15
    sparsity = np.zeros(NUM_OF_ITERATIONS)
    sum_sparse = np.zeros(NUM_OF_ITERATIONS)

    gridSize = [6, 8, 6]
    Y = np.zeros((vector_length, num_of_data))
    for iterCount in range(1, NUM_OF_ITERATIONS + 1):

        maxDisp = z_dim / gridSize[2] / 4
        print 'Iteration ' + str(iterCount) + ' lamda=%f' % lamda
        print 'Grid size: ', gridSize
        print 'Max Displacement: ', maxDisp
        a = time.clock()

        # prepare data matrix
        for i in range(num_of_data):
            im_file = os.path.join(result_folder, 'Iter' + str(iterCount - 1) + '_T1_' + str(i) + '.nrrd')
            tmp = sitk.ReadImage(im_file)
            tmp = sitk.GetArrayFromImage(tmp)
            Y[:, i] = tmp.reshape(-1)
            del tmp

        sparsity[iterCount - 1], sum_sparse[iterCount - 1] = runIteration(Y, iterCount, lamda, gridSize, maxDisp)
        # lamda += 0.025
        gc.collect()
        if iterCount % 2 == 0:
            if gridSize[0] < 10:
                gridSize = np.add(gridSize, [1, 1, 1])

        # a = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print 'Current memory usage :',a/1024.0/1024.0,'GB'

        # h = hpy()
        # print h.heap()
        b = time.clock()
        c = b - a
        print 'Iteration took  %f mins' % (c / 60.0)

    e = time.clock()
    l = e - s
    print 'Total running time:  %f mins' % (l / 60.0)

    # plot the sparsity curve
    plt.figure()
    plt.plot(range(NUM_OF_ITERATIONS), sparsity)
    plt.savefig(os.path.join(result_folder, 'sparsity.png'))

    plt.figure()
    plt.plot(range(NUM_OF_ITERATIONS), sum_sparse)
    plt.savefig(os.path.join(result_folder, 'sumSparse.png'))


if __name__ == "__main__":
    main()
