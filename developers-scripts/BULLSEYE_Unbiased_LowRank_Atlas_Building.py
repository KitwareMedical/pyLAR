#!/usr/bin/env python
"""Bulls eye unbiased low rank atlas building"""
import sys
import os
import inspect
import argparse
import shutil
import numpy as np
myfilepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
sys.path.insert(0, os.path.abspath(os.path.join(myfilepath, '../')))
from low_rank_atlas_iter import *

parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description=__doc__
)
parser.add_argument("--result_folder", nargs=1, type=str, required=True, help='Directory to store the output images')
parser.add_argument("--software", nargs=1, type=str, required=True, help='Software Configuration File')
parser.add_argument("--bullseye_data_folder", nargs=1, type=str, required=True, help="Folder containing bulls eye data,\
 generated with 'gen_3D_simulation_data.py'")
args = parser.parse_args()
software = loadConfiguration(args.software[0], 'software')
required_software = ['EXE_AverageImages',
                     'EXE_BRAINSDemonWarp',
                     'EXE_BSplineToDeformationField',
                     'EXE_BRAINSResample',
                     'EXE_ComposeMultiTransform'
                     ]
for i in required_software:
    if not hasattr(software, i):
        print "Requires " + i + "to be set in the software configuration file."
        sys.exit(1)
# global settings
# data_folder = '/Users/xiaoxiaoliu/work/data/BullEyeSimulation'
data_folder = args.bullseye_data_folder[0]
# result_folder = data_folder +'/low-rank-atlas-building'
result_folder = args.result_folder[0]
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
im_names = [os.path.join(data_folder, 'simu1.nrrd'),
            os.path.join(data_folder, 'simu2.nrrd'),
            os.path.join(data_folder, 'simu3.nrrd'),
            os.path.join(data_folder, 'simu4.nrrd')]
reference_im_name = os.path.join(data_folder, 'fMeanSimu.nrrd')

# data selection and global parameters, prepare iter0 data
selection = [0, 1, 2, 3]
num_of_data = len(selection)
for i in range(num_of_data):
    iter0fn = os.path.join(result_folder, 'Iter0' + '_simu_' + str(i) + '.nrrd')
    simufn = im_names[selection[i]]
    shutil.copyfile(simufn, iter0fn)

# profile data size, save into global variables
im_ref = sitk.ReadImage(reference_im_name)  # image in SITK format
im_ref_array = sitk.GetArrayFromImage(im_ref)  # get numpy array
z_dim, x_dim, y_dim = im_ref_array.shape  # get 3D volume shape
vector_length = z_dim * x_dim * y_dim

plt.figure()
implot = plt.imshow(im_ref_array[32, :, :], plt.cm.gray)
plt.title('healthy atlas')

slice_nr = 32  # just for vis purpose


###############################  the main pipeline #############################
def AverageImages(currentIter, atlasIm):
    executable = software.EXE_AverageImages  # '/Users/xiaoxiaoliu/work/bin/ANTS/bin/AverageImages'
    listOfImages = []
    for i in range(num_of_data):
        movingIm = os.path.join(result_folder, 'Iter' + str(currentIter) + '_LowRank_' + str(i) + '.nrrd')
        listOfImages.append(movingIm)
    arguments = ' 3 ' + atlasIm + '  0  ' + ' '.join(listOfImages)
    cmd = executable + ' ' + arguments
    tempFile = open(os.path.join(result_folder, 'average.log'), 'w')
    process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
    process.wait()
    tempFile.close()
    return


def runIteration(currentIter, lamda, gridSize=[3, 3, 3]):
    Y = np.zeros((vector_length, num_of_data))
    for i in range(num_of_data):
        im_file = os.path.join(result_folder, 'Iter%d_simu_%d.nrrd' % (currentIter - 1, i))
        tmp = sitk.ReadImage(im_file)
        # print 'read:'+ im_file
        tmp = sitk.GetArrayFromImage(tmp)
        Y[:, i] = tmp.reshape(-1)

    low_rank, sparse, n_iter, rank, sparsity, sumSparse = rpca(Y, lamda)
    saveImagesFromDM(low_rank, os.path.join(result_folder, 'Iter' + str(currentIter) + '_LowRank_'), reference_im_name)
    saveImagesFromDM(sparse, os.path.join(result_folder, 'Iter' + str(currentIter) + '_Sparse_'), reference_im_name)

    # Visualize and inspect
    fig = plt.figure(figsize=(15, 5))
    showSlice(Y, 'Iter' + str(currentIter) + ' Input', plt.cm.gray, 0, reference_im_name)
    showSlice(low_rank, 'Iter' + str(currentIter) + ' low rank', plt.cm.gray, 1, reference_im_name)
    showSlice(sparse, 'Iter' + str(currentIter) + ' sparse', plt.cm.gray, 2, reference_im_name)
    plt.savefig(os.path.join(result_folder, 'Iter' + str(currentIter) + '_w_' + str(lamda) + '.png'))
    plt.close(fig)

    # Register low-rank images to the reference (healthy) image, and update the input images to the next iteration
    atlas_im_name = os.path.join(result_folder, 'Iter' + str(currentIter) + '_atlas.nrrd')
    # Average lowrank images
    AverageImages(currentIter, atlas_im_name)

    # visualize the difference image  lowrank - reference
    im_ref_vec = im_ref_array.reshape(-1)
    plt.figure(figsize=(15, 5))
    for i in range(num_of_data):
        plt.subplot2grid((1, num_of_data + 1), (0, i))
        a = np.array((low_rank[:, i]).reshape(-1) - im_ref_vec)
        im = a.reshape(z_dim, x_dim, y_dim)
        implot = plt.imshow(im[slice_nr, :, :], plt.cm.gray)
        plt.axis('off')
        plt.title('Iter' + str(currentIter) + '_simu_' + str(i))
    # visulizat atlat difference    
    plt.subplot2grid((1, num_of_data + 1), (0, num_of_data))
    im_atlas = sitk.ReadImage(atlas_im_name)
    im_atlas_array = sitk.GetArrayFromImage(im_atlas)
    a = np.array(im_atlas_array.reshape(-1) - im_ref_vec)
    im = a.reshape(z_dim, x_dim, y_dim)
    implot = plt.imshow(im[slice_nr, :, :], plt.cm.gray)
    plt.axis('off')
    plt.title('Iter' + str(currentIter) + '_atlas_diff' + str(i))
    plt.colorbar()
    plt.savefig(os.path.join(result_folder, 'Differ_Lowrank_Iter' + str(currentIter) + '.png'))
    plt.close(fig)

    for i in range(num_of_data):
        movingIm = os.path.join(result_folder, 'Iter' + str(currentIter - 1) + '_simu_' + str(i) + '.nrrd')
        outputIm = os.path.join(result_folder, 'Iter' + str(currentIter) + '_deformed_' + str(i) + '.nrrd')
        outputTransform = os.path.join(result_folder, 'Iter' + str(currentIter) + '_Transform_' + str(i) + '.tfm')
        outputDVF = os.path.join(result_folder, 'Iter' + str(currentIter) + '_DVF_' + str(i) + '.nrrd')

        outputComposedDVFIm = os.path.join(result_folder,
                                           'Iter' + str(currentIter) + '_Composed_DVF_' + str(i) + '.nrrd')
        newInputImage = os.path.join(result_folder, 'Iter' + str(currentIter) + '_simu_' + str(i) + '.nrrd')
        initialInputImage = os.path.join(result_folder, 'Iter0_simu_' + str(i) + '.nrrd')

        logFile = open(os.path.join(result_folder, 'iteration' + str(i) + '.log'), 'w')

        cmd = DemonsReg(software.EXE_BRAINSDemonWarp, atlas_im_name, movingIm, outputIm, outputDVF)
        # cmd = BSplineReg(atlas_im_name,movingIm,outputIm, outputTransform,gridSize)
        # cmd = cmd + ';' + ConvertTransform(software.EXE_BSplineToDeformationField,
        #                          reference_im_name, outputTransform, outputDVF)

        # compose deformations then apply to the original input image
        if COMPOSE_DVF is True:
            DVFImageList = []
            for k in range(currentIter):
                DVFImageList.append(os.path.join(result_folder, 'Iter' + str(k + 1) + '_DVF_' + str(i) + '.nrrd'))

            cmd = cmd + ';' + composeMultipleDVFs(software.EXE_ComposeMultiTransform,
                                                  reference_im_name, DVFImageList, outputComposedDVFIm)
            cmd = cmd + ';' + updateInputImageWithDVF(software.EXE_BRAINSResample,
                                                      initialInputImage, reference_im_name, outputComposedDVFIm,
                                                      newInputImage)

            # cmd = cmd + ';' + WarpImageMultiDVF(initialInputImage,reference_im_name,DVFImageList,newInputImage)

        else:
            # update from previous Image
            previousInputImage = os.path.join(result_folder, 'Iter%d_simu_%d.nrrd' % (currentIter - 1, i))
            cmd2 = updateInputImageWithDVF(software.EXE_BRAINSResample,
                                           previousInputImage, reference_im_name, outputDVF, newInputImage)
            cmd = cmd1 + ";" + cmd2
        print cmd
        process = subprocess.Popen(cmd, stdout=logFile, shell=True)
        process.wait()
        logFile.close()

    return


# main
shutil.copy(inspect.getfile(inspect.currentframe()),result_folder)

NUM_OF_ITERATIONS = 15

COMPOSE_DVF = True
lamda = 1.5

for i in range(NUM_OF_ITERATIONS):
    print 'iter' + str(i + 1)
    runIteration(i + 1, lamda)
    # if lamda < 1.5:
    #    lamda = lamda +0.1
    # gridSize = np.add(gridSize , [2,2,2])

# check atlas similarities
MSE = np.zeros(NUM_OF_ITERATIONS)
for i in range(NUM_OF_ITERATIONS):
    atlasIm = os.path.join(result_folder, 'Iter' + str(i + 1) + '_atlas.nrrd')
    im = sitk.ReadImage(atlasIm)  # image in SITK format
    im_array = sitk.GetArrayFromImage(im)
    MSE[i] = np.sum(np.square(im_ref_array - im_array))
#        im= sitk.ReadImage(atlasIm) # image in SITK format
#       im_array = sitk.GetArrayFromImage(im) # get numpy array
#      figure()
#     implot = plt.imshow(im_array[32,:,:],cm.gray)
#    plt.title('Iter'+str(i)+ ' atlas')

plt.figure()
plt.plot(range(NUM_OF_ITERATIONS), MSE)
plt.savefig(os.path.join(result_folder, 'MSE.png'))
