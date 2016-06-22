#!/usr/bin/env python
import shutil
import sys
import os
import argparse
import numpy as np
import inspect
myfilepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
sys.path.insert(0, os.path.abspath(os.path.join(myfilepath, '../')))
from low_rank_atlas_iter import *


###############################  the main pipeline #############################
def runIteration(Y, currentIter, lamda, gridSize, maxDisp):
    low_rank, sparse, n_iter, rank, sparsity, sum_sparse = rpca(Y, lamda)
    saveImagesFromDM(low_rank, os.path.join(result_folder, 'Iter' + str(currentIter) + '_LowRank_'), reference_im_name)
    saveImagesFromDM(sparse, os.path.join(result_folder, 'Iter' + str(currentIter) + '_Sparse_'), reference_im_name)

    # Visualize and inspect
    fig = plt.figure(figsize=(15, 5))
    showSlice(Y, 'Iter' + str(currentIter) + ' Input', plt.cm.gray, 0, reference_im_name)
    showSlice(low_rank, 'Iter' + str(currentIter) + ' low rank', plt.cm.gray, 1, reference_im_name)
    showSlice(sparse, 'Iter' + str(currentIter) + ' sparse', plt.cm.gray, 2, reference_im_name)
    plt.savefig(os.path.join(result_folder, 'Iter' + str(currentIter) + '_w_' + str(lamda) + '.png'))
    fig.clf()
    plt.close(fig)

    num_of_data = Y.shape[1]
    del low_rank, sparse, Y

    atlas_im_name = os.path.join(result_folder, 'Iter' + str(currentIter) + '_atlas.nrrd')
    # Average low rank images
    listOfImages = []
    num_of_data = len(selection)
    for i in range(num_of_data):
        lrIm = os.path.join(result_folder, 'Iter' + str(currentIter) + '_LowRank_' + str(i) + '.nrrd')
        listOfImages.append(lrIm)
    AverageImages(software.EXE_AverageImages, listOfImages, atlas_im_name)

    ps = []
    for i in range(num_of_data):
        movingIm = os.path.join(result_folder, 'Iter' + str(currentIter) + '_LowRank_' + str(i) + '.nrrd')
        outputIm = os.path.join(result_folder, 'Iter' + str(currentIter) + '_Deformed_LowRank' + str(i) + '.nrrd')
        outputTransform = os.path.join(result_folder, 'Iter' + str(currentIter) + '_Transform_' + str(i) + '.tfm')
        outputDVF = os.path.join(result_folder, 'Iter' + str(currentIter) + '_DVF_' + str(i) + '.nrrd')
        previousInputImage = os.path.join(result_folder, 'Iter' + str(currentIter - 1) + '_Flair_' + str(i) + '.nrrd')
        outputComposedDVFIm = os.path.join(result_folder,
                                           'Iter' + str(currentIter) + '_Composed_DVF_' + str(i) + '.nrrd')
        initialInputImage = os.path.join(result_folder, 'Iter0_Flair_' + str(i) + '.nrrd')
        newInputImage = os.path.join(result_folder, 'Iter' + str(currentIter) + '_Flair_' + str(i) + '.nrrd')
        logFile = open(os.path.join(result_folder, 'Iter' + str(currentIter) + '_RUN_' + str(i) + '.log'), 'w')

        # pipe steps sequencially
        cmd = BSplineReg_BRAINSFit(software.EXE_BRAINSFit, atlas_im_name, movingIm, outputIm, outputTransform, gridSize, maxDisp)

        cmd += ';' + ConvertTransform(software.EXE_BSplineToDeformationField, atlas_im_name, outputTransform, outputDVF)

        outputComposedDVFIm = os.path.join(result_folder,
                                           'Iter' + str(currentIter) + '_Composed_DVF_' + str(i) + '.nrrd')
        initialInputImage = os.path.join(result_folder, 'Iter0_Flair_' + str(i) + '.nrrd')
        newInputImage = os.path.join(result_folder, 'Iter' + str(currentIter) + '_Flair_' + str(i) + '.nrrd')

        # compose deformations
        COMPOSE_DVF = True
        if COMPOSE_DVF:
            DVFImageList = []
            for k in range(currentIter):
                DVFImageList.append(os.path.join(result_folder, 'Iter' + str(k + 1) + '_DVF_' + str(i) + '.nrrd'))

                cmd += ';' + composeMultipleDVFs(software.EXE_ComposeMultiTransform, reference_im_name, DVFImageList, outputComposedDVFIm)

        cmd += ";" + updateInputImageWithDVF(software.EXE_BRAINSResample, initialInputImage, reference_im_name,\
                                             outputComposedDVFIm, newInputImage)
        print"--------------------------"
        print cmd
        print"--------------------------"
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
        outputIm = os.path.join(result_folder, 'Iter0_Flair_' + str(i) + '.nrrd')
        AffineReg(software.EXE_BRAINSFit, reference_im_name, im_names[selection[i]], outputIm)
    return


#######################################  main ##################################
parser = argparse.ArgumentParser(
        prog=sys.argv[0]
)
parser.add_argument("--result_folder", nargs=1, type=str, required=True, help='Directory to store the output images')
parser.add_argument("--reference_im_name", nargs=1, type=str, required=True, help='Reference Image')
parser.add_argument("--software", nargs=1, type=str, required=True, help='Software Configuration File')
parser.add_argument("--image_list", nargs=1, type=str, required=True, help='File Containing the List of Images')
args = parser.parse_args()
software = loadConfiguration(args.software[0], 'software')
required_software = ['EXE_BRAINSFit',
                     'EXE_BSplineToDeformationField',
                     'EXE_ComposeMultiTransform',
                     'EXE_BRAINSResample',
                     'EXE_AverageImages'
                     ]
for i in required_software:
    if not hasattr(software, i):
        print "Requires "+i+"to be set in the software configuration file."
        sys.exit(1)
# reference_im_name = '/Users/xiaoxiaoliu/work/data/SRI24/T1_Crop.nii.gz'
reference_im_name = args.reference_im_name[0]
im_names = readTxtIntoList(args.image_list[0])

lamda = 0.7
# result_folder = '/Users/xiaoxiaoliu/work/data/BRATS/BRATS-2/Image_Data/Unbiased_Atlas_Flair_w' + str(lamda)
selection = [0, 1, 3, 4, 6, 7, 9, 10]
for i in selection:
    if len(im_names) < i + 1:
        print "Image list must contain at least " + i + " elements for this test."
        sys.exit(1)

result_folder = args.result_folder[0]
print 'Results will be stored in:', result_folder
if not os.path.exists(result_folder):
    os.makedirs(result_folder)


# @profile
def main():
    import time

    global lamda
    s = time.clock()
    # save script to the result folder for paramter checkups
    currentPyFile = os.path.realpath(__file__)
    print currentPyFile
    shutil.copy(currentPyFile, result_folder)

    # showReferenceImage(reference_im_name)
    affineRegistrationStep()

    sys.stdout = open( os.path.join(result_folder, 'RUN.log'), "w")
    im_ref = sitk.ReadImage(reference_im_name)  # image in SITK format
    im_ref_array = sitk.GetArrayFromImage(im_ref)  # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape  # get 3D volume shape
    vector_length = z_dim * x_dim * y_dim
    del im_ref, im_ref_array

    num_of_data = len(selection)

    NUM_OF_ITERATIONS = 8
    sparsity = np.zeros(NUM_OF_ITERATIONS)
    sum_sparse = np.zeros(NUM_OF_ITERATIONS)

    gridSize = [6, 8, 6]
    Y = np.zeros((vector_length, num_of_data))
    for iterCount in range(1, NUM_OF_ITERATIONS + 1):

        maxDisp = z_dim / gridSize[2] / 4
        print 'Iteration ' + str(iterCount) + ' lamda=%f' % lamda
        print 'Grid size: ', gridSize
        print 'Max Displacement: ', maxDisp

        # prepare data matrix
        for i in range(num_of_data):
            im_file = os.path.join(result_folder, 'Iter' + str(iterCount - 1) + '_Flair_' + str(i) + '.nrrd')
            tmp = sitk.ReadImage(im_file)
            tmp = sitk.GetArrayFromImage(tmp)
            Y[:, i] = tmp.reshape(-1)
            del tmp

        sparsity[iterCount - 1], sum_sparse[iterCount - 1] = runIteration(Y, iterCount, lamda, gridSize, maxDisp)
        gc.collect()
        lamda += 0.025
        if iterCount % 2 == 0 and gridSize[0] < 10:
            gridSize = np.add(gridSize, [1, 1, 1])
            # a = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print 'Current memory usage :',a/1024.0/1024.0,'GB'
            # h = hpy()
            # print h.heap()

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

    for i in range(NUM_OF_ITERATIONS):
        atlasIm = os.path.join(result_folder, 'Iter' + str(i + 1) + '_atlas.nrrd')
        im = sitk.ReadImage(atlasIm)  # image in SITK format
        im_array = sitk.GetArrayFromImage(im)
        z_dim, x_dim, y_dim = im_array.shape  # get 3D volume shape
        plt.figure()
        implot = plt.imshow(im_array[z_dim / 2, :, :], plt.cm.gray)
        plt.title('Iter' + str(i) + ' atlas')
        plt.savefig(os.path.join(result_folder, 'Iter' + str(i) + '_atlas.png'))


if __name__ == "__main__":
    main()
