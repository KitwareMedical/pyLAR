# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys
sys.path.append('../')
from low_rank_atlas_iter import *

# <codecell>

# global variables
data_folder = ''
reference_im_name = ''
result_folder = ''
selection = []
im_names =[]
USE_HEALTHY_ATLAS = True

###############################  the main pipeline #############################
def runIteration(Y,currentIter,lamda,gridSize,maxDisp):
    low_rank, sparse, n_iter,rank, sparsity, sum_sparse = rpca(Y,lamda)
    saveImagesFromDM(low_rank,result_folder+'/'+ 'Iter'+str(currentIter) +'_LowRank_', reference_im_name)
    saveImagesFromDM(sparse,result_folder+'/'+ 'Iter'+str(currentIter) +'_Sparse_', reference_im_name)

    # Visualize and inspect
    fig = plt.figure(figsize=(15,5))
    showSlice(Y, 'Iter'+str(currentIter) +' Input',plt.cm.gray,0,reference_im_name)
    showSlice(low_rank,'Iter'+str(currentIter) +' low rank',plt.cm.gray,1, reference_im_name)
    showSlice(sparse,'Iter'+str(currentIter) +' sparse',plt.cm.gray,2, reference_im_name)
    plt.savefig(result_folder+'/'+'Iter'+ str(currentIter)+'_w_'+str(lamda)+'.png')
    fig.clf()
    plt.close(fig)

    num_of_data = Y.shape[1]
    del low_rank, sparse,Y

    
    if not USE_HEALTHY_ATLAS:
        reference_im_name = result_folder+'/Iter'+ str(currentIter) +'_atlas.nrrd'
      # Average lowrank images
        listOfImages = []
        num_of_data = len(selection)
        for i in range(num_of_data):
            lrIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_LowRank_' + str(i)  +'.nrrd'
            listOfImages.append(lrIm)
        AverageImages(listOfImages,reference_im_name)

    ps=[]
    for i in range(num_of_data):


        logFile = open(result_folder+'/Iter'+str(currentIter)+'_RUN_'+ str(i)+'.log', 'w')

        # pipe steps sequencially
        previousIterDVF = result_folder + '/'+ 'Iter'+ str(currentIter-1)+'_DVF_' + str(i) +  '.nrrd'
        inverseDVF = result_folder + '/'+ 'Iter'+ str(currentIter-1)+'_INV_DVF_' + str(i) +  '.nrrd'
        cmd = genInverseDVF(previousIterDVF,inverseDVF)


        lowRankIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_LowRank_' + str(i)  +'.nrrd'
        invWarpedlowRankIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_InvWarped_LowRank_' + str(i)  +'.nrrd'
        cmd += ";"+updateInputImageWithDVF( lowRankIm, reference_im_name, inverseDVF, invWarpedlowRankIm)


        outputIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Deformed_LowRank' + str(i)  + '.nrrd'
        outputTransform = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Transform_' + str(i) +  '.tfm'
        outputDVF = result_folder+'/'+ 'Iter'+ str(currentIter)+'_DVF_' + str(i) +  '.nrrd'

        initialInputImage= result_folder+'/Iter0_Flair_' +str(i) +  '.nrrd'
        newInputImage = result_folder+'/Iter'+ str(currentIter)+'_Flair_' +str(i) +  '.nrrd'
        cmd += ";"+BSplineReg_BRAINSFit(reference_im_name,invWarpedlowRankIm,outputIm,outputTransform,gridSize, maxDisp)
        cmd +=';'+ ConvertTransform(reference_im_name,outputTransform,outputDVF)


        initialInputImage= result_folder+'/Iter0_Flair_' +str(i) +  '.nrrd'
        newInputImage = result_folder+'/Iter'+ str(currentIter)+'_Flair_' +str(i) +  '.nrrd'
        cmd += ";" + updateInputImageWithDVF(initialInputImage,reference_im_name, \
                                       outputDVF,newInputImage)

        process = subprocess.Popen(cmd, stdout = logFile, shell = True)
        ps.append(process)
    for  p in ps:
        p.wait()
    return sparsity, sum_sparse


def showReferenceImage(reference_im_name):
    im_ref = sitk.ReadImage(reference_im_name) # image in SITK format
    im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
    vector_length = z_dim* x_dim*y_dim

    # display reference image
    fig = plt.figure(figsize=(15,5))
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

# Affine registering each input image to the reference(healthy atlas)  image
def affineRegistrationStep():
    num_of_data = len(selection)
    for i in range(num_of_data):
        outputIm =  result_folder+'/Iter0_Flair_' + str(i)  + '.nrrd'
        AffineReg(reference_im_name,im_names[selection[i]],outputIm)
    return


#######################################  main ##################################
reference_im_name = '/home/xiaoxiao/work/data/SRI24/T1_Crop.nii.gz'

data_folder= '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data'
im_names = readTxtIntoList(data_folder +'/Flair_FN.txt')

lamda = 0.7
result_folder = '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/Unbiased_Atlas_Flair_w'+str(lamda)
selection = [0,1,3,4,6,7,9,10]

print 'Results will be stored in:',result_folder
if not os.path.exists(result_folder):
	os.system('mkdir '+ result_folder)

#@profile
def main():
    import time
    import resource

    global lamda
    s = time.clock()
    # save script to the result folder for paramter checkups
    currentPyFile = os.path.realpath(__file__)
    print currentPyFile
    os.system('cp   ' + currentPyFile+' ' +result_folder)


    #showReferenceImage(reference_im_name)
    affineRegistrationStep()

    sys.stdout = open(result_folder+'/RUN.log', "w")
    im_ref = sitk.ReadImage(reference_im_name) # image in SITK format
    im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
    vector_length = z_dim * x_dim * y_dim
    del im_ref, im_ref_array

    num_of_data = len(selection)


    NUM_OF_ITERATIONS = 8
    sparsity = np.zeros(NUM_OF_ITERATIONS)
    sum_sparse = np.zeros(NUM_OF_ITERATIONS)

    gridSize = [6,8,6]
    Y = np.zeros((vector_length,num_of_data))
    for iterCount in range(1,NUM_OF_ITERATIONS + 1):


        maxDisp = z_dim/gridSize[2]/2
        print 'Iteration ' +  str(iterCount) + ' lamda=%f'  %lamda
        print 'Grid size: ', gridSize
        print 'Max Displacement: ', maxDisp 
        a = time.clock()


        # prepare data matrix
        for i in range(num_of_data) :
            im_file =  result_folder+'/'+ 'Iter'+str(iterCount-1)+'_Flair_' + str(i)  + '.nrrd'
            tmp = sitk.ReadImage(im_file)
            tmp = sitk.GetArrayFromImage(tmp)
            Y[:,i] = tmp.reshape(-1)
            del tmp


        sparsity[iterCount-1], sum_sparse[iterCount-1] = runIteration(Y, iterCount, lamda,gridSize, maxDisp)
        gc.collect()
        #lamda += 0.025
        if iterCount%2 == 0 :
          if gridSize[0] < 10:
            gridSize = np.add( gridSize,[1,1,1])

        #a = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        #print 'Current memory usage :',a/1024.0/1024.0,'GB'

        #h = hpy()
        #print h.heap()

    e = time.clock()
    l = e - s
    print 'Total running time:  %f mins'%(l/60.0)

    # plot the sparsity curve
    plt.figure()
    plt.plot(range(NUM_OF_ITERATIONS), sparsity)
    plt.savefig(result_folder+'/sparsity.png')

    plt.figure()
    plt.plot(range(NUM_OF_ITERATIONS), sum_sparse)
    plt.savefig(result_folder+'/sumSparse.png')

    if USE_HEALTHY_ATLAS:
      for i in range(NUM_OF_ITERATIONS):
          atlasIm = result_folder+'/'+ 'Iter'+str(i+1) +'_atlas.nrrd'
          im = sitk.ReadImage(atlasIm) # image in SITK format
          im_array = sitk.GetArrayFromImage(im)
          z_dim, x_dim, y_dim = im_array.shape # get 3D volume shape
          plt.figure()
          implot = plt.imshow(im_array[z_dim/2,:,:],plt.cm.gray)
          plt.title('Iter'+str(i)+ ' atlas')
          plt.savefig(result_folder+'/Iter'+str(i)+'_atlas.png')

if __name__ == "__main__":
    main()
