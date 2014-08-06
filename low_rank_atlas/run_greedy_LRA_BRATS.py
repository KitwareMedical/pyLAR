#############################################################################
#This script is the greedy impementation of the LRA framework and uses the BRATS
#dataset for experiments, as described in the MICCAI'14 paper.
#############################################################################



import sys
from low_rank_atlas_iter import *

###############################  the main pipeline #############################
def runIteration(Y,cIter,lamda,gridSize,maxDisp):
    low_rank, sparse, n_iter,rank, sparsity, sum_sparse = rpca(Y,lamda)
    saveImagesFromDM(low_rank,result_dir+'/'+ 'Iter'+str(cIter) +'_LowRank_', reference_im_fn)
    saveImagesFromDM(sparse,result_dir+'/'+ 'Iter'+str(cIter) +'_Sparse_', reference_im_fn)

    # Visualize and inspect
    fig = plt.figure(figsize=(15,5))
    showSlice(Y, ' Input',plt.cm.gray,0,reference_im_fn)
    showSlice(low_rank,' low rank',plt.cm.gray,1, reference_im_fn)
    showSlice(sparse,' sparse',plt.cm.gray,2, reference_im_fn)
    plt.savefig(result_dir+'/'+'Iter'+ str(cIter)+'_w_'+str(lamda)+'.png')
    fig.clf()
    plt.close(fig)

    num_of_data = Y.shape[1]
    del low_rank, sparse,Y

    print  'start image registrations'
    # Register low-rank images to the reference (healthy) image,
    # and update the input images to the next iteration
    ps=[]

    for i in range(num_of_data):
        movingIm = result_dir+'/'+ 'Iter'+ str(cIter)+'_LowRank_' + str(i)  +'.nrrd'
        outputIm = result_dir+'/'+ 'Iter'+ str(cIter)+'_Deformed_LowRank' + str(i)  + '.nrrd'
        outputTransform = result_dir+'/'+ 'Iter'+ str(cIter)+'_Transform_' + str(i) +  '.tfm'
        outputDVF = result_dir+'/'+ 'Iter'+ str(cIter)+'_DVF_' + str(i) +  '.nrrd'
        previousInputImage = result_dir+'/Iter'+str(cIter-1)+ '_Flair_' + str(i)  + '.nrrd'
        logFile = open(result_dir+'/Iter'+str(cIter)+'_RUN_'+ str(i)+'.log', 'w')

        # pipe steps sequencially
        cmd = BSplineReg_BRAINSFit(reference_im_fn,movingIm,outputIm,outputTransform,gridSize, maxDisp)

        cmd +=';'+ ConvertTransform(reference_im_fn,outputTransform,outputDVF)

        outputComposedDVFIm = result_dir+'/'+ 'Iter'+ str(cIter)+'_Composed_DVF_' + str(i) +  '.nrrd'
        initialInputImage= result_dir+'/Iter0_Flair_' +str(i) +  '.nrrd'
        newInputImage = result_dir+'/Iter'+ str(cIter)+'_Flair_' +str(i) +  '.nrrd'

        # compose deformations
        COMPOSE_DVF = True
        if COMPOSE_DVF:
          DVFImageList=[]
          for k in range(cIter):
              DVFImageList.append(result_dir+'/'+ 'Iter'+ str(k+1)+'_DVF_' + str(i) +  '.nrrd')

              cmd += ';'+ composeMultipleDVFs(reference_im_fn,DVFImageList,outputComposedDVFIm)

        cmd += ";" + updateInputImageWithDVF(initialInputImage,reference_im_fn, \
                                       outputComposedDVFIm,newInputImage)
        process = subprocess.Popen(cmd, stdout=logFile, shell = True)
        ps.append(process)

    for  p in ps:
        p.wait()

    return sparsity, sum_sparse


def showReferenceImage(reference_im_fn):
    im_ref = sitk.ReadImage(reference_im_fn) # image in SITK format
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
        outputIm =  result_dir+'/Iter0_Flair_' + str(i)  + '.nrrd'
        AffineReg(reference_im_fn,im_fns[selection[i]],outputIm)
    return


################### global paths and parameters ################
reference_im_fn = '$HOME/work/data/SRI24/T1_Crop.nii.gz'
data_dir = '$HOME/work/data/BRATS/BRATS-2/Image_Data'
im_fns = readTxtIntoList(data_dir +'/Flair_FN.txt')

selection = [0,1,3,4,6,7,9,10]
lamda = 0.8

result_dir = '$HOME/work/data/BRATS/BRATS-2/Image_Data/results/Flair_w'+str(lamda)
print 'Results will be stored in:',result_dir
if not os.path.exists(result_dir):
	os.system('mkdir '+ result_dir)

######################   main pipeline ##########################
#@profile
def main():
    import time
    import resource
    #from guppy import hpy
    global lamda

    s = time.clock()
    # save script to the result dir for paramter checkups
    currentPyFile = os.path.realpath(__file__)
    os.system('cp   ' + currentPyFile+' ' +result_dir)

    #showReferenceImage(reference_im_fn)
    affineRegistrationStep()

    sys.stdout = open(result_dir+'/RUN.log', "w")
    im_ref = sitk.ReadImage(reference_im_fn) # image in SITK format
    im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
    vector_length = z_dim * x_dim * y_dim
    del im_ref, im_ref_array

    num_of_data = len(selection)

    NUM_OF_ITERATIONS = 12
    sparsity = np.zeros(NUM_OF_ITERATIONS)
    sum_sparse = np.zeros(NUM_OF_ITERATIONS)

    gridSize = [6,8,6]
    Y = np.zeros((vector_length,num_of_data))
    for iterCount in range(1,NUM_OF_ITERATIONS + 1):

        maxDisp = z_dim/gridSize[2]/4
        print 'Iteration ' +  str(iterCount) + ' lamda=%f'  %lamda
        print 'Grid size: ', gridSize
        print 'Max Displacement: ', maxDisp 
        a = time.clock()

        # prepare data matrix
        for i in range(num_of_data) :
            im_file =  result_dir+'/'+ 'Iter'+str(iterCount-1)+'_Flair_' + str(i)  + '.nrrd'
            tmp = sitk.ReadImage(im_file)
            tmp = sitk.GetArrayFromImage(tmp)
            Y[:,i] = tmp.reshape(-1)
            del tmp

        sparsity[iterCount-1], sum_sparse[iterCount-1] = runIteration(Y, iterCount, lamda,gridSize, maxDisp)
        #lamda += 0.025
        gc.collect()
        if iterCount%2 == 0 :
          if gridSize[0] < 10:
            gridSize = np.add( gridSize,[1,1,1])

        #a = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        #print 'Current memory usage :',a/1024.0/1024.0,'GB'

        #h = hpy()
        #print h.heap()
        b = time.clock()
        c = b-a
        print 'Iteration took  %f mins'%(c/60.0)

    e = time.clock()
    l = e - s
    print 'Total running time:  %f mins'%(l/60.0)

    # plot the sparsity curve
    plt.figure()
    plt.plot(range(NUM_OF_ITERATIONS), sparsity)
    plt.savefig(result_dir+'/sparsity.png')

    plt.figure()
    plt.plot(range(NUM_OF_ITERATIONS), sum_sparse)
    plt.savefig(result_dir+'/sumSparse.png')


###################################################
if __name__ == "__main__":
    main()
