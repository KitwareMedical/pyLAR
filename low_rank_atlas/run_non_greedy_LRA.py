import sys
from low_rank_atlas_iter import *
import imp

##########################################################
# assign global parameters from the input config txt file
configFN = sys.argv[1]
f = open(configFN)
config  = imp.load_source('config', '', f)
f.close()

USE_HEALTHY_ATLAS = config.USE_HEALTHY_ATLAS
USE_BLUR          = config.USE_BLUR
reference_im_fn   = config.reference_im_fn
data_dir          = config.data_dir
fileListFN        = config.fileListFN
lamda             = config.lamda
result_dir        = config.result_dir
selection         = config.selection
sigma             = config.sigma
NUM_OF_ITERATIONS_PER_LEVEL = config.NUM_OF_ITERATIONS_PER_LEVEL
NUM_OF_LEVELS               = config.NUM_OF_LEVELS # multiscale bluring (coarse-to-fine)
REGISTRATION_TYPE           = config.REGISTRATION_TYPE

gridSize = [0,0,0]
if REGISTRATION_TYPE =='BSpline':
  gridSize = config.gridSize

antsParams = {None:None}
if REGISTRATION_TYPE == 'ANTS':
   antsParams = config.antsParams

im_fns = readTxtIntoList(data_dir +'/'+ fileListFN)
print 'Results will be stored in:',result_dir
if not os.path.exists(result_dir):
	os.system('mkdir '+ result_dir)

# For reproducibility: save all parameters into the result dir
os.system('cp   ' + configFN+' ' +result_dir)
os.system('cp   ' + data_dir +'/'+fileListFN + ' ' +result_dir)
currentPyFile = os.path.realpath(__file__)
os.system('cp   ' + currentPyFile+' ' +result_dir)

############################################## #############################
def runIteration(vector_length,level,currentIter,lamda,sigma, gridSize,maxDisp):
    global reference_im_fn

    # prepare data matrix
    num_of_data = len(selection)
    Y = np.zeros((vector_length,num_of_data))
    for i in range(num_of_data) :
          im_file =  result_dir+'/L'+ str(level) +'_Iter'+str(currentIter-1)+'_Flair_' + str(i)  + '.nrrd'
          inIm = sitk.ReadImage(im_file)
          tmp = sitk.GetArrayFromImage(inIm)
          if USE_BLUR:
              if sigma > 0:
                srg = sitk.SmoothingRecursiveGaussianImageFilter()
                srg.SetSigma(sigma)
                outIm = srg.Execute(inIm)
                tmp = sitk.GetArrayFromImage(outIm)
          Y[:,i] = tmp.reshape(-1)
          del tmp

    # Low-rank and sparse decomposition
    low_rank, sparse, n_iter,rank, sparsity, sum_sparse = rpca(Y,lamda)

    saveImagesFromDM(low_rank,result_dir+'/L'+ str(level)+'_Iter'+str(currentIter) +'_LowRank_', reference_im_fn)
    saveImagesFromDM(sparse,result_dir+'/L'+str(level)+ '_Iter'+str(currentIter) +'_Sparse_', reference_im_fn)

    # Visualize and inspect
    fig = plt.figure(figsize=(15,5))
    showSlice(Y,'L'+str(level)+ '_'+str(currentIter) +' Input',plt.cm.gray,0,reference_im_fn)
    showSlice(low_rank,'L'+str(level)+ '_'+str(currentIter) +' low rank',plt.cm.gray,1, reference_im_fn)
    showSlice(sparse,'L'+str(level)+ '_'+str(currentIter)+' sparse',plt.cm.gray,2, reference_im_fn)
    plt.savefig(result_dir+'/'+'L'+str(level)+'_Iter'+ str(currentIter)+'.png')
    fig.clf()
    plt.close(fig)

    del low_rank, sparse,Y

    # unbiased atlas building
    if not USE_HEALTHY_ATLAS:
        reference_im_fn = result_dir+'/L'+str(level)+'_Iter'+ str(currentIter) +'_atlas.nrrd'
      # Average lowrank images
        listOfImages = []
        num_of_data = len(selection)
        for i in range(num_of_data):
            lrIm = result_dir+'/L'+str(level)+ '_Iter'+ str(currentIter)+'_LowRank_' + str(i)  +'.nrrd'
            listOfImages.append(lrIm)
        AverageImages(listOfImages,reference_im_fn)
        im = sitk.ReadImage(reference_im_fn) # image in SITK format
        im_array = sitk.GetArrayFromImage(im)
        z_dim, x_dim, y_dim = im_array.shape # get 3D volume shape
        plt.figure()
        implot = plt.imshow(im_array[z_dim/2,:,:],plt.cm.gray)
        plt.title('Level'+str(i)+ ' atlas')
        plt.savefig(result_dir+'/atlas_L'+str(level)+'_Iter'+str(currentIter)+'.png')


    ps = []
    for i in range(num_of_data):
        logFile = open(result_dir+'/L'+str(level)+'_Iter'+str(currentIter)+'_RUN_'+ str(i)+'.log', 'w')

        # pipe steps sequencially
        cmd = ''
        invWarpedlowRankIm = result_dir+'/L'+ str(level)+'_Iter'+ str(currentIter)+'_LowRank_' + str(i)  +'.nrrd'
        if currentIter > 1:
            lowRankIm = result_dir+'/L'+ str(level)+'_Iter'+ str(currentIter)+'_LowRank_' + str(i)  +'.nrrd'
            invWarpedlowRankIm = result_dir+'/L'+ str(level)+'_Iter'+ str(currentIter)+'_InvWarped_LowRank_' + str(i)  +'.nrrd'
            if REGISTRATION_TYPE == 'BSpline' or REGISTRATION_TYPE == 'Demons':
              previousIterDVF = result_dir + '/L'+str(level)+ '_Iter'+ str(currentIter-1)+'_DVF_' + str(i) +  '.nrrd'
              inverseDVF = result_dir + '/L'+str(level)+ '_Iter'+ str(currentIter-1)+'_INV_DVF_' + str(i) +  '.nrrd'
              genInverseDVF(previousIterDVF,inverseDVF, True)
              updateInputImageWithDVF( lowRankIm, reference_im_fn, inverseDVF, invWarpedlowRankIm,True)
            if REGISTRATION_TYPE == 'ANTS':
              outputTransformPrefix = result_dir+'/L'+ str(level)+'_Iter'+ str(currentIter-1)+'_'+str(i)+'_'
              ANTSWarpImage(lowRankIm,invWarpedlowRankIm, reference_im_fn,outputTransformPrefix,True, True)


        outputIm = result_dir+'/L'+ str(level)+'_Iter'+ str(currentIter)+'_Deformed_LowRank' + str(i)  + '.nrrd'
        outputTransform = result_dir+'/L'+ str(level)+'_Iter'+ str(currentIter)+'_Transform_' + str(i) +  '.tfm'
        outputDVF = result_dir+'/L'+ str(level)+'_Iter'+ str(currentIter)+'_DVF_' + str(i) +  '.nrrd'


        movingIm = invWarpedlowRankIm
        fixedIm =  reference_im_fn

        initialInputImage= result_dir+'/L'+str(level)+'_Iter0_Flair_' +str(i) +  '.nrrd'
        newInputImage = result_dir+'/L'+str(level)+'_Iter'+ str(currentIter)+'_Flair_' +str(i) +  '.nrrd'

        if REGISTRATION_TYPE == 'BSpline':
          cmd += BSplineReg_BRAINSFit(fixedIm,movingIm,outputIm,outputTransform,gridSize, maxDisp)
          cmd +=';'+ ConvertTransform(reference_im_fn,outputTransform,outputDVF)
          cmd += ";" + updateInputImageWithDVF(initialInputImage,reference_im_fn, outputDVF,newInputImage)
        elif REGISTRATION_TYPE == 'Demons':
          cmd += DemonsReg(fixedIm,movingIm,outputIm,outputDVF)
          cmd += ";" + updateInputImageWithDVF(initialInputImage,reference_im_fn, outputDVF,newInputImage)
        elif REGISTRATION_TYPE == 'ANTS':
          # will generate a warp(DVF) file and an affine file
          outputTransformPrefix = result_dir+'/L'+ str(level)+'_Iter'+ str(currentIter) +'_'+str(i)+'_'
          if currentIter > 1:
            initialTransform = result_dir+'/L'+str(level)+'_Iter'+str(currentIter-1)+'_'+str(i)+'_0Warp.nii.gz'
          else:
            initialTransform = None
          antsParams['Metric'] = antsParams['Metric'].replace('fixedIm', fixedIm)
          antsParams['Metric'] = antsParams['Metric'].replace('movingIm', movingIm)
          cmd += ANTS(fixedIm,movingIm,outputTransformPrefix,antsParams, initialTransform)
          cmd += ";" + ANTSWarpImage(initialInputImage,newInputImage, reference_im_fn,outputTransformPrefix)
          #print cmd
        else:
          print "unrecognized registration type:", REGISTRATION_TYPE


        process = subprocess.Popen(cmd, stdout = logFile, shell = True)
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
        outputIm =  result_dir+'/L0_Iter0_Flair_' + str(i)  + '.nrrd'
        AffineReg(reference_im_fn,im_fns[selection[i]],outputIm)
    return


#######################################  main ##################################
#@profile
def main():
    import time
    import resource

    global lamda, gridSize, sigma
    s = time.clock()


    #showReferenceImage(reference_im_fn)
    affineRegistrationStep()

    sys.stdout = open(result_dir+'/RUN.log', "w")
    im_ref = sitk.ReadImage(reference_im_fn) # image in SITK format
    im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
    vector_length = z_dim * x_dim * y_dim
    del im_ref, im_ref_array

    num_of_data = len(selection)
    #BSpline max displacement constrain, 0.5 refers to half of the grid size
    factor = 0.5
    for level in range(0, NUM_OF_LEVELS):
        for iterCount in range(1,NUM_OF_ITERATIONS_PER_LEVEL+1):
            maxDisp = -1
            print 'Level: ', level
            print 'Iteration ' +  str(iterCount) + ' lamda=%f'  %lamda
            if REGISTRATION_TYPE == 'BSpline':
              print 'Grid size: ', gridSize
              maxDisp = z_dim/gridSize[2]*factor

            print 'Sigma: ', sigma

            runIteration(vector_length,level, iterCount, lamda,sigma, gridSize, maxDisp)

            if REGISTRATION_TYPE == 'BSpline' and  gridSize[0] < 10:
                 gridSize = np.add( gridSize,[1,2,1])
            if sigma > 0:
                 sigma = sigma - 0.5
            gc.collect()


        # update the input image at each iteration by composing deformations fields from previous iterations
        if NUM_OF_ITERATIONS_PER_LEVEL > 1:
            for i in range(num_of_data):
                newLevelInitIm = result_dir + '/L'+str(level+1)+'_Iter0_Flair_'+str(i)+'.nrrd'
                initialInputImage = result_dir + '/L0_Iter0_Flair_'+str(i)+'.nrrd'
                outputComposedDVFIm = result_dir + '/L'+str(level) + '_Composed_DVF_'+str(i)+'.nrrd'
                DVFImageList=[]
                for k in range(level+1):
                    DVFImageList.append(result_dir+'/L'+ str(k)+'_Iter'+ str(NUM_OF_ITERATIONS_PER_LEVEL)+'_DVF_' + str(i) +  '.nrrd')
                composeMultipleDVFs(reference_im_fn,DVFImageList,outputComposedDVFIm, True)
                updateInputImageWithDVF(initialInputImage,reference_im_fn, \
                                           outputComposedDVFIm, newLevelInitIm, True)
                finalDVFIm =  result_dir + '/L'+str(level)+'_Iter'+ str(NUM_OF_ITERATIONS_PER_LEVEL)+'_DVF_' + str(i) +  '.nrrd'

            if gridSize[0] < 10:
                 gridSize = np.add( gridSize,[1,2,1])
            if sigma > 0:
                 sigma = sigma - 1
            factor = factor*0.5

        #a = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        #print 'Current memory usage :',a/1024.0/1024.0,'GB'

        #h = hpy()
        #print h.heap()

    e = time.clock()
    l = e - s
    print 'Total running time:  %f mins'%(l/60.0)





if __name__ == "__main__":
    main()
