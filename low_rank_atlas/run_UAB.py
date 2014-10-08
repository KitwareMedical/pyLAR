import sys
from low_rank_atlas_iter import *
import imp

##########################################################
##########################################################
# assign global parameters from the input config txt file
configFN = sys.argv[1]
f = open(configFN)
config  = imp.load_source('config', '', f)
f.close()

USE_HEALTHY_ATLAS = config.USE_HEALTHY_ATLAS
reference_im_fn   = config.reference_im_fn
data_dir          = config.data_dir
result_dir        = config.result_dir
fileListFN        = config.fileListFN
selection         = config.selection

NUM_OF_ITERATIONS_PER_LEVEL = config.NUM_OF_ITERATIONS_PER_LEVEL
NUM_OF_LEVELS               = config.NUM_OF_LEVELS # multiscale bluring (coarse-to-fine)
REGISTRATION_TYPE           = config.REGISTRATION_TYPE

antsParams = {None:None}
if REGISTRATION_TYPE == 'ANTS':
   antsParams = config.antsParams

im_fns = readTxtIntoList(data_dir + '/' + fileListFN)
print 'Results will be stored in:',result_dir
if not os.path.exists(result_dir):
	os.system('mkdir ' + result_dir)

# For reproducibility: save all parameters into the result dir
os.system('cp   ' + configFN+ ' ' +result_dir)
os.system('cp   ' + data_dir + '/' +fileListFN + ' ' +result_dir)
currentPyFile = os.path.realpath(__file__)
os.system('cp   ' + currentPyFile+ ' ' +result_dir)

###########################################################################
###########################################################################
# Iterative Low-rank Atlas-to-Image Registration
def runIteration(vector_length,level,currentIter):
    global reference_im_fn

    # average the low-rank images to produce the Atlas
    atlasIm = result_dir+ '/L' + str(level) + '_Iter' + str(currentIter-1) + '_atlas.nrrd'
    listOfImages = [ ]
    num_of_data = len(selection)
    for i in range(num_of_data):
        lrIm = result_dir+ '/L' + str(level) + '_Iter' + str(currentIter-1) + '_' + str(i)  + '.nrrd'
        listOfImages.append(lrIm)
    AverageImages(listOfImages,atlasIm)

    im = sitk.ReadImage(atlasIm)
    im_array = sitk.GetArrayFromImage(im)
    z_dim, x_dim, y_dim = im_array.shape
    plt.figure()
    implot = plt.imshow(im_array[z_dim/2,:,:],plt.cm.gray)
    plt.title('L' + str(level) + '_Iter' + str(currentIter-1) + ' atlas')
    plt.savefig(result_dir+ '/atlas_L' + str(level) + '_Iter' + str(currentIter-1) + '.png')
    reference_im_fn = atlasIm



    ps = [] # to use multiple processors
    for i in range(num_of_data):
        logFile = open(result_dir+ '/L' + str(level) + '_Iter' + str(currentIter) + '_RUN_' + str(i) + '.log', 'w')

        cmd = ''

        initialInputImage= result_dir+ '/L' + str(level) + '_Iter0_' + str(i) +  '.nrrd'
        newInputImage = result_dir+ '/L' + str(level) + '_Iter' + str(currentIter) + '_' + str(i) +  '.nrrd'

          # will generate a warp(DVF) file and an affine file
        outputTransformPrefix = result_dir+ '/L' + str(level) + '_Iter' + str(currentIter) + '_' + str(i) + '_'
        fixedIm = atlasIm
        movingIm = initialInputImage
        cmd += ANTS(fixedIm,movingIm,outputTransformPrefix,antsParams)
        cmd += ";" + ANTSWarpImage(initialInputImage, newInputImage, reference_im_fn, outputTransformPrefix)
        process = subprocess.Popen(cmd, stdout = logFile, shell = True)
        ps.append(process)
    for  p in ps:
        p.wait()
    return


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
        outputIm =  result_dir+ '/L0_Iter0_' + str(i)  + '.nrrd'
        AffineReg(reference_im_fn,im_fns[selection[i]],outputIm)
    return

# histogram matching preprocessing
def  histogramMatchingStep():
    num_of_data = len(selection)
    for i in range(1,num_of_data):
        inIm =  result_dir+ '/L0_Iter0_' + str(i)  + '.nrrd'
        refIm =  result_dir+ '/L0_Iter0_' + str(0)  + '.nrrd'
        outIm =  result_dir+ '/L0_Iter0_' + str(i)  + '.nrrd'
        inputIm =  sitk.ReadImage(inIm)
        referenceIm =  sitk.ReadImage(refIm)
        histMatchingFilter = sitk.HistogramMatchingImageFilter()
        histMatchingFilter.SetNumberOfHistogramLevels( 1024 );
        histMatchingFilter.SetNumberOfMatchPoints( 7 );
        histMatchingFilter.ThresholdAtMeanIntensityOff();
        outputIm = histMatchingFilter.Execute(inputIm, referenceIm)
        sitk.WriteImage(outputIm, outIm)
    return

def  normalizeIntensityStep():
    num_of_data = len(selection)
    for i in range(num_of_data):
        inIm =  result_dir+ '/L0_Iter0_' + str(i)  + '.nrrd'
        outIm =  result_dir+ '/L0_Iter0_' + str(i)  + '.nrrd'
        normalizeFilter = sitk.NormalizeImageFilter()
        inputIm =  sitk.ReadImage(inIm)
        outputIm = normalizeFilter.Execute(inputIm)
        sitk.WriteImage(outputIm, outIm)
    return
#######################################  main ##################################
#@profile
def main():
    import time
    #import resource

    global  sigma
    s = time.time()

    #showReferenceImage(reference_im_fn)
    affineRegistrationStep()
   # normalizeIntensityStep()
    #histogramMatchingStep()

    sys.stdout = open(result_dir+ '/RUN.log', "w")
    im_ref = sitk.ReadImage(reference_im_fn)
    im_ref_array = sitk.GetArrayFromImage(im_ref)
    z_dim, x_dim, y_dim = im_ref_array.shape
    vector_length = z_dim * x_dim * y_dim
    del im_ref, im_ref_array

    num_of_data = len(selection)
    for level in range(0, NUM_OF_LEVELS):
        for iterCount in range(1,NUM_OF_ITERATIONS_PER_LEVEL+1):
            print 'Level: ', level
            print 'Iteration ' +  str(iterCount)  

            runIteration(vector_length,level, iterCount)

            gc.collect() # garbage collection


        if NUM_OF_LEVELS > 1:
            print 'WARNING: No need for multiple levels! TO BE REMOVED!'
            for i in range(num_of_data):
                newLevelInitIm = result_dir + '/L' + str(level+1) + '_Iter0_' + str(i) + '.nrrd'

    atlasIm = result_dir+ '/L' + str(level) + '_Iter' + str(NUM_OF_ITERATIONS_PER_LEVEL) + '_atlas.nrrd'
    listOfImages = [ ]
    num_of_data = len(selection)
    for i in range(num_of_data):
        lrIm = result_dir+ '/L' + str(level) + '_Iter' + str(NUM_OF_ITERATIONS_PER_LEVEL) + '_' + str(i)  + '.nrrd'
        listOfImages.append(lrIm)
    AverageImages(listOfImages,atlasIm)

    im = sitk.ReadImage(atlasIm)
    im_array = sitk.GetArrayFromImage(im)
    z_dim, x_dim, y_dim = im_array.shape
    plt.figure()
    implot = plt.imshow(np.flipud(im_array[z_dim/2,:,:]),plt.cm.gray)
    plt.title('L' + str(level) + '_Iter' + str(NUM_OF_ITERATIONS_PER_LEVEL) + ' atlas')
    plt.savefig(result_dir+ '/atlas_L' + str(level) + '_Iter' + str(NUM_OF_ITERATIONS_PER_LEVEL) + '.png')


    e = time.time()
    l = e - s
    print 'Total running time:  %f mins'%(l/60.0)



if __name__ == "__main__":
    main()
