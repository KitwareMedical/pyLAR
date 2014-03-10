# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys
sys.path.append('/home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples')
from low_rank_atlas_iter import *

# <codecell>

# global variables
data_folder = ''
reference_im_name = ''
result_folder = ''
selection = []
im_names =[]

# <codecell>

###############################  the main pipeline #############################
def runIteration(Y,currentIter,lamda,gridSize,bsplineIterationNum):
    low_rank, sparse, n_iter,rank, sparsity = rpca(Y,lamda)
    saveImagesFromDM(low_rank,result_folder+'/'+ 'Iter'+str(currentIter) +'_LowRank_', reference_im_name)
    saveImagesFromDM(sparse,result_folder+'/'+ 'Iter'+str(currentIter) +'_Sparse_', reference_im_name)

    # Visualize and inspect
    fig = plt.figure(figsize=(15,5))
    showSlice(Y, ' Input',plt.cm.gray,0,reference_im_name)
    showSlice(low_rank,' low rank',plt.cm.gray,1, reference_im_name)
    showSlice(sparse,' sparse',plt.cm.gray,2, reference_im_name)
    plt.savefig(result_folder+'/'+'Iter'+ str(currentIter)+'_w_'+str(lamda)+'.png')
    fig.clf()
    plt.close(fig)

    num_of_data = Y.shape[1]
    del low_rank, sparse,Y

    print  'start image registrations'
    # Register low-rank images to the reference (healthy) image,
    # and update the input images to the next iteration
    ps=[]

    for i in range(num_of_data):
        movingIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_LowRank_' + str(i)  +'.nrrd'
        outputIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Deformed_LowRank' + str(i)  + '.nrrd'
        outputTransform = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Transform_' + str(i) +  '.tfm'
        outputDVF = result_folder+'/'+ 'Iter'+ str(currentIter)+'_DVF_' + str(i) +  '.nrrd'
        previousInputImage = result_folder+'/Iter'+str(currentIter-1)+ '_Flair_' + str(i)  + '.nrrd'
        logFile = open(result_folder+'/Iter'+str(currentIter)+'_RUN_'+ str(i)+'.log', 'w')

        # pipe steps sequencially
       # cmd = BSplineReg_Legacy(reference_im_name,movingIm,outputIm,outputDVF,gridSize, bsplineIterationNum)
        cmd = BSplineReg_BRAINSFit(reference_im_name,movingIm,outputIm,outputTransform,gridSize)

        cmd +=';'+ ConvertTransform(reference_im_name,outputTransform,outputDVF)

        outputComposedDVFIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Composed_DVF_' + str(i) +  '.nrrd'
        initialInputImage= result_folder+'/Iter0_Flair_' +str(i) +  '.nrrd'
        newInputImage = result_folder+'/Iter'+ str(currentIter)+'_Flair_' +str(i) +  '.nrrd'

        # compose deformations
        DVFImageList=[]
        for k in range(currentIter):
            DVFImageList.append(result_folder+'/'+ 'Iter'+ str(k+1)+'_DVF_' + str(i) +  '.nrrd')

        cmd += ';'+ composeMultipleDVFs(reference_im_name,DVFImageList,outputComposedDVFIm)

        cmd += ";" + updateInputImageWithDVF(initialInputImage,reference_im_name, \
                                       outputComposedDVFIm,newInputImage)
        process = subprocess.Popen(cmd, stdout=logFile, shell = True)
        ps.append(process)

    count = 0
    for  p in ps:
        p.wait()
        count = count + 1
        print  count, ' registration done'

    return sparsity


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


# <codecell>

# Data info
def useData_BRATS_Challenge():
    global data_folder,result_folder,im_names,selection,reference_im_name
    data_folder = '/home/xiaoxiao/work/data/BRATS/Challenge'

    im_names = [ \
     data_folder+'/HG/0301/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17569.mha',
     data_folder+'/HG/0302/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17573.mha',
     data_folder+'/HG/0303/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17577.mha',
     data_folder+'/HG/0304/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17581.mha',
     data_folder+'/HG/0305/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17585.mha',
     data_folder+'/HG/0306/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17589.mha',
     data_folder+'/HG/0307/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17593.mha',
     data_folder+'/HG/0308/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17597.mha',
     data_folder+'/HG/0309/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17601.mha',
     data_folder+'/HG/0310/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17605.mha']

    result_folder = '/home/xiaoxiao/work/data/BRATS/Challenge/LRA_Results_T1_Legacy_w1.0'
    os.system('mkdir '+ result_folder)

    # data selection
    selection = [0,1,2,3,4,5,6,7]
    reference_im_name = '/home/xiaoxiao/work/data/SRI24/T1_Crop.nii.gz'
    return (data_folder,result_folder,im_names,selection,reference_im_name)


def useData_BRATS2():
    global data_folder,result_folder,im_names,selection,reference_im_name
    data_folder = '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data'

    im_names = [ \
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0001/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.684.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0002/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.691.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0003/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.697.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0004/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.703.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0005/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.709.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0006/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.715.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0007/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.721.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0008/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.727.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0009/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.733.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0010/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.739.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0011/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.745.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0012/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.751.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0013/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.757.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0014/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.763.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0015/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.769.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0022/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.775.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0024/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.781.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0025/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.787.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0026/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.793.mha',
'/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/HG/0027/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.799.mha'
    ]
    result_folder = '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/LRA_Results_FLAIR_w0.7'
    os.system('mkdir '+ result_folder)
    # data selection
    selection = [0,1,2,3,4,5,6,7,8,9]
    reference_im_name = '/home/xiaoxiao/work/data/SRI24/T1_Crop.nii.gz'
    return


# <codecell>

#######################################  main ##################################
#@profile
def main():
    import time
    import resource
    #from guppy import hpy


    ##CropImage(data_folder +'/'+'SRI24/Flair.nii.gz',data_folder +'/'+'SRI24/Flair_Crop.nii.gz',[50,20,0],[50,30,0])

    #useData_BRATS_Challenge()
    useData_BRATS2()

    s = time.clock()
    # save script to the result folder for paramter checkups
    os.system('cp /home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples/lowrankflair.py   ' +result_folder)

    #showReferenceImage(reference_im_name)
    affineRegistrationStep()


    sys.stdout = open(result_folder+'/RUN.log', "w")
    im_ref = sitk.ReadImage(reference_im_name) # image in SITK format
    im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
    vector_length = z_dim * x_dim * y_dim
    del im_ref, im_ref_array

    num_of_data = len(selection)


    NUM_OF_ITERATIONS = 15
    lamda = 0.7
    sparsity = np.zeros(NUM_OF_ITERATIONS)

    gridSize = [3,5,3]
    bsplineIterationNum = 20
    Y = np.zeros((vector_length,num_of_data))
    for iterCount in range(1,NUM_OF_ITERATIONS + 1):


        print 'Iteration ' +  str(iterCount) + ' lambda=%f'  %lamda
        a = time.clock()

        # prepare data matrix
        for i in range(num_of_data) :
            im_file =  result_folder+'/'+ 'Iter'+str(iterCount-1)+'_Flair_' + str(i)  + '.nrrd'
            tmp = sitk.ReadImage(im_file)
            tmp = sitk.GetArrayFromImage(tmp)
            Y[:,i] = tmp.reshape(-1)
            del tmp

        sparsity[iterCount-1] = runIteration(Y, iterCount, lamda,gridSize, bsplineIterationNum)
        gc.collect()
        # if lamda < 1.0:
        #  lamda = lamda + 0.1
        gridSize = np.add( gridSize,[1,2,1])
        bsplineIterationNum = bsplineIterationNum + 2


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
    plt.savefig(result_folder+'/sparsity.png')


    #visulaize
    for i in range(num_of_data):
       for currentIter in range(1,NUM_OF_ITERATIONS):
            outputComposedDVFIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_DVF_' + str(i) +  '.nrrd'
            deformedInputIm = result_folder+'/'+ 'Iter'+ str(currentIter+1)+'_Flair_' + str(i) +  '.nrrd'
            gridVisDVF(outputComposedDVFIm,z_dim/2,'DVF_Vis_Flair_'+str(i) +'_Iter'+str(currentIter+1),\
                       result_folder,deformedInputIm,20)



if __name__ == "__main__":
    main()

