# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys
sys.path.append('/home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples')
from low_rank_atlas_iter import *


def useData_BRATS2_FLAIR():
    global data_folder,result_folder,im_names,selection,reference_im_name

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
    return

# Data info
def useData_BRATS2_Synthetic():
    global data_folder,result_folder,im_names,selection,reference_im_name
    data_folder = '/home/xiaoxiao/work/data/BRATS/BRATS-2/Synthetic_Data/HG'
    im_names = [ \
    data_folder +'/0001/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.866.mha',
    data_folder +'/0002/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.872.mha',
    data_folder +'/0003/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.878.mha',
    data_folder +'/0004/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.884.mha',
    data_folder +'/0005/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.890.mha',
    data_folder +'/0006/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.896.mha',
    data_folder +'/0007/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.902.mha',
    data_folder +'/0008/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.908.mha',
    data_folder +'/0009/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.914.mha',
    data_folder +'/0010/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.920.mha',
    data_folder +'/0011/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.926.mha',
    data_folder +'/0012/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.932.mha',
    data_folder +'/0013/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.938.mha',
    data_folder +'/0014/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.944.mha',
    data_folder +'/0015/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.950.mha',
    data_folder +'/0016/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.956.mha',
    data_folder +'/0017/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.962.mha',
    data_folder +'/0018/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.968.mha',
    data_folder +'/0019/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.974.mha',
    data_folder +'/0020/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.980.mha',
    data_folder +'/0021/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.986.mha',
    data_folder +'/0022/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.992.mha',
    data_folder +'/0023/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.998.mha',
    data_folder +'/0024/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.1004.mha',
    data_folder +'/0025/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.1010.mha'
    ]



    return 


def useData_BRATS2():
    global data_folder,im_names,reference_im_name
    data_folder = '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data'

    im_names = [ \
    data_folder+'/HG/0001/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.685.mha',
    data_folder+'/HG/0002/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.692.mha',
    data_folder+'/HG/0003/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.698.mha',
    data_folder+'/HG/0004/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.704.mha',
    data_folder+'/HG/0005/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.710.mha',
    data_folder+'/HG/0006/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.716.mha',
    data_folder+'/HG/0007/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.722.mha',
    data_folder+'/HG/0008/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.728.mha',
    data_folder+'/HG/0009/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.734.mha',
    data_folder+'/HG/0010/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.740.mha',
    data_folder+'/HG/0011/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.746.mha',
    data_folder+'/HG/0012/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.752.mha',
    data_folder+'/HG/0013/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.758.mha',
    data_folder+'/HG/0014/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.764.mha',
    data_folder+'/HG/0015/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.770.mha',
    data_folder+'/HG/0022/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.776.mha',
    data_folder+'/HG/0024/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.782.mha',
    data_folder+'/HG/0025/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.788.mha',
    data_folder+'/HG/0026/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.794.mha',
    data_folder+'/HG/0027/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.800.mha'
    ]

    return




# <codecell>

###############################  the main pipeline #############################
def runIteration(Y,currentIter,lamda,gridSize,maxDisp):
    low_rank, sparse, n_iter,rank, sparsity, sum_sparse = rpca(Y,lamda)
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
        previousInputImage = result_folder+'/Iter'+str(currentIter-1)+ '_T1_' + str(i)  + '.nrrd'
        logFile = open(result_folder+'/Iter'+str(currentIter)+'_RUN_'+ str(i)+'.log', 'w')

        # pipe steps sequencially
        cmd = BSplineReg_BRAINSFit(reference_im_name,movingIm,outputIm,outputTransform,gridSize, maxDisp)

        cmd +=';'+ ConvertTransform(reference_im_name,outputTransform,outputDVF)

        outputComposedDVFIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Composed_DVF_' + str(i) +  '.nrrd'
        initialInputImage= result_folder+'/Iter0_T1_' +str(i) +  '.nrrd'
        newInputImage = result_folder+'/Iter'+ str(currentIter)+'_T1_' +str(i) +  '.nrrd'

        # compose deformations
        COMPOSE_DVF = True
        if COMPOSE_DVF:
          DVFImageList=[]
          for k in range(currentIter):
              DVFImageList.append(result_folder+'/'+ 'Iter'+ str(k+1)+'_DVF_' + str(i) +  '.nrrd')

              cmd += ';'+ composeMultipleDVFs(reference_im_name,DVFImageList,outputComposedDVFIm)

        cmd += ";" + updateInputImageWithDVF(initialInputImage,reference_im_name, \
                                       outputComposedDVFIm,newInputImage)
        process = subprocess.Popen(cmd, stdout=logFile, shell = True)
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
        outputIm =  result_folder+'/Iter0_T1_' + str(i)  + '.nrrd'
        AffineReg(reference_im_name,im_names[selection[i]],outputIm)
    return




#######################################  manual setting  ##################################
# global variables
im_names =[]
reference_im_name = '/home/xiaoxiao/work/data/SRI24/T1_Crop.nii.gz'
######  set data folders

lamda = 0.9

useData_BRATS2()
selection = range(8)
result_folder = '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/RegulateBspline_w'+str(lamda)



#useData_BRATS2_Synthetic()
#result_folder = '/home/xiaoxiao/work/data/BRATS/BRATS-2/Synthetic_Data/RegulateBSpline_w'+str(lamda)
#selection = range(8)

#os.system('mkdir '+ result_folder)

#@profile
def main():
    import time
    import resource
    #from guppy import hpy
    global lamda

    s = time.clock()
    # save script to the result folder for paramter checkups
    os.system('cp /home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples/Low_Rank_Atlas_Iter_BRATS.py   ' +result_folder)

    #showReferenceImage(reference_im_name)
    affineRegistrationStep()


    #sys.stdout = open(result_folder+'/RUN.log', "w")
    im_ref = sitk.ReadImage(reference_im_name) # image in SITK format
    im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
    z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
    vector_length = z_dim * x_dim * y_dim
    del im_ref, im_ref_array

    num_of_data = len(selection)


    NUM_OF_ITERATIONS = 15
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
            im_file =  result_folder+'/'+ 'Iter'+str(iterCount-1)+'_T1_' + str(i)  + '.nrrd'
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
    plt.savefig(result_folder+'/sparsity.png')
    


    plt.figure()
    plt.plot(range(NUM_OF_ITERATIONS), sum_sparse)
    plt.savefig(result_folder+'/sumSparse.png')


###################################################
###################################################
###################################################
###################################################



if __name__ == "__main__":
    main()

