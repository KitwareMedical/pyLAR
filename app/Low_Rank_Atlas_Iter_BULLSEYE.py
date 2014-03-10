
# In[ ]:
import sys
sys.path.append('./')
from low_rank_atlas_iter import *
import gc


# In[ ]:

# global settings
data_folder = '/home/xiaoxiao/work/data/BullEyeSimulation'
result_folder = data_folder +'/try1_greedyDVF'
os.system('mkdir '+result_folder)
im_names = [  data_folder+'/simu1.nrrd',
 data_folder+'/simu2.nrrd',
 data_folder+'/simu3.nrrd',
 data_folder+'/simu4.nrrd']
reference_im_name = data_folder +'/fMeanSimu.nrrd'

COMPOSE_DVF = True

# In[ ]:

# data selection and global parameters, prepare iter0 data
selection = [0,1,2,3]
num_of_data = len(selection)
for i in xrange(num_of_data):
    iter0fn = result_folder+'/Iter0'+ '_simu_' + str(i)  + '.nrrd'
    simufn = im_names[selection[i]]
    os.system('cp '+ simufn +' ' + iter0fn)


# In[ ]:

# profile data size, save into global variables
im_ref = sitk.ReadImage(reference_im_name) # image in SITK format
im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
vector_length = z_dim* x_dim*y_dim

implot = plt.imshow(im_ref_array[32,:,:],plt.cm.gray)
plt.title('healthy atlas')

slice_nr = 32  # just for vis purpose


# In[ ]:

###############################  the main pipeline #############################
def runIteration(currentIter,lamda,gridSize=[3,3,3]):
    # run RPCA
    Y = np.zeros((vector_length,num_of_data))
    for i in range(num_of_data) :
        im_file =  result_folder+'/'+ 'Iter'+str(currentIter - 1)+'_simu_' + str(i)  + '.nrrd'
        tmp = sitk.ReadImage(im_file)
        #print 'read:'+ im_file
        tmp = sitk.GetArrayFromImage(tmp)
        Y[:,i] = tmp.reshape(-1)

    low_rank, sparse, n_iter,rank, sparsity = rpca(Y,lamda)
    saveImagesFromDM(low_rank,result_folder+'/'+ 'Iter'+str(currentIter) +'_LowRank_', reference_im_name)
    saveImagesFromDM(sparse,result_folder+'/'+ 'Iter'+str(currentIter) +'_Sparse_', reference_im_name)

    # Visualize and inspect
    fig = plt.figure(figsize=(15,5))
    showSlice(Y, 'Iter'+str(currentIter) +' Input',plt.cm.gray,0, reference_im_name)
    showSlice(low_rank,'Iter'+str(currentIter) +' low rank',plt.cm.gray,1, reference_im_name)
    showSlice(sparse,'Iter'+str(currentIter) +' sparse',plt.cm.gray,2, reference_im_name)
    plt.savefig(result_folder+'/'+'Iter'+ str(currentIter)+'.png')
    fig.clf()
    plt.close(fig)

    # visualize the difference image  lowrank - reference
    im_ref_vec  = im_ref_array.reshape(-1)
    fig = plt.figure(figsize=(15,5))
    for i  in xrange(num_of_data):
        plt.subplot2grid((1,num_of_data),(0,i))
        a = np.array((low_rank[:,i]).reshape(-1) - im_ref_vec)
        im = a.reshape(z_dim,x_dim,y_dim)
        implot = plt.imshow(im[slice_nr,:,:],plt.cm.gray)
        plt.axis('off')
        plt.title('Iter'+str(currentIter)+'_simu_'+str(i))
    plt.savefig(result_folder+'/'+'Differ_Lowrank_Iter'+str(currentIter)+'.png')
    fig.clf()
    plt.close(fig)

     # Register low-rank images to the reference (healthy) image, and update the input images to the next iteration

    for i in xrange(num_of_data):
        movingIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_LowRank_' + str(i)  +'.nrrd'
        outputIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Deformed_LowRank' + str(i)  + '.nrrd'
        outputTransform = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Transform_' + str(i) +  '.tfm'
        outputDVF = result_folder+'/'+ 'Iter'+ str(currentIter)+'_DVF_' + str(i) +  '.nrrd'

        outputComposedDVFIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Composed_DVF_' + str(i) +  '.nrrd'
        newInputImage = result_folder+'/Iter'+ str(currentIter)+'_simu_' +str(i) +  '.nrrd'
        initialInputImage= result_folder+'/Iter0_simu_' +str(i) +  '.nrrd'

        logFile = open(result_folder+'/iteration'+ str(i)+'.log', 'w')


        cmd = DemonsReg(reference_im_name,movingIm,outputIm, outputDVF)
        #cmd = BSplineReg(reference_im_name,movingIm,outputIm, outputTransform,gridSize)
        #cmd = cmd + ';' +ConvertTransform(reference_im_name,outputTransform,outputDVF)


        # compose deformations then apply to the original input image
        if COMPOSE_DVF is True:
            DVFImageList=[]
            for k in xrange(currentIter):
                DVFImageList.append(result_folder+'/'+ 'Iter'+ str(k+1)+'_DVF_' + str(i) +  '.nrrd')

            cmd = cmd + ';' + composeMultipleDVFs(reference_im_name,DVFImageList,outputComposedDVFIm)
            cmd = cmd + ';' + updateInputImageWithDVF(initialInputImage,reference_im_name,                                                       outputComposedDVFIm,newInputImage)

            #cmd = cmd + ';' + WarpImageMultiDVF(initialInputImage,reference_im_name,DVFImageList,newInputImage)

        else:
            # update from previous Image
            previousInputImage = result_folder+'/Iter'+str(currentIter-1)+ '_simu_' + str(i)  + '.nrrd'
            cmd = cmd + ';' + updateInputImageWithDVF(previousInputImage,reference_im_name, outputDVF,newInputImage)

        process = subprocess.Popen(cmd, stdout = logFile, shell = True)
        process.wait()
        logFile.close()
    #gc.collect()
    return sparsity


def computeHealthyDeformations():
    #Validation & Comparison: register healthy simulation images to frechet mean
    healthyIm_names = [      data_folder+'/healthySimu1.nrrd',
     data_folder+'/healthySimu2.nrrd',
     data_folder+'/healthySimu3.nrrd',
     data_folder+'/healthySimu4.nrrd']

    for i in xrange(num_of_data):
        fn = healthyIm_names[selection[i]]
        outputIm = data_folder+'/healthy/healthyDeformed_' + str(i)  + '.nrrd'
        outputDVF = data_folder+'/healthy/healthyDVF_' + str(i) +  '.nrrd'

        cmd = DemonsReg(reference_im_name,fn,outputIm, outputDVF)

        logFile = open(data_folder+'/healthy/healthy_'+ str(i)+'.log', 'w')

        process = subprocess.Popen(cmd, stdout = logFile, shell = True)
        process.wait()
        logFile.close()
    return

# In[ ]:
def main():
  # main
  #computeHealthyDeformations()
  os.system('cp /home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples/Low_Rank_Atlas_Iter_BULLSEYE.py  '+result_folder)

  NUM_OF_ITERATIONS = 10
  lamda = 1.0
  sparsity = np.zeros(NUM_OF_ITERATIONS)
  # gridSize = [3,3,3]
  for i in xrange(0,NUM_OF_ITERATIONS):
      print 'iter'+str(i+1)
      sparsity[i]= runIteration(i+1,lamda)
      gc.collect()

      lamda = lamda +0.1
      #gridSize[0] =gridSize[0] + 2
      #gridSize[1] =gridSize[1] + 2
      #gridSize[2] =gridSize[2] + 2

  #visulaize DVFs
  for i in xrange(num_of_data):
     for currentIter in xrange(1,NUM_OF_ITERATIONS):
          outputComposedDVFIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_DVF_' + str(i) +  '.nrrd'
          deformedInputIm = result_folder+'/'+ 'Iter'+ str(currentIter+1)+'_simu_' + str(i) +  '.nrrd'
          gridVisDVF(outputComposedDVFIm,32,'DVF_Vis_Simu'+str(i) +'_Iter'+str(currentIter+1),result_folder,deformedInputIm)



  plt.figure()
  plt.plot(xrange(NUM_OF_ITERATIONS), sparsity)
  plt.savefig(result_folder+'/Sparsity.png')
  # In[ ]:

  # check atlas similarities
  # check atlas similarities

  MSE = np.zeros((num_of_data, NUM_OF_ITERATIONS))
  for i in xrange(NUM_OF_ITERATIONS):
      for j in xrange(num_of_data):
          lowrankIm = result_folder+'/'+ 'Iter'+str(i+1) +'_LowRank_' +str(j) + '.nrrd'
          im = sitk.ReadImage(lowrankIm) # image in SITK format
          im_array = sitk.GetArrayFromImage(im)
          MSE[j,i] = np.sum(np.square(im_ref_array - im_array))


  plt.figure()
  plt.plot(xrange(NUM_OF_ITERATIONS),sum(MSE,0) )
  plt.savefig(result_folder+'/MSE.png')
  return



if __name__ == "__main__":
    main()






