
# In[25]:
import sys
sys.path.append('./')
from low_rank_atlas_iter import *


# In[26]:

# global settings
data_folder = '/home/xiaoxiao/work/data/BullEyeSimulation'
result_folder = data_folder +'/low-rank-atlas-building'
os.system('mkdir '+result_folder)
im_names = [  data_folder+'/simu1.nrrd',
 data_folder+'/simu2.nrrd',
 data_folder+'/simu3.nrrd',
 data_folder+'/simu4.nrrd']
reference_im_name = data_folder +'/fMeanSimu.nrrd'


# In[27]:

# data selection and global parameters, prepare iter0 data
selection = [0,1,2,3]    
num_of_data = len(selection)  
for i in range(num_of_data):
    iter0fn = result_folder+'/Iter0'+ '_simu_' + str(i)  + '.nrrd'
    simufn = im_names[selection[i]]
    os.system('cp '+ simufn +' ' + iter0fn)


# In[28]:

# profile data size, save into global variables
im_ref = sitk.ReadImage(reference_im_name) # image in SITK format
im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
vector_length = z_dim* x_dim*y_dim

plt.figure()
implot = plt.imshow(im_ref_array[32,:,:],plt.cm.gray)
plt.title('healthy atlas')

slice_nr = 32  # just for vis purpose


# Out[28]:

# image file:

# In[29]:

###############################  the main pipeline #############################
def AverageImages(currentIter,atlasIm):
    executable = '/home/xiaoxiao/work/bin/ANTS/bin/AverageImages'
    listOfImages = []
    for i in range(num_of_data):
        movingIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_LowRank_' + str(i)  +'.nrrd'
        listOfImages.append(movingIm)
    arguments = ' 3 ' + atlasIm +'  0  ' +  ' '.join(listOfImages)
    cmd = executable + ' ' + arguments
    tempFile = open(result_folder+'/average.log', 'w')
    process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
    process.wait()
    tempFile.close()
    return

def runIteration(currentIter,lamda,gridSize=[3,3,3]):
    Y = np.zeros((vector_length,num_of_data))
    for i in range(num_of_data) :
        im_file =  result_folder+'/'+ 'Iter%d_simu_%d.nrrd' %(currentIter-1, i)
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
    plt.savefig(result_folder+'/'+'Iter'+ str(currentIter)+'_w_'+str(lamda)+'.png')
    plt.close(fig)
    
    # Register low-rank images to the reference (healthy) image, and update the input images to the next iteration
    atlas_im_name = result_folder+'/Iter'+ str(currentIter) +'_atlas.nrrd'
    # Average lowrank images
    AverageImages(currentIter,atlas_im_name)  
    
    # visualize the difference image  lowrank - reference
    im_ref_vec  = im_ref_array.reshape(-1)
    plt.figure(figsize=(15,5))
    for i  in range(num_of_data):
        plt.subplot2grid((1,num_of_data+1),(0,i))
        a = np.array((low_rank[:,i]).reshape(-1) - im_ref_vec)
        im = a.reshape(z_dim,x_dim,y_dim)
        implot = plt.imshow(im[slice_nr,:,:],plt.cm.gray)
        plt.axis('off')
        plt.title('Iter'+str(currentIter)+'_simu_'+str(i))
    # visulizat atlat difference    
    plt.subplot2grid((1,num_of_data+1),(0,num_of_data))
    im_atlas = sitk.ReadImage(atlas_im_name)
    im_atlas_array = sitk.GetArrayFromImage(im_atlas)
    a = np.array(im_atlas_array.reshape(-1) - im_ref_vec)
    im = a.reshape(z_dim,x_dim,y_dim)
    implot = plt.imshow(im[slice_nr,:,:],plt.cm.gray)
    plt.axis('off')
    plt.title('Iter'+str(currentIter)+'_atlas_diff'+str(i))  
    plt.colorbar()
    plt.savefig(result_folder+'/'+'Differ_Lowrank_Iter'+str(currentIter)+'.png')
    plt.close(fig)
    
        
    for i in range(num_of_data):
        movingIm = result_folder+'/'+ 'Iter'+ str(currentIter-1)+'_simu_' + str(i)  +'.nrrd'
        outputIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_deformed_' + str(i)  + '.nrrd'
        outputTransform = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Transform_' + str(i) +  '.tfm'
        outputDVF = result_folder+'/'+ 'Iter'+ str(currentIter)+'_DVF_' + str(i) +  '.nrrd'
      
        outputComposedDVFIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Composed_DVF_' + str(i) +  '.nrrd'
        newInputImage = result_folder+'/Iter'+ str(currentIter)+'_simu_' +str(i) +  '.nrrd'
        initialInputImage= result_folder+'/Iter0_simu_' +str(i) +  '.nrrd'
        
        logFile = open(result_folder+'/iteration'+ str(i)+'.log', 'w') 
   
        
            
        cmd = DemonsReg(atlas_im_name,movingIm,outputIm, outputDVF)
        #cmd = BSplineReg(atlas_im_name,movingIm,outputIm, outputTransform,gridSize)
        cmd = cmd + ';' +ConvertTransform(reference_im_name,outputTransform,outputDVF)
       
        
        # compose deformations then apply to the original input image
        if COMPOSE_DVF is True:
            DVFImageList=[]
            for k in range(currentIter):
                DVFImageList.append(result_folder+'/'+ 'Iter'+ str(k+1)+'_DVF_' + str(i) +  '.nrrd')
            
            cmd = cmd + ';' + composeMultipleDVFs(reference_im_name,DVFImageList,outputComposedDVFIm)
            cmd = cmd + ';' + updateInputImageWithDVF(initialInputImage,reference_im_name, outputComposedDVFIm,newInputImage)
         
            #cmd = cmd + ';' + WarpImageMultiDVF(initialInputImage,reference_im_name,DVFImageList,newInputImage)
          
        else:
            # update from previous Image
            previousInputImage = result_folder+'/Iter%d_simu_%d.nrrd' % (currentIter-1,i)
            cmd2 = updateInputImageWithDVF(previousInputImage,reference_im_name, outputDVF,newInputImage)
            cmd = cmd1 + ";" + cmd2
        
        process = subprocess.Popen(cmd, stdout = logFile, shell = True)
        process.wait()
        logFile.close()
        
    return
    


# In[30]:

# main
os.system('cp /home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples/BULLSEYE_Unbiased_LowRank_Atlas_Building.py  '+result_folder)

NUM_OF_ITERATIONS = 15

COMPOSE_DVF = True
lamda = 1.5

for i in range(NUM_OF_ITERATIONS):
    print 'iter'+str(i+1)
    runIteration(i+1,lamda)
   # if lamda < 1.5:
    #    lamda = lamda +0.1
    #gridSize = np.add(gridSize , [2,2,2])
    





# check atlas similarities
MSE = zeros( NUM_OF_ITERATIONS)
for i in range(NUM_OF_ITERATIONS):
        atlasIm = result_folder+'/'+ 'Iter'+str(i+1) +'_atlas.nrrd'
        im = sitk.ReadImage(atlasIm) # image in SITK format
        im_array = sitk.GetArrayFromImage(im)
        MSE[i] = np.sum(np.square(im_ref_array - im_array))
#        im= sitk.ReadImage(atlasIm) # image in SITK format
 #       im_array = sitk.GetArrayFromImage(im) # get numpy array
  #      figure()
   #     implot = plt.imshow(im_array[32,:,:],cm.gray)
    #    plt.title('Iter'+str(i)+ ' atlas')

    
plt.figure()
plt.plot(range(NUM_OF_ITERATIONS),MSE)
plt.savefig(result_folder+'/MSE.png')
