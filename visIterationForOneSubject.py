import sys
sys.path.append('/home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples')
from low_rank_atlas_iter import *


result_folder ='/home/xiaoxiao/work/data/BRATS/BRATS-2/Synthetic_Data/Flair_w0.9'
outputPNGFolder = result_folder
os.system('mkdir '+outputPNGFolder)
madality = 'Flair'
NUM_OF_ITERATIONS = 10


def showIteraionSlices(typename, row, numList, t):
     ii = 0
     vmin = 0
     vmax = 0
     for i in numList:
          inputIm = result_folder+'/'+ 'Iter'+ str(i)+'_'+typename+'_' + str(inputNumber) +  '.nrrd'
          im= sitk.ReadImage(inputIm) # image in SITK format
          im_array = sitk.GetArrayFromImage(im) # get numpy array
          z_dim, x_dim, y_dim = im_array.shape # get 3D volume shape

          plt.subplot2grid((3,NUM_OF_ITERATIONS),(row,ii))
          implot = plt.imshow(np.flipud(im_array[z_dim/2,:,:]),plt.cm.gray)
          if ii == 0:
            vmin, vmax = plt.gci().get_clim()
          ii = ii + 1
          plt.clim([vmin,vmax])
          plt.axis('off')
          plt.title('Iter'+str(ii) +' '+t)



for inputNumber in range(8):
     fig = plt.figure(figsize=(15,5))
     showIteraionSlices('Flair',0, range(0,NUM_OF_ITERATIONS),'D') #iteraion i's input image is output of iter(i-!)
     showIteraionSlices('LowRank',1, range(1,NUM_OF_ITERATIONS+1),'L')
     showIteraionSlices('Sparse',2,range(1,NUM_OF_ITERATIONS+1),'S')
 
     plt.savefig(outputPNGFolder+'/'+'Input'+str(inputNumber) +'All'+str(NUM_OF_ITERATIONS)+'Iterations.png')
     fig.clf()
     plt.close(fig)

