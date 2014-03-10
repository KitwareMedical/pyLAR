import sys
sys.path.append('/home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples')
from low_rank_atlas_iter import *


result_folder ='/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/RegulateBspline_grid10_w0.7' 
outputPNGFolder = result_folder
inputNumber = 0
slicerNum = 77
madality = 'T1'
NUM_OF_ITERATIONS = 15

for i in [inputNumber]:
     for currentIter in range(1,NUM_OF_ITERATIONS+1):
          composedDVFIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Composed_DVF_' + str(i) +  '.nrrd'
          #DVFIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_DVF_' + str(i) +  '.nrrd'
          deformedInputIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_T1_' + str(i) +  '.nrrd'
          gridVisDVF(composedDVFIm,slicerNum,'CDVF_Vis_T1_'+str(i) +'_Iter'+str(currentIter), outputPNGFolder,deformedInputIm,20)

          #gridVisDVF(DVFIm,slicerNum,'DVF_Vis_T1_'+str(i) +'_Iter'+str(currentIter), outputPNGFolder,deformedInputIm,20)
