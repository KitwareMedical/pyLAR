import sys
from low_rank_atlas_iter import *
import imp
import time

lamda =0.8
result_folder = '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/ANTS_Flair_w'+str(lamda)
reference_im_name = '/home/xiaoxiao/work/data/SRI24/T1_Crop.nii.gz'

level = 0
currentIter =1
i = 0


fixedIm = reference_im_name
movingIm = result_folder+'/L'+ str(level)+'_Iter'+ str(currentIter)+'_LowRank_' + str(i)  +'.nrrd'
outputTransformPrefix = result_folder+'/L'+ str(level)+'_Iter'+ str(currentIter)+'_'+str(i)#+'_'



s = time.clock()
cmd = ANTS(fixedIm,movingIm,outputTransformPrefix)
print cmd
#os.system(cmd)
e = time.clock()
l = e - s
print 'ANTS running time:  %f seconds'%(l)

DVFIm = outputTransformPrefix +'0Warp.nii.gz' 
InvDVFIm = outputTransformPrefix +'0InverseWarp.nii.gz' 

outputIm = outputTransformPrefix + '_Warped.nrrd'
sliceNum =77
modality = 'Flair'
gridVisDVF(DVFIm,sliceNum,'DVF_Vis_'+modality+'_'+str(i) +'_Iter'+str(currentIter), result_folder,outputIm,20)
gridVisDVF(InvDVFIm,sliceNum,'InvDVF_Vis_'+modality+'_'+str(i) +'_Iter'+str(currentIter), result_folder,outputIm,20)


#visDF
