# Testing scripts to run ANTS for image registration on BRATS images

import sys
sys.path.append('./')
from low_rank_atlas_iter import *
import imp
import time

lamda =0.8
result_folder = '/Users/xiaoxiaoliu/work/data/BRATS/BRATS-2/Image_Data/results/LAB_ANTS_Flair'
reference_im_name = '/Users/xiaoxiaoliu/work/data/SRI24/T1_Crop.nii.gz'

level = 0
currentIter =1
i = 0


fixedIm = reference_im_name
movingIm = result_folder+'/L'+ str(level)+'_Iter'+ str(currentIter)+'_LowRank_' + str(i)  +'.nrrd'
outputTransformPrefix = result_folder+'/test_L'+ str(level)+'_Iter'+ str(currentIter)+'_'+str(i)+'_'


antsParams = {'Convergence' : '[100x50x25,1e-6,10]',\
          'Dimension': 3,\
          'ShrinkFactors' : '4x2x1',\
          'SmoothingSigmas' : '2x1x0vox',\
          'Transform' :'SyN[0.5]',\
          'Metric': 'Mattes[fixedIm,movingIm,1,50,Regular,0.95]'}


s = time.time()
cmd=''
antsParams['Metric'] = antsParams['Metric'].replace('fixedIm', fixedIm)
antsParams['Metric'] = antsParams['Metric'].replace('movingIm', movingIm)
cmd = ANTS(fixedIm,movingIm,outputTransformPrefix, antsParams, None)
print cmd
os.system(cmd)
e = time.time()
l = e - s
print 'ANTS running time:  %f seconds'%(l)
print 'Output to %s'%(result_folder)

DVFIm = outputTransformPrefix +'0Warp.nii.gz'
InvDVFIm = outputTransformPrefix +'0InverseWarp.nii.gz'

outputIm = outputTransformPrefix + 'Warped.nrrd'
sliceNum =77
modality = 'Flair'
gridVisDVF(DVFIm,sliceNum,'DVF_Vis_'+modality+'_'+str(i) +'_Iter'+str(currentIter), result_folder,outputIm,20)
gridVisDVF(InvDVFIm,sliceNum,'InvDVF_Vis_'+modality+'_'+str(i) +'_Iter'+str(currentIter), result_folder,outputIm,20)


s = time.time()
initialTransform = DVFIm
outputTransformPrefix = result_folder+'/L'+ str(level)+'_Iter'+ str(currentIter)+'_'+str(i)+'_further_'
cmd = ANTS(fixedIm,movingIm,outputTransformPrefix, antsParams,initialTransform)
print cmd
os.system(cmd)
e = time.time()
l = e - s
print 'ANTS running time:  %f seconds'%(l)
#visDF
