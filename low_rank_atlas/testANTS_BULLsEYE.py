import sys
from low_rank_atlas_iter import *
import imp
import time

result_folder = '/home/xiaoxiao/work/data/BullEyeSimulation/testANTS'

fixedIm= '/home/xiaoxiao/work/data/BullEyeSimulation/fMeanSimu.nrrd'
movingIm = '/home/xiaoxiao/work/data/BullEyeSimulation/healthySimu1.nrrd'
outputTransformPrefix = result_folder+'/output_simu1_'


params = {'SyNConvergence' : '[70x50x20,1e-6,10]',\
          'SyNShrinkFactors' : '4x2x1',\
          'SynSmoothingSigmas' : '2x1x0vox',\
          'Transform' :'SyN[1.0]',\
          'Metric': 'CC[%s,%s,1,3]' %(fixedIm,movingIm)}
#          'Metric': 'MI[%s,%s,1,32]' %(fixedIm,movingIm)}
cmd = ANTS(fixedIm,movingIm,outputTransformPrefix, params)
print cmd
os.system(cmd)
print 'Output to %s'%(result_folder)

DVFIm = outputTransformPrefix +'0Warp.nii.gz'
InvDVFIm = outputTransformPrefix +'0InverseWarp.nii.gz'

outputIm = outputTransformPrefix + 'Warped.nrrd'
sliceNum =32
gridVisDVF(DVFIm,sliceNum,'DVF_Vis_simu1', result_folder,outputIm,40)
gridVisDVF(InvDVFIm,sliceNum,'InvDVF_Vis_simu1', result_folder,outputIm,40)


initialTransform = DVFIm
outputTransformPrefix = result_folder+'/output_simu1_further_'

#cmd = ANTS(fixedIm,movingIm,outputTransformPrefix,params, initialTransform)
