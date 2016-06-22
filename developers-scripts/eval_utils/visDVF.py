import sys
import imp
sys.path.insert(0, './')
from low_rank_atlas_iter import *

configFN = sys.argv[1]
f = open(configFN)
config  = imp.load_source('config', '', f)
f.close()


result_folder = config.result_folder
outputPNGFolder = result_folder

inputNumber = 0
sliceNum = 77
level = 0
modality = config.modality
NUM_OF_ITERATIONS = config.NUM_OF_ITERATIONS_PER_LEVEL
REGISTRATION_TYPE = config.REGISTRATION_TYPE

for i in [inputNumber]:
     for currentIter in range(1,NUM_OF_ITERATIONS+1):
	 if REGISTRATION_TYPE == "BSpline":
            composedDVFIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Composed_DVF_' + str(i) +  '.nrrd'
            DVFIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_DVF_' + str(i) +  '.nrrd'
            deformedInputIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_'+modality+'_' + str(i) +  '.nrrd'
             #gridVisDVF(composedDVFIm,slicerNum,'CDVF_Vis_T1_'+str(i) +'_Iter'+str(currentIter), outputPNGFolder,deformedInputIm,20)

            gridVisDVF(DVFIm,slicerNum,'DVF_Vis_'+modality+'_'+str(i) +'_Iter'+str(currentIter), outputPNGFolder,deformedInputIm,20)
         if REGISTRATION_TYPE == "ANTS":
            outputTransformPrefix = result_folder+'/L'+ str(level)+'_Iter'+ str(currentIter)+'_'+str(i)+'_'
            DVFIm = outputTransformPrefix +'0Warp.nii.gz'
	    InvDVFIm = outputTransformPrefix +'0InverseWarp.nii.gz'
            deformedInputIm = result_folder+'/L'+str(level)+ '_Iter'+ str(currentIter)+'_'+modality+'_' + str(i) +  '.nrrd'
           
            gridVisDVF(DVFIm,sliceNum,'DVF_Vis_'+modality+'_'+str(i) +'_Iter'+str(currentIter), result_folder, deformedInputIm,20)
            print outputPNGFolder+'/DVF_Vis_'+modality+'_'+str(i) +'_Iter'+str(currentIter)+'.png'
