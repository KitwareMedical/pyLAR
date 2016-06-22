#!/usr/bin/env python
"""Testing scripts to run ANTS for image registraiton on simulated Bull's eye images."""

import sys
import os
import inspect
import argparse

from low_rank_atlas.low_rank_atlas_iter import loadConfiguration

myfilepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
sys.path.insert(0, os.path.abspath(os.path.join(myfilepath, '../')))
from low_rank_atlas_iter import *


parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description=__doc__,
)
parser.add_argument("--result_folder", nargs=1, type=str, required=True, help='Directory to store the output images')
parser.add_argument("--fixed_image", nargs=1, type=str, required=True, help='Fixed Image')
parser.add_argument("--moving_image", nargs=1, type=str, required=True, help='Moving Image')
parser.add_argument("--software", nargs=1, type=str, required=True, help='Software Configuration File')
args = parser.parse_args()

result_folder = args.result_folder[0]
if not os.path.exists(result_folder):
  os.makedirs(result_folder)
fixedIm= args.fixed_image[0]  # '/Users/xiaoxiaoliu/work/data/BullEyeSimulation/fMeanSimu.nrrd'
movingIm = args.moving_image[0]  # '/Users/xiaoxiaoliu/work/data/BullEyeSimulation/healthySimu1.nrrd'
outputTransformPrefix = os.path.join(result_folder,'output_simu1_')
software = loadConfiguration(args.software[0], 'software')

if not hasattr(software,'EXE_ANTS'):
  print "Path to ANTS not given in software configuration file."
  sys.exit(1)

params = {'Dimension' : 3,\
          'Convergence' : '[70x50x20,1e-6,10]',\
          'ShrinkFactors' : '4x2x1',\
          'SmoothingSigmas' : '2x1x0vox',\
          'Transform' :'SyN[1.0]',\
          'Metric': 'CC[%s,%s,1,3]' %(fixedIm,movingIm)}
#          'Metric': 'MI[%s,%s,1,32]' %(fixedIm,movingIm)}
cmd = ANTS(software.EXE_ANTS, fixedIm,movingIm,outputTransformPrefix, params)
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
