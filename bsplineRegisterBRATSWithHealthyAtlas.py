import sys
sys.path.append('/home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples')
from low_rank_atlas_iter import *



reference_im_name = '/home/xiaoxiao/work/data/SRI24/T1_Crop.nii.gz'
tissues_Image =  '/home/xiaoxiao/work/data/SRI24/tissues_crop.nrrd'
data_folder = '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/flair_bsplineRegToSRI'

sys.stdout = open(data_folder+'/RUN.log', "w")
ps=[]
num_of_data = 8
CSF = np.zeros(num_of_data)
GM = np.zeros(num_of_data)
WM = np.zeros(num_of_data)
for i in range(num_of_data):
    print i
    movingIm = data_folder+'/Iter0_Flair_'+str(i)+'.nrrd'
    outputIm = data_folder+'/deformed_Flair_'+str(i)+'.nrrd'
    outputTransform = data_folder+'/Flair_'+str(i)+'.tfm'
    outputDVF = data_folder +'/DVF_Flair_'+str(i)+'.nrrd'
  

    logFile = open(data_folder+'_RUN_'+ str(i)+'.log', 'w')
    gridSize = [10,12,10]
    maxDisp = 0
    #cmd = BSplineReg_BRAINSFit(reference_im_name,movingIm,outputIm,outputTransform,gridSize, maxDisp)

    #cmd +=';'+ ConvertTransform(reference_im_name,outputTransform,outputDVF)
    
    #process = subprocess.Popen(cmd, stdout=logFile, shell = True) 
    #process.wait()

    tumorMaskImage= '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/TumorMask/affine3more_' +str(i) +  '.nrrd'
    deformedTumorMaskImage= '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/TumorMask/bspline_deformed_3more_' +str(i) +  '.nrrd'
    updateInputImageWithDVF(tumorMaskImage,tumorMaskImage,outputDVF, deformedTumorMaskImage,True)
   
    stats = computeLabelStatistics(outputIm, tissues_Image, deformedTumorMaskImage)
    metricInx =1
    CSF[i] = stats[1,metricInx]
    WM[i] = stats[2,metricInx]
    GM[i] = stats[3,metricInx]

    gridVisDVF(outputDVF,77,'DVF_Vis_Flair_'+str(i), data_folder,outputIm,20)
print 'CSF',CSF
print 'WM',WM
print 'GM',GM

