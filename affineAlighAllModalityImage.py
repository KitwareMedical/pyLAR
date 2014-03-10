import sys
sys.path.append('/home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples')

from low_rank_atlas_iter import *

data_folder= '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data'



t1_im_names = readTxtIntoList(data_folder +'/T1_FN.txt')
t2_im_names = readTxtIntoList(data_folder +'/T2_FN.txt')
flair_im_names = readTxtIntoList(data_folder +'/Flair_FN.txt')
tumor_im_names = readTxtIntoList(data_folder +'/Tumor_FN.txt')


selection = [0,1,3,4,6,7,9,10]
reference_im_name = '/home/xiaoxiao/work/data/SRI24/T1_Crop.nii.gz'

t1_result_folder = data_folder +'/T1_affineRegistered8Inputs'
t2_result_folder = data_folder +'/T2_affineRegistered8Inputs'
flair_result_folder = data_folder +'/Flair_affineRegistered8Inputs'
tumor_result_folder = data_folder +'/Tumor_affineRegistered8Inputs'
os.system('mkdir '+t1_result_folder)
os.system('mkdir '+t2_result_folder)
os.system('mkdir '+flair_result_folder)
os.system('mkdir '+tumor_result_folder)
num_of_data = len(selection)

for i in range(num_of_data):
        t1Im = t1_im_names[selection[i]] 
        outputIm = t1_result_folder+'/affine_T1_'+str(i)+'.nrrd'
        outputTransform = t1_result_folder +'/affine_'+str(i)+'.tfm'
        AffineReg(reference_im_name,t1Im,outputIm, outputTransform)

        t2Im = t2_im_names[selection[i]] 
        outputIm =  t2_result_folder+'/affine_T2_' + str(i)  + '.nrrd'
        applyLinearTransform(t2Im, reference_im_name, outputTransform,outputIm, True)

        flairIm = flair_im_names[selection[i]] 
        outputIm =  flair_result_folder+'/affine_Flair_' + str(i)  + '.nrrd'
        applyLinearTransform(flairIm, reference_im_name, outputTransform,outputIm, True)

        tumorIm = tumor_im_names[selection[i]] 
        outputIm =  tumor_result_folder+'/affine_Tumor_' + str(i)  + '.nrrd'
        applyLinearTransform(tumorIm, reference_im_name, outputTransform,outputIm, True)
