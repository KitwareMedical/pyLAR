import sys
sys.path.append('/home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples')

from low_rank_atlas_iter import *

data_folder= '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data'
sparse_result_folder = data_folder +'/sparse_InvDef_8inputs'

os.system('mkdir '+ sparse_result_folder)

reference_im_name = '/home/xiaoxiao/work/data/SRI24/T1_Crop.nii.gz'

result_folder = data_folder +'/Flair_w0.8'

num_of_data = 8
for i in range(num_of_data):
        SparseImage = result_folder+'/Iter10_Sparse_'+str(i)+'.nrrd'
        DeformedSparseImage = sparse_result_folder+'/deformedIter10_Sparse_'+str(i)+'.nrrd'
        DVFImage = result_folder+'/Iter9_Composed_DVF_'+str(i)+'.nrrd'
        inverseDVFImage = result_folder+'/Iter9_Composed_DVF_INV_'+str(i)+'.nrrd'
        genInverseDVF(DVFImage, inverseDVFImage, True)
        updateInputImageWithDVF(SparseImage,SparseImage,inverseDVFImage, DeformedSparseImage,True)

