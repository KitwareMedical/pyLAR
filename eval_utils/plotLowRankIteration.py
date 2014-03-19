# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys
sys.path.append('/home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples')
from low_rank_atlas_iter import *



lamda=0.9
selection = range(8)
result_folder = '/home/xiaoxiao/work/data/BRATS/BRATS-2/Synthetic_Data/Flair_w'+str(lamda)



reference_im_name = '/home/xiaoxiao/work/data/SRI24/T1_Crop.nii.gz'
im_ref = sitk.ReadImage(reference_im_name) # image in SITK format
im_ref_array = sitk.GetArrayFromImage(im_ref) # get numpy array
z_dim, x_dim, y_dim = im_ref_array.shape # get 3D volume shape
vector_length = z_dim* x_dim*y_dim


def loaddata(num_of_data, iterCount,typename):# prepare data matri
    Y = np.zeros((vector_length, num_of_data))
    for i in range(num_of_data) :
        im_file =  result_folder+'/'+ 'Iter'+str(iterCount)+'_'+typename+'_' + str(i)  + '.nrrd'
        tmp = sitk.ReadImage(im_file)
        tmp = sitk.GetArrayFromImage(tmp)
        Y[:,i] = tmp.reshape(-1)
        del tmp
    return Y

def main():

    num_of_data = len(selection)


    NUM_OF_ITERATIONS = 10

    gridSize = [6,8,6]
    Y = np.zeros((vector_length,num_of_data))
    for iterCount in range(1,NUM_OF_ITERATIONS + 1):

        Y= loaddata(num_of_data,iterCount-1,'Flair')
        low_rank = loaddata(num_of_data,iterCount,'LowRank')
        sparse= loaddata(num_of_data,iterCount,'Sparse')
        fig = plt.figure(figsize=(15,5))
        showSlice(Y, ' Input',plt.cm.gray,0,reference_im_name)
        showSlice(low_rank,' low rank',plt.cm.gray,1, reference_im_name)
        showSlice(sparse,' sparse',plt.cm.gray,2, reference_im_name)
        plt.savefig(result_folder+'/'+'Iter'+ str(iterCount)+'_w_'+str(lamda)+'.png')
        fig.clf()
        plt.close(fig)


if __name__ == "__main__":
    main()

