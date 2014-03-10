# -*- coding: utf-8 -*-
import sys
sys.path.append('./')
from low_rank_atlas_iter import *
import pickle


# global variables
data_folder = ''
reference_im_name = ''
result_folder = ''
selection = []
im_names =[]

def main():
    global result_folder, NUM_OF_ITERATIONS, num_of_data 
    result_folder = '/home/xiaoxiao/work/data/BRATS/BRATS-2/Image_Data/Flair_w0.8'
    num_of_data = 8
    NUM_OF_ITERATIONS = 10

    sum_sparse = np.zeros((num_of_data, NUM_OF_ITERATIONS))
    for inputNum in range(num_of_data):
        for currentIter in range(1,NUM_OF_ITERATIONS+1):
          sparseIm = result_folder+'/'+ 'Iter'+str(currentIter) +'_Sparse_'+str(inputNum)+'.nrrd'
          im = sitk.ReadImage(sparseIm)
          im_array= sitk.GetArrayFromImage(im) # get numpy array
          sum_sparse[inputNum,currentIter-1] = np.sum(np.abs(im_array))

    with open(result_folder+ '/sumsparse_dump.txt', 'wb') as f:
           pickle.dump(sum_sparse,f)
           f.close()

    plt.figure()
    plt.boxplot(sum_sparse)
    plt.title('Sum of Absolute Sparse Image')
    plt.xlabel('Iterations')
    plt.savefig(result_folder+'/SumSparse.png')

    return


if __name__ == "__main__":
    main()
