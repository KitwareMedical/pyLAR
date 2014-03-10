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
tissues_Image =  '/home/xiaoxiao/work/data/SRI24/tissues_crop.nrrd'


#CropImage('/home/xiaoxiao/work/data/SRI24/tissues.nrrd',tissues_Image,[50,20,0],[50,30,0])

###############################  the main pipeline #############################
def collectStatstics(InputNum, NUM_OF_ITERATIONS):
    allStats = [ ]
    for currentIter in range(1,NUM_OF_ITERATIONS+1):
        outputComposedDVFIm = result_folder+'/'+ 'Iter'+ str(currentIter)+'_Composed_DVF_' + str(InputNum) +  '.nrrd'
        inputImage = result_folder+'/Iter'+ str(currentIter)+'_Flair_' +str(InputNum) +  '.nrrd'
        tumorMaskImage= '/home/xiaoxiao/work/data/BRATS/BRATS-2/Synthetic_Data/TumorMask/affine3more_' +str(InputNum) +  '.nrrd'
        deformedTumorMaskImage= '/home/xiaoxiao/work/data/BRATS/BRATS-2/Synthetic_Data/TumorMask/deformed_3more_' +str(InputNum) +  '.nrrd'


        #outputTissueImage = result_folder+'/tissues_'+str(InputNum) + '_Iter'+ str(currentIter) +  '.nrrd'
        logFile = open(result_folder+'/Iter'+str(currentIter)+'_TissueStats_'+ str(InputNum)+'.log', 'w')

        updateInputImageWithDVF(tumorMaskImage,tumorMaskImage,outputComposedDVFIm, deformedTumorMaskImage,True)

        # stats is a matrix of the statistics, including four metrics: mean, std, var,min, max
        stats = computeLabelStatistics(inputImage, tissues_Image, deformedTumorMaskImage)

        allStats.append(stats)

    return allStats


#######################################  main ##################################
def main():

    global result_folder, NUM_OF_ITERATIONS, num_of_data 
    result_folder = '/home/xiaoxiao/work/data/BRATS/BRATS-2/Synthetic_Data/Flair_w0.9'

    num_of_data = 8
    NUM_OF_ITERATIONS = 10

    # save script to the result folder for paramter checkups
    os.system('cp /home/xiaoxiao/work/src/TubeTK/Base/Python/pyrpca/examples/TissueStatisticsValidation.py   ' +result_folder)
    #sys.stdout = open(result_folder+'/RUN_tissue_stats.log', "w")

    # collect label statistics and save into txt files
    CALCULATE = True 
    if CALCULATE:
      for inputNum in range(num_of_data):
         # a list of stats matrix ( numOfLables  *  5)
         allStats = collectStatstics(inputNum, NUM_OF_ITERATIONS)
         with open(result_folder+ '/input'+str(inputNum)+'_label_stats.txt', 'wb') as f:
            pickle.dump(allStats,f)
            f.close()

    # plot label statistics for each input over the iterations
    PLOT_EACH_SUBJECT = False
    if PLOT_EACH_SUBJECT:
      for inputNum in range(num_of_data):
          # visualize std
          with open(result_folder+ '/input'+str(inputNum)+'_label_stats.txt', 'r') as f:
              allstats =  pickle.load(f)
              plotIterStats(inputNum,allstats,1,'STD',NUM_OF_ITERATIONS)
              #plotIterStats(inputNum,allstats,2,'VAR',NUM_OF_ITERATIONS)
              #plotIterStats(inputNum,allstats,0,'MEAN',NUM_OF_ITERATIONS)
              #plotIterStats(inputNum,allstats,3,'MIN',NUM_OF_ITERATIONS)
              #plotIterStats(inputNum,allstats,4,'MAX',NUM_OF_ITERATIONS)
              f.close()

    # plot population label statistics 
    plotAllStats(num_of_data,NUM_OF_ITERATIONS,1,'STD')
    #plotAllStats(num_of_data,NUM_OF_ITERATIONS,2,'VAR')
    #plotAllStats(num_of_data,NUM_OF_ITERATIONS,0,'MEAN')
    #plotAllStats(num_of_data,NUM_OF_ITERATIONS,3,'MIN')
    #plotAllStats(num_of_data,NUM_OF_ITERATIONS,4,'MAX')

    #plotAllStats_Seperate(num_of_data,NUM_OF_ITERATIONS,1,'STD')
    return

def plotAllStats_Seperate(num_of_data,NUM_OF_ITERATIONS,metricInx,metricType):
    CSF = np.zeros((num_of_data,NUM_OF_ITERATIONS))
    WM = np.zeros((num_of_data,NUM_OF_ITERATIONS))
    GM = np.zeros((num_of_data,NUM_OF_ITERATIONS))
    for inputNum in range(num_of_data):
        with open(result_folder+ '/input'+str(inputNum)+'_label_stats.txt', 'r') as f:
          allstats =  pickle.load(f)
          for j in range(NUM_OF_ITERATIONS):
             iterStats = allstats[j]
             CSF[inputNum,j] = iterStats[1, metricInx]
             WM[inputNum,j]  = iterStats[2, metricInx]
             GM[inputNum,j]  = iterStats[3, metricInx]
          f.close()
    print 'WM',WM[:,NUM_OF_ITERATIONS-1]
    print 'GM',GM[:,NUM_OF_ITERATIONS-1]
    print 'CSF',CSF[:,NUM_OF_ITERATIONS-1]
    plt.figure()
    for i in range(num_of_data):
      plt.plot(CSF[i,:])
    plt.title('CSF label '+ metricType)
    plt.xlabel('Iteration')
    plt.savefig(result_folder +'/CSF_label_'+metricType+'.png')
    plt.close()

    plt.figure()
    for i in range(num_of_data):
      plt.plot(GM[i,:])
    plt.title('GM label '+ metricType)
    plt.xlabel('Iteration')
    plt.savefig(result_folder +'/GM_label_'+metricType+'.png')
    plt.close()

    plt.figure()
    for i in range(num_of_data):
      plt.plot(WM[i,:])
    plt.title('WM label '+ metricType)
    plt.xlabel('Iteration')
    plt.savefig(result_folder +'/WM_label_'+metricType+'.png')
    plt.close()
    return

def plotAllStats(num_of_data,NUM_OF_ITERATIONS,metricInx,metricType):
    CSF = np.zeros((num_of_data,NUM_OF_ITERATIONS))
    WM = np.zeros((num_of_data,NUM_OF_ITERATIONS))
    GM = np.zeros((num_of_data,NUM_OF_ITERATIONS))
    for inputNum in range(num_of_data):
        with open(result_folder+ '/input'+str(inputNum)+'_label_stats.txt', 'r') as f:
          allstats =  pickle.load(f)
          for j in range(NUM_OF_ITERATIONS):
             iterStats = allstats[j]
             CSF[inputNum,j] = iterStats[1, metricInx]
             WM[inputNum,j]  = iterStats[2, metricInx]
             GM[inputNum,j]  = iterStats[3, metricInx]
          f.close()
    print 'WM',WM[:,NUM_OF_ITERATIONS-1]
    print 'GM',GM[:,NUM_OF_ITERATIONS-1]
    print 'CSF',CSF[:,NUM_OF_ITERATIONS-1]
    plt.figure()
    plt.boxplot(CSF)
    plt.title('CSF label '+ metricType)
    plt.xlabel('Iteration')
    plt.savefig(result_folder +'/CSF_label_'+metricType+'.png')
    plt.close()

    plt.figure()
    plt.boxplot(GM)
    plt.title('GM label '+ metricType)
    plt.xlabel('Iteration')
    plt.savefig(result_folder +'/GM_label_'+metricType+'.png')
    plt.close()

    plt.figure()
    plt.boxplot(WM)
    plt.title('WM label '+ metricType)
    plt.xlabel('Iteration')
    plt.savefig(result_folder +'/WM_label_'+metricType+'.png')
    plt.close()
    return

def plotIterStats(inputNum, allStats,metricInx,metricType,NUM_OF_ITERATIONS):
    fig = plt.figure()
    CSF = np.zeros(NUM_OF_ITERATIONS)
    GM = np.zeros(NUM_OF_ITERATIONS)
    WM = np.zeros(NUM_OF_ITERATIONS)
    for j in range(NUM_OF_ITERATIONS):
         iterStats = allStats[j]
         CSF[j] = iterStats[1,metricInx]
         WM[j] = iterStats[2,metricInx]
         GM[j] = iterStats[3,metricInx]
    plt.plot(CSF)
    plt.plot(WM)
    plt.plot(GM)
    plt.legend(['CSF','WM','GM'])
    plt.title('input'+str(inputNum)+' tissue label stats:' + metricType)
    plt.xlabel('iteration')
    plt.savefig(result_folder+ '/input'+str(inputNum)+'_label_'+metricType+'.png')
    plt.close(fig)

if __name__ == "__main__":
    main()
