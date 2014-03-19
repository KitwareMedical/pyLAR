
# In[1]:

import numpy as np # Numpy for general purpose processing
import SimpleITK as sitk # SimpleITK to load images
import sys
import os
import matplotlib.pyplot as plt


# In[2]:

def sphereImage(radius, x_dim,y_dim,z_dim, label):
    sData = np.zeros((x_dim,y_dim,z_dim))
    for i in range(x_dim):
        for j in range (y_dim):
            for k in range (z_dim):
                if ( pow(i-x_dim/2.0,2) + pow(j-y_dim/2.0,2) + pow(k-z_dim/2.0,2) )< pow(radius,2) :
                    sData[i,j,k] = label
    return sData

def tumorImage(radius, x_dim,y_dim,z_dim, label,l_x,l_y,l_z):
    tData = np.zeros((x_dim,y_dim,z_dim))
    for i in range(x_dim):
        for j in range (y_dim):
            for k in range (z_dim):
                if ( pow(i-l_x,2) + pow(j-l_y,2) + pow(k-l_z,2) )< pow(radius,2) :
                    tData[i,j,k] = label
    return tData





def simuGen(radii,baseIntensity,outputImageFileName,tumorLocation =[0,0,0],tumorRadius = 0, gaussianSigma = 1.0):
    s1 = sphereImage(radii[2],size,size,size,baseIntensity*3)
    s2 = sphereImage(radii[1],size,size,size,baseIntensity*2)
    s3 = sphereImage(radii[0],size,size,size,baseIntensity)

    t = tumorImage(tumorRadius,size,size,size,baseIntensity*5, tumorLocation[0],tumorLocation[1],tumorLocation[2])
    s_final = s1 + s2 + s3 +t
    
    img = sitk.GetImageFromArray(s_final)
    img.SetOrigin([0,0,0])
    img.SetSpacing([1.0,1.0,1.0]) 
    
    # apply gaussian blur
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma (gaussianSigma )
    img_blur = gaussian.Execute (img)
    s_final = sitk.GetArrayFromImage(img_blur)
    sitk.WriteImage(img_blur,outputImageFileName)
    return s_final


# In[3]:

size = 64
smallSphereRadius = size/8
largeSphereRadius = size*3/8
midSphereRaidus = (smallSphereRadius + largeSphereRadius )/2
imArray = simuGen([smallSphereRadius,midSphereRaidus,largeSphereRadius],10.0,                  '/home/xiaoxiao/work/data/BullEyeSimulation/fMeanSimu.nrrd')
plt.figure()
plt.imshow(imArray[size/2,:,:],plt.cm.gray)
plt.title('healthy atlas (fMean)')


# Out[3]:

#     <matplotlib.text.Text at 0x4353790>

# image file:

# In[4]:

num_of_simulation = 4

stepRadius =(largeSphereRadius-smallSphereRadius )/(num_of_simulation +1)


# In[5]:


tumorRadius = 6.0
#import random
dataFolder = '/home/xiaoxiao/work/data/BullEyeSimulation_largeTumor'
os.system('mkdir  '+dataFolder)

#tumorLocation: random.randrange(size/8,size-size/8)
t = np.zeros((4,2))
t[0] = [15,15]
t[1] = [20,45]
t[2] = [50,15] 
t[3] = [45,45]

fig = plt.figure(figsize=(15,5))
for i in range (num_of_simulation):
    simuFileName =dataFolder + '/simu'+str(i+1)+'.nrrd'
    midSphereRaidus = smallSphereRadius + stepRadius*(i+1)
    #tumorLocation = [size/2,random.randrange(size/8,size-size/8), random.randrange(size/8,size-size/8)]
    tumorLocation = [size/2,t[i][0],t[i][1]]
    imArray = simuGen([smallSphereRadius,midSphereRaidus,largeSphereRadius],10.0,simuFileName,            tumorLocation, tumorRadius)
    fig.add_subplot(1,num_of_simulation,i+1)
    implot = plt.imshow(imArray[size/2,:,:],plt.cm.gray)
    plt.title('simu'+str(i+1))



# Out[5]:

# image file:

# In[12]:

# generate healthy bull's eye images
fig = plt.figure(figsize=(15,5))
for i in range (num_of_simulation):
    simuFileName = dataFolder + 'healthySimu'+str(i+1)+'.nrrd'
    midSphereRaidus = smallSphereRadius + stepRadius*(i+1)
    imArray = simuGen([smallSphereRadius,midSphereRaidus,largeSphereRadius],10.0,simuFileName,            [0,0,0], 0)
    fig.add_subplot(1,num_of_simulation,i+1)
    implot = plt.imshow(imArray[size/2,:,:],plt.cm.gray)
    plt.title('healthy simu'+str(i+1))


# Out[12]:

# image file:
