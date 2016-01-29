#!/usr/bin/env python

# Library: pyLAR
#
# Copyright 2014 Kitware Inc. 28 Corporate Drive,
# Clifton Park, NY, 12065, USA.
#
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np # Numpy for general purpose processing
import SimpleITK as sitk # SimpleITK to load images
import sys
import os
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description=" This script generates 3D simulation data",
)
parser.add_argument('result_dir', nargs=1, type=str, help='Directory to store the output images')
args = parser.parse_args()
result_dir = args.result_dir[0]
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def circleImage(radius, x_dim, y_dim, label):
    sData = np.zeros((x_dim, y_dim))
    for i in range(x_dim):
        for j in range(y_dim):
                if (pow(i-x_dim/2.0, 2) + pow(j-y_dim/2.0, 2)) < pow(radius, 2):
                    sData[i, j] = label
    return sData


def randomObject(radius, x_dim, y_dim, label, l_x, l_y):
    tData = np.zeros((x_dim, y_dim))
    for i in range(x_dim):
        for j in range(y_dim):
                if (pow(i-l_x, 2) + pow(j-l_y, 2)) < pow(radius, 2):
                    tData[i, j] = label
    return tData


def simuGen(radii, baseIntensity, outputImageFileName, objectLocation =[0,0], objectRadius = 0, gaussianSigma = 1.0):
    s1 = circleImage(radii[2], size, size, baseIntensity*3)
    s2 = circleImage(radii[1], size, size, baseIntensity*2)
    s3 = circleImage(radii[0], size, size, baseIntensity)

    t = randomObject(objectRadius, size, size, baseIntensity*5, objectLocation[0], objectLocation[1])
    s_final = s1 + s2 + s3 +t

    img = sitk.GetImageFromArray(s_final)
    img.SetOrigin([0, 0])
    img.SetSpacing([1.0, 1.0])

    # apply gaussian blur
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(gaussianSigma)
    img_blur = gaussian.Execute(img)
    s_final = sitk.GetArrayFromImage(img_blur)
    sitk.WriteImage(img_blur, outputImageFileName)
    return s_final


size = 64
smallSphereRadius = size/8
largeSphereRadius = size*3/8
midSphereRadius = (smallSphereRadius + largeSphereRadius)/2

num_of_simulation = 8
stepRadius = (largeSphereRadius-smallSphereRadius)/(num_of_simulation + 1)
objectRadius = 6.0
t = np.zeros((num_of_simulation, 2))
t[0] = [15, 15]
t[1] = [20, 45]
t[2] = [50, 15]
t[3] = [45, 45]
t[4] = [30, 20]
t[5] = [15, 15]
t[6] = [20, 45]
t[7] = [50, 15]

r = [3, 5, 2, 4, 2, 3, 5, 2]

fig = plt.figure(figsize=(15, 5))
for i in range(num_of_simulation):
    simuFileName = os.path.join(result_dir, 'simu'+str(i)+'.nrrd')
    midSphereRadius = smallSphereRadius + stepRadius*(i+1)
    objectLocation = [t[i][0], t[i][1]]
    objectRadius = r[i]
    imArray = simuGen([smallSphereRadius, midSphereRadius, largeSphereRadius], 10.0,
                      simuFileName, objectLocation, objectRadius)
    fig.add_subplot(1, num_of_simulation, i+1)
    implot = plt.imshow(imArray, plt.cm.gray)
    plt.axis('off')
    plt.title('simu'+str(i))

# generate healthy bull's eye images
fig = plt.figure(figsize=(15, 5))
for i in range(num_of_simulation):
    simuFileName = os.path.join(result_dir, 'healthySimu'+str(i)+'.nrrd')
    midSphereRadius = smallSphereRadius + stepRadius*(i+1)
    imArray = simuGen([smallSphereRadius, midSphereRadius, largeSphereRadius], 10.0, simuFileName, [0, 0], 0)
    fig.add_subplot(1, num_of_simulation, i+1)
    implot = plt.imshow(imArray, plt.cm.gray)
    plt.axis('off')
    plt.title('healthy simu'+str(i))

midSphereRadius = smallSphereRadius + stepRadius*(1+2+3+4+5+6+7+8)/8
imArray = simuGen([smallSphereRadius, midSphereRadius, largeSphereRadius], 10.0,
                  os.path.join(result_dir, 'fMeanSimu.nrrd'), [0, 0], 0)
plt.figure()
plt.imshow(imArray, plt.cm.gray, vmin=0, vmax=90)
plt.title('healthy atlas (fMean)')
plt.axis('off')

