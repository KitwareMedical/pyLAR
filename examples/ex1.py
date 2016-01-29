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

"""ex1.py

Load a checkerboard image, specify the percentage of outliers
(as the fraction of corrupted pixel) and then seperate the low
rank from the sparse (corrupted) part using Candes et al., 2011
formulation of RPCA, solved via IALM.

CAUTION: only grayscale images are supported at that point.
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


import os
import sys
import random
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    print "SimpleITK is required to run this example!"
    sys.exit(-1)

# make sure ialm is found and import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import pyLAR.alm.ialm as ialm


def main(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) != 5:
        print "Usage: python %s <CheckerboardImage> <OutlierFraction> <CorruptedImage> <LowRankImage>" % sys.argv[0]
        sys.exit(1)

    # outlier fraction
    p = float(sys.argv[2])

    # read image
    I = sitk.ReadImage(argv[1])
    # data for processing
    X = sitk.GetArrayFromImage(I)
    # number of pixel
    N = np.prod(X.shape)

    eps = np.round(np.random.uniform(-10, 10, 100))
    idx = np.random.random_integers(0, N-1, np.round(N*p))
    X.ravel()[idx] = np.array(200+eps, dtype=np.uint8)

    # write outlier image
    J = sitk.GetImageFromArray(X)
    sitk.WriteImage(J, sys.argv[3], True)

    # decompose X into L+S
    L, S, _, _, _, _ = ialm.recover(X)

    C = sitk.GetImageFromArray(np.asarray(L, dtype=np.uint8))
    sitk.WriteImage(C, sys.argv[4], True)

    # compute mean-square error and Frobenius norm
    print "MSE: %.4g" % np.sqrt(np.asmatrix((L-sitk.GetArrayFromImage(I))**2).sum())
    print "Frobenius-Norm: %.4g" % np.linalg.norm(L-sitk.GetArrayFromImage(I),ord='fro')


if __name__ == '__main__':
    main()
