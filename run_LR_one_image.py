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

""" Load a given input image, and then seperate the low
rank from the sparse (corrupted) part using Candes et al., 2011
formulation of RPCA, solved via IALM.

CAUTION: only grayscale images are supported at that point.
"""


import sys
import numpy as np
import SimpleITK as sitk
import pyLAR.alm.ialm as ialm
import argparse


def main(argv=None):
    """Parsing command line arguments and reading input files."""
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(
            prog=argv[0],
            description=__doc__
    )
    parser.add_argument('-i', "--inputImage", required=True,
                        help="Input image on which low-rank decomposition is computed")
    parser.add_argument('-l', "--lowRank", required=True, help="Low-rank output image")
    parser.add_argument('-s', "--Sparse", required=True,
                        help="Sparse output image")
    args = parser.parse_args(argv[1:])

    # read image
    I = sitk.ReadImage(args.inputImage)
    # data for processing
    X = sitk.GetArrayFromImage(I)

    # decompose X into L+S
    L, S, _, _, _, _ = ialm.recover(X)

    L_image = sitk.GetImageFromArray(np.asarray(L, dtype=np.uint8))
    sitk.WriteImage(L_image, args.lowRank, True)
    S_image = sitk.GetImageFromArray(np.asarray(S, dtype=np.uint8))
    sitk.WriteImage(S_image, args.Sparse, True)


if __name__ == '__main__':
    main()
