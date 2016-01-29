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

from ..alm import ialm
import gc
import time
import numpy as np


def rpca(Y, lamda):
    """Run Robust PCA via ialm implementation in alm."""
    t_begin = time.clock()

    gamma = lamda * np.sqrt(float(Y.shape[1]) / Y.shape[0])
    low_rank, sparse, n_iter, rank, sparsity, sum_sparse = ialm.recover(Y, gamma)
    gc.collect()

    t_end = time.clock()
    t_elapsed = t_end - t_begin
    print 'RPCA takes:%f seconds' % t_elapsed

    return low_rank, sparse, n_iter, rank, sparsity, sum_sparse
