#!/usr/bin/env python

#  Library: pyLAR
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


"""test_ealm.py
"""


import os
import sys
import unittest
import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'..'))
import ealm


class EALMTesting(unittest.TestCase):
    """Testcase for the EALM method.
    """

    def setUp(self):
        """Load data.
        """
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "im_outliers.dat")
        self._data = np.genfromtxt(file_path)


    def test_recover(self):
        """Test recovery from outliers.
        """
        # run recovery
        lr, sp, _ = ealm.recover(self._data, None)
        # load baseline (no outliers)
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "im_baseline.dat")
        baseline = np.genfromtxt(file_path)
        # Frobenius norm between recovered mat. and baseline
        d = np.linalg.norm(np.round(lr)-baseline, ord='fro')
        self.assertTrue(np.allclose(d,0.0))


if __name__ == '__main__':
    unittest.main()
