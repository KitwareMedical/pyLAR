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


"""test_batch.py
"""


import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import files


class filesTesting(unittest.TestCase):
    """Testcase for the files method.
    """

    def setUp(self):
        """Load data.
        """
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self._data = data_path


    def test_loadConfiguration(self):
        """Test load configuration
        """
        file_path = os.path.join(self._data, "config-file-loadConfiguration-test1.txt")
        res = files.loadConfiguration(file_path, "config")
        baseline = type('obj', (object,), {'value1': 1, 'value2': "value"})
        self.assertTrue(baseline.value1 == res.value1 and baseline.value2 == res.value2)




if __name__ == '__main__':
    unittest.main()


