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

"""test_files.py
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
        baseline = type('obj', (object,), {'value1': 1, 'value2': "value", 'value3': True,
                                           'value4': False, 'value7': 2.5})
        setattr(baseline,'value5',[0,1])
        setattr(baseline,'value6',{'vdic1':'test'})
        self.assertTrue(baseline.value1 == res.value1 and baseline.value2 == res.value2
                        and baseline.value3 == res.value3 and baseline.value4 == res.value4
                        and baseline.value5 == res.value5 and baseline.value6 == res.value6
                        and baseline.value7 == res.value7)

    def test_saveConfiguration(self):
        """Test save configuration
        """
        baseline = ["value1 = 1", "value2 = \'value\'"]
        data_folder = tempfile.mkdtemp()
        data = type('obj', (object,), {'value1': 1, 'value2': "value"})
        output_file = os.path.join(data_folder, "config_file-test_saveConfiguration.txt")
        print "Output File: " + output_file
        files.saveConfiguration(output_file, data)
        with open(output_file, 'r') as f:
            for i, val in enumerate(f.readlines()):
                print "Baseline: " + baseline[i]
                print "Value: " + val.rstrip('\r\n')
                self.assertTrue(val.rstrip('\r\n') == baseline[i])

    def test_readTxtIntoList(self):
        """Test readTxtIntoList
        """
        baseline = ["file1", "file2"]
        file_path = os.path.join(self._data, "readTxtIntoList-file-test1.txt")
        res = files.readTxtIntoList(file_path)
        for i, var in enumerate(res):
            self.assertTrue(var == baseline[i])


    def test_writeTxtFromList(self):
        """Test writeTxtFromList
        """
        baseline=["value1", "value2"]
        data_folder = tempfile.mkdtemp()
        output_file = os.path.join(data_folder, "list-file-test_writeTxtFromList.txt")
        print "Output File: " + output_file
        files.writeTxtFromList(output_file, baseline)
        with open(output_file,'r') as f:
            for i,val in enumerate(f.readlines()):
                print "Baseline: " + baseline[i]
                print "Value: " + val.strip('\r\n')
                self.assertTrue(val.strip('\r\n') == baseline[i])

    def test_containsRequirements(self):
        config = type('obj', (object,), {'value1': 1, 'value2': "value"})
        requirements = ['value1', 'value2']
        files.containsRequirements(config, requirements)
        requirements.append('value3')
        with self.assertRaisesRegexp(Exception, 'Requires.*'):
            self.assertFalse(files.containsRequirements(config, requirements))


if __name__ == '__main__':
    unittest.main()


