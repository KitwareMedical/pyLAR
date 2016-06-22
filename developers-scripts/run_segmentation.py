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

#!/usr/bin/env python

import sys
import imp
import os
import subprocess


if len(sys.argv) < 5:
  print "Usage: " + sys.argv[0] + " scriptToRun configFN imageToSegment outputFolder [additionalArgs]"
  sys.exit()

# Script to run
scriptToRun = sys.argv[1]
# assign global parameters from the input config txt file
configFN = sys.argv[2]
f = open(configFN)
config  = imp.load_source('config', '', f)
f.close()

data_dir          = config.data_dir
result_dir        = config.result_dir
fileListFN        = config.fileListFN
selection         = config.selection

# Image to segment
imageToSegment = sys.argv[3]

#Output folder
outputFolder = sys.argv[4]
if os.path.isabs(outputFolder):
  outputDir = outputFolder
else:
  outputDir = os.path.join(result_dir,outputFolder)

# Create output directory
segmentationConfigDir = os.path.join(outputDir,"Segmentation")
if not os.path.exists(segmentationConfigDir):
    os.makedirs(segmentationConfigDir)

selection.append(len(selection))

# Creates new filelistFN with additional file
fileListName = os.path.join(data_dir,fileListFN)
flist = []
with open(fileListName) as f:
  flist = f.read().splitlines()
flist.insert(0,imageToSegment)
baseFileListFN = os.path.splitext(fileListFN)[0]+"_seg.txt"
newFileListFN = os.path.join(segmentationConfigDir,baseFileListFN)
outputFileListFN = open(newFileListFN,"w")
for item in flist:
  outputFileListFN.write("%s\n" % item)

# Copies initial configuration file and replaces certain values (data_dir, fileListFN, selection)
outputConfigFN = os.path.join(segmentationConfigDir,os.path.splitext(os.path.basename(configFN))[0]+"_seg.txt")
print outputConfigFN
with open(configFN) as old_f:
  with open(outputConfigFN,"w") as new_f:
    for line in old_f:
      if "selection =" in line:
        new_f.write("selection = %s\n" % selection)
      elif "data_dir =" in line:
        new_f.write("data_dir = \'%s\'\n" % segmentationConfigDir)
      elif "fileListFN =" in line:
        new_f.write("fileListFN = \'%s\'\n" % baseFileListFN )
      elif "result_dir =" in line:
        new_f.write("result_dir = \'%s\'\n" % outputDir )
      else:
        new_f.write(line)

# Runs process with new configuration file
cmd = scriptToRun + " " + outputConfigFN + " " + " ".join(sys.argv[5:])
print "Segmentation: "+cmd
process = subprocess.Popen(cmd, shell = True)
process.wait()

