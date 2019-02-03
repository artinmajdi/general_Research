import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
sys.path.append('/array/ssd/msmajdi/code/general')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import params, smallFuncs
params.preprocess.Mode = False
params.directories.Train.address
subF = smallFuncs.listSubFolders(params.directories.Train.address)

import numpy as np
a = np.zeros((10,20))
