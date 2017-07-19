import numpy as np
import IPython
import matplotlib.pyplot as plt
import sys
import caffe
# display plots in this notebook

# What is in this folder???
# sys.path.insert(0, 'python')

# Are the required files in the folder?
import os
if os.path.isfile('models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'

# Use only CPUs in this specific instance
caffe.set_mode_cpu()

# Instantiate the Solver
solver = caffe.SGDSolver('mnist/lenet_auto_solver.prototxt')

# each output is (batch size, feature dim, spatial dim)
print [(k, v.data.shape) for k, v in solver.net.blobs.items()]

# just print the weight sizes (we'll omit the biases)
print [(k, v[0].data.shape) for k, v in solver.net.params.items()]
