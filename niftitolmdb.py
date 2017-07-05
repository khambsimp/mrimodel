import numpy as np
import lmdb
import caffe

def nifti_lmdb(volumes):
    """
    Function to convert data for 100 training examples from
    nifti to lmdb

    """
    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problems after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    size = volumes.nbytes * 10
    matlmdb = lmdb.open('mylmdb', map_size=map_size)
    with matlmdb.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.height = X.shape[1]
        datum.width = X.shape[2]
        datum.depth = X.shape[3]
        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(y[i]) #???
        str_id = '{:08}'.format(i) #???
