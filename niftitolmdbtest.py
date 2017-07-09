import nibabel as nib
import pylab
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from transforms3d.euler import euler2mat, mat2euler
import scipy.ndimage
import lmdb
import caffe

epi_img = nib.load('training_axial_crop_pat0.nii')
epi_img_data = epi_img.get_data()
print epi_img_data.shape
# Do preprocessing for MRI volume normalize and rotate
def norm_volume(volume):
    """ Function to normalize volume zero mean unit variance """ # how are these normalized on a single volume
    zero_mean = epi_img_data - np.mean(volume,keepdims=True)
    norm = zero_mean/np.std(volume,keepdims=True)
    return norm
def aug_volume(volume):
    """
    Function to augment volume with 90, 180, and 270 degree rotations
    and a flip (left to right)

    """
    rot_90 = np.rot90(volume)
    rot_180 = np.rot90(rot_90)
    rot_270 = np.rot90(rot_180)
    flip = np.fliplr(volume)
    return rot_90, rot_180, rot_270, flip

# If rotate doesn't work
# rot_90 = scipy.ndimage.interpolation.rotate(norm_ax0,90)

# Create all normalized and rotated volumes, remember the constant.
norm = norm_volume(epi_img_data)
norm_90, norm_180, norm_270, flip = aug_volume(norm)

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
    vollmdb = lmdb.open('mrilmdbtest', map_size=size)
    with vollmdb.begin(write=True) as txn:
    # txn is a Transaction object
        datum = caffe.proto.caffe_pb2.Datum()
        datum.height = volumes.shape[0]
        datum.width = volumes.shape[1]
        datum.data = volumes.tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(1)
        # str_id = '{:08}'.format(i) #???

# Convert one LMDB to a full on volume that has been rotated
nifti_lmdb(flip)

lmdb_env = lmdb.open('mrilmdbtest')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    for l, d in zip(label, data):
            print l, d
print label
print data
