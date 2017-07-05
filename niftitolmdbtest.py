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
    zero_mean_ax0 = epi_img_data - np.mean(volume,axis=0,keepdims=True)
    norm_ax0 = zero_mean_ax0/np.std(volume,axis=0,keepdims=True)
    zero_mean_ax1 = epi_img_data - np.mean(volume,axis=1,keepdims=True)
    norm_ax1 = zero_mean_ax1/np.std(volume,axis=1,keepdims=True)
    zero_mean_ax2 = epi_img_data - np.mean(volume,axis=2,keepdims=True)
    norm_ax2 = zero_mean_ax2/np.std(volume,axis=2,keepdims=True)
    return norm_ax0,norm_ax1,norm_ax2
def aug_volume(volume):
    """
    Function to augment volume with 90, 180, and 270 degree rotations
    and a flip (left to right)

    """
    rot_90 = np.rot90(volume)
    rot_180 = np.rot90(rot_90)
    rot_270 = np.rot90(rot_180)
    flip = np.fliplr(volume)
    return rot_90,rot_180,rot_270,flip

# # If rotate doesn't work
# # rot_90 = scipy.ndimage.interpolation.rotate(norm_ax0,90)

norm_array_axis = norm_volume(epi_img_data)
print norm_array_axis

# def nifti_lmdb(volumes):
#     """
#     Function to convert data for 100 training examples from
#     nifti to lmdb
#
#     """
#     # We need to prepare the database for the size. We'll set it 10 times
#     # greater than what we theoretically need. There is little drawback to
#     # setting this too big. If you still run into problems after raising
#     # this, you might want to try saving fewer entries in a single
#     # transaction.
#     size = volumes.nbytes * 10
#     matlmdb = lmdb.open('mylmdb', map_size=map_size)
#     with matlmdb.begin(write=True) as txn:
#     # txn is a Transaction object
#         for i in range(N):
#             datum = caffe.proto.caffe_pb2.Datum()
#             datum.height = X.shape[1]
#             datum.width = X.shape[2]
#             datum.depth = X.shape[3]
#             datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
#             datum.label = int(y[i]) #???
#             str_id = '{:08}'.format(i) #???
