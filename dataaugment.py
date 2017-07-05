import numpy as np
from sklearn import preprocessing
# import scipy.ndimage

# Do preprocessing for MRI volume normalize and rotate volume
def norm_volume(volume):
    """ Function to normalize volume zero mean unit variance """
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

# If rotate doesn't work
# rot_90 = scipy.ndimage.interpolation.rotate(norm_ax0,90)
