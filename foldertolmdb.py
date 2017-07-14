import nibabel as nib
import numpy as np
import lmdb
import caffe
import os
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum

# Do preprocessing for MRI volume normalize and rotate
def norm_volume(volume):
    """ Function to normalize volume zero mean unit variance """
    zero_mean = volume - np.mean(volume,keepdims=True)
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

# Convert one volume to a full lmdb that has been rotated
def write_images_to_lmdb(file, db_name):
    """
    Function takes volumes created above and imports them into a single,
    labelled lmdb file.

    """
    for root, dirs, filename in os.walk(file, topdown = False):
        # MacOS puts a system file .DS_Store in all folders ignore it
        del filename[0]
        if root != file:
            continue
        map_size = int(1e10)
        env = lmdb.Environment(db_name, map_size=map_size)
        txn = env.begin(write=True,buffers=True)
        for idx, name in enumerate(filename):
            epi_img = nib.load(os.path.join(root, name))
            X = epi_img.get_data()
            # y is used to figure which folder file we are looking at.
            y = idx
            idx = idx * 5
            # Create all normalized and rotated volumes,
            # remember the constant.
            vol_90, vol_180, vol_270, flip = aug_volume(X)
            norm_x = norm_volume(X)
            norm_90 = norm_volume(vol_90)
            norm_180 = norm_volume(vol_180)
            norm_270 = norm_volume(vol_270)
            norm_flip = norm_volume(flip)
            # Put multiple augmented arrays in the lmdb at once
            datum1 = array_to_datum(norm_x,y)
            datum2 = array_to_datum(norm_90,y)
            datum3 = array_to_datum(norm_180,y)
            datum4 = array_to_datum(norm_270,y)
            datum5 = array_to_datum(norm_flip,y)
            str_id1 = '{:08}'.format(idx)
            str_id2 = '{:08}'.format(idx + 1)
            str_id3 = '{:08}'.format(idx + 2)
            str_id4 = '{:08}'.format(idx + 3)
            str_id5 = '{:08}'.format(idx + 4)
            txn.put(str_id1.encode('ascii'), datum1.SerializeToString())
            txn.put(str_id2.encode('ascii'), datum2.SerializeToString())
            txn.put(str_id3.encode('ascii'), datum3.SerializeToString())
            txn.put(str_id4.encode('ascii'), datum4.SerializeToString())
            txn.put(str_id5.encode('ascii'), datum5.SerializeToString())
    txn.commit()
    env.close()
    print 'Writing to', db_name, 'done'
write_images_to_lmdb('fractal_train_raw','fractal_train_lmdb')
