import tensorflow as tf
import numpy as np
import neurite as ne
import voxelmorph as vxm


def vxm_data_generator(x_data, y_data, batch_size=4):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
#    vol_shape = x_data.shape[1:-1] # RGB
    vol_shape = x_data.shape[1:] # extract data shape

    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        fixed_images = y_data[idx1, ..., np.newaxis]

#        moving_images = x_data[idx1] #RGB
#        fixed_images = y_data[idx1] #RGB


        #idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        #fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = (moving_images, fixed_images)

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = (fixed_images, zero_phi)
#        print('outtttt', outputs[0].shape, outputs[1].shape)
#        print('typeeeeeeeeeeeeeeeeeeeee', inputs[1].shape)
#        print('tttttttttttttt', len(outputs))

        yield (inputs, outputs)
