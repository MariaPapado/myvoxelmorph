import numpy as np
import neurite as ne
import voxelmorph as vxm
#import sys
#sys.path.append('/home/mariapap/ENVS/env1/lib/python3.10/site-packages/keras/_tf_keras/keras/losses/')
from tools import *
import imageio
import cv2
import os

val_data_1 = np.load('val_data_1.npy')
val_data_2 = np.load('val_data_2.npy')

# fix data
val_data_1 = val_data_1.astype('float')/255
val_data_2 = val_data_2.astype('float')/255

# verify
print('training maximum value', val_data_1.max())


# configure unet input shape (concatenation of moving and fixed images)
ndim = 2
unet_input_features = 2
inshape = (*val_data_1.shape[1:], unet_input_features)

# configure unet features
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]




# build model using VxmDense
inshape = val_data_1.shape[1:]
#vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0, src_feats=3,trg_feats=3)
vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0,  src_feats=3,trg_feats=3)



vxm_model.load_weights('./saved_weights/checkpoint.weights.h5')

#let's get some data
#val_generator = vxm_data_generator(val_data_1, val_data_2, batch_size = 1)
#val_input, _ = next(val_generator)

id= '12_25297_BD_0_3.png'
id = '12_26449_BD_0_1.png'
id = '12_25297_BD_0_3.png'

im1 = imageio.imread('/home/mariapap/CODE/voxelmorph/hir/hir_data/val/T1_rgb/{}'.format(id))
im2 = imageio.imread('/home/mariapap/CODE/voxelmorph/hir/hir_data/val/T2_rgb/{}'.format(id))
print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAim111111111', im1.shape)
#im1 = np.expand_dims(im1, 0)/255. #RGB
#im2 = np.expand_dims(im2, 0)/255. #RGB

#im1 = np.expand_dims(im1, 0)/255. #
#im2 = np.expand_dims(im2, 0)/255. #

im1 = np.expand_dims(im1, 0)/255. #
im2 = np.expand_dims(im2, 0)/255. #



print('22222', im1.shape)
val_input = (im1,im2)

#print('aaaa', val_input[0].shape, val_input[1].shape)

val_pred = vxm_model.predict(val_input)


images = [img[0, :, :, 0] for img in list(val_input) + list(val_pred)]
#print('unique', np.unique(images[0]))
cv2.imwrite('./RESULTS/{}'.format(id[:-4] + '_0.png'), images[0]*255)
cv2.imwrite('./RESULTS/{}'.format(id[:-4] + '_1.png'), images[1]*255)
cv2.imwrite('./RESULTS/{}'.format(id[:-4] + '_2.png'), images[2]*255)
cv2.imwrite('./RESULTS/{}'.format(id[:-4] + '_f.png'), images[3]*255)
titles = ['im1', 'im2', 'im1_moved', 'pixel_flow']


ne.plot.flow([val_pred[1].squeeze()], width=5);
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);
