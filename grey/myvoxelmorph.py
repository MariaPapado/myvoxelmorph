import numpy as np
import neurite as ne
import voxelmorph as vxm
#import sys
#sys.path.append('/home/mariapap/ENVS/env1/lib/python3.10/site-packages/keras/_tf_keras/keras/losses/')
#from tensorflow.keras.datasets import mnist
from tools import *
import imageio
import cv2
import os
from tqdm import tqdm
from PIL import Image



train_data_1 = np.load('train_data_1.npy')
train_data_2 = np.load('train_data_2.npy')


# fix data
train_data_1 = train_data_1.astype('float')/255.
train_data_2 = train_data_2.astype('float')/255.

# verify
print('training maximum value', train_data_1.max(), train_data_2.max())

# verify
print('shape of training data', train_data_1.shape)

# configure unet features 
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]


inshape = train_data_1.shape[1:]
vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
#print(vxm_model.summary())

#<KerasTensor shape=(None, 256, 256, 1), dtype=float32, sparse=None, name=vxm_dense_target_input>
#print('type', type(vxm_model.inputs))
#print('bbbbbbbbbbbbbbbbbb')
#vxm_model.inputs[0] = tf.keras.KerasTensor(shape=(None, 256, 256, 3), name='vxm_dense_target_input', sparse=None, record_history=True)
#vxm_model.inputs[1] = tf.keras.KerasTensor(shape=(None, 256, 256, 3), name='vxm_dense_target_input', sparse=None, record_history=True)
#vxm_model.inputs[1] = tf.keras.layers.InputLayer(shape=(None, 256, 256, 3), trainable=True)
#print('shpw', vxm_model.inputs[0])


####################################################################################

#print(vxm_model.layers[4])

#print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
#print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)
#print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', x_train.shape)  #(4421, 32, 32) numpy array
train_generator = vxm_data_generator(train_data_1, train_data_2)

#hist = vxm_model.fit(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2);



# let's test it
in_sample, out_sample = next(train_generator)
#print('iiiiii', out_sample[0].shape)
# visualize
images = [img[0, :, :, 0] for img in in_sample + out_sample]

#print('unique', np.unique(images[0]))
#cv2.imwrite('im0.png', images[0]*255)
#cv2.imwrite('im1.png', images[1]*255)
#cv2.imwrite('im2.png', images[2]*255)
#cv2.imwrite('im3.png', images[3]*255)

#print('aaaaaaaaaa', len(images))
#titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
#print('len', len(images))


########################################################################################################################################################
checkpoint_filepath = './saved_weights/checkpoint.weights.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max')
#    save_best_only=True)



nb_epochs = 10
steps_per_epoch = 500
hist = vxm_model.fit(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2, callbacks=[model_checkpoint_callback])


'''
# let's get some data
val_generator = vxm_data_generator(x_val, batch_size = 1)
val_input, _ = next(val_generator)


val_pred = vxm_model.predict(val_input)


# visualize
images = [img[0, :, :, 0] for img in list(val_input) + list(val_pred)]
print('unique', np.unique(images[0]))
cv2.imwrite('im0.png', images[0]*255)
cv2.imwrite('im1.png', images[1]*255)
cv2.imwrite('im2.png', images[2]*255)
cv2.imwrite('im3.png', images[3]*255)

'''
