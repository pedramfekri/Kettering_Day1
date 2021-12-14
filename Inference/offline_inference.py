import tensorflow as tf
from Model import DL_model
from PIL import Image
import numpy as np
import pathlib

data_dir_mask = pathlib.Path('../Data/SmallDataset/Mask-resized/')

Mask = list(data_dir_mask.glob('*.jpg'))

model = DL_model.face_model(200, 200, 3)
model.load_weights('../Train/SavedModel/Weights/2.hdf5')

print('the model was loaded properly')

image = Image.open(Mask[0])
image = image.resize((200, 200))
image = np.asarray(image)
image = image.astype('float32')
# image /=255.0
# (h, w, c) --> (b, h, w, c)
image = image[np.newaxis, ...]

clas = ['Mask', 'NoMask', 'NotPers']
pred = model.predict(image)

score = tf.nn.softmax(pred)
cls = np.argmax(pred)

print(pred, score, clas[cls])