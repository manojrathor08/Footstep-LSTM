# Footstep recognition using LSTM
from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
from traindata_new import max_size1
from validation import max_size2
from traindata_new import trainingdata_Xy
from validation import validationdata_Xy
from testing import testingdata_Xy
import numpy as np
from keras.optimizers import Adam, RMSprop
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time
import os.path
from matplotlib import pyplot
data_path = '/home/manoj/Desktop/footstep_recognition/data1'
data_path_dir = os.listdir(data_path)
np.random.seed(7)
batch_size = 32
hidden_size = 50
use_dropout=True
model = Sequential()
vocabulary = 6
data_type = 'features'
seq_length = 60
class_limit =  6
image_shape = None
data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )
# generator = data.frame_generator(batch_size, 'train', data_type)
# # for f in generator:
# #     print(f)
# val_generator = data.frame_generator(batch_size, 'test', data_type)
X, y = data.get_all_sequences_in_memory('train', data_type)
print(X.shape)
print(y.shape)
X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
print(X_test.shape)
print(y_test.shape)
model.add(LSTM(hidden_size, input_shape= (60,18), return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
#model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
      model.add(Dropout(0.5))
if use_dropout:
      model.add(Dropout(0.5))

#model.add(Dense(4))
#model.add(TimeDistributed(Dense(vocabulary)))
model.add(Flatten())
model.add(Dense(6))
model.add(Activation('softmax'))
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# #checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
#num_epochs = 2
# model.fit_generator(generator, 4, num_epochs,
#                     validation_data=val_generator,
#                     validation_steps=1)
nb_epoch =50
history = model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            epochs=nb_epoch)
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model train vs validation acc')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
#model.save("/home/manoj/Desktop/assignment3/toy/lstm.h5")
#model1 = load_model("/home/manoj/Desktop/assignment3/toy/lstm.h5")
#print(np.argmax(model1.predict(testing),axis = 0))