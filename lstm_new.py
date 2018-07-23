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
#print(data_path_dir)
"""for folder in data_path_dir:
    print(folder)
    for files in os.listdir(os.path.join('/home','manoj','Desktop','footstep_recognition','data1',folder)):
        print(files)"""
"""def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=None, nb_epoch=None): # data_type = features, seq_len = 160, model = lstm, class_limit = 6
# image shape = None, batch_size = 1, nb_epoch = 2
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('/home','manoj','Desktop','footstep_recognition','data1', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('/home','manoj','Desktop','footstep_recognition','data1', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('/home','manoj','Desktop','footstep_recognition','data1', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type)
        val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)
    print("LOAD MODEL")
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=4)

def main():
    #These are the main training settings. Set each before running
    this file.
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'lstm'
    saved_model = None  # None or weights file
    class_limit =  6 # int, can be 1-101 or None
    seq_length = 60
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 32
    nb_epoch = 10
# Chose images or features and image shape based on network.
    data_type = 'features'
    image_shape = None
    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)
if __name__ == '__main__':
    main()"""

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