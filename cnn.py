# runs a CNN to generate a model to predict the modality of the radiological images
# written to run on AWS
# Please note: Default name of data file is image_dict.pickle, modality_dict.pickle

import os
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
# use keras a wrapper on tensorflow
from keras.models import Sequential
from keras.layers import Lambda, Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy

# set tensorflow as keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['image_data_format'] = 'channels_last'

# define flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir','/home/ec2-user/data', "location of data folder")
flags.DEFINE_string('save_dir','/home/ec2-user/save', "location to save model")
flags.DEFINE_integer('batch_size', 128, "size of each batch")
flags.DEFINE_integer('epochs', 50, "number of epochs to run the training")
flags.DEFINE_float('learn_rate', 0.0001, "learning rate to use for training")

if __name__ == '__main__':

    os.chdir(FLAGS.data_dir)

    # read in data (assumes binary files):
    with open('./image_dict.pickle','rb') as f: X = list(pickle.load(f).values())
    with open('./modality_dict.pickle','rb') as f: y = list(pickle.load(f).values())

    assert len(X) == len(y), "X,y size mismatch. Check data."

    # prepare data
    # train, valid, test splits
    num_classes = len(set(y))
    y = pd.Series(y)
    # TODO: Find a more efficient way to onehot encode
    y = np.array(pd.get_dummies(y)) # onehot encoding y
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, stratify=y,
                                        test_size=0.1)
    X_train, X_valid, y_train, y_valid = train_test_split(
                                    X_train, y_train,
                                    stratify=y_train, test_size=0.15)


    X_train, X_test, X_valid = np.float32(X_train), np.float32(X_test), np.float32(X_valid)
    y_train, y_test, y_valid = np.array(y_train), np.array(y_test), np.array(y_valid)

    # convert to tensors
    X_train = X_train.reshape(list(X_train.shape)+[1])
    X_test = X_test.reshape(list(X_test.shape)+[1])
    X_valid = X_valid.reshape(list(X_valid.shape)+[1])

    #### create network graph ####
    model = Sequential()
    row, col, ch = 200, 300, 1

    # TODO: add some preprocessing layers
    # convolutional layers:
    # conv1 (need to specify input since it is first layer)
    # NOTE: that we used same padding to have output of same size
    model.add(Conv2D(filters=24, kernel_size=(5,5),
                     input_shape=(row,col, ch),
                     strides=(2,2), padding='same'))
    model.add(Activation('relu'))
    # conv2
    model.add(Conv2D(filters=36, kernel_size=(5,5),
                     strides=(2,2), padding='same'))
    model.add(Activation('relu'))
    # conv3
    model.add(Conv2D(filters=48, kernel_size=(5,5),
                    strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    # maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1),
                           padding='same'))
    model.add(Activation('relu'))
    # conv4
    model.add(Conv2D(filters=36, kernel_size=(5,5),
                    strides=(2,2), padding='valid'))
    model.add(Activation('relu'))

    # fully connected layers
    model.add(Flatten())
    # fully connected 1
    model.add(Dense(500))
    # Dropout
    model.add(Dropout(0.1)) # prob of dropout
    # fully connected 2
    model.add(Dense(100))

    # output layer (using softmax + cross entropy)
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    #### network graph is built ####

    # create and run model
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metric=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              verbose=True,
              validation_data=(X_valid, y_valid))

    # print performance on test data
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=False)
    predictions = model.predict(X_test)
    predictions_sum = np.sum(X_test, 1) # to see if it is always predicting one type
    print('Accuracy on test data: ', test_acc)
    print('Cross Entropy Loss on test data: ', test_loss)
    print(predictions_sum)

    # save model for future use
    model.save(FLAGS.save_dir + '/model.h5')
