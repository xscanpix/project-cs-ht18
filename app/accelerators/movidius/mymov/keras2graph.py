#!/usr/bin/python3
import os

import tensorflow as tf
import keras.backend as K

from helpers import load_settings

# For generating model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation
from keras import optimizers


# Generate a model to a tensor file model format.
def gen_model(tf_model_path):
    K.set_learning_phase(0)
    sess = K.get_session()
    newmodel = Sequential()
    newmodel.add(Dense(64, input_shape=(28, ), activation='relu', kernel_initializer='lecun_uniform', name='input'))
    newmodel.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform', name='dense_1'))
    newmodel.add(Dense(5, kernel_initializer='lecun_uniform', name='dense_2'))
    newmodel.add(Activation('linear', name='output'))
    newmodel.compile(loss='mse', optimizer=optimizers.RMSprop(lr=0.2,rho=0.9,epsilon=1e-06))

    newmodel.summary()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, tf_model_path)

    return newmodel

# Compile the model files to movidius graph format
def compile_tf(jsonData):
    exists = os.path.exists(jsonData['ncsdkGraphPath'])

    os.system("mvNCCompile {}.meta -s {} -in {} -on {} -o {}".format(jsonData['tfOutputPath'], jsonData['shaves'], jsonData['inputLayerName'], jsonData['outputLayerName'], jsonData['ncsdkGraphPath']))
