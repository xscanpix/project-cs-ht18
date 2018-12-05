#!/usr/bin/python3
import sys, os, shutil
import numpy as np

import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation
from keras import optimizers

from mymov.helpers import load_settings


def gen_model(tf_model_path, test):
    K.set_learning_phase(0)
    sess = K.get_session()
    newmodel = Sequential()
    newmodel.add(Dense(test['neurons'], input_shape=(28, ), activation='relu', kernel_initializer='lecun_uniform', name='input'))
    for i in range(test['layers']):
        newmodel.add(Dense(test['neurons'], activation='relu', kernel_initializer='lecun_uniform', name='dense_{}'.format(i+1)))
    newmodel.add(Dense(5, kernel_initializer='lecun_uniform', name='dense_{}'.format(test['layers'] + 1)))
    newmodel.add(Activation('linear', name='output'))
    newmodel.compile(loss='mse', optimizer=optimizers.RMSprop(lr=0.2,rho=0.9,epsilon=1e-06))

    newmodel.summary()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, tf_model_path + "_{}_{}".format(test['layers'], test['neurons']))

    return newmodel


def compile_tf(jsonData, test):
    exists = os.path.exists(jsonData['ncsdkGraphPath'])

    os.system("mvNCCompile {}_{}_{}.meta -s {} -in {} -on {} -o {}_{}_{}_{}".format(jsonData['tfOutputPath'], test['layers'], test['neurons'], test['shaves'], jsonData['inputLayerName'], jsonData['outputLayerName'], jsonData['ncsdkGraphPath'], test['layers'], test['neurons'], test['shaves']))
