#!/usr/bin/python3
import sys, os, shutil

import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation
from keras import optimizers, models

from code.helpers import load_settings

VALID_H5   = ['.h5']


def gen_new_model(model, tf_model_path):
    '''
    Define a new model from model description.
    Takes in the keras model with weights and adds them
    to the new model.
    '''
    newmodel = Sequential()
    newmodel.add(Dense(64, input_shape=(28, ), activation='relu', kernel_initializer='lecun_uniform', name='input'))
    newmodel.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform', name='dense_2'))
    newmodel.add(Dense(5, kernel_initializer='lecun_uniform', name='dense_3'))
    newmodel.add(Activation('linear', name='output'))
    newmodel.compile(loss='mse', optimizer=optimizers.RMSprop(lr=0.2,rho=0.9,epsilon=1e-06))

    newmodel.layers[0].set_weights(model.layers[0].get_weights())
    newmodel.layers[1].set_weights(model.layers[1].get_weights())
    newmodel.layers[2].set_weights(model.layers[3].get_weights())
    newmodel.layers[3].set_weights(model.layers[4].get_weights())

    save_tf_model(tf_model_path)

    return newmodel


def save_tf_model(outputpath):
    """Saves the tf model, must be run after gen_new_model"""
    K.set_learning_phase(0)
    saver = tf.train.Saver()
    sess = K.get_session()
    saver.save(sess, outputpath)


def load_model_keras(filepath):
    model = models.load_model(filepath)

    return model


def keras_to_tf(jsonData, force):
    exists = os.path.exists(jsonData['tfOutputPath'] + ".meta")

    if(exists and force):
        shutil.rmtree(jsonData['outputDir'])

    model = load_model_keras(jsonData['kerasModelPath'])
    newmodel = gen_new_model(model, jsonData['tfOutputPath'])

    return newmodel


def compile_tf(jsonData, force):
    exists = os.path.exists(jsonData['ncsdkGraphPath'])
    
    if(exists and force):
        os.remove(ncsdkGraphPath)

    os.system("mvNCCompile {0}.meta -in {1} -on {2} -o {3}".format(jsonData['tfOutputPath'], jsonData['inputLayerName'], jsonData['outputLayerName'], jsonData['ncsdkGraphPath']))
