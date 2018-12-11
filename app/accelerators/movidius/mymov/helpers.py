#!/usr/bin/python3
import os
import json

import tensorflow as tf
import keras.backend as K

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers.core import Activation
from keras import optimizers

def load_settings(filepath):

    basepath = os.environ['PROJ_DIR']

    with open(filepath) as file:
        try:
            json_data = json.load(file)
        except:
            print("Cannot load settings file: {}\nExiting...".format(basepath + "/" + filepath))
            exit()

    kerasModelPath = json_data['kerasModelPath']
    tfOutputPath = json_data['tfOutputPath']
    ncsdkGraphPath = json_data['ncsdkGraphPath']
    outputDir = json_data['outputDir']

    json_data['outputDir'] = basepath + "/" + outputDir
    json_data['kerasModelPath'] = basepath + "/" + kerasModelPath
    json_data['tfOutputPath'] = basepath + "/" + tfOutputPath
    json_data['ncsdkGraphPath'] = basepath + "/" + ncsdkGraphPath

    return json_data

def prepare_keras_model(jsonData):
    K.set_learning_phase(0)
    sess = K.get_session()
    newmodel = Sequential()
    newmodel.add(Dense(64, input_shape=(28, ), activation='relu', kernel_initializer='lecun_uniform', name='input'))
    newmodel.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform', name='dense_1'))
    newmodel.add(Dense(5, kernel_initializer='lecun_uniform', name='dense_2'))
    newmodel.add(Activation('linear', name='output'))
    newmodel.compile(loss='mse', optimizer=optimizers.RMSprop(lr=0.2,rho=0.9,epsilon=1e-06))

    loaded_model = load_model(jsonData['kerasModelPath'])

    newmodel.layers[0].set_weights(loaded_model.layers[0].get_weights())
    newmodel.layers[1].set_weights(loaded_model.layers[1].get_weights())
    newmodel.layers[2].set_weights(loaded_model.layers[3].get_weights())


    saver = tf.train.Saver()
    saver.save(sess, jsonData['tfOutputPath'])

    return newmodel

def gen_model(jsonData):
    K.set_learning_phase(0)
    sess = K.get_session()
    newmodel = Sequential()
    newmodel.add(Dense(64, input_shape=(28, ), activation='relu', kernel_initializer='lecun_uniform', name='input'))
    newmodel.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform', name='dense_1'))
    newmodel.add(Dense(5, kernel_initializer='lecun_uniform', name='dense_2'))
    newmodel.add(Activation('linear', name='output'))
    newmodel.compile(loss='mse', optimizer=optimizers.RMSprop(lr=0.2,rho=0.9,epsilon=1e-06))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, jsonData['tfOutputPath'])

    return newmodel

def compile_tf(jsonData):
    if(os.path.exists(jsonData['ncsdkGraphPath'])):
        return
    
    os.system("mvNCCompile {}.meta -in {} -on {} -o {}".format(jsonData['tfOutputPath'], jsonData['inputLayerName'], jsonData['outputLayerName'], jsonData['ncsdkGraphPath']))
