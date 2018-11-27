#!/usr/bin/python3
import sys, os, shutil
import numpy as np

import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation
from keras import optimizers, models

from app.helpers import load_settings



VALID_H5   = ['.h5']


def gen_model(tf_model_path, numHidden=1, numNodes=64):
    K.set_learning_phase(0)

    sess = K.get_session()

    newmodel = Sequential()
    newmodel.add(Dense(numNodes, input_shape=(28, ), activation='relu', kernel_initializer='lecun_uniform', name='input'))
    for i in range(numHidden):
        newmodel.add(Dense(numNodes, activation='relu', kernel_initializer='lecun_uniform', name='dense_{}'.format(i+1)))
    newmodel.add(Dense(5, kernel_initializer='lecun_uniform', name='dense_{}'.format(numHidden + 1)))
    newmodel.add(Activation('linear', name='output'))
    newmodel.compile(loss='mse', optimizer=optimizers.RMSprop(lr=0.2,rho=0.9,epsilon=1e-06))

    newmodel.summary()

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.save(sess, tf_model_path + "_{}_{}".format(numHidden, numNodes))

    return newmodel


def load_model_keras(filepath):
    model = models.load_model(filepath)

    return model


def keras_to_tf(jsonData, force, numHidden=1, numNodes=64):
    exists = os.path.exists(jsonData['tfOutputPath'] + "_{}_{}.meta".format(numHidden, numNodes))

    if(exists and force):
        shutil.rmtree(jsonData['outputDir'])

    #model = load_model_keras(jsonData['kerasModelPath'])
    #newmodel = gen_new_model(model, jsonData['tfOutputPath'], jsonData['extraLayers'], jsonData['nodesHidden'])
    newmodel = gen_model(jsonData['tfOutputPath'], numHidden, numNodes)

    return newmodel


def compile_tf(jsonData, force, numHidden=1, numNodes=64):
    exists = os.path.exists(jsonData['ncsdkGraphPath'])
    
    if(exists and force):
        os.remove(ncsdkGraphPath)

    print("mvNCCompile {}_{}_{}.meta -in {} -on {} -o {}_{}_{}".format(jsonData['tfOutputPath'], numHidden, numNodes, jsonData['inputLayerName'], jsonData['outputLayerName'], jsonData['ncsdkGraphPath'], numHidden, numNodes))
    os.system("mvNCCompile {}_{}_{}.meta -in {} -on {} -o {}_{}_{} -ec".format(jsonData['tfOutputPath'], numHidden, numNodes, jsonData['inputLayerName'], jsonData['outputLayerName'], jsonData['ncsdkGraphPath'], numHidden, numNodes))
