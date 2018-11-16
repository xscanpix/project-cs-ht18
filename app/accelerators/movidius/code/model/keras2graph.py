#!/usr/bin/python3
import sys, os

import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation
from keras import optimizers, models

from helpers import get_extension, load_settings

VALID_H5   = ['.h5']
VALID_JSON = ['.json']
VALID_YAML = ['.yml', '.yaml']


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


def save_model_arch(model, filepath):
    """Save the model architecture to file"""
    ext = get_extension(filepath)

    if(ext in VALID_JSON):
        string = model.to_json()
    elif(ext in VALID_YAML):
        string = model.to_yaml()
    else:
        print("Invalid file extension: {0}.".format(ext))
        exit()

    with open(filepath, "w") as outfile:
        outfile.write(string)

    return True


def load_model_arch(filepath):
    """Load the model architecture from file and return the model"""
    ext = get_extension(filepath)

    with open(filepath, "r") as infile:
        string = infile.read()

    if(ext in VALID_JSON):
        model = models.model_from_json(string)
    elif(ext in VALID_YAML):
        model = models.model_from_yaml(string)
    else:
        print("Invalid file extension: {0}.".format(ext))
        exit()

    return model


def save_model_keras(model, filepath):
    """Save a keras model to a HDF5 file"""
    ext = get_extension(filepath)
    if(ext not in VALID_H5):
        print("Invalid file extension: {0}.".format(ext))
        exit()

    model.save(filepath)

    return True


def load_model_keras(filepath):
    """Load keras model from HDF5 file"""
    ext = get_extension(filepath)

    if(ext not in VALID_H5):
        print("Invalid file extension: {0}.".format(ext))
        exit()

    model = models.load_model(filepath)

    return model

def main():
    if(len(sys.argv) != 2):
        print("Usage: python3 run settings_file")
        print("Example: python3 run settings.json")
        exit()

    print("\n")
    print("[INFO] Loading settings from {}...".format(sys.argv[1]))
    jsonData = load_settings(sys.argv[1])
    print(jsonData)

    kerasModelPath = jsonData['kerasModelPath']
    tfOutputPath = jsonData['tfOutputPath']
    inputLayerName = jsonData['inputLayerName']
    outputLayerName = jsonData['outputLayerName']
    ncsdkGraphPath = jsonData['ncsdkGraphPath']

    print("[INFO] Loading model from keras file {}".format(kerasModelPath))
    model = load_model_keras(kerasModelPath)
    print("[INFO] Loaded model: ")
    model.summary()
    print("\n")

    print("[INFO] Generating new model without Dropout layer...")
    newModel = gen_new_model(model, tfOutputPath)
    print("[INFO] New model: ")
    newModel.summary()
    print("\n")

    print("Compiling with mvNNCompile... ")
    os.system("mvNCCompile {0}.meta -in {1} -on {2} -o {3}".format(tfOutputPath, inputLayerName, outputLayerName, ncsdkGraphPath))
    print("Graph generated at {}".format(ncsdkGraphPath))


if __name__ == '__main__':
    main()