#! /usr/bin/env python
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import keras as keras
from keras.models import load_model
import numpy as np
import timeit
from tensorflow.python.client import device_lib

config = tf.ConfigProto(intra_op_parallelism_threads = 2, \
                        inter_op_parallelism_threads = 2, \
                        allow_soft_placement=True, \
                        device_count={"CPU": 2})

#keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#print(device_lib.list_local_devices())


class SampleStates:
    def __init__(self, model_name, input_name):
	
        self.input_file = input_name
        self.model = load_model(model_name)

    def read_data_from_file(self):
        with open(self.input_file, 'r') as file:
       	    pl = np.fromstring(file.read().replace("[", "").replace("]", ""), sep="   ").reshape(-1, 28)
        return pl

    def predict_q_value(self, state_):
        return sample_states.model.predict(state_.reshape(1, 28))

    def calc(self):
	
        for state in states:
	    
            predicted_q_values = self.predict_q_value(state)

           # print(predicted_q_values)
    
    


if __name__ == '__main__':
    sample_states = SampleStates('model_2380.h5', 'sample_input.txt')
    
    states = sample_states.read_data_from_file()

    #timing func, number is equal to iteration count
    print(timeit.timeit(sample_states.calc, number = 10))
	

    
