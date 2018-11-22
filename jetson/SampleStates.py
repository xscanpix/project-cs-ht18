#! /usr/bin/env python
import os
import sys


import tensorflow as tf
import keras as keras
from keras.models import load_model
import numpy as np
import timeit
from tensorflow.python.client import device_lib

states = None
sample_states = None

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

def print_info():
    print "Usage: python Samplestates.py modelname inputfile mode iterations"
    print "       Use python Samplestates.py -h for more help"

def print_help():
    print "Usage: python Samplestates.py modelname inputfile mode iterations"
    print "       modelname is the name of a modelfile of file extension .h5"
    print "       inputfile is the name of the inputfile of file extentsion .txt"
    print "       mode is Either GPU or CPU"
    print "       iterations is the amount of iterations we should benchmark on and should be a whole number"

def print_final(model_name, input_file, mode, iterations, runtime):
    print_runtime = str(round(runtime, 5))
    print_average = str(round(runtime/iterations, 5))
    print_iterations = str(iterations)
    print "\n" * 5
    print " " + ("_" * 21) + "FINALIZED RUN" + ("_" * 21) + " "
    print "|" + (" " * 55) + "|"
    print "|   Total runtime:          " + print_runtime + " seconds" + (" " * (20 - len(print_runtime))) + "|"
    print "|   Average runtime:        " + print_average + " seconds" + (" " * (20 - len(print_average))) + "|"
    print "|   Model:                  " + model_name[:-3] + (" " * 18) + "|"
    print "|   Input File:             " + input_file[:-4] + (" " * 16) + "|"
    print "|   Mode:                   " + mode + (" " * 25) + "|"
    print "|   Number of iterations:   " + print_iterations + (" " * (28 - len(print_iterations))) + "|"
    print "|" + ("_" * 55) + "|\n"
           
def main():
    global states
    global sample_states

    if len(sys.argv) < 2:
        print_info()
        return
    
    model_name = str(sys.argv[1])


    #If user uses -h to request help, this if statement will print info about the script
    if (model_name == '-h'):
        print_help()
        return

    if len(sys.argv) < 5:
        print_info()
        return
        
    input_file = str(sys.argv[2])
    mode = str(sys.argv[3])

    try:
        iterations = int(sys.argv[4])
    except ValueError:
        print_info()
        print "       Argument iterations was not a whole  number."
        return

    #This if statements catches any invalid user inputs and prompts the user how to use the script
    if (model_name[-3:] != '.h5' or input_file[-4:] != '.txt' or (mode != 'CPU' and  mode != 'GPU')):
        print_info()
        return

    if mode == 'CPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    sample_states = SampleStates(model_name, input_file)
    
    states = sample_states.read_data_from_file()

    #timing func, number is equal to iteration count
    runtime = timeit.timeit(sample_states.calc, number = iterations)

    print_final(model_name, input_file, mode, iterations, runtime)


if __name__ == '__main__':
    main()
	

    
