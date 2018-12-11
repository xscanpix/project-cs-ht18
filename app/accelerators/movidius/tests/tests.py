#!/usr/bin/python3

from abc import ABC, abstractmethod
import os
import time

import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation
from keras import optimizers

from mymov.mymovidius import MyMovidius

from tests.helpers import save_result


class TestClass(ABC):
    @abstractmethod
    def __init__(self, jsonData, testconfig, test, inputs):
        self.jsonData = jsonData
        self.testconfig = testconfig
        self.test = testconfig['tests'][test]
        self.inputs = inputs

    @abstractmethod
    def test_start(self):
        pass

    @abstractmethod
    def run_setup(self):
        pass

    @abstractmethod
    def run_inference(self, input):
        pass

    @abstractmethod
    def run_cleanup(self):
        pass

    @abstractmethod
    def test_end(self):
        pass


class CpuTest(TestClass):
    def __init__(self, jsonData, testconfig, test, inputs):
        super().__init__(jsonData, testconfig, test, inputs)

    def test_start(self):
        super().test_start()

    def run_setup(self):
        super().run_setup()
        self.model = gen_model(self.jsonData['tfOutputPath'])
        pass

    def run_inference(self, input):
        super().run_inference(input)
        self.model.predict(input)
        return (0, 0)

    def run_cleanup(self):
        super().run_cleanup()
        pass

    def test_end(self):
        super().test_end()
        pass

from mvnc import mvncapi as mvnc

class MovidiusTest(TestClass):
    def __init__(self, jsonData, testconfig, test, inputs):
        super().__init__(jsonData, testconfig, test, inputs)

    def test_start(self):
        super().test_start()

    def run_setup(self):
        super().run_setup()
        try:
            self.myMovidius = MyMovidius()
            self.myMovidius.init_devices(self.jsonData['numDevices'])
            self.myMovidius.load_graph_device_index(self.jsonData['defaultDeviceIndex'], self.jsonData['graphName']+"_{}_{}_{}".format(self.test['layers'], self.test['neurons'], self.test['shaves']), self.jsonData['ncsdkGraphPath']+"_{}_{}_{}".format(self.test['layers'], self.test['neurons'], self.test['shaves']))
        except Exception as error:
            print("Error:", error)
            exit()

    def run_inference(self, input):
        super().run_inference(input)
        
        (output, user_obj) = self.myMovidius.run_inference_device_index(self.jsonData['defaultDeviceIndex'], self.jsonData['graphName']+"_{}_{}_{}".format(self.test['layers'], self.test['neurons'], self.test['shaves']), input)
        (times, sub) = self.myMovidius.get_inference_time(self.myMovidius.get_device_by_index(self.jsonData['defaultDeviceIndex']), self.jsonData['graphName']+"_{}_{}_{}".format(self.test['layers'], self.test['neurons'], self.test['shaves']))

        return (times, sub)

    def run_cleanup(self):
        super().run_cleanup()
        try:
            self.myMovidius.cleanup()
        except Exception as error:
            print("Error:", error)
            exit()

    def test_end(self):
        super().test_end()


def run_tests(testclass):
    totaltimes = []
    movidiustimes = []

    for i in range(testclass.testconfig['iterations']):
        start = time.perf_counter()
        times = testclass.run_inference(testclass.inputs[i])
        end = time.perf_counter()

        if times != None:
            movidiustimes.append(times[0])
        totaltimes.append(((end - start) - times[1]) * 1000)

    save_result(testclass.testconfig, testclass.test, [totaltimes, movidiustimes])


def gen_model(tf_model_path, test):
    if(os.path.exists(tf_model_path + "_{}_{}.meta".format(test['layers'], test['neurons']))):
        return

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


def compile_tf(jsonData, test):
    if(os.path.exists("{}_{}_{}_{}".format(jsonData['ncsdkGraphPath'], test['layers'], test['neurons'], test['shaves']))):
        return
    
    os.system("mvNCCompile {}_{}_{}.meta -s {} -in {} -on {} -o {}_{}_{}_{}".format(jsonData['tfOutputPath'], test['layers'], test['neurons'], test['shaves'], jsonData['inputLayerName'], jsonData['outputLayerName'], jsonData['ncsdkGraphPath'], test['layers'], test['neurons'], test['shaves']))
