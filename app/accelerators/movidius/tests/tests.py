#!/usr/bin/python3

from abc import ABC, abstractmethod

import time, pprint
import numpy as np
import matplotlib.pyplot as plt

from code.model.keras2graph import gen_new_model, load_model_keras
from code.movidius.mymovidius import MyMovidius


class TestClass(ABC):
    @abstractmethod
    def __init__(self, jsonData, testInfo, logging):
        self.jsonData = jsonData
        self.testInfo = testInfo
        self.logging = logging


    @abstractmethod
    def test_start(self):
        print("-- Printing Test Configuration --")
        print("JsonData:")
        pprint.pprint(self.jsonData)
        print("TestInfo:")
        pprint.pprint(self.testInfo)
        print("")
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
    def __init__(self, jsonData, testInfo, logging):
        super().__init__(jsonData, testInfo, logging)
        pass

    def test_start(self):
        super().test_start()
        print("-- Start CpuTest --")
        pass

    def run_setup(self):
        super().run_setup()
        self.model = gen_new_model(load_model_keras(self.jsonData['kerasModelPath']), self.jsonData['tfOutputPath'])
        pass

    def run_inference(self, input):
        super().run_inference(input)
        return self.model.predict(input)

    def run_cleanup(self):
        super().run_cleanup()
        pass

    def test_end(self):
        super().test_end()
        print("-- End CpuTest --")
        pass


class MovidiusTest(TestClass):
    def __init__(self, jsonData, testInfo, logging):
        super().__init__(jsonData, testInfo, logging)
        pass

    def test_start(self):
        super().test_start()
        print("-- Start MovidiusTest --")
        pass

    def run_setup(self):
        super().run_setup()
        self.myMovidius = MyMovidius(self.jsonData['numDevices'])
        self.myMovidius.load_graph_device_index(self.jsonData['defaultDeviceIndex'], self.jsonData['graphName'], self.jsonData['ncsdkGraphPath'])
        pass

    def run_inference(self, input):
        super().run_inference(input)
        return self.myMovidius.run_inference_device_index(self.jsonData['defaultDeviceIndex'], self.jsonData['graphName'], input)

    def run_cleanup(self):
        super().run_cleanup()
        self.myMovidius.cleanup()
        pass

    def test_end(self):
        super().test_end()
        print("-- End MovidiusTest --")
        pass


def run_tests(testclass):
    times = []

    for j in range(testclass.testInfo['runs']):
        times.append([])
        setup = testclass.run_setup()
        for i in range(testclass.testInfo['iterations']):
            start = time.time()
            result = testclass.run_inference(np.random.uniform(0,1,28).reshape(1,28).astype(np.float32))
            end = time.time()
            times[j].append(end - start)
        testclass.run_cleanup()

    return times


def plot_result(testInfo, times):
    if(testInfo['createGraph']):
        plt.plot(np.linspace(testInfo['throwFirst'],testInfo['iterations'], testInfo['iterations'] - testInfo['throwFirst']), [x * 1000.0 for x in times[0][testInfo['throwFirst']:]])
        plt.xlabel("Iteration")
        plt.ylabel("Execution time")
        plt.show()
