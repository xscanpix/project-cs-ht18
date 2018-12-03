#!/usr/bin/python3

from abc import ABC, abstractmethod
from pprint import pprint
import os

from tests.testkeras2graph import gen_model
from mymov.mymovidius import MyMovidius

import logging

logger = logging.getLogger('(Tests)')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class TestClass(ABC):
    @abstractmethod
    def __init__(self, jsonData, testconfig, test, inputs):
        self.jsonData = jsonData
        self.testconfig = testconfig
        self.test = testconfig['tests'][test]
        self.inputs = inputs

    @abstractmethod
    def test_start(self):
        logging.info("-- Printing Test Configuration --")
        logging.info("JsonData:")
        pprint(self.jsonData)
        logging.info("Testconfig:")
        pprint(self.testconfig)
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
        pass

    def test_start(self):
        super().test_start()
        logger.info("-- Start CpuTest --")
        pass

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
        logger.info("-- End CpuTest --")
        pass

from mvnc import mvncapi as mvnc

class MovidiusTest(TestClass):
    def __init__(self, jsonData, testconfig, test, inputs):
        super().__init__(jsonData, testconfig, test, inputs)
        pass

    def test_start(self):
        super().test_start()
        logger.info("-- Start MovidiusTest --")
        pass

    def run_setup(self):
        super().run_setup()
        self.myMovidius = MyMovidius()
        self.myMovidius.init_devices(self.jsonData['numDevices'])
        self.myMovidius.load_graph_device_index(self.jsonData['defaultDeviceIndex'], self.jsonData['graphName']+"_{}_{}_{}".format(self.test['layers'], self.test['neurons'], self.test['shaves']), self.jsonData['ncsdkGraphPath']+"_{}_{}_{}".format(self.test['layers'], self.test['neurons'], self.test['shaves']))
        pass

    def run_inference(self, input):
        super().run_inference(input)
        (output, user_obj) = self.myMovidius.run_inference_device_index(self.jsonData['defaultDeviceIndex'], self.jsonData['graphName']+"_{}_{}_{}".format(self.test['layers'], self.test['neurons'], self.test['shaves']), input)
        (times, sub) = self.myMovidius.get_inference_time(self.myMovidius.get_device_by_index(self.jsonData['defaultDeviceIndex']), self.jsonData['graphName']+"_{}_{}_{}".format(self.test['layers'], self.test['neurons'], self.test['shaves']))
        # (ms, seconds)
        return (times, sub)

    def run_cleanup(self):
        super().run_cleanup()
        self.myMovidius.cleanup()
        pass

    def test_end(self):
        super().test_end()
        logger.info("-- End MovidiusTest --")
        pass

import time

def run_tests(testclass):
    totaltimes = []
    movidiustimes = []

    for i in range(testclass.testconfig['iterations']):
        start = time.perf_counter()
        times = testclass.run_inference(testclass.inputs[i])
        # time = (ms, seconds)
        end = time.perf_counter()
        
        if times != None:
            movidiustimes.append(times[0])
        totaltimes.append(((end - start) - times[1]) * 1000)

    plot_result(testclass.testconfig, testclass.test, [totaltimes, movidiustimes])


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def plot_result(testconfig, test, times):
    if(testconfig['savegraphs']):
        n = testconfig['smoothing']
        b = [1.0 / n] * n
        a = 1

        y0 = lfilter(b,a,times[0])[testconfig['throwfirst']:]
        x0 = np.arange(len(y0))
        avg_0 = np.average(y0)
        std_0 = np.std(y0)
        mean_0 = np.median(y0)

        fig, ax1 = plt.subplots()
        ax1.set_title("Execution times")
        ax1.set_xlim(0, len(y0))
        ax1.set_ylabel("Execution time ms")
        ax1.set_ylim([0,2])
        ax1.set_xlabel("Iteration")
        line_1, = ax1.plot(x0, y0, label="Total ms, avg(%.5f ms), std(%.5f ms), median(%.5f ms)" % (avg_0, std_0, mean_0), color='b')
        
        if(len(times[1]) > 0):
            y1 = lfilter(b,a,times[1])[testconfig['throwfirst']:]
            x1 = np.arange(len(y1))
            avg_1 = np.average(y1)
            std_1 = np.std(y1)
            mean_1 = np.median(y1)
            line_2, = ax1.plot(x1, y1, label= "Movidius ms, avg(%.5f ms), std(%.5f ms), median(%.5f ms)" % (avg_1, std_1, mean_1), color='r')
            ax2 = ax1.twinx()
            y3 = np.divide(y1, y0)
            avg_2 = np.average(y3)
            std_2 = np.std(y3)
            mean_2 = np.median(y3)
            line_3, = ax2.plot(x0, y3, label="%% of total, avg(%.5f %%), std(%.5f %%), median(%.5f %%)" % (avg_2, std_2, mean_2), color='g')
            ax2.set_ylabel("% of total")
            ax2.set_ylim([0,0.5])
            ax1.legend(handles=[line_1, line_2, line_3])

        else:
            ax1.legend(handles=[line_1])


        fig.tight_layout()
        #plt.show()
        #plt.savefig(os.environ["PROJ_DIR"]+"/resources/graphs/graph" + test['graphNameSuffix'])

        info = "_{}_{}_{}".format(test['layers'], test['neurons'], test['shaves']).split("_")[1:]

        if(not os.path.exists(os.environ["PROJ_DIR"]+"/resources/testresults/testdata_shave_{}.txt".format(info[2]))):
            writeflag = "w"
        else:
            writeflag = "a"
        with open(os.environ["PROJ_DIR"]+"/resources/testresults/testdata_shave_{}.txt".format(info[2]), writeflag) as file:
            file.write("!{}:{}:{}\n".format(info[0], info[1], info[2]))
            file.write("% total\n")
            file.write("{} {} {}\n".format(avg_0, std_0, mean_0))
            file.write("% movidius\n")
            file.write("{} {} {}\n".format(avg_1, std_1, mean_1))
            file.write("% percent\n")
            file.write("{} {} {}\n".format(avg_2, std_2, mean_2))
