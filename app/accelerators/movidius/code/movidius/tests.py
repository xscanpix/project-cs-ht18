#!/usr/bin/python3
import time

def run_tests(testInfos, repetitions, myMovidius, myDevice, graphName, input):

    for info in testInfos:

        print("Running test %d runs for %d iterations." % (info['runs'], repetitions))
        start = time.time()
        for j in range(repetitions):
            for k in range(info['runs']):
                myMovidius.run_inference(myDevice, graphName, input)
        stop = time.time()
        info['result'] = (stop - start) / float(info['runs'])
        print("Result: %f\n" % (info['result']))

    return testInfos

