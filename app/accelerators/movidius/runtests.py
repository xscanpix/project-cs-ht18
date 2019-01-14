import argparse
import os
from pprint import pprint

import numpy as np

from mymov.helpers import load_settings
from tests.helpers import load_test_config
from tests.tests import run_tests, gen_model, compile_tf, MovidiusTest

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tc", "--testconfig", help="Supply test config or tensorflow model", required=True)
    parser.add_argument("-s", "--settings", help="Environment settings file.", required=True)
    parser.add_argument("-ti", "--testindex", help="Test index")
    args = parser.parse_args()

    os.environ['PROJ_DIR'] = os.getcwd()
    
    try:
        jsonData = load_settings(args.settings)
        testconfig = load_test_config(args.testconfig)
    except Exception as error:
        print("Error loading file:", error)
        exit()

    inputs = []

    for _ in range(int(testconfig['iterations'])):
        inputs.append(np.random.uniform(0,1,testconfig['inputs']).reshape(1,testconfig['inputs']).astype(np.float32))

    if args.testindex:
        print("Test:")
        pprint(testconfig['tests'][int(args.testindex)-1])
        gen_model(jsonData['tfOutputPath'], testconfig['tests'][int(args.testindex)-1])
        compile_tf(jsonData, testconfig['tests'][int(args.testindex)-1])
        # Fix some reset mechanism that works...
        #os.system("./usbreset $(lsusb | grep 03e7 | awk \'{print \"/dev/bus/usb/\" $2 \"/\" substr($4,1,3)}\')")
        testclass = MovidiusTest(jsonData, testconfig, int(args.testindex)-1, inputs)
        testclass.run_setup()
        for i in range(int(testconfig['runs'])):
            run_tests(testclass)
            print("Subtest #{} done".format(i + 1))
        testclass.run_cleanup()
    else:
        for index, test in enumerate(testconfig['tests']):
            print("Test:")
            pprint(test)
            gen_model(jsonData['tfOutputPath'], test)
            compile_tf(jsonData, test)
            # Fix some reset mechanism that works...
            #os.system("./usbreset $(lsusb | grep 03e7 | awk \'{print \"/dev/bus/usb/\" $2 \"/\" substr($4,1,3)}\')")
            testclass = MovidiusTest(jsonData, testconfig, index, inputs)
            testclass.run_setup()
            for i in range(int(testconfig['runs'])):
                run_tests(testclass)
                print("Subtest #{} done".format(i + 1))
            testclass.run_cleanup()
