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
        inputs.append(np.random.uniform(0,1,28).reshape(1,28).astype(np.float32))

    for index, test in enumerate(testconfig['tests']):
        gen_model(jsonData['tfOutputPath'], test)
        compile_tf(jsonData, test)
        testclass = MovidiusTest(jsonData, testconfig, index, inputs)
        print("Test:")
        pprint(test)
        testclass.run_setup()
        for i in range(int(testconfig['runs'])):
            run_tests(testclass)
            print("Subtest #{} done".format(i + 1))
        testclass.run_cleanup()