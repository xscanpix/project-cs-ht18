#!/usr/bin/python3

import argparse, os, time
from pprint import pprint
import numpy as np

from mymov.helpers import load_settings
from tests.testkeras2graph import compile_tf, gen_model

from tests.tests import run_tests, plot_result, CpuTest, MovidiusTest
from tests.helpers import load_test_config

def main():
    os.environ['PROJ_DIR'] = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=['cpu', 'movidius'], help="Which mode to run.", required=True)
    parser.add_argument("-ms", "--modelsource", choices=['tensorfile', 'generate'], help="Run with supplied tensorflow model file or generate from method according to testfile.", required=True)
    parser.add_argument("-tf", "--testfile", help="Supply test config or tensorflow model")
    parser.add_argument("-ti", "--testindex")
    parser.add_argument("-s", "--settings", help="Environment settings file.", required=True)
    args = parser.parse_args()

    try:
        jsonData = load_settings(args.settings)
    except Exception as error:
        print(error)
        exit()

    testconfig = None

    if args.modelsource == "generate":
        if not args.testfile:
            parser.print_help()
            exit()

        testconfig = load_test_config(args.testfile)

        if args.testindex:
            index = int(args.testindex) - 1
            gen_model(jsonData['tfOutputPath'], testconfig['tests'][index])
        else:
            pass

    # Compile Tensorflow model to Movidius graph
    if args.mode == 'movidius':
        assert(testconfig != None)

        inputs = []

        for _ in range(int(testconfig['iterations'])):
            inputs.append(np.random.uniform(0,1,28).reshape(1,28).astype(np.float32))

        if args.testindex:
            index = int(args.testindex) - 1
            compile_tf(jsonData, testconfig['tests'][index])
            testclass = MovidiusTest(jsonData, testconfig, index, inputs)
            #testclass.test_start()
            print("Test:")
            pprint(testconfig['tests'][index])
            testclass.run_setup()
            for i in range(int(testconfig['runs'])):
                run_tests(testclass)
                print("Subtest #{} done".format(i + 1))
                time.sleep(1)
            testclass.run_cleanup()
            #testclass.test_end()
        else:
            pass


if __name__ == '__main__':
    main()