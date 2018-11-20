#!/usr/bin/python3

import argparse, os

from code.helpers import load_settings
from code.model.keras2graph import compile_tf, keras_to_tf

from tests.tests import run_tests, plot_result, CpuTest, MovidiusTest

def main():
    os.environ['PROJ_DIR'] = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("-k2tf", "--kerastotf", action="store_true", help="Convert first to tf")
    parser.add_argument("-c", "--compile", action="store_true", help="Compile to a Movidius graph?")
    parser.add_argument("-tmc", "--testmodecpu", action="store_true", help="Run tests on CPU")
    parser.add_argument("-tmm", "--testmodemovidius", action="store_true", help="Run tests on Movidius")
    parser.add_argument("-tf", "--testfile", help="Sample file to test")
    parser.add_argument("-s", "--settings", help="Path to settings file")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite files")
    parser.add_argument("-it", "--iterations", type=int, help="Iterations")
    args = parser.parse_args()

    try:
        jsonData = load_settings(args.settings)
    except:
        exit()

    # Convert Keras model to Tensorflow model, save Tensorflowmodel.
    if args.kerastotf:
        keras_to_tf(jsonData, args.force)

    # Compile Tensorflow model to Movidius graph
    if args.compile:
        compile_tf(jsonData, args.force)

    if args.testmodecpu or args.testmodemovidius:
        testInfo = {
            "throwFirst": 1,
            "runs": 1,
            "iterations": 1000,
            "createGraph": True
        }
        # Run inference with CPU
        if args.testmodecpu:
            testclass = CpuTest(jsonData, testInfo, False)

        # Run inference with Movidius
        elif args.testmodemovidius:
            testclass = MovidiusTest(jsonData, testInfo, False)

        testclass.test_start()
        times = run_tests(testclass)
        testclass.test_end()
        plot_result(testInfo, times)


if __name__ == '__main__':
    main()