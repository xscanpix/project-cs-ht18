#!/usr/bin/python3

import argparse, os, time, logging

from app.helpers import load_settings
from app.keras2graph import compile_tf, keras_to_tf, gen_model

from tests.tests import run_tests, plot_result, save_results, CpuTest, MovidiusTest

def main():


    os.environ['PROJ_DIR'] = os.getcwd()
    print(os.environ['PROJ_DIR'])

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

    testtuples = [  (0, 64)]#, (1, 64), (2, 64), (3, 64), (4, 64), (5, 64), (6, 64), (7, 64), (8, 64), (9, 64),
                    #(0, 128), (1, 128), (2, 128), (3, 128), (4, 128)]

    # Convert Keras model to Tensorflow model, save Tensorflowmodel.
    if args.kerastotf:
        #keras_to_tf(jsonData, args.force)
        for tup in testtuples:
            gen_model(jsonData['tfOutputPath'], tup[0], tup[1]).summary()

    # Compile Tensorflow model to Movidius graph
    if args.compile:
        for tup in testtuples:
            compile_tf(jsonData, args.force, tup[0], tup[1])

    if args.testmodecpu or args.testmodemovidius:
        for tup in testtuples:
            testInfo = {
                "throwFirst": 100,
                "smoothing": 2,
                "runs": 1,
                "iterations": 1000,
                "createGraph": True,
                "graphNameSuffix": "_{}_{}".format(tup[0], tup[1])
            }
            
            # Run inference with CPU
            if args.testmodecpu:
                testclass = CpuTest(jsonData, testInfo)

            # Run inference with Movidius
            elif args.testmodemovidius:
                testclass = MovidiusTest(jsonData, testInfo)

            testclass.test_start()
            result = run_tests(testclass)
            testclass.test_end()
            #save_results(testInfo, result)
            plot_result(testInfo, result)

            time.sleep(1)


if __name__ == '__main__':
    main()