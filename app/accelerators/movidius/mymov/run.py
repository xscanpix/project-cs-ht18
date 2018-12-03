#!/usr/bin/python3

import argparse, os

import numpy as np

from helpers import load_settings
from keras2graph import compile_tf, gen_model

from mymovidius import MyMovidius

def main():
    os.environ['PROJ_DIR'] = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="Run tests.", action='store_true')
    parser.add_argument("settings", help="Environment settings file.")
    args = parser.parse_args()

    try:
        jsonData = load_settings(args.settings)
    except Exception as error:
        print(error)
        exit()

    gen_model(jsonData['tfOutputPath'])
    compile_tf(jsonData)

    if args.test:
        mov = MyMovidius()
        mov.init_devices(1)
        mov.load_graph_device_index(0, jsonData['graphName'], jsonData['ncsdkGraphPath'])
        (result, _) = mov.run_inference_device_index(0, jsonData['graphName'], np.random.uniform(0,1,28).reshape(1,28).astype(np.float32))
        print("Result: ", result)
        mov.cleanup()


if __name__ == '__main__':
    main()