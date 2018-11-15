#!/usr/bin/python3
import sys
import numpy as np

from helpers import load_settings
from code.movidius.setup import setup_device, load_graph, run_inference, cleanup

def main():
    if(len(sys.argv) != 2):
        print("Usage: python3 run_model.py settings_file")
        print("Example: python3 run_model.py settings.json")
        exit()

    print("[INFO]: Loading settings...")
    json_data = load_settings(sys.argv[1])
    path_to_graph = json_data['ncsdk_graph_path']
    print("[INFO]: Settings loaded.")

    print("[INFO]: Settings up device...")
    device = setup_device()
    print("[INFO]: Device ready.")
    print("[INFO]: Loading graph onto device...")
    (graph, fifoIn, fifoOut) = load_graph(device, path_to_graph)
    print("[INFO]: Graph loaded.")

    input = np.random.uniform(0,1,28).reshape(1,28).astype(np.float32)
    print("[INFO]: Running test on Tensor:\n{}".format(input))
    (output, user_obj) = run_inference(graph, fifoIn, fifoOut, input)
    print("[INFO]: Test done. Output: {}".format(output))

    print("[INFO]: Cleaning up...")
    cleanup(fifoIn, fifoOut, graph, device)


if __name__ == '__main__':
    main()