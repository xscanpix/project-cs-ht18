#!/usr/bin/python3

import sys

from mvnc import mvncapi as mvnc
import numpy as np

def setup_device(mvnc_log=bool):
    if(mvnc_log):
        mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)

    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    device = mvnc.Device(devices[0])
    device.open()

    return device


def load_graph(device, graph_dir, graph_filename):
    with open("{0}/{1}".format(graph_dir, graph_filename), mode="rb") as file:
        graphFileBuff = file.read()

    graph = mvnc.Graph("graph1")
    fifoIn, fifoOut = graph.allocate_with_fifos(device, graphFileBuff)

    return (graph, fifoIn, fifoOut)


def run_inference(graph, fifoIn, fifoOut, input):
    input = np.random.uniform(0,1,28).reshape(1,28).astype(np.float32)
    graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, input, 'user object')
    output, user_obj = fifoOut.read_elem()

    return (output, user_obj)


def cleanup(fifoIn, fifoOut, graph, device):
    fifoIn.destroy()
    fifoOut.destroy()
    graph.destroy()
    device.close()


def main():
    if(len(sys.argv) != 3):
        print("Usage: python3 run_model.py graph_dir graph_filename")
        print("Example: python3 run_model.py TF_Model graph")
        exit()

    device = setup_device(True)
    (graph, fifoIn, fifoOut) = load_graph(device, sys.argv[1], sys.argv[2])
    
    input = np.random.uniform(0,1,28).reshape(1,28).astype(np.float32)
    (output, user_obj) = run_inference(graph, fifoIn, fifoOut, input)
    print("({0}, {1})".format(output, user_obj))

    cleanup(fifoIn, fifoOut, graph, device)


if __name__ == '__main__':
    main()