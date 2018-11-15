#!/usr/bin/python3
from mvnc import mvncapi as mvnc

def setup_device(mvnc_log=True):
    if(mvnc_log):
        mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)

    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    device = mvnc.Device(devices[0])
    device.open()

    return device


def load_graph(device, path_to_graph):
    with open(path_to_graph, mode="rb") as file:
        graphFileBuff = file.read()

    graph = mvnc.Graph("graph1")
    fifoIn, fifoOut = graph.allocate_with_fifos(device, graphFileBuff)

    return (graph, fifoIn, fifoOut)


def run_inference(graph, fifoIn, fifoOut, input):
    graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, input, 'user object')
    output, user_obj = fifoOut.read_elem()

    return (output, user_obj)


def cleanup(fifoIn, fifoOut, graph, device):
    fifoIn.destroy()
    fifoOut.destroy()
    graph.destroy()
    device.close()