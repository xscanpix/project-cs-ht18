#!/usr/bin/python3
from mvnc import mvncapi as mvnc

import time
import numpy as np

class MyGraph:

    '''
    Class for a NCSDK Graph and its FIFO queues
    Arguments:
        device: DeviceClass
        graphName: string
        pathToGraph: string
    '''
    def __init__(self, device, graphName, pathToGraph):
        self.graphName = None
        self.graph = None
        self.fifoIn = None
        self.fifoOut = None

        with open(pathToGraph, mode="rb") as file:
            graphFileBuff = file.read()

        self.graphName = graphName
        try:
            self.graph = mvnc.Graph(self.graphName)
        except Exception as error:
            print("Error creating graph:", error)
            exit()
        try:
            self.fifoIn, self.fifoOut = self.graph.allocate_with_fifos(device, graphFileBuff)
        except Exception as error:
            print("Error creating fifo:", error)
            exit()

    def cleanup(self):
        try:
            self.fifoIn.destroy()
            self.fifoOut.destroy()
            self.graph.destroy()
        except Exception as error:
            print("Error creating destroying graph or fifos:", error)
            exit()


class MyDevice:
    '''
    Class for a NCSDK Device and its graphs
    Arguments: 
        device: mvnc enumerated device
    '''
    def __init__(self, device):
        try:
            self.device = mvnc.Device(device)
        except Exception as error:
            print("Error creating device:", error)
            exit()
        self.graphs = []


    '''
    Opens the device
    '''
    def open_device(self):
        try:
            self.device.open()
        except Exception as error:
            print("Error opening device:", error)
            exit()

    '''
    Loads a graph into itself
    '''
    def allocate_graph(self, graphName, pathToGraph):
        graph = MyGraph(self.device, graphName, pathToGraph)
        self.graphs.append(graph)

        return (graph.graph, graph.fifoIn, graph.fifoOut)


    def deallocate_graph(self, graphName):
        for key, graph in enumerate(self.graphs):
            if(graph.graphName == graphName):
                graph.cleanup()
                del self.graphs[key]

    '''
    Get a graph referenced by graph name
    '''
    def get_graph_by_name(self, graphName):
        rv = None

        for graph in self.graphs:
            if(graph.graphName == graphName):
                rv = graph

        return rv


    '''
    Cleanup the device
    '''
    def cleanup(self):
        for graph in self.graphs: 
            graph.cleanup()
        
        try:
            self.device.close()
            self.device.destroy()
        except Exception as error:
            print("Error closing or destroying device:",error)
            exit()

class MyMovidius:
    '''
    Class for all Movidius devices
    '''
    def __init__(self):
        #mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, mvnc.LogLevel.INFO)

        self.deviceHandles = []
        self.devices = []

        self.deviceHandles = mvnc.enumerate_devices()
        if len(self.deviceHandles) == 0:
            exit()

    def init_devices(self, numDevices):
        if(len(self.deviceHandles) < numDevices):
            exit()

        # Create devices
        for i in range(numDevices):
            newDevice = MyDevice(self.deviceHandles[i])
            self.devices.append(newDevice)

        # Open devices
        for device in self.devices:
            device.open_device()


    '''
    Returns the MyDevice object at deviceIndex
    '''
    def get_device_by_index(self, deviceIndex):
        if(deviceIndex < 0 or not deviceIndex < len(self.devices)):
            self.cleanup()
            exit()

        return self.devices[deviceIndex]


    '''
    Loads a graph onto the device at deviceIndex
    '''
    def load_graph_device_index(self, deviceIndex, graphName, pathToGraph):
        device = self.get_device_by_index(deviceIndex)

        self.load_graph(device, graphName, pathToGraph)


    def deallocate_graph_device_index(self, deviceIndex, graphName):
        device = self.get_device_by_index(deviceIndex)
        self.deallocate_graph(device, graphName)


    def deallocate_graph(self, device, graphName):
        device.deallocate_graph(graphName)

    '''
    Loads a graph onto the MyDevice object
    '''
    def load_graph(self, device, graphName, pathToGraph):
        device.allocate_graph(graphName, pathToGraph)


    def run_inference_device_index(self, deviceIndex, graphName, input):
        device = self.get_device_by_index(deviceIndex)

        return self.run_inference(device, graphName, input)


    '''
    Runs inference with input on the MyDevice object
    '''
    def run_inference(self, device, graphName, input):
        graphclass = device.get_graph_by_name(graphName)

        assert(graphclass != None)

        try:
            graphclass.graph.queue_inference_with_fifo_elem(graphclass.fifoIn, graphclass.fifoOut, input, None)
            output, userObj = graphclass.fifoOut.read_elem()
        except Exception as error:
            print("Error qeueing or reading:", error)
            exit()

        return (output, userObj)


    def get_inference_time(self, device, graphName):
        start = time.perf_counter()
        res = 0.0
        try:
            time.sleep(0.001)
            if(device.get_graph_by_name(graphName).graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN_ARRAY_SIZE) == 0):
                print("Can happen??")
            times = device.get_graph_by_name(graphName).graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN)
        except Exception as error:
            print("Error reading time:", error)
            exit()
        for subtime in np.nditer(times):
            res += subtime
        end = time.perf_counter()
        # (ms, seconds)
        return (res, end - start)


    def get_inference_time_device_index(self, deviceIndex, graphName):
        return self.get_inference_time(self.get_device_by_index(deviceIndex), graphName)

    '''
    Cleans up all devices
    '''
    def cleanup(self):
        for device in self.devices:
            device.cleanup()


##############################################################3
