#!/usr/bin/python3
from mvnc import mvncapi as mvnc

import time
import numpy as np

import logging

logger = logging.getLogger('(MyMovidius)')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

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
        self.graph = mvnc.Graph(self.graphName)
        self.fifoIn, self.fifoOut = self.graph.allocate_with_fifos(device, graphFileBuff)
    
    def __repr__(self):
        return "<MyGraph>\nGraphName: {}\nGraph: {}\nFifoIn: {}\nFifoOut: {}".format(self.graphName, self.graph, self.fifoIn, self.fifoOut)

    def cleanup(self):
        logger.debug("Cleaning up graph %s." % (self))
        self.fifoIn.destroy()
        self.fifoOut.destroy()
        self.graph.destroy()



class MyDevice:
    '''
    Class for a NCSDK Device and its graphs
    Arguments: 
        device: mvnc enumerated device
    '''
    def __init__(self, device):
        self.device = mvnc.Device(device)
        self.graphs = []
        logger.debug("Creating device %s" % (self.device))


    '''
    Opens the device
    '''
    def open_device(self):
        logger.debug("Opening device %s" % (self.device))
        self.device.open()


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
        for graph in self.graphs:
            if(graph.graphName == graphName):
                return graph


    '''
    Cleanup the device
    '''
    def cleanup(self):
        logger.debug("Cleaning up device %s" % (self.device))
        for graph in self.graphs: 
            graph.cleanup()
            
        self.device.close()
        self.device.destroy()


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
            logger.error('No devices found')
            quit()

        logger.debug("Creating movidius object. Device handles found: %d" % (len(self.deviceHandles)))


    def init_devices(self, numDevices):
        if(len(self.deviceHandles) < numDevices):
            logger.error("Not enough devices available.")
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
            logger.error("Device index is wrong. Exiting...")
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

        graphclass.graph.queue_inference_with_fifo_elem(graphclass.fifoIn, graphclass.fifoOut, input, None)
        output, userObj = graphclass.fifoOut.read_elem()
        return (output, userObj)


    def get_inference_time(self, device, graphName):
        start = time.perf_counter()
        times = np.sum(device.get_graph_by_name(graphName).graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN))
        end = time.perf_counter()

        # (ms, seconds)
        return (times, end - start)


    def get_inference_time_device_index(self, deviceIndex, graphName):
        return self.get_inference_time(self.get_device_by_index(deviceIndex), graphName)

    '''
    Cleans up all devices
    '''
    def cleanup(self):
        logger.debug("Cleaning up movidius.")
        for device in self.devices:
            device.cleanup()


##############################################################3
