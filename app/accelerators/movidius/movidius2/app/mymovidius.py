#!/usr/bin/python3
from mvnc import mvncapi as mvnc

import numpy as np
import time

import logging

logger = logging.getLogger('(MyMovidius)')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
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

        logger.info("Creating and allocating graph %s. Name: %s, Path: %s" % (self.graph, self.graphName, pathToGraph))


    def get_graph(self):
        return self.graph


    def get_fifoIn(self):
        return self.fifoIn


    def get_fifoOut(self):
        return self.fifoOut


    def cleanup(self):
        logger.info("Cleaning up graph %s." % (self.graph))
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
        logger.info("Creating device %s" % (self.device))


    '''
    Opens the device
    '''
    def open_device(self):
        logger.info("Opening device %s" % (self.device))
        self.device.open()


    '''
    Loads a graph into itself
    '''
    def load_graph(self, graphName, pathToGraph):
        graph = MyGraph(self.device, graphName, pathToGraph)
        self.graphs.append(graph)

        return (graph.graph, graph.fifoIn, graph.fifoOut)


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
        logger.info("Cleaning up device %s" % (self.device))
        for graph in self.graphs: 
            graph.cleanup()
            
        self.device.destroy()


class MyMovidius:
    '''
    Class for all Movidius devices
    '''
    def __init__(self):
        self.deviceHandles = []
        self.devices = []

        self.deviceHandles = mvnc.enumerate_devices()
        if len(self.deviceHandles) == 0:
            logger.error('No devices found')
            quit()

        logger.info("Creating movidius object. Device handles found: %d" % (len(self.deviceHandles)))


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


    '''
    Loads a graph onto the MyDevice object
    '''
    def load_graph(self, device, graphName, pathToGraph):
        device.load_graph(graphName, pathToGraph)


    def run_inference_device_index(self, deviceIndex, graphName, input):
        device = self.get_device_by_index(deviceIndex)

        return self.run_inference(device, graphName, input)


    '''
    Runs inference with input on the MyDevice object
    '''
    def run_inference(self, device, graphName, input):
        graphclass = device.get_graph_by_name(graphName)

        graphclass.get_graph().queue_inference_with_fifo_elem(graphclass.get_fifoIn(), graphclass.get_fifoOut(), input, None)
        output, userObj = graphclass.get_fifoOut().read_elem()
        return (output, userObj)


    def get_inference_time(self, device, graphName):
        start = time.perf_counter()
        times = np.sum(device.get_graph_by_name(graphName).graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN))
        end = time.perf_counter()

        return (times, end - start)


    def get_inference_time_device_index(self, deviceIndex, graphName):
        return self.get_inference_time(self.get_device_by_index(deviceIndex), graphName)

    '''
    Cleans up all devices
    '''
    def cleanup(self):
        logger.info("Cleaning up movidius.")
        for device in self.devices:
            device.cleanup()


##############################################################3
