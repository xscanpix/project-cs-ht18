#!/usr/bin/python3
from mvnc import mvncapi as mvnc

import time
import numpy as np


api_version = None

try: # Try to use function from API v2
    api_version = mvnc.global_get_option(mvnc.GlobalOption.RO_API_VERSION)[0]
except:
    try: # Try to use function from API v1
        mvnc.GetGlobalOption(mvnc.mvncGlobalOption.LOG_LEVEL)
        api_version = 1
    except:
        print("Neither API v1 or v2.")
finally:
    print("Api version", api_version)


class MyGraph:
    def __init__(self, device, graphName, pathToGraph):
        self.graphName = None
        self.graph = None
        self.fifoIn = None
        self.fifoOut = None

        with open(pathToGraph, mode="rb") as file:
            graphFileBuff = file.read()

        self.graphName = graphName
        try:
            if api_version == 2:
                self.graph = mvnc.Graph(self.graphName)
                self.fifoIn, self.fifoOut = self.graph.allocate_with_fifos(device, graphFileBuff)
            else:
                self.graph = device.AllocateGraph(graphFileBuff)
        except Exception as error:
            print("Error creating graph or fifo:", error)
            exit()

    def cleanup(self):
        try:
            if api_version == 2:
                self.fifoIn.destroy()
                self.fifoOut.destroy()
                self.graph.destroy()
            else:
                self.graph.DeallocateGraph()
        except Exception as error:
            print("Error creating destroying graph or fifos:", error)
            exit()


class MyDevice:
    def __init__(self, device):
        try:
            self.device = mvnc.Device(device)
        except Exception as error:
            print("Error creating device:", error)
            exit()
        self.graphs = []

    def open_device(self):
        try:
            if api_version == 2:
                self.device.open()
            else:
                self.device.OpenDevice()
        except Exception as error:
            print("Error opening device:", error)
            exit()

    def allocate_graph(self, graphName, pathToGraph):
        graph = MyGraph(self.device, graphName, pathToGraph)
        self.graphs.append(graph)

        return (graph.graph, graph.fifoIn, graph.fifoOut)


    def deallocate_graph(self, graphName):
        for key, graph in enumerate(self.graphs):
            if(graph.graphName == graphName):
                graph.cleanup()
                del self.graphs[key]

    def get_graph_by_name(self, graphName):
        rv = None

        for graph in self.graphs:
            if(graph.graphName == graphName):
                rv = graph

        return rv

    def cleanup(self):
        for graph in self.graphs: 
            graph.cleanup()
        
        try:
            if api_version == 2:
                self.device.close()
                self.device.destroy()
            else:
                self.device.CloseDevice()
        except Exception as error:
            print("Error closing or destroying device:",error)
            exit()


class MyMovidius:
    def __init__(self):
        self.deviceHandles = []
        self.devices = []

        if api_version == 2:
            self.deviceHandles = mvnc.enumerate_devices()
        else:
            self.deviceHandles = mvnc.EnumerateDevices()

        if len(self.deviceHandles) == 0:
            print("No devices found")
            exit()

    def init_devices(self, numDevices):
        if(len(self.deviceHandles) < numDevices):
            exit()

        for i in range(numDevices):
            newDevice = MyDevice(self.deviceHandles[i])
            self.devices.append(newDevice)

        for device in self.devices:
            device.open_device()

    def get_device_by_index(self, deviceIndex):
        if(deviceIndex < 0 or not deviceIndex < len(self.devices)):
            self.cleanup()
            exit()

        return self.devices[deviceIndex]

    def load_graph_device_index(self, deviceIndex, graphName, pathToGraph):
        device = self.get_device_by_index(deviceIndex)

        self.load_graph(device, graphName, pathToGraph)

    def deallocate_graph_device_index(self, deviceIndex, graphName):
        device = self.get_device_by_index(deviceIndex)
        self.deallocate_graph(device, graphName)


    def deallocate_graph(self, device, graphName):
        device.deallocate_graph(graphName)

    def load_graph(self, device, graphName, pathToGraph):
        device.allocate_graph(graphName, pathToGraph)

    def run_inference_device_index(self, deviceIndex, graphName, input):
        device = self.get_device_by_index(deviceIndex)

        return self.run_inference(device, graphName, input)

    def run_inference(self, device, graphName, input):
        graphclass = device.get_graph_by_name(graphName)

        assert(graphclass != None)

        try:
            if api_version == 2:
                graphclass.graph.queue_inference_with_fifo_elem(graphclass.fifoIn, graphclass.fifoOut, input, None)
                output, userObj = graphclass.fifoOut.read_elem()
            else:
                graphclass.graph.LoadTensor(input, None)
                output, userObj = graphclass.graph.GetResult()
        except Exception as error:
            print("Error qeueing or reading:", error)
            exit()

        return (output, userObj)

    def get_inference_time(self, device, graphName):
        start = time.perf_counter()
        res = 0.0
        try:
            time.sleep(0.001)
            if api_version == 2:
                times = device.get_graph_by_name(graphName).graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN)
            else:
                times = device.get_graph_by_name(graphName).graph.GetGraphOption(mvnc.mvncGraphOption.TIME_TAKEN)
        except Exception as error:
            print("Error reading time:", error)
            exit()
        for subtime in np.nditer(times):
            res += subtime
        end = time.perf_counter()
        return (res, end - start)

    def get_inference_time_device_index(self, deviceIndex, graphName):
        return self.get_inference_time(self.get_device_by_index(deviceIndex), graphName)

    def cleanup(self):
        for device in self.devices:
            device.cleanup()


##############################################################3
