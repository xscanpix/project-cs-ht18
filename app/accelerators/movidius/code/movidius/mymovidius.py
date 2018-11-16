#!/usr/bin/python3
from mvnc import mvncapi as mvnc

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


    def get_graph(self):
        return self.graph


    def get_fifoIn(self):
        return self.fifoIn


    def get_fifoOut(self):
        return self.fifoOut


    def cleanup(self):
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


    '''
    Opens the device
    '''
    def open_device(self):
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
        for graph in self.graphs: 
            graph.cleanup()

        self.device.destroy()


class MyMovidius:
    '''
    Class for all Movidius devices
    Arguments:
        numDevicesToUse: number of devices to use
    '''
    def __init__(self, numDevicesToUse):
        self.devices = []

        devices = mvnc.enumerate_devices()
        if len(devices) == 0:
            print('No devices found')
            quit()

        numFound = len(devices)

        if(len(devices) < numDevicesToUse):
            print("Not enough devices found. Expected {}, found {}.".format(numDevices, numFound))
            print("Using found amount")


        # Create devices
        for device in devices:
            newDevice = MyDevice(device)
            self.devices.append(newDevice)

        
        # Open devices
        for device in self.devices:
            device.open_device()

    '''
    Returns the MyDevice object at deviceIndex
    '''
    def get_device_by_index(self, deviceIndex):
        if(deviceIndex < 0 or not deviceIndex < len(self.devices)):
            print("Device index is wrong. Exiting...")
            self.cleanup()
            exit()

        return self.devices[deviceIndex]


    '''
    Loads a graph onto the device at deviceIndex
    '''
    def load_graph_device_index(self, deviceIndex, graphName, pathToGraph):
        device = self.get_device(deviceIndex)

        self.load_graph(device, graphName, pathToGraph)


    '''
    Loads a graph onto the MyDevice object
    '''
    def load_graph(self, device, graphName, pathToGraph):
        device.load_graph(graphName, pathToGraph)


    def run_inference_device_index(self, deviceIndex, graphName, input):
        device = self.get_device(deviceIndex)

        return self.run_inference(device, graphName, input)


    '''
    Runs inference with input on the MyDevice object
    '''
    def run_inference(self, device, graphName, input):
        graphclass = device.get_graph_by_name(graphName)
        graphclass.get_graph().queue_inference_with_fifo_elem(graphclass.get_fifoIn(), graphclass.get_fifoOut(), input, 'user object')
        output, userObj = graphclass.get_fifoOut().read_elem()

        return (output, userObj)


    '''
    Cleans up all devices
    '''
    def cleanup(self):
        for device in self.devices:
            device.cleanup()

##############################################################3
