from mvnc import mvncapi as mvnc
import numpy as np
import time

from mymov.Device import Device

def timer(fun):
    def wrapper(*argv, **kwargs):
        begin = time.time()
        rv = fun(*argv, **kwargs)
        end = time.time()
        return rv, (end - begin) * 1000.0
    return wrapper

class Movidius:
    def __init__(self):
        self.mvnc_api_version = None
        self.mvnc_device_handles = None
        self.devices = []

        try: # Try to use function from API v2
            mvnc.global_get_option(mvnc.GlobalOption.RO_LOG_LEVEL)
            self.mvnc_api_version = 2
        except:
            try: # Try to use function from API v1
                mvnc.GetGlobalOption(mvnc.mvncGlobalOption.LOG_LEVEL)
                self.mvnc_api_version = 1
            except:
                print("Neither API v1 or v2.")
        finally:
            print("Api version", self.mvnc_api_version)

        self.mvnc_device_handles = self.enumerate_devices()

        if len(self.mvnc_device_handles) == 0:
            print("No devices found")
            exit()


    def enumerate_devices(self):
        device_handles = None

        if self.mvnc_api_version == 1:
            device_handles = mvnc.EnumerateDevices()
        elif self.mvnc_api_version == 2:
            device_handles = mvnc.enumerate_devices()

        return device_handles


    def init_devices(self):
        for handle in self.mvnc_device_handles:
            device = Device(handle, self.mvnc_api_version)
            device.open_device()
            self.devices.append(device)


    def load_graph(self, device, graph_name, path_to_graph):
        device.allocate_graph(graph_name, path_to_graph)


    def load_graph_device_index(self, index, graph_name, path_to_graph):
        self.load_graph(self.devices[index], graph_name, path_to_graph)


    def deallocate_graph(self, device, graph_name):
        device.deallocate_graph(graph_name)


    def deallocate_graph_device_index(self, index, graph_name):
        self.deallocate_graph(self.devices[index], graph_name)


    def run_inference_device_index(self, index, graph_name, input):
        return self.run_inference(self.devices[index], graph_name, input)


    def run_inference(self, device, graph_name, input):
        graph = device.get_graph_by_name(graph_name)

        try:
            if self.mvnc_api_version == 1:
                graph.graph.LoadTensor(input, None)
                output, userObj = graph.graph.GetResult()
            elif self.mvnc_api_version == 2:
                graph.graph.queue_inference_with_fifo_elem(graph.fifoIn, graph.fifoOut, input, None)
                output, userObj = graph.fifoOut.read_elem()
        except Exception as error:
            print("Error qeueing or reading:", error)
            exit()

        return (output, userObj)


    def get_inference_time_device_index(self, index, graph_name):
        return self.get_inference_time(self.devices[index], graph_name)


    @timer
    def get_inference_time(self, device, graph_name):
        try:
            if self.mvnc_api_version == 1:
                times = device.get_graph_by_name(graph_name).graph.GetGraphOption(mvnc.mvncGraphOption.TIME_TAKEN)
            elif self.mvnc_api_version == 2:
                times = device.get_graph_by_name(graph_name).graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN)
        except Exception as error:
            print("Error reading time:", error)
            exit()

        return np.sum(times)


    def cleanup(self):
        for device in self.devices:
            device.cleanup()