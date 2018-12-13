from mvnc import mvncapi as mvnc

from mymov.Graph import Graph

class Device:
    def __init__(self, device_handle, mvnc_api_version):
        assert(mvnc_api_version in [1, 2])
        self.mvnc_api_version = mvnc_api_version

        try:
            self.mvnc_device = mvnc.Device(device_handle)
        except Exception as error:
            print("Error creating mvnc device:", error)
            exit()
        self.graphs = []

    def open_device(self):
        try:
            if self.mvnc_api_version == 1:
                self.mvnc_device.OpenDevice()
            elif self.mvnc_api_version == 2:
                self.mvnc_device.open()
        except Exception as error:
            print("Error opening mvnc device:", error)
            exit()

    def allocate_graph(self, graph_name, path_to_graph):
        graph = Graph(self.mvnc_device, graph_name, path_to_graph, self.mvnc_api_version)
        self.graphs.append(graph)

        if self.mvnc_api_version == 1:
            return graph.graph
        elif self.mvnc_api_version == 2:
            return graph.graph, graph.fifo_in, graph.fifo_out

    def deallocate_graph(self, graph_name):
        for key, graph in enumerate(self.graphs):
            if(graph.graph_name == graph_name):
                graph.cleanup()
                del self.graphs[key]

    def get_graph_by_name(self, graph_name):
        rv = None

        for graph in self.graphs:
            if(graph.graph_name == graph_name):
                rv = graph

        return rv

    def cleanup(self):
        for graph in self.graphs: 
            graph.cleanup()
        
        try:
            if self.mvnc_api_version == 1:
                self.mvnc_device.CloseDevice()
            elif self.mvnc_api_version == 2:
                self.mvnc_device.close()
                self.mvnc_device.destroy()
        except Exception as error:
            print("Error closing or destroying mvnc device:", error)
            exit()