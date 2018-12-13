from mvnc import mvncapi as mvnc

class Graph:
    def __init__(self, device, graph_name, path_to_graph, api_version):
        self.api_version = api_version
        self.graph_name = None
        self.graph = None
        self.fifo_in = None
        self.fifo_out = None

        with open(path_to_graph, mode="rb") as file:
            graph_file_buff = file.read()

        self.graph_name = graph_name
        try:
            if self.api_version == 1:
                self.graph = device.AllocateGraph(graph_file_buff)
            elif self.api_version == 2:
                self.graph = mvnc.Graph(self.graph_name)
                self.fifo_in, self.fifo_out = self.graph.allocate_with_fifos(device, graph_file_buff)
        except Exception as error:
            print("Error creating graph or fifo:", error)
            exit()


    def cleanup(self):
        try:
            if self.api_version == 1:
                self.graph.DeallocateGraph()
            elif self.api_version == 2:
                self.fifo_in.destroy()
                self.fifo_out.destroy()
                self.graph.destroy()
        except Exception as error:
            print("Error creating destroying graph or fifos:", error)
            exit()
