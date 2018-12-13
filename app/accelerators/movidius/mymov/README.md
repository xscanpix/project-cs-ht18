./usbreset $(lsusb | grep 03e7 | awk '{print "/dev/bus/usb/" $2 "/" substr($4,1,3)}')

# Usage guide

## Preparations

### Step 1.
Install python 3 (latest)
sudo apt-get install python3


### Step 2.
Install python requirements.
sudo apt-get install python3-pip
sudo -H pip3 install -r requirements.txt


### Step 3.
Clone the repoository ncsdk v2 from https://github.com/movidius/ncsdk/tree/ncsdk2
git clone -b ncsdk2 https://github.com/movidius/ncsdk.git

Install ncsdk v2 by following their installation guide

## API Usage

### Step 1.

Edit the settings.json file
- "outputDir" 
General directory for where generated files will be stored

- "kerasModelPath"
Path to the keras model that will be converted
Only use this if you know that the model is supported by the movidius compiler

- "tfOutputPath"
Intermediate directory where the TensorFlow model files are stored
Specify the base name of the model files, 
they are generated as:
    {base name}.meta
    {base name}.index
    {base name}.data-00000-of-00001

- "inputLayerName"
The name of the input layer

- "outputLayerName"
The name of the output layer

- "ncsdkGraphPath"
Path to the compiled movidius graph file (after beng generated)

- "graphName"
Name of the graph to allocated on the device

- "defaultDeviceIndex"
Which device should be used (default 0)

- "numDevices"
Num devices

Example settings.json:
{
    "outputDir": "outputs", 
    "kerasModelPath": "resources/model/model.h5",
    "tfOutputPath": "outputs/tf_model/model",
    "inputLayerName": "input_input",
    "outputLayerName": "output/Identity",
    "ncsdkGraphPath": "outputs/graph",
    "graphName": "graph",
    "defaultDeviceIndex": 0,
    "numDevices": 1
}


### Step 2.

Generate the tensorflow model files if not done already
Make sure the layers are supported by the ncsdk v2

Construct the model and set the weights manually as seen in keras2graph.py/gen_model

(TODO: Support keras model files)


### Step 3.

Generate movidius graph

either run the mvNCCompile command manually or by using the run.py file


### Step 4.

Use the mymovidius.py module to setup the movidius environment and run inference

Example:

Init
mymov = MyMovidius()
mymov.init_devices(1)
mymov.load_graph_device_index(0, "graph", "graphs/graph")

Run inferences
input = numpy.random.uniform(0,1,28).reshape(1,28)
(result, myobject) = mymov.run_inference_device_index(0, "graph", input)

Cleanup
mymov.cleanup()
