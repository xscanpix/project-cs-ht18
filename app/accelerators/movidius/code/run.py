#!/usr/bin/python3
import sys
import numpy as np

from helpers import load_settings
from movidius.mymovidius import MyMovidius

def main():
    if(len(sys.argv) != 2):
        print("Usage: python3 run_model.py settings_file")
        print("Example: python3 run_model.py settings.json")
        exit()

    print("[INFO]: Loading settings...")
    jsonData = load_settings(sys.argv[1])
    ncsdkGraphPath = jsonData['ncsdkGraphPath']
    defaultDeviceIndex = jsonData['defaultDeviceIndex']
    numDevices = jsonData['numDevices']
    print("[INFO]: Settings loaded.")

    print("[INFO]: Settings up devices...")
    myMovidius = MyMovidius(numDevices)
    myDevice = myMovidius.get_device_by_index(defaultDeviceIndex)
    
    myMovidius.load_graph(myDevice, "graph1", ncsdkGraphPath)

    input = np.random.uniform(0,1,28).reshape(1,28).astype(np.float32)
    print("[INFO]: Running test on Tensor:\n{}".format(input))
    (output, userObj) = myMovidius.run_inference(myDevice, "graph1", input)
    print("[INFO]: Test done. Output: {}".format(output))

    print("[INFO]: Cleaning up...")
    myMovidius.cleanup()


if __name__ == '__main__':
    main()