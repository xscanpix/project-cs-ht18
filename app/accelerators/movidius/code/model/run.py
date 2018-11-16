#!/usr/bin/python3
import sys, os

from helpers import load_settings
from code.model.keras2graph import load_model_keras, gen_new_model

def main():
    if(len(sys.argv) != 2):
        print("Usage: python3 run settings_file")
        print("Example: python3 run settings.json")
        exit()

    print("\n")
    print("[INFO] Loading settings from {}...".format(sys.argv[1]))
    jsonData = load_settings(sys.argv[1])
    print(jsonData)
    
    kerasModelPath = jsonData['kerasModelPath']
    tfOutputPath = jsonData['tfOutputPath']
    inputLayerName = jsonData['inputLayerName']
    outputLayerName = jsonData['outputLayerName']
    ncsdkGraphPath = jsonData['ncsdkGraphPath']

    print("[INFO] Loading model from keras file {}".format(kerasModelPath))
    model = load_model_keras(kerasModelPath)
    print("[INFO] Loaded model: ")
    model.summary()
    print("\n")

    print("[INFO] Generating new model without Dropout layer...")
    newModel = gen_new_model(model, tfOutputPath)
    print("[INFO] New model: ")
    newModel.summary()
    print("\n")

    print("Compiling with mvNNCompile... ")
    os.system("mvNCCompile {0}.meta -in {1} -on {2} -o {3}".format(tfOutputPath, inputLayerName, outputLayerName, ncsdkGraphPath))
    print("Graph generated at {}".format(ncsdkGraphPath))

if __name__ == '__main__':
    main()