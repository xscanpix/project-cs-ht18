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
    json_data = load_settings(sys.argv[1])
    print(json_data)
    
    path_to_model = json_data['keras_model_path']
    path_to_tf_model = json_data['tf_output_path']
    input_name = json_data['input_layer_name']
    output_name = json_data['output_layer_name']
    path_to_graph = json_data['ncsdk_graph_path']

    print("[INFO] Loading model from keras file {}".format(path_to_model))
    model = load_model_keras(path_to_model)
    print("[INFO] Loaded model: ")
    model.summary()
    print("\n")

    print("[INFO] Generating new model without Dropout layer...")
    newmodel = gen_new_model(model, path_to_tf_model)
    print("[INFO] New model: ")
    newmodel.summary()
    print("\n")

    print("Compiling with mvNNCompile... ")
    os.system("mvNCCompile {0}.meta -in {1} -on {2} -o {3}".format(path_to_tf_model, input_name, output_name, path_to_graph))
    print("Graph generated at {}".format(path_to_graph))

if __name__ == '__main__':
    main()