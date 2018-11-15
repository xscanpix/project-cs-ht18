#!/usr/bin/python3
import os, json

def get_extension(filepath):
    filename, file_extension = os.path.splitext(filepath)

    return file_extension


def load_settings(filepath):

    basepath = os.path.abspath(".") + "/"

    with open(filepath) as file:
        try:
            json_data = json.load(file)
        except:
            print("Cannot load settings file: {}\nExiting...".format(filepath))
            exit()

    keras_model_path = json_data['keras_model_path']
    tf_output_path = json_data['tf_output_path']
    input_layer_name = json_data['input_layer_name']
    output_layer_name = json_data['output_layer_name']
    ncsdk_graph_path = json_data['ncsdk_graph_path']

    json_data['keras_model_path'] = basepath + keras_model_path
    json_data['tf_output_path'] = basepath + tf_output_path
    json_data['ncsdk_graph_path'] = basepath + ncsdk_graph_path

    return json_data