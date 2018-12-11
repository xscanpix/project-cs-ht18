#!/usr/bin/python3
import os
import json

def load_settings(filepath):

    basepath = os.environ['PROJ_DIR']

    with open(filepath) as file:
        try:
            json_data = json.load(file)
        except:
            print("Cannot load settings file: {}\nExiting...".format(basepath + "/" + filepath))
            exit()

    kerasModelPath = json_data['kerasModelPath']
    tfOutputPath = json_data['tfOutputPath']
    ncsdkGraphPath = json_data['ncsdkGraphPath']
    outputDir = json_data['outputDir']

    json_data['outputDir'] = basepath + "/" + outputDir
    json_data['kerasModelPath'] = basepath + "/" + kerasModelPath
    json_data['tfOutputPath'] = basepath + "/" + tfOutputPath
    json_data['ncsdkGraphPath'] = basepath + "/" + ncsdkGraphPath

    return json_data