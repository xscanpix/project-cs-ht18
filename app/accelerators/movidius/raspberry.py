import numpy as np
import os

import mymov.mymovidius
from mymov.mymovidius import MyMovidius
from mymov.helpers import load_settings, prepare_keras_model, compile_tf, gen_model


from mymov.Movidius import Movidius

if __name__ == '__main__':
    os.environ['PROJ_DIR'] = os.getcwd()

    try:
        jsonData = load_settings("settings.json")
    except Exception as error:
        print("Error loading file:", error)
        exit()

    input = np.random.uniform(0,1,28).reshape(1,28).astype(np.float16)

    mov = Movidius()
    mov.init_devices()
    model = prepare_keras_model(jsonData)
    print("Predicted TF:", model.predict(input))
    compile_tf(jsonData)
    mov.load_graph_device_index(0, jsonData['graphName'], jsonData['ncsdkGraphPath'])

    ## Inference

    result, user_obj = mov.run_inference_device_index(0, jsonData['graphName'], input)

    print("Movidius:", result)

    mov.cleanup()
