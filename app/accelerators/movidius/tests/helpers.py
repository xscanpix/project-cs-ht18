import os
import numpy as np

def load_test_config(filepath):
    testconfig = {
        "iterations": None,
        "throwfirst": None,
        "runs": None,
        "tests": []
    }

    with open(filepath, 'r') as file:
        try:
            lines = file.readlines()
        except:
            print("Cannot load testconfig file: {}\nExiting...".format(filepath))
            exit()

    for line in lines:
        line = line.rstrip()
        if len(line) == 0:
            continue
        if line[0] == "!":
            values = line[1:].split(" ")
            
            if values[0] == 'iterations':
                testconfig['iterations'] = int(values[1])
            elif values[0] == 'throwfirst':
                testconfig['throwfirst'] = int(values[1])
            elif values[0] == "runs":
                testconfig['runs'] = int(values[1])
            else:
                print("Unknown parameter in test config {}".format(line))
                pass

        elif line[0] == "#":
            values = line[1:].replace(" ", "").split(",")
            test = {"layers": None, "neurons": None, "shaves": None}
            test["layers"] = int(values[0])
            test["neurons"] = int(values[1])

            if len(values) == 3:
                test["shaves"] = int(values[2])

            testconfig['tests'].append(test)

        else:
            pass
    
    return testconfig

def save_result(testconfig, test, times):
    y0 = times[0][testconfig['throwfirst']:]
    avg_0 = np.average(y0)
    std_0 = np.std(y0)
    mean_0 = np.median(y0)

    y1 = times[1][testconfig['throwfirst']:]
    avg_1 = np.average(y1)
    std_1 = np.std(y1)
    mean_1 = np.median(y1)

    y3 = np.divide(y1, y0)
    avg_2 = np.average(y3)
    std_2 = np.std(y3)
    mean_2 = np.median(y3)

    info = "_{}_{}_{}".format(test['layers'], test['neurons'], test['shaves']).split("_")[1:]

    if(not os.path.exists(os.environ["PROJ_DIR"]+"/tests/testresults/testdata_shave_{}.txt".format(info[2]))):
        writeflag = "w"
    else:
        writeflag = "a"
    with open(os.environ["PROJ_DIR"]+"/tests/testresults/testdata_shave_{}.txt".format(info[2]), writeflag) as file:
        file.write("!{}:{}:{}\n".format(info[0], info[1], info[2]))
        file.write("% total\n")
        file.write("{} {} {}\n".format(avg_0, std_0, mean_0))
        file.write("% movidius\n")
        file.write("{} {} {}\n".format(avg_1, std_1, mean_1))
        file.write("% percent\n")
        file.write("{} {} {}\n".format(avg_2, std_2, mean_2))
