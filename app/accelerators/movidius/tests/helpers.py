import os

def load_test_config(filepath):
    assert(os.path.exists(filepath))

    testconfig = {
        "iterations": None,
        "savegraphs": None,
        "smoothing": None,
        "throwfirst": None,
        "runs": None,
        "tests": []
    }

    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.rstrip()
            if len(line) == 0:
                continue
            if line[0] == "!":
                values = line[1:].split(" ")
                
                if values[0] == 'iterations':
                    testconfig['iterations'] = int(values[1])
                elif values[0] == 'savegraphs':
                    testconfig['savegraphs'] = bool(values[1])
                elif values[0] == 'smoothing':
                    testconfig['smoothing'] = int(values[1])
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