import sys

max_val = 0.0

def print_info():
    print("info to be displayed")

def open_file_stream(file_name):
    return open(file_name, "r")

def clean_line(line):

    line_c = line.replace("[", "").replace("]", "").replace("  ", " ").replace("  ", " ").replace("\n", "")
    line_c = line_c.split(" ")
    line_c = filter(None, line_c)
    #print(line_c)
    return line_c

def compare_line(line1, line2):
    global max_val
    line1_c = clean_line(line1)
    line2_c = clean_line(line2)
    new_line = ""
    for i in range(0, 5):
	fval = float(line1_c[i]) - float(line2_c[i])
	if(abs(fval) > max_val):
		max_val = abs(fval)

        new_line += str(fval) + " "

    return new_line

def compare_files(file_stream_1, file_stream_2):

    line_file_1 = file_stream_1.readline()
    line_file_2 = file_stream_2.readline()
    compared_lines = ""
    while (line_file_1 != "" and line_file_2 != ""):
        compared_lines += "["+ compare_line(line_file_1, line_file_2) +"] \n"

        line_file_1 = file_stream_1.readline()
        line_file_2 = file_stream_2.readline()

    return compared_lines

def main():
    #print (sys.argv)

    if len(sys.argv) != 3:
        print_info()
        return

    file_1 = sys.argv[1]
    file_2 = sys.argv[2]

    file_stream_1 = open_file_stream(file_1)
    file_stream_2 = open_file_stream(file_2)

    files_comped = compare_files(file_stream_1, file_stream_2)

    print(files_comped)
    print("\n val max diff: " + str(max_val))
	


if __name__ == "__main__":
    main()
