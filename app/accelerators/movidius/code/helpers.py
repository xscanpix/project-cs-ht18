#!/usr/bin/python3

import os

# Get the extension of a file
def get_extension(filepath):
    filename, file_extension = os.path.splitext(filepath)

    return file_extension

