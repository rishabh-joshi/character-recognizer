import pytest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath('.'))
sys.path.append('../../..')
from app.helper import *

# reading the metadata file
metadata_file_path = "../../metadata.yaml"
metadata = yaml.load(open(metadata_file_path))


def test_compress_image_type():
    # should return numpy ndarray
    
    # creating a test input
    base = 28
    size = 196
    test_input = ("0,"*size*size).strip(',')
    result = compress_image(pixel_list, size, base)
    assert isinstance(result, np.ndarray)


def test_read_metadata_IOError():
    # checking to see if the metadata file exists or not
    # if the file does not exist then the function will raise a SystemExit exception
    try:
        read_metadata(metadata_file_path)
        assert True
    except SystemExit:
        assert False