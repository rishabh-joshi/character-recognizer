import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.append('../../../app')
import pytest
import numpy as np
import keras
import json
from __init__ import db
from create_db import Prediction
import yaml
import numpy as np

from helper import read_metadata, compress_image

# reading the metadata file
metadata_file_path = "../../metadata.yaml"
metadata = yaml.load(open(metadata_file_path))


def test_compress_image_type():
    # should return numpy ndarray
    
    # creating a test input
    base = 28
    size = 196
    test_input = ("0,"*size*size).strip(',')
    result = compress_image(test_input, size, base)
    assert isinstance(result, np.ndarray)


def test_read_metadata_IOError():
    # checking to see if the metadata file exists or not
    # if the file does not exist then the function will raise a SystemExit exception
    try:
        read_metadata(metadata_file_path)
        assert True
    except SystemExit:
        assert False