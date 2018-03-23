import keras
import json
from __init__ import db
from create_db import Prediction
import yaml
import numpy as np


def read_metadata(metadata_file_path):
    """Read the YAML file containing metadata

    Args:
        metadata_file_path (str): The path of the file containing the training data.

    Returns:
        File Object: File object of the YAML metadata file.

    """
    # reading and loading metadata from YAML file
    try:
        with open(metadata_file_path, 'r') as metadata_file:
            metadata = yaml.load(metadata_file)
    except IOError:
        raise SystemExit("The file you are trying to read does not exist: " + metadata_file_path)

    # returning the metadata dictionary
    return metadata


def compress_image(pixel_list, size, base):
    """Convert a list of pixels into a size x size image and compress it into a base x base image.

    Args:
        pixel_list (str): A comma separated string containing pixel values.
        size (int): Height and width of the image corresponding to the pixel list.
        base (int): Height and width of the compressed image.

    Returns:
        numpy.ndarray: A base x base 2d numpy array corresponding to the compressed image.

    """
    # parsing the pixels into a size x size matrix
    large_image = np.array([float(elem) for elem in pixel_list.split(',')]).reshape(size, size)

    # creating a base x base matrix to store the compressed image
    small_image = np.zeros(base * base).reshape(base, base)

    # factor by which to shrink the image
    factor = int(size / base)

    # iterating over the large image to compress it 
    for i in range(base):
        for j in range(base):
            # taking mean of only the positive values
            small_image[i, j] = large_image[factor * i:factor * (i + 1), factor * j:factor * (j + 1)][np.where(
                large_image[factor * i:factor * (i + 1), factor * j:factor * (j + 1)] > 0)].mean()

    # eliminating the nan values and reshaping the matrix
    small_image[np.isnan(small_image)] = 0.0
    small_image = small_image.reshape(base * base)
    return small_image


def predict_character(model_path, image):
    """Predict the character drawn in a 28x28 grayscale image.

    Args:
        model_path (str): Path to the model file used to load the keras model.
        image (numpy.ndarray): A 28 x 28 numpy array with the grayscale pixel values of the image.

    Returns:
        str: The predicted character or the string "Failed to recognize!" if no match found.

    """
    # loading the CNN model saved to disk
    model = keras.models.load_model(model_path)

    # number of classes in the prediction model
    num_classes = 62

    # transforming/normalizing the input to be acceptable by the model
    inp = np.array(image).reshape(1, 28, 28, 1) / 255

    # predicting on the input
    prediction = model.predict(inp)
    prediction = str(prediction.reshape(num_classes).argsort()[-1])
    labels = json.load(open("app/labels.json"))
    prediction = labels[prediction]
    return prediction


def store_prediction(prediction):
    """Store the predicted character in the database.

    Args:
        prediction (str): The predicted character to be stored in the database.

    """
    # saving the prediction into RDS
    instance = Prediction(character=prediction)
    db.session.add(instance)
    db.session.commit()