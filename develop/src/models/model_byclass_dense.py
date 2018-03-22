import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import yaml


def read_transform_data(train_file, test_file):
    """Read in the training and test data and reshape them for the neural network model.

    Args:
        train_file (str): The path of the file containing the training data.
        test_file (str): The path of the file containing the testing data.

    Returns:
        tuple: A 4-tuple of numpy arrays containing training predictors and response and testing predictors and response respectively.

    """
    logging.info('Executing function read_transform_data().')

    # reading the train and test data
    logging.info('Reading training and testing data.')
    try:
        with open(train_file, 'r') as f:
            train = pd.read_csv(train_file, header=None)
    except IOError:
        raise SystemExit('The file you are trying to read does not exist: {0}'.format(train_file))

    try:
        with open(test_file, 'r') as f:
            test = pd.read_csv(test_file, header=None)
    except IOError:
        raise SystemExit('The file you are trying to read does not exist: {0}'.format(test_file))

    # separate out the train data from the response variable
    logging.info('Separating out the train data from the response variable.')
    x_train = train.iloc[:, 1:]
    x_test = test.iloc[:, 1:]

    # separate out the response variable from the data
    logging.info('Separating out the response variable from the data.')
    y_train = train[0]
    y_test = test[0]

    # converting the pandas data frame to numpy matrices
    logging.info('Converting the pandas data frame to numpy matrices.')
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values

    # reshaping the data into the format which can be passed to the neural network
    logging.info('Reshaping the data into the format which can be passed to the neural network.')
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # converting the data type to float32
    logging.info('Converting the data type to float32.')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalizing the predictors
    logging.info('Normalizing the predictors.')
    x_train /= 255
    x_test /= 255

    # Defining the number of classes in the response variable
    logging.info('Defining the number of classes in the response variable.')
    number_of_classes = 62

    # One hot encoding the response variable
    logging.info('One hot encoding the response variable.')
    y_train = to_categorical(y_train, number_of_classes)
    y_test = to_categorical(y_test, number_of_classes)

    # Returning the training and testing predictors and response. Function succesfully completed
    logging.info('Returning the training and testing predictors and response.'
                 'Function read_transform_data() succesfully completed.')

    return x_train, y_train, x_test, y_test


def build_convnet():
    """Read in the training and test data and reshape them for the neural network model.

    Returns:
        Keras Sequential Model: A convolutional neural network model.

    """
    logging.info('Executing function build_convnet().')

    # Defining the number of classes in the response variable
    logging.info('Defining the number of classes in the response variable.')
    number_of_classes = 62

    # Adding the first set of convolutional and pooling layers with ReLu activation
    logging.info('Adding the first set of convolutional and pooling layers with ReLu activation.')
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding the second set of convolutional and pooling layers with ReLu activation
    logging.info('Adding the second set of convolutional and pooling layers with ReLu activation.')
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # Adding fully connected layers with softmax activation and 20% dropout
    logging.info('Adding fully connected layers with softmax activation and 0.20 dropout.')
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))

    # Compiling the model with categorical crossentropy loss function to handle multiple classes
    logging.info('Compiling the model with categorical crossentropy loss function to handle multiple classes.')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Returning the completed model
    logging.info('Returning the completed model. Function build_convnet() executed successfully.')
    return model


def train_test(model, x_train, y_train, x_test, y_test, model_name):
    """Fit the convolutional neural network model on the training data, evaluate on the test data, and save it to disk.

    Args:
        model (Keras model): The path of the file containing the training data.
        x_train (ndarray): Numpy array with shape (n,28,28,1) containing the predictors for n images in the train data.
        y_train (ndarray): Numpy array with shape (n,) containing the response classes for n images in the train data.
        x_test (ndarray): Numpy array with shape (n,28,28,1) containing the predictors for n images in the test data.
        y_test (ndarray): Numpy array with shape (n,) containing the response classes for n images in the train data.
        model_name (str): The filename to save the keras model to.

    """
    logging.info('Executing function train_test().')

    # Setting the batch size and number of epochs
    logging.info('Setting the batch size and number of epochs.')
    batch_size = 256
    epochs = 10

    # Training the model on the train data
    logging.info('Training the model on the train data.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    # Evaluating the model on the test data
    logging.info('Evaluating the model on the test data.')
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Saving the model to disk
    logging.info('Saving the model to disk.')
    model.save('develop/models/' + model_name)
    logging.info('Function train_test() executed succesfully.')


def main(metadata):
    """Read the train and test data, build, train, evaluate and save the conv net.

    Args:
        metadata (file object): The file object of the metadata file.

    """
    logging.info('Executing function main().')

    # train and test file locations
    train_file = 'develop/data/' + metadata['train_file']
    test_file = 'develop/data/' + metadata['test_file']

    # defining the model name
    model_name = metadata['model_name']

    # creating the training and testing set
    x_train, y_train, x_test, y_test = read_transform_data(train_file, test_file)

    # building the model
    model = build_convnet()

    # training, evaluating, and saving the model
    train_test(model, x_train, y_train, x_test, y_test, model_name)
    logging.info('Function main() executed successfully.')


def read_metadata(metadata_file_path):
    """Read the YAML file containing metadata

    Args:
        metadata_file_path (str): The path of the file containing the training data.

    Returns:
        File Object: File object of the YAML metadata file.

    """
    logging.info('Executing function read_metadata().')

    # reading and loading metadata from yaml file
    try:
        with open(metadata_file_path, 'r') as metadata_file:
            metadata = yaml.load(metadata_file)
    except IOError:
        raise SystemExit("The file you are trying to read does not exist: " + metadata_file_path)

    # returning the metadata dictionary
    logging.info("Function read_metadata() executed successfully.")
    return metadata


if __name__ == '__main__':
    # setting up the logging file and formats
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    date_fmt = '%m/%d/%Y %I:%M:%S %p'
    logging.basicConfig(filename='develop/logs/model_byclass_dense.log', filemode='w', level=logging.DEBUG,
                        format=log_fmt, datefmt=date_fmt)

    # defining the metadata file path
    metadata_file_path = "develop/metadata.yaml"

    # reading the metadata file
    metadata = read_metadata(metadata_file_path)

    # calling the main function
    main(metadata)