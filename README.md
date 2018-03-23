# Character Recognizer

Character recognition project for the course MSiA423: Analytics Value Chain.

- ### Developer: Rishabh Joshi 

- Product Owner: Lauren Gardiner

- Quality Assurance: Vincent Wang

## Project Charter

Predict the character drawn by the user on the web app by using a neural network model based on the grayscale pixel values of the image to assist in building a handwriting input keyboard system that can be employed on different smartphones.

### Vision 

Assist in building a handwriting input keyboard system that can be employed on different smartphones.

### Mission 

Predict the character drawn by the user on the web app by using a neural network model based on the grayscale pixel values of the image.

### Success criteria 

Accuracy of classification more than 85% for the neural network on a pre-determined test set of grayscale character images.

## Overall Approach

Optical Character Recognition (OCR) is a field of research in pattern recognition, computer vision and artificial intelligence. It is used to capture texts from scanned documents or photos. The Extended Modified NIST (EMNIST) dataset, derived from NIST Special Database 19 is a set of handwritten character digits of [0-9], [a-z] and [A-Z]. Images of those handwritten characters are converted to 28x28 pixels.

We started by building a simple neural network model with a single hidden layers and more than a thousand hidden nodes. The accuracy for this model was only 15.84%. Simple neural networks tend not to perform that well on image data becuase they treat the information in all the pixels independently, not accounting for their relative position in the image. This becomes the core idea behind convolutional neural networks because they try to create features out of pixels that are present close by (for example, a vertical/horizontal line, a curve, etc). The accuracy from the convolutional neural network was 88.39%, which passed the success criteria.

## Getting Started

### Dependencies

Dependencies are listed in [requirements.txt](requirements.txt)

### Documentation

This project was documented using Sphinx and the documentation files can be found in the [develop/docs/\_build/html](develop/docs/_build/html) folder.

### Repository Structure

The two main folders in this repository are [app](app) and [develop](develop) which contains the code to deploy the app and the code for model development respectively.

### Steps to Deploy Application

1. Install python 3.6
```
    sudo yum install python36
```
2. Create a virtual environment.
```
    virtualenv -p python3 turnover
    source turnover/bin/activate
```
3. Install Git.
```
    sudo yum install git
```
4. Clone the repository
```
    git clone https://github.com/rishabh-joshi/character-recognizer.git
```
5. Change directory
```
    cd character-recognizer
```
6. Create a file "config" in the app folder which stores database credentials as described below.
```
    SQLALCHEMY_DATABASE_URI = 'postgresql://<db_user>:<db_password>@<endpoint>/<db_url>'
    SQLALCHEMY_TRACK_MODIFICATIONS = True
```
7. Export environment variables from the config file.
```
    export APP_SETTINGS="config"
```
8. Download the following csv data files from Kaggle and extract them into the [develop/data](develop/data) folder.
    - [EMNIST Balanced Train Data](https://www.kaggle.com/crawford/emnist/downloads/emnist-balanced-train.csv/3)
    - [EMNIST Balanced Test Data](https://www.kaggle.com/crawford/emnist/downloads/emnist-balanced-test.csv/3)
    - [EMNIST By Class Train Data](https://www.kaggle.com/crawford/emnist/downloads/emnist-byclass-train.csv/3)
    - [EMNIST By Class Test Data](https://www.kaggle.com/crawford/emnist/downloads/emnist-byclass-test.csv/3)
9. [OPTIONAL STEP] Run the unit tests to ensure that there are no bugs in the model development and deployment code. WARNING: The tests take a long time to run because they have to build the model again to check if the model performs as expected. It is recommended to not perform testing if the model files have not been altered.
```
    cd develop/src/tests
    pytest
    cd ../../..
```
10. There are four convolutional neural network models to choose from for the prediction of characters. These are called `sparse_cnn_balanced`, `dense_cnn_balanced`, `sparse_cnn_byclass`, and `dense_cnn_byclass`. Because these models take a long time to train, they have been trained and provided as h5 files in the [develop/models](develop/models) directory. They can be read in through Keras.
11. [OPTIONAL STEP] The best performing model from the above four models is `sparse_cnn_byclass` which is chosen by default. If the models are to be retrained, modify the [develop/metadata.yaml](develop/metadata.yaml) with the name of the model to be retrained as follows. Replace the name of the model and the corresponding data file in the YAML file while preserving the extension of the files.
```
    train_file : emnist-byclass-train.csv
    test_file : emnist-byclass-test.csv
    model_name : sparse_cnn_byclass.h5
```
12. [OPTIONAL STEP] For every model that needs retraining, run the following make command by appropriately changing the model name.
```
    make sparse_cnn_byclass
```
13. Finally, once the models have been fitted, specify the model that should be used for prediction purposes in the [develop/metadata.yaml](develop/metadata.yaml) file by changing the name of the h5 file in which the model is saved with the model you want to use.
```
    model_name : sparse_cnn_byclass.h5
```
14. We are now in a position to deploy the app.
```
    python36 app/application.py
```
15. Check out the app by visiting the IP address that appeared on the consol after the app has been successfully deployed.


## Pivotal Tracker Project URL

[https://www.pivotaltracker.com/n/projects/2142491](https://www.pivotaltracker.com/n/projects/2142491)
