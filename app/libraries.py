from flask import Flask, redirect, url_for, request, render_template, jsonify
import random
import json
import re

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix

labels = json.load(open("labels.json"))