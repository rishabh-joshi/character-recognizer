from flask import Flask, redirect, url_for, request, render_template
from __init__ import app, db
from create_db import Prediction
from showtable import getHtmlTable

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

labels = json.load(open("app/labels.json"))


@app.route('/', methods=['GET'])
def home_page():
    # redirect("127.0.0.1/5000")
    return render_template('index.html', output="")


@app.route('/recognize', methods=['POST'])
def recognize():
    """View that process a POST with new song input

    :return: redirect to index page
    """

    size = 196
    base = 28
    num_classes = 62
    factor = int(size/base)
    # list = request.args.get('hiddenBox')
    list = request.form["hiddenBox"]
    large_image = np.array([float(elem) for elem in list.split(',')]).reshape(size,size)
    small_image = np.zeros(base*base).reshape(base, base)

    for i in range(base):
        for j in range(base):
            print(i,j)
            # mean of the positive values
            small_image[i,j] = large_image[factor*i:factor*(i+1), factor*j:factor*(j+1)][np.where(large_image[factor*i:factor*(i+1), factor*j:factor*(j+1)]>0)].mean()
            
            # 255 if any value is greater than 0
            # small_image[i,j] = 255 if large_image[factor*i:factor*(i+1), factor*j:factor*(j+1)].sum()>0 else 0



    small_image[np.isnan(small_image)] = 0.0
    small_image = small_image.reshape(base*base)


    new_model = load_model('develop/models/sparse_cnn_byclass.h5') 
    input = np.array(small_image).reshape(1,28,28,1)/255
    prediction = new_model.predict(input)
    prediction = str(prediction.reshape(num_classes).argsort()[-1])
    prediction = labels[prediction]
    if prediction == 'null' : 
        output = "Failed to recognize, try again!"
    else:
        output = "Predicted " + prediction + "!"

        # # Uncomment the following lines for saving the prediction results in RDS
        character = prediction
        instance = Prediction(character=character)
        db.session.add(instance)
        db.session.commit()
    
    # return redirect(request.path,code=302)
    
    return render_template('recognize.html', output = output)


@app.route('/history',methods=["GET"])
def show_history():
    db.session.commit()
    results = db.session.execute('SELECT * FROM PREDICTION;')
    table = getHtmlTable(results)
    return render_template("history.html",table=table)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000, debug=True)
    # app.run(debug = True)